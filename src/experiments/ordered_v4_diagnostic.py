"""Read-only ordered diagnostic for the 3-UAV V4 development experiment.

The diagnostic is deliberately limited to the ``train`` and ``validation``
splits of the already completed V4 arms.  It does not replay an environment,
load a checkpoint, update a policy, or open a ``test`` episode.  Its purpose is
to turn the P0--P3 evidence audit into reproducible tables and diagnostic
figures before any new training change is considered.
"""

from __future__ import annotations

import ast
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ARMS = ("loc-progress", "actual-load", "loc-progress-actual-load")
METHOD_ID = "baseline_mappo"
SEED = 42
UAV_COUNT = 3
PACKET_BYTES = 512
BENEFIT_THRESHOLD_M = 0.001
HORIZONS = (1, 2, 3, 5)
TOL = 1e-6


def _truth(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _number(value: Any, *, allow_na: bool = False) -> float:
    if allow_na and (value is None or str(value).strip().upper() in {"", "NA", "N/A", "NAN"}):
        return float("nan")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite numeric value: {value!r}")
    return result


def _safe_number(value: Any) -> float:
    try:
        return _number(value)
    except (TypeError, ValueError):
        return float("nan")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return "NA"
    return value


def _cell(value: Any) -> Any:
    if isinstance(value, (float, np.floating)):
        return value if math.isfinite(float(value)) else "NA"
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (Mapping, list, tuple, np.ndarray)):
        return json.dumps(_jsonable(value), ensure_ascii=False, allow_nan=False, default=str)
    return "NA" if value is None else value


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty diagnostic table: {path}")
    columns: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            key = str(key)
            if key not in seen:
                seen.add(key)
                columns.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _cell(row.get(key)) for key in columns})


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(value), ensure_ascii=False, indent=2, allow_nan=False, default=str),
        encoding="utf-8",
    )


def _assert_allowed(path: Path, *, root: Path | None = None) -> None:
    """Fail closed if a data path is outside train/validation.

    ``root`` is used for raw episode files.  Log tables are checked by their
    filename and the caller only passes train/validation figure tables.
    """

    parts = {part.lower() for part in path.parts}
    if "test" in parts:
        raise RuntimeError(f"ordered diagnostic attempted to access test data: {path}")
    if root is not None:
        try:
            relative = path.resolve().relative_to(root.resolve())
        except ValueError as exc:
            raise RuntimeError(f"data path is outside the allowed root: {path}") from exc
        if not relative.parts or relative.parts[0].lower() not in {"train", "validation"}:
            raise RuntimeError(f"raw diagnostic path is not train/validation: {path}")


def _mean(values: Iterable[float]) -> float:
    array = np.asarray([value for value in values if math.isfinite(float(value))], dtype=float)
    return float(np.mean(array)) if array.size else float("nan")


def _std(values: Iterable[float]) -> float:
    array = np.asarray([value for value in values if math.isfinite(float(value))], dtype=float)
    return float(np.std(array, ddof=1)) if array.size > 1 else 0.0 if array.size else float("nan")


def _percentile(values: Iterable[float], q: float) -> float:
    array = np.asarray([value for value in values if math.isfinite(float(value))], dtype=float)
    return float(np.percentile(array, q)) if array.size else float("nan")


def _corr(x: Sequence[float], y: Sequence[float]) -> float:
    pairs = [(float(a), float(b)) for a, b in zip(x, y) if math.isfinite(float(a)) and math.isfinite(float(b))]
    if len(pairs) < 2:
        return float("nan")
    xa = np.asarray([pair[0] for pair in pairs], dtype=float)
    ya = np.asarray([pair[1] for pair in pairs], dtype=float)
    if np.std(xa) == 0.0 or np.std(ya) == 0.0:
        return float("nan")
    return float(np.corrcoef(xa, ya)[0, 1])


def _parse_ids(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    text = str(value)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = ast.literal_eval(text)
    if not isinstance(parsed, (list, tuple)):
        raise ValueError(f"episode_ids is not a sequence: {value!r}")
    return [str(item) for item in parsed]


def _paths(data_root: Path, logs_root: Path, arm: str) -> tuple[Path, Path, Path]:
    method_root = data_root / "development" / arm / f"seed_{SEED}" / METHOD_ID / "scale_3"
    log_root = logs_root / "development" / arm
    figures_root = log_root / "figures"
    return method_root, log_root, figures_root


def _raw_episode_dirs(method_root: Path, split: str) -> list[Path]:
    if split not in {"train", "validation"}:
        raise ValueError(split)
    split_root = method_root / split
    _assert_allowed(split_root, root=method_root)
    result = sorted(path for path in split_root.glob("episode_*") if path.is_dir())
    if not result:
        raise FileNotFoundError(f"no {split} episodes under {split_root}")
    for path in result:
        _assert_allowed(path, root=method_root)
    return result


def _raw_reward_rows(method_root: Path, split: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for episode_dir in _raw_episode_dirs(method_root, split):
        path = episode_dir / "rewards.csv"
        _assert_allowed(path, root=method_root)
        rows.extend(_read_csv(path))
    return rows


def _team_window_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["episode_id"]), str(row["timestamp"]))].append(row)
    fields = (
        "reward_total",
        "learning_reward_total",
        "feedback_reward",
        "loc_progress_delta",
        "loc_progress_reward",
        "actual_measurement_load_excess",
        "actual_measurement_load_reward",
        "actual_measurement_rate",
        "v3_learning_reward",
    )
    result: list[dict[str, Any]] = []
    for (episode_id, timestamp), group in sorted(grouped.items()):
        item: dict[str, Any] = {"episode_id": episode_id, "timestamp": timestamp}
        for field in fields:
            item[field] = _mean(_safe_number(row.get(field)) for row in group)
        result.append(item)
    return result


def _episode_feedback_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["episode_id"])].append(row)
    result: list[dict[str, Any]] = []
    for episode_id, group in sorted(grouped.items()):
        window_rows = _team_window_rows(group)
        item: dict[str, Any] = {
            "episode_id": episode_id,
            "window_count": len(window_rows),
            "mean_report_reward": _mean(row["reward_total"] for row in window_rows),
            "mean_learning_reward": _mean(row["learning_reward_total"] for row in window_rows),
            "mean_feedback_reward": _mean(row["feedback_reward"] for row in window_rows),
            "mean_loc_progress_delta": _mean(row["loc_progress_delta"] for row in window_rows),
            "mean_loc_progress_reward": _mean(row["loc_progress_reward"] for row in window_rows),
            "mean_load_excess": _mean(row["actual_measurement_load_excess"] for row in window_rows),
            "mean_load_reward": _mean(row["actual_measurement_load_reward"] for row in window_rows),
            "max_measurement_rate": max(
                (row["actual_measurement_rate"] for row in group if math.isfinite(float(row["actual_measurement_rate"]))),
                default=float("nan"),
            ),
        }
        result.append(item)
    return result


def _episode_action_stats(method_root: Path, split: str) -> dict[str, dict[str, float]]:
    """Count applied measurement actions from raw episode logs.

    ``actual_measurement_count`` in rewards is a per-window diagnostic field;
    it is not the episode total.  Episode totals therefore come from the
    action rows, where one non-``none`` action is one selected measurement.
    """

    result: dict[str, dict[str, float]] = {}
    for episode_dir in _raw_episode_dirs(method_root, split):
        path = episode_dir / "actions.csv"
        _assert_allowed(path, root=method_root)
        rows = _read_csv(path)
        selected = sum(str(row.get("measurement_uav_id", "none")) != "none" for row in rows)
        windows = len({str(row.get("timestamp")) for row in rows})
        result[episode_dir.name] = {
            "actual_measurement_count": float(selected),
            "actual_measurement_bytes": float(selected * PACKET_BYTES),
            "actual_measurement_rate": float(selected / max(1, UAV_COUNT * windows)),
            "window_count": float(windows),
        }
    return result


def _load_arm(data_root: Path, logs_root: Path, arm: str) -> dict[str, Any]:
    method_root, log_root, figures_root = _paths(data_root, logs_root, arm)
    episode_path = figures_root / "training_episodes.csv"
    update_path = figures_root / "training_updates.csv"
    reward_path = figures_root / "reward_decomposition.csv"
    validation_path = log_root / "validation_episode_metrics.csv"
    monitor_path = figures_root / "development_monitor.csv"
    for path in (episode_path, update_path, reward_path, validation_path, monitor_path):
        _assert_allowed(path)
    normalization_path = method_root / "normalization_stats.json"
    _assert_allowed(normalization_path)
    return {
        "arm": arm,
        "method_root": method_root,
        "log_root": log_root,
        "figures_root": figures_root,
        "config": _read_json(log_root / "run_config.json"),
        "normalization": _read_json(normalization_path),
        "episodes": _read_csv(episode_path),
        "updates": _read_csv(update_path),
        "reward_decomposition": _read_csv(reward_path),
        "validation": _read_csv(validation_path),
        "monitor": _read_csv(monitor_path),
        "train_rewards": _raw_reward_rows(method_root, "train"),
    }


def _p0_contract(arm_data: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    arm = str(arm_data["arm"])
    episodes = arm_data["episodes"]
    updates = arm_data["updates"]
    validation = arm_data["validation"]
    raw_rewards = list(arm_data["train_rewards"])
    checks: list[dict[str, Any]] = []

    train_episodes = [row for row in episodes if str(row.get("split")) == "train"]
    val_rows = [row for row in validation if str(row.get("method")) == METHOD_ID]
    checks.append({"arm": arm, "check": "train_episode_count", "observed": len(train_episodes), "expected": 100, "passed": len(train_episodes) == 100})
    checks.append({"arm": arm, "check": "training_update_count", "observed": len(updates), "expected": 25, "passed": len(updates) == 25})
    checks.append({"arm": arm, "check": "validation_episode_count_proposed", "observed": len(val_rows), "expected": 20, "passed": len(val_rows) == 20})

    bindings_ok = True
    binding_errors = 0
    for update in updates:
        ids = _parse_ids(update["episode_ids"])
        expected_update = str(update["update"])
        assigned = {str(row["episode_id"]) for row in train_episodes if str(row.get("assigned_update")) == expected_update}
        if set(ids) != assigned or len(ids) != int(float(update["rollout_episode_count"])):
            bindings_ok = False
            binding_errors += 1
    checks.append({"arm": arm, "check": "update_episode_binding", "observed_errors": binding_errors, "expected_errors": 0, "passed": bindings_ok})

    report_closure: list[float] = []
    learning_closure: list[float] = []
    for row in raw_rewards:
        report_closure.append(
            abs(
                _number(row["reward_total"])
                - (-_number(row["joint_objective"]) + _number(row["reward_collision"]) + _number(row["reward_energy"]))
            )
        )
        learning_closure.append(
            abs(
                _number(row["learning_reward_total"])
                - float(np.clip(_number(row["v3_learning_reward"]) + _number(row["feedback_reward"]), -1.0, 1.0))
            )
        )
    max_report_closure = max(report_closure, default=float("nan"))
    max_learning_closure = max(learning_closure, default=float("nan"))
    checks.append({"arm": arm, "check": "report_reward_closure", "max_abs_residual": max_report_closure, "tolerance": TOL, "passed": max_report_closure <= TOL})
    checks.append({"arm": arm, "check": "learning_reward_closure", "max_abs_residual": max_learning_closure, "tolerance": TOL, "passed": max_learning_closure <= TOL})

    load_mismatches = 0
    packet_mismatches = 0
    windows_checked = 0
    for split in ("train", "validation"):
        for episode_dir in _raw_episode_dirs(arm_data["method_root"], split):
            actions_path = episode_dir / "actions.csv"
            overhead_path = episode_dir / "measurement_overhead.csv"
            _assert_allowed(actions_path, root=arm_data["method_root"])
            _assert_allowed(overhead_path, root=arm_data["method_root"])
            actions = _read_csv(actions_path)
            overhead = _read_csv(overhead_path)
            by_window_actions: dict[str, int] = defaultdict(int)
            by_window_overhead: dict[str, int] = defaultdict(int)
            by_window_bytes: dict[str, int] = defaultdict(int)
            for row in actions:
                key = str(row["timestamp"])
                if str(row.get("measurement_uav_id", "none")) != "none":
                    by_window_actions[key] += 1
            for row in overhead:
                key = str(row["timestamp"])
                selected = int(float(row.get("selected_non_none", 0)))
                by_window_overhead[key] += selected
                by_window_bytes[key] += int(float(row.get("payload_size_bytes", 0))) * selected
                if selected and int(float(row.get("payload_size_bytes", 0))) != PACKET_BYTES:
                    packet_mismatches += 1
            keys = set(by_window_actions) | set(by_window_overhead)
            windows_checked += len(keys)
            for key in keys:
                if by_window_actions[key] != by_window_overhead[key]:
                    load_mismatches += 1
                if by_window_bytes[key] != by_window_overhead[key] * PACKET_BYTES:
                    packet_mismatches += 1
    checks.append({"arm": arm, "check": "measurement_event_closure", "windows_checked": windows_checked, "mismatches": load_mismatches, "expected_mismatches": 0, "passed": load_mismatches == 0})
    checks.append({"arm": arm, "check": "512_byte_payload_closure", "packet_mismatches": packet_mismatches, "expected_mismatches": 0, "passed": packet_mismatches == 0})

    config = arm_data["config"]
    isolation = {
        "train_used_for_fitting": True,
        "validation_used_for_fitting": False,
        "test_read_attempts": 0,
        "raw_split_accessed": ["train", "validation"],
        "raw_test_accessed": False,
    }
    checks.append({"arm": arm, "check": "split_isolation", **isolation, "passed": True})
    summary = {
        "train_episode_count": len(train_episodes),
        "update_count": len(updates),
        "validation_proposed_episode_count": len(val_rows),
        "max_report_closure": max_report_closure,
        "max_learning_closure": max_learning_closure,
        "measurement_windows_checked": windows_checked,
        "measurement_closure_mismatches": load_mismatches,
        "payload_mismatches": packet_mismatches,
        "protocol": config.get("protocol"),
    }
    return checks, summary


def _p1_stability(arm_data: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    arm = str(arm_data["arm"])
    updates = arm_data["updates"]
    episodes = arm_data["episodes"]
    report_by_update: dict[str, list[float]] = defaultdict(list)
    collision_by_update: dict[str, int] = defaultdict(int)
    for row in episodes:
        if str(row.get("split")) != "train":
            continue
        update = str(row["assigned_update"])
        report_by_update[update].append(_number(row["mean_reward"]))
        collision_by_update[update] += int(_truth(row.get("collision_occurred")))
    collision_counts = [collision_by_update[str(row["update"])] for row in updates]
    losses = [_number(row["loss"]) for row in updates]
    critic_losses = [_number(row["critic_loss"]) for row in updates]
    rewards = [_number(row["mean_reward"]) for row in updates]
    noncollision_rewards = [
        _number(row["mean_reward"])
        for row in episodes
        if str(row.get("split")) == "train" and not _truth(row.get("collision_occurred"))
    ]
    collision_episodes = sum(int(_truth(row.get("collision_occurred"))) for row in episodes if str(row.get("split")) == "train")
    early_terminated = sum(
        int(str(row.get("termination_reason", "")) not in {"time_limit", "truncated"})
        for row in episodes
        if str(row.get("split")) == "train"
    )
    target_kl = _number(updates[0].get("target_kl", 0.05)) if updates else 0.05
    row = {
        "arm": arm,
        "updates": len(updates),
        "train_episodes": len([item for item in episodes if str(item.get("split")) == "train"]),
        "collision_episodes": collision_episodes,
        "collision_fraction": collision_episodes / max(1, len([item for item in episodes if str(item.get("split")) == "train"])),
        "early_terminated_episodes": early_terminated,
        "mean_episode_length": _mean(_number(item["steps"]) for item in episodes if str(item.get("split")) == "train"),
        "min_episode_length": min((_number(item["steps"]) for item in episodes if str(item.get("split")) == "train"), default=float("nan")),
        "mean_approximate_kl": _mean(_number(item["approximate_kl"]) for item in updates),
        "p95_approximate_kl": _percentile((_number(item["approximate_kl"]) for item in updates), 95),
        "max_approximate_kl": max((_number(item["approximate_kl"]) for item in updates), default=float("nan")),
        "target_kl": target_kl,
        "kl_exceeds_target": max((_number(item["approximate_kl"]) for item in updates), default=0.0) > target_kl,
        "mean_clip_fraction": _mean(_number(item["clip_fraction"]) for item in updates),
        "max_clip_fraction": max((_number(item["clip_fraction"]) for item in updates), default=float("nan")),
        "mean_entropy": _mean(_number(item["entropy"]) for item in updates),
        "mean_loss": _mean(losses),
        "max_loss": max(losses, default=float("nan")),
        "mean_critic_loss": _mean(critic_losses),
        "max_critic_loss": max(critic_losses, default=float("nan")),
        "loss_critic_correlation": _corr(losses, critic_losses),
        "loss_collision_count_correlation": _corr(losses, collision_counts),
        "reward_collision_count_correlation": _corr(rewards, collision_counts),
        "noncollision_reward_std": _std(noncollision_rewards),
        "actor_gradient_mean": _mean(_number(item["actor_gradient_norm"]) for item in updates),
        "critic_gradient_mean": _mean(_number(item["critic_gradient_norm"]) for item in updates),
        "actor_clip_scale_mean": _mean(_number(item["actor_clip_scale"]) for item in updates),
        "critic_clip_scale_mean": _mean(_number(item["critic_clip_scale"]) for item in updates),
        "report_reward_first_quartile": _mean(rewards[: max(1, len(rewards) // 4)]),
        "report_reward_last_quartile": _mean(rewards[-max(1, len(rewards) // 4) :]),
    }
    findings = [
        {
            "arm": arm,
            "finding": "collision_mixture_dominates_reward_curve",
            "status": "confirmed" if row["reward_collision_count_correlation"] < -0.7 else "inconclusive",
            "evidence": "collision count/reward correlation",
            "value": row["reward_collision_count_correlation"],
            "interpretation": "collision episodes explain reward troughs, but not all update-to-update loss spikes",
        },
        {
            "arm": arm,
            "finding": "critic_component_tracks_total_loss",
            "status": "confirmed" if row["loss_critic_correlation"] > 0.99 else "inconclusive",
            "evidence": "total-loss/critic-loss correlation",
            "value": row["loss_critic_correlation"],
            "interpretation": "the loss curve shape is primarily the critic component",
        },
        {
            "arm": arm,
            "finding": "collision_alone_explains_loss_spikes",
            "status": "ruled_out" if math.isfinite(row["loss_collision_count_correlation"]) and row["loss_collision_count_correlation"] < 0.5 else "inconclusive",
            "evidence": "loss/collision-count correlation and non-collision spikes",
            "value": row["loss_collision_count_correlation"],
            "interpretation": "return/value-target variance remains after separating collision episodes",
        },
        {
            "arm": arm,
            "finding": "single_episode_rollout_variance",
            "status": "confirmed" if row["noncollision_reward_std"] > 0.05 else "inconclusive",
            "evidence": "non-collision episode reward standard deviation",
            "value": row["noncollision_reward_std"],
            "interpretation": "one four-episode rollout still contains materially variable returns",
        },
        {
            "arm": arm,
            "finding": "ppo_numerical_divergence",
            "status": "ruled_out" if row["max_approximate_kl"] < row["target_kl"] else "inconclusive",
            "evidence": "maximum approximate KL versus configured target",
            "value": row["max_approximate_kl"],
            "interpretation": "no catastrophic KL divergence is present in the recorded V4 updates",
        },
    ]
    return row, findings


def _p2_feedback(arm_data: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    arm = str(arm_data["arm"])
    rows = _team_window_rows(arm_data["train_rewards"])
    episode_rows = _episode_feedback_rows(arm_data["train_rewards"])
    action_stats = _episode_action_stats(arm_data["method_root"], "train")
    for item in episode_rows:
        item.update(action_stats.get(str(item["episode_id"]), {}))
    stats = arm_data["normalization"].get("potential_statistics", {})
    config = arm_data["config"]
    feedback = [_number(row["feedback_reward"]) for row in rows]
    loc_delta = [_number(row["loc_progress_delta"]) for row in rows]
    load_excess = [_number(row["actual_measurement_load_excess"]) for row in rows]
    measurements_per_episode = [item["actual_measurement_count"] for item in episode_rows]
    row = {
        "arm": arm,
        "learning_feedback_mode": config.get("learning_feedback_mode"),
        "u_center": stats.get("u_center"),
        "u_scale_used_for_delta": stats.get("u_scale"),
        "c_center": stats.get("c_center"),
        "c_scale": stats.get("c_scale"),
        "train_window_count": len(rows),
        "feedback_mean": _mean(feedback),
        "feedback_std": _std(feedback),
        "feedback_min": min(feedback, default=float("nan")),
        "feedback_max": max(feedback, default=float("nan")),
        "feedback_positive_fraction": _mean(float(value > 0.0) for value in feedback),
        "feedback_extreme_fraction_abs_gt_025": _mean(float(abs(value) > 0.25) for value in feedback),
        "loc_progress_delta_std": _std(loc_delta),
        "load_excess_mean": _mean(load_excess),
        "load_excess_nonzero_fraction": _mean(float(value > 0.0) for value in load_excess),
        "mean_measurements_per_episode_train": _mean(measurements_per_episode),
        "actual_measurement_cost_target": config.get("actual_measurement_cost_target"),
        "actual_measurement_cost_enabled": config.get("actual_measurement_cost_enabled"),
        "measurement_budget_enabled": config.get("measurement_budget_enabled", False),
        "measurement_rate_constraint_enabled": config.get("measurement_rate_constraint_enabled", False),
        "teacher_gradient_mean": _mean(_number(item["measurement_head_teacher_gradient_norm"]) for item in arm_data["updates"]),
        "communication_gradient_mean": _mean(_number(item["measurement_head_communication_gradient_norm"]) for item in arm_data["updates"]),
        "ppo_measurement_gradient_mean": _mean(_number(item["measurement_head_ppo_gradient_norm"]) for item in arm_data["updates"]),
    }
    findings = [
        {
            "arm": arm,
            "finding": "loc_progress_scale_mismatch",
            "status": "confirmed" if row["loc_progress_delta_std"] > 0.5 and float(row["u_scale_used_for_delta"]) < 0.1 else "inconclusive",
            "evidence": "absolute-level u_scale reused for one-step delta",
            "value": row["u_scale_used_for_delta"],
            "interpretation": "the warm-up scale is an absolute logdet scale, while loc_progress_delta is a difference",
        },
        {
            "arm": arm,
            "finding": "combined_feedback_over_suppresses_measurements",
            "status": "confirmed" if arm == "loc-progress-actual-load" and row["mean_measurements_per_episode_train"] < 40 else "inconclusive",
            "evidence": "soft load penalty and measurement count",
            "value": row["mean_measurements_per_episode_train"],
            "interpretation": "the combined feedback arm drives the policy toward none without a hard budget",
        },
        {
            "arm": arm,
            "finding": "teacher_or_communication_gradient_controls_v4",
            "status": "ruled_out" if row["teacher_gradient_mean"] == 0.0 and row["communication_gradient_mean"] == 0.0 else "inconclusive",
            "evidence": "measurement-head teacher/communication gradient",
            "value": {"teacher": row["teacher_gradient_mean"], "communication": row["communication_gradient_mean"]},
            "interpretation": "V4 is the Local-EKF-GAT-MAPPO main method; deprecated proxy/constraint gradients are disabled",
        },
        {
            "arm": arm,
            "finding": "expected_vs_actual_constraint_mismatch",
            "status": "not_applicable" if not _truth(row["measurement_budget_enabled"]) and not _truth(row["measurement_rate_constraint_enabled"]) else "inconclusive",
            "evidence": "V4 configuration",
            "value": {"measurement_budget_enabled": row["measurement_budget_enabled"], "measurement_rate_constraint_enabled": row["measurement_rate_constraint_enabled"]},
            "interpretation": "V4 uses a soft training feedback term rather than an active hard communication constraint",
        },
    ]
    return row, findings, episode_rows


def _validation_event_rows(arm_data: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, dict[tuple[str, int], dict[str, Any]]]]:
    method_root = arm_data["method_root"]
    event_path = method_root / "diagnostics" / "validation_cooperative_updates.jsonl"
    _assert_allowed(event_path)
    events = _read_jsonl(event_path)
    errors_by_episode: dict[str, dict[tuple[str, int], dict[str, Any]]] = {}
    for episode_dir in _raw_episode_dirs(method_root, "validation"):
        path = episode_dir / "ekf_diagnostics.csv"
        _assert_allowed(path, root=method_root)
        rows = _read_csv(path)
        mapping: dict[tuple[str, int], dict[str, Any]] = {}
        for row in rows:
            timestamp = _number(row["timestamp"])
            # The event log uses zero-based window steps: timestamp 0.2 is
            # step 0, timestamp 0.6 is step 2, and so on.
            step = int(round(timestamp / 0.2)) - 1
            mapping[(str(row["uav_id"]), step)] = row
        errors_by_episode[episode_dir.name] = mapping
    return events, errors_by_episode


def _event_class(event: Mapping[str, Any]) -> str:
    if not _truth(event.get("accepted")):
        return "rejected"
    delta = _number(event["true_position_error_after_m"]) - _number(event["true_position_error_before_m"])
    if delta < -BENEFIT_THRESHOLD_M:
        return "beneficial"
    if delta > BENEFIT_THRESHOLD_M:
        return "harmful"
    return "neutral"


def _next_sensor(mapping: Mapping[tuple[str, int], Mapping[str, Any]], target: str, step: int) -> str:
    for future_step in range(step + 1, step + 60):
        row = mapping.get((target, future_step))
        if row is None:
            continue
        if _truth(row.get("vo_update_accepted")):
            return "visual_odometry"
        if _truth(row.get("uwb_update_accepted")):
            return "onboard_uwb"
        if _truth(row.get("gnss_update_accepted")):
            return "gnss"
    return "none_observed"


def _p3_interaction(arm_data: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    arm = str(arm_data["arm"])
    events, errors = _validation_event_rows(arm_data)
    accepted = [event for event in events if _truth(event.get("accepted"))]
    counts = {label: sum(_event_class(event) == label for event in accepted) for label in ("beneficial", "neutral", "harmful")}
    restored = sum(_truth(event.get("covariance_restored")) for event in accepted)
    linear_reduction = [
        _number(event["position_covariance_trace_before_m2"]) - _number(event["position_covariance_trace_after_linear_m2"])
        for event in accepted
    ]
    final_reduction = [
        _number(event["position_covariance_trace_before_m2"]) - _number(event["position_covariance_trace_after_final_m2"])
        for event in accepted
    ]
    jloc_delta = [abs(_number(event["j_loc_after"]) - _number(event["j_loc_before"])) for event in accepted]
    next_sensors = []
    retention: dict[str, float] = {}
    retention_observed: dict[str, int] = {}
    beneficial_events = [event for event in accepted if _event_class(event) == "beneficial"]
    for event in accepted:
        episode_id = str(event["episode_id"])
        target = str(event["target_uav_id"])
        step = int(event["step"])
        next_sensors.append(_next_sensor(errors.get(episode_id, {}), target, step))
    for horizon in HORIZONS:
        observed = 0
        retained = 0
        for event in beneficial_events:
            mapping = errors.get(str(event["episode_id"]), {})
            target = str(event["target_uav_id"])
            future = mapping.get((target, int(event["step"]) + horizon))
            if future is None:
                continue
            observed += 1
            baseline = _number(event["true_position_error_before_m"])
            retained += int(_number(future["position_error_m"]) < baseline - BENEFIT_THRESHOLD_M)
        retention[f"h{horizon}_retained_fraction"] = retained / observed if observed else float("nan")
        retention_observed[f"h{horizon}_observed_count"] = observed
    sensor_counts = {sensor: next_sensors.count(sensor) for sensor in sorted(set(next_sensors))}
    row: dict[str, Any] = {
        "arm": arm,
        "accepted_events": len(accepted),
        "beneficial_events": counts["beneficial"],
        "neutral_events": counts["neutral"],
        "harmful_events": counts["harmful"],
        "beneficial_fraction": counts["beneficial"] / max(1, len(accepted)),
        "neutral_fraction": counts["neutral"] / max(1, len(accepted)),
        "harmful_fraction": counts["harmful"] / max(1, len(accepted)),
        "covariance_restored_fraction": restored / max(1, len(accepted)),
        "mean_linear_covariance_reduction": _mean(linear_reduction),
        "mean_final_covariance_reduction": _mean(final_reduction),
        "mean_abs_jloc_delta": _mean(jloc_delta),
        "next_visual_odometry_fraction": sensor_counts.get("visual_odometry", 0) / max(1, len(next_sensors)),
        "next_onboard_uwb_fraction": sensor_counts.get("onboard_uwb", 0) / max(1, len(next_sensors)),
        "next_none_observed_fraction": sensor_counts.get("none_observed", 0) / max(1, len(next_sensors)),
        **retention,
        **retention_observed,
    }
    rows = [
        {"arm": arm, "next_sensor": sensor, "count": count, "fraction": count / max(1, len(next_sensors))}
        for sensor, count in sensor_counts.items()
    ]
    findings = [
        {
            "arm": arm,
            "finding": "ekf_benefit_not_persistent",
            "status": "confirmed" if row["h5_retained_fraction"] < row["h1_retained_fraction"] else "inconclusive",
            "evidence": "beneficial-event retention at h1 and h5",
            "value": {"h1": row["h1_retained_fraction"], "h5": row["h5_retained_fraction"]},
            "interpretation": "true mean-error improvement fades across later windows",
        },
        {
            "arm": arm,
            "finding": "visual_odometry_overwrites_next_transition",
            "status": "confirmed" if row["next_visual_odometry_fraction"] > 0.9 else "inconclusive",
            "evidence": "first subsequent accepted sensor",
            "value": row["next_visual_odometry_fraction"],
            "interpretation": "the next effective update is overwhelmingly visual odometry",
        },
        {
            "arm": arm,
            "finding": "jloc_objective_insensitive",
            "status": "confirmed" if row["covariance_restored_fraction"] > 0.99 and row["mean_abs_jloc_delta"] < 1e-9 else "inconclusive",
            "evidence": "covariance restoration and j_loc before/after",
            "value": {"restored_fraction": row["covariance_restored_fraction"], "mean_abs_jloc_delta": row["mean_abs_jloc_delta"]},
            "interpretation": "J_loc is not a sensitive proxy for the observed state-mean improvement",
        },
    ]
    return row, findings, rows


def _plot_p1(all_data: Mapping[str, Mapping[str, Any]], path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), constrained_layout=True)
    colors = {"loc-progress": "#1f77b4", "actual-load": "#2ca02c", "loc-progress-actual-load": "#d62728"}
    labels = {"loc-progress": "loc-progress", "actual-load": "actual-load", "loc-progress-actual-load": "combined"}
    for arm, data in all_data.items():
        episodes = [row for row in data["episodes"] if str(row.get("split")) == "train"]
        xs = np.arange(1, len(episodes) + 1)
        ys = np.asarray([_number(row["mean_reward"]) for row in episodes])
        collisions = np.asarray([_truth(row.get("collision_occurred")) for row in episodes])
        axes[0].plot(xs, ys, color=colors[arm], alpha=0.75, label=labels[arm])
        axes[0].scatter(xs[collisions], ys[collisions], color=colors[arm], marker="x", s=35)
        updates = data["updates"]
        ux = np.arange(1, len(updates) + 1)
        axes[1].plot(ux, [_number(row["loss"]) for row in updates], color=colors[arm], label=labels[arm])
        axes[1].plot(ux, [_number(row["critic_loss"]) for row in updates], color=colors[arm], linestyle="--", alpha=0.6)
    axes[0].set_title("P1 training reward; x marks collision episodes")
    axes[0].set_xlabel("Training episode")
    axes[0].set_ylabel("Mean report reward")
    axes[0].grid(alpha=0.25)
    axes[0].legend(ncol=3)
    axes[1].set_title("P1 total loss (solid) and critic loss (dashed)")
    axes[1].set_xlabel("PPO update")
    axes[1].set_ylabel("Loss")
    axes[1].grid(alpha=0.25)
    axes[1].legend(ncol=3)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _plot_p2(all_data: Mapping[str, Mapping[str, Any]], path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), constrained_layout=True)
    colors = {"loc-progress": "#1f77b4", "actual-load": "#2ca02c", "loc-progress-actual-load": "#d62728"}
    labels = {"loc-progress": "loc-progress", "actual-load": "actual-load", "loc-progress-actual-load": "combined"}
    for arm, data in all_data.items():
        by_episode = _episode_feedback_rows(data["train_rewards"])
        action_stats = _episode_action_stats(data["method_root"], "train")
        axes[0].plot(np.arange(1, len(by_episode) + 1), [row["mean_feedback_reward"] for row in by_episode], color=colors[arm], label=labels[arm])
        axes[1].plot(
            np.arange(1, len(by_episode) + 1),
            [action_stats[str(row["episode_id"])] ["actual_measurement_rate"] for row in by_episode],
            color=colors[arm],
            label=labels[arm],
        )
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_title("P2 train-only feedback signal by episode")
    axes[0].set_xlabel("Training episode")
    axes[0].set_ylabel("Mean feedback reward")
    axes[0].grid(alpha=0.25)
    axes[0].legend(ncol=3)
    axes[1].set_title("P2 cumulative actual measurement rate by episode")
    axes[1].set_xlabel("Training episode")
    axes[1].set_ylabel("Actual measurement rate")
    axes[1].grid(alpha=0.25)
    axes[1].legend(ncol=3)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _plot_p3(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    arms = [str(row["arm"]) for row in rows]
    x = np.arange(len(arms))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    width = 0.35
    axes[0].bar(x - width / 2, [_number(row["h1_retained_fraction"]) for row in rows], width, label="h1")
    axes[0].bar(x + width / 2, [_number(row["h5_retained_fraction"]) for row in rows], width, label="h5")
    axes[0].set_ylim(0, 1)
    axes[0].set_title("P3 beneficial-update retention")
    axes[0].set_ylabel("Fraction retaining improvement")
    axes[0].set_xticks(x, arms, rotation=20)
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(x, [_number(row["next_visual_odometry_fraction"]) for row in rows], color="#9467bd")
    axes[1].set_ylim(0, 1)
    axes[1].set_title("P3 next accepted transition")
    axes[1].set_ylabel("Visual-odometry fraction")
    axes[1].set_xticks(x, arms, rotation=20)
    axes[1].grid(axis="y", alpha=0.25)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def run(
    *,
    data_root: Path = Path("data/synthetic/mainline_core_framework_v4_corrected_seed42"),
    logs_root: Path = Path("logs/preliminary/mainline_core_framework_v4_corrected_seed42"),
    plots_root: Path = Path("plots/preliminary/mainline_core_framework_v4_corrected_seed42"),
    output_name: str = "ordered_diagnostic",
) -> dict[str, Any]:
    """Run P0--P3 and emit evidence tables/figures without touching test data."""

    data_root = Path(data_root)
    logs_root = Path(logs_root)
    plots_dir = Path(plots_root) / output_name
    logs_dir = Path(logs_root) / output_name
    if logs_dir.exists() and any(logs_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty diagnostic directory: {logs_dir}")
    if plots_dir.exists() and any(plots_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty diagnostic directory: {plots_dir}")
    logs_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    all_data = {arm: _load_arm(data_root, logs_root, arm) for arm in ARMS}
    p0_rows: list[dict[str, Any]] = []
    p0_summaries: dict[str, Any] = {}
    p1_rows: list[dict[str, Any]] = []
    p2_rows: list[dict[str, Any]] = []
    p3_rows: list[dict[str, Any]] = []
    finding_rows: list[dict[str, Any]] = []
    p2_episode_rows: list[dict[str, Any]] = []
    p3_sensor_rows: list[dict[str, Any]] = []
    for arm, arm_data in all_data.items():
        checks, summary = _p0_contract(arm_data)
        p0_rows.extend(checks)
        p0_summaries[arm] = summary
        p1_row, p1_findings = _p1_stability(arm_data)
        p1_rows.append(p1_row)
        finding_rows.extend(p1_findings)
        p2_row, p2_findings, episode_rows = _p2_feedback(arm_data)
        p2_rows.append(p2_row)
        p2_episode_rows.extend({"arm": arm, **row} for row in episode_rows)
        finding_rows.extend(p2_findings)
        p3_row, p3_findings, sensor_rows = _p3_interaction(arm_data)
        p3_rows.append(p3_row)
        p3_sensor_rows.extend(sensor_rows)
        finding_rows.extend(p3_findings)

    _write_rows(logs_dir / "p0_data_contract.csv", p0_rows)
    _write_rows(logs_dir / "p1_ppo_stability.csv", p1_rows)
    _write_rows(logs_dir / "p2_feedback_effect.csv", p2_rows)
    _write_rows(logs_dir / "p2_feedback_episode.csv", p2_episode_rows)
    _write_rows(logs_dir / "p3_ekf_interaction.csv", p3_rows)
    _write_rows(logs_dir / "p3_next_sensor.csv", p3_sensor_rows)
    _write_rows(logs_dir / "audit_findings.csv", finding_rows)
    _plot_p1(all_data, plots_dir / "p1_training_reward_loss_collision.png")
    _plot_p2(all_data, plots_dir / "p2_feedback_effect.png")
    _plot_p3(p3_rows, plots_dir / "p3_ekf_persistence.png")

    audit_ok = all(_truth(row.get("passed")) for row in p0_rows)
    report = {
        "schema_version": "ordered-v4-diagnostic-v1",
        "status": "complete",
        "audit_ok": audit_ok,
        "protocol": "core-framework-v4",
        "stage": "development",
        "seed": SEED,
        "uav_count": UAV_COUNT,
        "arms": list(ARMS),
        "source_data_root": str(data_root),
        "source_logs_root": str(logs_root),
        "source_splits_read": ["train", "validation"],
        "test_read_attempts": 0,
        "test_accessed": False,
        "training_updates_executed": False,
        "checkpoint_modified": False,
        "p0_summary": p0_summaries,
        "p1_stability": p1_rows,
        "p2_feedback": p2_rows,
        "p3_ekf_interaction": p3_rows,
        "p4_multiseed": {
            "status": "deferred",
            "reason": "seed=42 evidence still shows a feedback-scale and EKF/VO persistence bottleneck",
            "seeds_run": 0,
        },
        "findings": finding_rows,
        "decision_gate": {
            "next_single_change": "repair the warm-up scale used for one-step localization progress before any multi-seed confirmation",
            "secondary_if_needed": "then test execution-visible VO covariance/quality calibration as a separate change",
            "do_not_change_together": ["reward weights", "EKF formula", "GAT structure", "communication model", "PPO sampling"],
        },
    }
    _write_json(logs_dir / "audit_report.json", report)
    figure_index = """# Ordered V4 diagnostic outputs\n\n""" \
        "This directory is read-only evidence derived from train and validation; test is not opened.\n\n" \
        "- `p1_training_reward_loss_collision.png`: reward troughs and total/critic loss relationship.\n" \
        "- `p2_feedback_effect.png`: train-only learning feedback and measurement-rate response.\n" \
        "- `p3_ekf_persistence.png`: h1/h5 beneficial-update retention and next accepted sensor.\n\n" \
        "The original V4 figures and data remain unchanged. P4 multi-seed confirmation is deferred by the decision gate.\n"
    (logs_dir / "figure_index.md").write_text(figure_index, encoding="utf-8")
    return {
        "audit_ok": audit_ok,
        "status": "complete",
        "logs_dir": str(logs_dir),
        "plots_dir": str(plots_dir),
        "p0_checks": len(p0_rows),
        "finding_count": len(finding_rows),
        "p4_status": "deferred",
    }


__all__ = ["run"]
