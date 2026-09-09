"""Diagnostic audit of the quality-aware MAPPO measurement mechanism.

This module deliberately does not train a policy and does not write episode
files.  It reloads frozen checkpoints from the completed overhead experiment,
replays validation episodes deterministically, and records the complete
candidate-level measurement distribution.  The replay is therefore a
mechanism audit, not a new performance experiment.
"""

from __future__ import annotations

import csv
import gc
import json
import math
import shutil
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ..learning.measurement_proxy import information_gain_features
from ..learning.measurement_quality import quality_features
from ..learning.mappo import MAPPOConfig, MAPPOTrainer
from ..learning.normalization import FeatureNormalizer, ObservationNormalizer
from ..synthetic.episode_generator import _jsonable
from .formal import (
    FormalExperimentConfig,
    _make_environment,
    _scale_seed,
    _scenario_for,
)
from .pilot import _append_jsonl, _assert_action_feasible, _seed_everything


DEFAULT_SEEDS = (2026, 3035, 4044, 5053, 6062)
SUPPORTED_BRANCHES = (
    "baseline_mappo",
    "quality_aware_mappo",
    "budgeted_quality_aware_mappo",
    "adaptive_budgeted_quality_aware_mappo",
    "proxy_guided_adaptive_budgeted_quality_aware_mappo",
)
DEFAULT_BRANCHES = SUPPORTED_BRANCHES[:2]
TRANSITION_MIN_DENOMINATOR = 30
TRANSITION_ACTIONABLE_RATE = 0.05
_MANAGED_FILES = (
    "measurement_policy_trace.jsonl",
    "policy_feature_summary.csv",
    "action_probability_summary.csv",
    "measurement_overhead_summary.csv",
    "transition_invalidation_summary.csv",
    "budget_usage_summary.csv",
    "seed_level_mechanism_summary.csv",
    "audit_report.json",
)


@dataclass(frozen=True)
class PolicyMechanismAuditConfig:
    """Configuration for a validation-only policy mechanism audit."""

    reference_root: Path
    output_dir: Path = Path(
        "logs/preliminary/python_sim_quality_aware_mappo_overhead_seed2026/"
        "policy_mechanism_audit"
    )
    seeds: tuple[int, ...] = DEFAULT_SEEDS
    branches: tuple[str, ...] = DEFAULT_BRANCHES
    split: str = "validation"
    plots_dir: Path | None = None
    bootstrap_samples: int = 10_000
    bootstrap_seed: int = 2026
    device: str = "cpu"
    force: bool = False
    progress: bool = True

    def __post_init__(self) -> None:
        if self.split != "validation":
            raise ValueError("policy mechanism audit is validation-only; test is sealed")
        if not self.seeds:
            raise ValueError("at least one base seed is required")
        if not self.branches:
            raise ValueError("at least one branch is required")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be unique")
        unknown = sorted(set(self.branches) - set(SUPPORTED_BRANCHES))
        if unknown:
            raise ValueError(
                "unsupported branch(es): " + ", ".join(unknown)
            )
        if len(set(self.branches)) != len(self.branches):
            raise ValueError("branches must be unique")
        if self.device != "cpu":
            raise ValueError("the deterministic audit currently supports device='cpu' only")
        if self.bootstrap_samples < 100:
            raise ValueError("bootstrap_samples must be at least 100")


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
            if isinstance(value, dict):
                rows.append(value)
    return rows


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, columns: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: _csv_value(row.get(column, "")) for column in columns})


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple, np.ndarray)):
        return json.dumps(_jsonable(value), ensure_ascii=False, separators=(",", ":"))
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    return value


def _float(value: Any, default: float = float("nan")) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _array(value: Any, *, dtype=float) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=dtype).reshape(-1)
    except (TypeError, ValueError):
        return np.empty(0, dtype=dtype)
    return result


def _safe_mean(values: Iterable[float], default: float = 0.0) -> float:
    values = np.asarray(list(values), dtype=float)
    values = values[np.isfinite(values)]
    return float(values.mean()) if values.size else default


def _pearson(x: Sequence[float], y: Sequence[float]) -> float:
    left = np.asarray(x, dtype=float)
    right = np.asarray(y, dtype=float)
    mask = np.isfinite(left) & np.isfinite(right)
    left = left[mask]
    right = right[mask]
    if left.size < 3 or np.std(left) <= 1e-12 or np.std(right) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def compute_risk_probability_correlation(
    risks: Sequence[float], probabilities: Sequence[float], valid: Sequence[float] | None = None
) -> float:
    """Return the candidate risk/probability Pearson correlation.

    A negative value is the expected direction: lower communication risk is
    associated with a larger measurement probability.  This is a diagnostic
    association, not a causal or performance test.
    """

    risk = np.asarray(risks, dtype=float).reshape(-1)
    probability = np.asarray(probabilities, dtype=float).reshape(-1)
    length = min(risk.size, probability.size)
    mask = np.ones(length, dtype=bool)
    if valid is not None:
        validity = np.asarray(valid, dtype=float).reshape(-1)
        mask &= validity[:length] > 0.5
    return _pearson(risk[:length][mask], probability[:length][mask])


def _finite_trace_value(value: Any) -> bool:
    if isinstance(value, Mapping):
        return all(_finite_trace_value(item) for item in value.values())
    if isinstance(value, (list, tuple, np.ndarray)):
        try:
            array = np.asarray(value, dtype=float)
        except (TypeError, ValueError):
            return all(_finite_trace_value(item) for item in value)
        return bool(np.isfinite(array).all())
    if isinstance(value, (float, np.floating)):
        return math.isfinite(float(value))
    return True


def _prepare_output(path: Path, *, force: bool) -> None:
    path = path.resolve()
    if path.exists() and any(path.iterdir()) and not force:
        raise FileExistsError(f"audit output is not empty: {path}; use --force to replace it")
    path.mkdir(parents=True, exist_ok=True)
    if force:
        for name in _MANAGED_FILES:
            target = path / name
            if target.exists():
                target.unlink()


def _manifest_for(reference_root: Path, seed: int, branch: str) -> tuple[Path, dict[str, Any]]:
    scale_root = reference_root / f"seed_{seed}" / branch / "scale_3"
    manifest_path = scale_root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing reference scale manifest: {manifest_path}")
    return scale_root, _read_json(manifest_path)


def _formal_config(manifest: Mapping[str, Any], branch: str) -> FormalExperimentConfig:
    raw = dict(manifest.get("config", {}))
    names = {field.name for field in fields(FormalExperimentConfig)}
    values: dict[str, Any] = {name: raw[name] for name in names if name in raw}
    values["output_dir"] = Path(str(values.get("output_dir", ".")))
    if values.get("scenario_profile_path"):
        values["scenario_profile_path"] = Path(str(values["scenario_profile_path"]))
    if values.get("metric_scales") is not None:
        values["metric_scales"] = {
            str(key): float(value) for key, value in dict(values["metric_scales"]).items()
        }
    values["mappo_variant"] = branch
    values["force"] = False
    values["progress"] = False
    return FormalExperimentConfig(**values)


def _load_quality_model(scale_root: Path) -> dict[str, Any]:
    path = scale_root / "calibration" / "measurement_quality_model.json"
    if not path.exists():
        raise FileNotFoundError(f"missing warm-up quality model: {path}")
    model = _read_json(path)
    if model.get("source_split") != "warmup":
        raise ValueError(f"quality model is not warm-up-only: {path}")
    return model


def _load_proxy_model(scale_root: Path) -> dict[str, Any]:
    """Load the train-only information-gain proxy used by proxy checkpoints."""

    path = scale_root / "calibration" / "measurement_proxy_model.json"
    if not path.exists():
        raise FileNotFoundError(f"missing train-fitted measurement proxy model: {path}")
    model = _read_json(path)
    if model.get("source_split") != "train":
        raise ValueError(f"measurement proxy is not train-only: {path}")
    if model.get("test_used_for_fitting", False) or model.get("validation_used_for_fitting", False):
        raise ValueError(f"measurement proxy records non-train fitting data: {path}")
    return model


def _teacher_probabilities(
    gain: Sequence[float],
    cost: Sequence[float],
    valid: Sequence[float],
    action_mask: Sequence[float],
    *,
    temperature: float,
    communication_cost_weight: float,
) -> np.ndarray:
    """Reproduce the detached proxy teacher distribution without torch.

    This is deliberately kept byte-for-byte in semantic alignment with
    ``MAPPOPolicy.measurement_proxy_teacher_loss``: invalid candidates and
    masked actions receive a large negative logit, while ``none`` has utility
    zero and remains available.
    """

    count = max(len(gain), len(cost), len(valid))
    if count <= 0:
        return np.asarray([1.0], dtype=np.float64)
    gain_array = np.zeros(count, dtype=np.float64)
    cost_array = np.ones(count, dtype=np.float64)
    valid_array = np.zeros(count, dtype=np.float64)
    gain_array[: len(gain)] = np.asarray(gain, dtype=float)[:count]
    cost_array[: len(cost)] = np.asarray(cost, dtype=float)[:count]
    valid_array[: len(valid)] = np.asarray(valid, dtype=float)[:count]
    utilities = gain_array - float(communication_cost_weight) * cost_array
    utilities = np.where(valid_array > 0.5, utilities, -1.0e9)
    logits = np.concatenate([utilities, np.asarray([0.0])])
    mask = np.zeros(count + 1, dtype=bool)
    provided = np.asarray(action_mask, dtype=bool).reshape(-1)
    mask[: min(mask.size, provided.size)] = provided[: mask.size]
    if not bool(mask.any()):
        mask[-1] = True
    logits = np.where(mask, logits, -1.0e9)
    scaled = logits / max(float(temperature), 1.0e-12)
    scaled -= np.max(scaled)
    probabilities = np.exp(np.clip(scaled, -745.0, 0.0))
    probabilities[~mask] = 0.0
    total = float(probabilities.sum())
    if not math.isfinite(total) or total <= 0.0:
        probabilities = np.zeros(count + 1, dtype=np.float64)
        probabilities[-1] = 1.0
        return probabilities
    return probabilities / total


def _load_trainer(
    config: FormalExperimentConfig,
    scale_root: Path,
    manifest: Mapping[str, Any],
    environment: Any,
    *,
    seed: int,
) -> tuple[MAPPOTrainer, dict[str, Any]]:
    import torch

    checkpoint_path = scale_root / "checkpoints" / "final.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"missing final checkpoint: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError(f"invalid checkpoint payload: {checkpoint_path}")
    trainer_config_raw = dict(payload.get("trainer_config", {}))
    trainer_fields = {field.name for field in fields(MAPPOConfig)}
    trainer_config = {
        key: value for key, value in trainer_config_raw.items() if key in trainer_fields
    }
    quality_model = _load_quality_model(scale_root)
    proxy_model: dict[str, Any] | None = None
    use_quality_features = bool(trainer_config.get("use_measurement_quality_features", False))
    use_proxy_features = bool(trainer_config.get("use_measurement_proxy_features", False))
    if use_proxy_features:
        proxy_model = _load_proxy_model(scale_root)
    trainer = MAPPOTrainer.from_environment(
        environment,
        seed=seed,
        normalizer=ObservationNormalizer.from_state_dict(
            payload["normalization_stats"]
        ),
        central_normalizer=FeatureNormalizer.from_state_dict(
            payload["central_normalization_stats"]
        ),
        measurement_quality_model=quality_model if use_quality_features else None,
        measurement_proxy_model=proxy_model if use_proxy_features else None,
        **trainer_config,
    )
    trainer.policy.load_state_dict(payload["policy_state_dict"])
    trainer.policy.eval()
    return trainer, quality_model


def _parse_action(row: Mapping[str, Any]) -> dict[str, Any]:
    raw = row.get("action_raw", "{}")
    try:
        action = json.loads(str(raw))
    except (TypeError, ValueError, json.JSONDecodeError):
        action = {}
    if not isinstance(action, Mapping):
        action = {}
    return {
        "measurement_uav_id": str(action.get("measurement_uav_id", row.get("measurement_uav_id", "none"))),
        "bandwidth": {
            str(key): float(value)
            for key, value in dict(action.get("bandwidth", {})).items()
        },
        "delta_v": _array(action.get("delta_v", (0.0, 0.0, 0.0)), dtype=float),
    }


def _actions_match(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    if str(left.get("measurement_uav_id", "none")) != str(right.get("measurement_uav_id", "none")):
        return False
    left_bandwidth = {str(key): float(value) for key, value in dict(left.get("bandwidth", {})).items()}
    right_bandwidth = {str(key): float(value) for key, value in dict(right.get("bandwidth", {})).items()}
    if set(left_bandwidth) != set(right_bandwidth):
        return False
    if any(not math.isclose(left_bandwidth[key], right_bandwidth[key], rel_tol=1e-6, abs_tol=1e-7) for key in left_bandwidth):
        return False
    return bool(np.allclose(_array(left.get("delta_v", [])), _array(right.get("delta_v", [])), rtol=1e-6, atol=1e-7))


def _reference_actions(scale_root: Path, split: str, episode_id: str) -> dict[tuple[int, str], dict[str, Any]]:
    rows = _read_csv(scale_root / split / episode_id / "actions.csv")
    result: dict[tuple[int, str], dict[str, Any]] = {}
    for row in rows:
        try:
            # Formal actions use time stamps in seconds and a 0.2 s window.
            step = max(0, int(round(float(row.get("timestamp", "0")) / 0.2)) - 1)
        except (TypeError, ValueError):
            step = 0
        result[(step, str(row.get("uav_id", "")))] = _parse_action(row)
    return result


def _reference_overhead(scale_root: Path, split: str, episode_id: str) -> list[dict[str, str]]:
    return _read_csv(scale_root / split / episode_id / "measurement_overhead.csv")


def _event_for(events: Sequence[Mapping[str, Any]], source: str, target: str) -> Mapping[str, Any] | None:
    for event in events:
        if str(event.get("source_uav_id", "")) == source and str(event.get("target_uav_id", "")) == target:
            return event
    return None


def _edge_pairs(value: Any) -> set[tuple[str, str]]:
    """Normalize edge collections from native info objects or CSV/JSON text."""

    if value is None:
        return set()
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return set()
        try:
            return _edge_pairs(json.loads(text))
        except (TypeError, ValueError, json.JSONDecodeError):
            pairs: set[tuple[str, str]] = set()
            for token in text.split(","):
                parts = token.strip().replace("(", "").replace(")", "").split("->")
                if len(parts) == 2:
                    pairs.add((parts[0].strip().strip("'\""), parts[1].strip().strip("'\"")))
            return pairs
    if isinstance(value, Mapping):
        value = value.get("edges", value.get("edge_index", []))
    pairs = set()
    try:
        items = list(value)
    except TypeError:
        return pairs
    for item in items:
        if isinstance(item, str):
            parts = item.strip().replace("(", "").replace(")", "").split("->")
            if len(parts) == 2:
                pairs.add((parts[0].strip().strip("'\""), parts[1].strip().strip("'\"")))
        elif isinstance(item, Sequence) and not isinstance(item, (bytes, bytearray)) and len(item) >= 2:
            pairs.add((str(item[0]), str(item[1])))
    return pairs


def _event_float(event: Mapping[str, Any] | None, key: str, default: float = -1.0) -> float:
    return _float(event.get(key), default) if event is not None else default


def _event_int(event: Mapping[str, Any] | None, key: str, default: int = 0) -> int:
    if event is None:
        return default
    try:
        return int(bool(event.get(key))) if isinstance(event.get(key), bool) else int(float(event.get(key)))
    except (TypeError, ValueError):
        return default


def _branch_order(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    present = {str(row.get("branch", "")) for row in rows}
    return [branch for branch in SUPPORTED_BRANCHES if branch in present]


def _replay_branch(
    *,
    reference_root: Path,
    base_seed: int,
    branch: str,
    split: str,
    trace_path: Path,
    progress: bool,
    diagnostic_proxy_model: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    scale_root, manifest = _manifest_for(reference_root, base_seed, branch)
    config = _formal_config(manifest, branch)
    metric_scales = dict(config.metric_scales or {})
    model = _load_quality_model(scale_root)
    # The proxy is always used for offline candidate diagnostics when the
    # calibration artifact exists.  It is passed into the actor only when the
    # frozen checkpoint declares proxy features enabled (see _load_trainer).
    # Legacy references may not have this artifact, so keep the old audit
    # compatible by treating it as optional outside the proxy branch.
    proxy_model: dict[str, Any] | None = None
    if diagnostic_proxy_model is not None:
        # Candidate-level diagnostics use one train-fitted proxy per seed so
        # that baseline, adaptive, and proxy-guided replays are compared on
        # an identical execution-visible scale.  This does not feed the
        # proxy into the actor unless the frozen checkpoint declares the
        # proxy feature branch.
        proxy_model = dict(diagnostic_proxy_model)
    else:
        proxy_model_path = scale_root / "calibration" / "measurement_proxy_model.json"
        if proxy_model_path.exists():
            proxy_model = _load_proxy_model(scale_root)
    all_trace_rows: list[dict[str, Any]] = []
    action_mismatches = 0
    reference_action_rows = 0
    overhead_mismatches = 0
    reference_overhead_rows = 0
    replay_overhead_rows = 0
    episode_count = int(config.validation_episodes)
    trainer: MAPPOTrainer | None = None
    uav_ids: tuple[str, ...] = tuple(f"uav_{index:02d}" for index in range(config.uav_count))

    for episode_index in range(episode_count):
        episode_seed = _scale_seed(config.seed, config.uav_count, split, episode_index)
        if progress:
            print(
                f"[policy-audit] seed={base_seed} branch={branch} "
                f"validation episode {episode_index + 1}/{episode_count}",
                flush=True,
            )
        _seed_everything(episode_seed)
        environment = _make_environment(
            config,
            episode_seed,
            metric_scales=metric_scales,
            measurement_overhead_enabled=config.measurement_overhead_enabled,
            measurement_budget_enabled=config.measurement_budget_enabled,
            measurement_budget_by_uav=config.measurement_budget_by_uav,
        )
        if trainer is None:
            trainer, model = _load_trainer(
                config,
                scale_root,
                manifest,
                environment,
                seed=base_seed,
            )
            uav_ids = tuple(environment.uav_ids)
        observations, info = environment.reset(
            seed=episode_seed,
            scenario_config=_scenario_for(config, episode_seed),
        )
        episode_id = f"episode_{400000 + episode_index + 1:06d}"
        expected_actions = _reference_actions(scale_root, split, episode_id)
        expected_overhead = _reference_overhead(scale_root, split, episode_id)
        reference_action_rows += len(expected_actions)
        reference_overhead_rows += len(expected_overhead)
        episode_action_rows = 0
        episode_replay_overhead: list[Mapping[str, Any]] = []
        for step in range(config.max_steps):
            pre_central = dict(info.get("central_state") or {})
            pre_potential = _edge_pairs(pre_central.get("potential_edges", []))
            pre_candidate = _edge_pairs(pre_central.get("candidate_edges", []))
            pre_uwb = _edge_pairs(pre_central.get("uwb_edges", []))
            pre_limits = dict(info.get("measurement_budget_limits") or {})
            pre_remaining = dict(info.get("measurement_budget_remaining") or {})
            pre_used = dict(info.get("measurement_budget_used") or {})

            # The audit is inference-only.  Explicitly disable autograd for
            # the complete actor call so no transient policy graph can be
            # retained across replay steps.
            import torch

            with torch.no_grad():
                decision = trainer.act(
                    observations,
                    info.get("central_state"),
                    deterministic=True,
                )
            _assert_action_feasible(environment, decision.actions)
            submitted_actions = {
                uav_id: dict(decision.actions[uav_id]) for uav_id in uav_ids
            }
            update_event_start = len(getattr(environment, "relative_range_update_events", []))
            next_observations, _rewards, terminated, truncated, next_info = environment.step(
                decision.actions
            )
            post_central = dict(next_info.get("central_state") or {})
            post_potential = _edge_pairs(post_central.get("potential_edges", []))
            post_candidate = _edge_pairs(post_central.get("candidate_edges", []))
            post_uwb = _edge_pairs(post_central.get("uwb_edges", []))
            events = list(next_info.get("measurement_overhead_events", []))
            invalid_events = list(next_info.get("invalid_measurement_action_events", []))
            transition_events = list(next_info.get("measurement_transition_invalidated_events", []))
            gate_events = list(next_info.get("measurement_gate_events", []))
            # ``next_info`` exposes the episode-cumulative update history.
            # For a step-level replay diagnostic, use only the events appended
            # by this environment transition.  Otherwise cumulative events
            # would be counted repeatedly in later windows.
            update_events = list(
                getattr(environment, "relative_range_update_events", [])[update_event_start:]
            )
            episode_replay_overhead.extend(events)
            replay_overhead_rows += len(events)
            final_actions = dict(next_info.get("actions") or {})
            objective_components = dict(next_info.get("objective_components") or {})
            final_remaining = dict(next_info.get("measurement_budget_remaining") or {})
            final_used = dict(next_info.get("measurement_budget_used") or {})

            updates_by_target: dict[str, list[Mapping[str, Any]]] = {}
            for update in update_events:
                updates_by_target.setdefault(str(update.get("target_uav_id", "")), []).append(update)
            accepted_updates = [update for update in update_events if bool(update.get("accepted", False))]
            harmful_updates = [
                update for update in accepted_updates
                if _float(update.get("true_position_error_after_m"), 0.0)
                - _float(update.get("true_position_error_before_m"), 0.0) > 0.001
            ]

            for uav_id in uav_ids:
                raw_local = observations.get(uav_id, {})
                risk, valid = quality_features(raw_local, uav_ids, model)
                sample = decision.samples[uav_id]
                probabilities = _array(sample.measurement_probabilities, dtype=float)
                if proxy_model is not None:
                    computed_proxy_gain, computed_proxy_cost, computed_proxy_valid = information_gain_features(
                        raw_local, uav_ids, proxy_model
                    )
                else:
                    computed_proxy_gain = np.empty(0, dtype=float)
                    computed_proxy_cost = np.empty(0, dtype=float)
                    computed_proxy_valid = np.empty(0, dtype=float)
                # For a proxy-enabled checkpoint the sample arrays are the
                # exact values consumed by its measurement head.  For legacy
                # branches, use the same train-fitted proxy offline so all
                # three branches have a common candidate-level diagnostic.
                sample_proxy_gain = _array(sample.measurement_proxy_gain, dtype=float)
                sample_proxy_cost = _array(sample.measurement_proxy_cost, dtype=float)
                sample_proxy_valid = _array(sample.measurement_proxy_valid, dtype=float)
                proxy_gain = (
                    sample_proxy_gain
                    if sample_proxy_gain.size >= len(uav_ids)
                    else computed_proxy_gain
                )
                proxy_cost = (
                    sample_proxy_cost
                    if sample_proxy_cost.size >= len(uav_ids)
                    else computed_proxy_cost
                )
                proxy_valid = (
                    sample_proxy_valid
                    if sample_proxy_valid.size >= len(uav_ids)
                    else computed_proxy_valid
                )
                submitted = str(submitted_actions[uav_id].get("measurement_uav_id", "none"))
                final_action = dict(final_actions.get(uav_id, {}))
                final_selected = str(final_action.get("measurement_uav_id", "none"))
                submitted_index = uav_ids.index(submitted) if submitted in uav_ids else config.uav_count
                final_index = uav_ids.index(final_selected) if final_selected in uav_ids else config.uav_count
                candidate_ids = sorted(
                    {
                        str(value)
                        for value in raw_local.get("candidate_measurement_sources", [])
                        if str(value) in uav_ids
                    }
                )
                candidate_indices = [uav_ids.index(value) for value in candidate_ids]
                action_mask_values = _array(raw_local.get("action_mask", []), dtype=float)
                proxy_teacher = _teacher_probabilities(
                    proxy_gain,
                    proxy_cost,
                    proxy_valid,
                    action_mask_values,
                    temperature=float(getattr(trainer.config, "measurement_proxy_temperature", 0.25)),
                    communication_cost_weight=float(
                        getattr(trainer.config, "measurement_proxy_communication_cost_weight", 1.0)
                    ),
                )
                teacher_index = int(np.argmax(proxy_teacher)) if proxy_teacher.size else len(uav_ids)
                teacher_choice = (
                    uav_ids[teacher_index] if teacher_index < len(uav_ids) else "none"
                )
                candidate_link_metrics = []
                candidate_links = raw_local.get("candidate_incoming_links", {})
                if isinstance(candidate_links, Mapping):
                    for source_id in candidate_ids:
                        link = candidate_links.get(source_id, {})
                        if not isinstance(link, Mapping):
                            link = {}
                        candidate_link_metrics.append(
                            {
                                "source_uav_id": source_id,
                                "rssi_dbm": _float(link.get("rssi_dbm"), -100.0),
                                "snr_db": _float(link.get("snr_db"), 0.0),
                                "packet_loss_pred": _float(link.get("packet_loss"), 1.0),
                                "delay_pred_s": _float(link.get("delay_s"), 0.0),
                                "aoi_pred_s": _float(link.get("aoi_s"), 0.0),
                                "queue_length_pred": _float(link.get("queue_length_packets"), 0.0),
                                "message_age_s": _float(link.get("aoi_s"), 0.0),
                                "link_state": str(link.get("link_state", "unknown")),
                            }
                        )
                submitted_pair = (submitted, uav_id)
                final_pair = (final_selected, uav_id)
                invalid_event = (
                    _event_for(invalid_events, submitted, uav_id)
                    if submitted != "none" else None
                )
                transition_event = (
                    _event_for(transition_events, submitted, uav_id)
                    if submitted != "none" else None
                )
                event = (
                    _event_for(events, final_selected, uav_id)
                    if final_selected != "none" else None
                )
                gate = (
                    _event_for(gate_events, final_selected, uav_id)
                    if final_selected != "none" else None
                )
                target_updates = updates_by_target.get(uav_id, [])
                target_accepted = [update for update in target_updates if bool(update.get("accepted", False))]
                target_harmful = [
                    update for update in target_accepted
                    if _float(update.get("true_position_error_after_m"), 0.0)
                    - _float(update.get("true_position_error_before_m"), 0.0) > 0.001
                ]
                budget_limit = _float(pre_limits.get(uav_id), -1.0)
                budget_remaining_before = _float(pre_remaining.get(uav_id), -1.0)
                budget_used_before = _float(pre_used.get(uav_id), -1.0)
                budget_remaining_after = _float(final_remaining.get(uav_id), -1.0)
                budget_used_after = _float(final_used.get(uav_id), -1.0)
                budget_consumed = max(0.0, budget_used_after - budget_used_before) if budget_used_before >= 0.0 else 0.0
                post_motion_edge = int(submitted != "none" and submitted_pair in post_potential)
                final_potential_edge = int(final_selected != "none" and final_pair in post_potential)
                final_candidate_edge = int(final_selected != "none" and final_pair in post_candidate)
                final_uwb_available = int(final_selected != "none" and final_pair in post_uwb)
                effective_measurement = int(final_potential_edge and final_selected != "none")
                row: dict[str, Any] = {
                    "base_seed": int(base_seed),
                    "episode_seed": int(episode_seed),
                    "split": split,
                    "phase": "validation",
                    "branch": branch,
                    "episode_id": episode_id,
                    "step": int(step),
                    "timestamp": float((step + 1) * environment.config.motion.dt_s),
                    "uav_id": uav_id,
                    "candidate_uav_ids": candidate_ids,
                    "quality_risk": risk.tolist(),
                    "quality_valid": valid.tolist(),
                    "candidate_quality_risk": risk[candidate_indices].tolist() if candidate_indices else [],
                    "candidate_quality_valid": valid[candidate_indices].tolist() if candidate_indices else [],
                    "measurement_probabilities": probabilities.tolist(),
                    "candidate_measurement_probabilities": probabilities[candidate_indices].tolist() if candidate_indices else [],
                    "candidate_link_metrics": candidate_link_metrics,
                    "measurement_proxy_gain": _array(proxy_gain, dtype=float).tolist(),
                    "measurement_proxy_cost": _array(proxy_cost, dtype=float).tolist(),
                    "measurement_proxy_valid": _array(proxy_valid, dtype=float).tolist(),
                    "measurement_proxy_utility": (
                        _array(proxy_gain, dtype=float)
                        - float(getattr(trainer.config, "measurement_proxy_communication_cost_weight", 1.0))
                        * _array(proxy_cost, dtype=float)
                    ).tolist()
                    if _array(proxy_gain, dtype=float).size == _array(proxy_cost, dtype=float).size
                    else [],
                    "candidate_proxy_gain": _array(proxy_gain, dtype=float)[candidate_indices].tolist() if candidate_indices and _array(proxy_gain, dtype=float).size > max(candidate_indices) else [],
                    "candidate_proxy_cost": _array(proxy_cost, dtype=float)[candidate_indices].tolist() if candidate_indices and _array(proxy_cost, dtype=float).size > max(candidate_indices) else [],
                    "candidate_proxy_valid": _array(proxy_valid, dtype=float)[candidate_indices].tolist() if candidate_indices and _array(proxy_valid, dtype=float).size > max(candidate_indices) else [],
                    "candidate_proxy_utility": (
                        (
                            _array(proxy_gain, dtype=float)
                            - float(getattr(trainer.config, "measurement_proxy_communication_cost_weight", 1.0))
                            * _array(proxy_cost, dtype=float)
                        )[candidate_indices].tolist()
                        if candidate_indices
                        and _array(proxy_gain, dtype=float).size > max(candidate_indices)
                        and _array(proxy_cost, dtype=float).size > max(candidate_indices)
                        else []
                    ),
                    "proxy_model_source": "train" if proxy_model is not None else "unavailable",
                    "proxy_features_in_actor": int(bool(trainer.config.use_measurement_proxy_features)),
                    "proxy_teacher_probabilities": proxy_teacher.tolist(),
                    "proxy_teacher_choice": teacher_choice,
                    "proxy_teacher_choice_probability": float(proxy_teacher[teacher_index]) if teacher_index < proxy_teacher.size else 0.0,
                    "proxy_teacher_actor_agreement": int(teacher_choice == submitted),
                    "none_probability": float(probabilities[-1]) if probabilities.size > config.uav_count else 0.0,
                    # Backward-compatible name: this is the submitted action.
                    "selected_measurement_uav_id": submitted,
                    "submitted_measurement_uav_id": submitted,
                    "final_measurement_uav_id": final_selected,
                    "selected_probability": float(probabilities[submitted_index]) if submitted_index < probabilities.size else 0.0,
                    "selected_quality_risk": float(risk[submitted_index]) if submitted_index < risk.size else -1.0,
                    "selected_quality_valid": float(valid[submitted_index]) if submitted_index < valid.size else 0.0,
                    "quality_feature_in_actor": int(bool(trainer.config.use_measurement_quality_features)),
                    "quality_model_source": "warmup",
                    "action_mask": action_mask_values.astype(int).tolist(),
                    "action_mask_allows_submitted": int(
                        submitted == "none"
                        or submitted_index < _array(raw_local.get("action_mask", []), dtype=int).size
                        and bool(_array(raw_local.get("action_mask", []), dtype=int)[submitted_index])
                    ),
                    "pre_action_potential_edge": int(submitted != "none" and submitted_pair in pre_potential),
                    "pre_action_candidate_edge": int(submitted != "none" and submitted_pair in pre_candidate),
                    "pre_action_uwb_available": int(submitted != "none" and submitted_pair in pre_uwb),
                    "post_motion_potential_edge": post_motion_edge,
                    "final_potential_edge": final_potential_edge,
                    "final_candidate_edge": final_candidate_edge,
                    "final_uwb_available": final_uwb_available,
                    "effective_measurement": effective_measurement,
                    "invalid_measurement_action": int(invalid_event is not None),
                    "invalid_measurement_action_reason": str(invalid_event.get("reason", "")) if invalid_event else "",
                    "measurement_transition_invalidated": int(transition_event is not None),
                    "measurement_transition_reason": str(transition_event.get("reason", "")) if transition_event else "",
                    "overhead_event_present": int(event is not None),
                    "payload_size_bytes": int(event.get("payload_size_bytes", 0)) if event else 0,
                    "load_packets_requested": _event_float(event, "load_packets_requested", 0.0),
                    "load_packets_applied": _event_float(event, "load_packets_applied", 0.0),
                    "potential_edge": _event_int(event, "potential_edge", final_potential_edge),
                    "candidate_edge": _event_int(event, "candidate_edge", final_candidate_edge),
                    "effective_edge": _event_int(event, "effective_edge", int(final_selected != "none" and final_pair in _edge_pairs(post_central.get("effective_edges", [])))),
                    "queue_before_packets": _event_float(event, "queue_before_packets", -1.0),
                    "queue_with_measurement_packets": _event_float(event, "queue_with_measurement_packets", -1.0),
                    "queue_next_packets": _event_float(event, "queue_next_packets", -1.0),
                    "channel_bandwidth_hz": _event_float(event, "channel_bandwidth_hz", -1.0),
                    "rate_bps": _event_float(event, "rate_bps", -1.0),
                    "packet_loss": _event_float(event, "packet_loss", -1.0),
                    "delay_s": _event_float(event, "delay_s", -1.0),
                    "aoi_s": _event_float(event, "aoi_s", -1.0),
                    "link_state": str(event.get("link_state", "not_selected")) if event else "not_selected",
                    "uwb_update_opportunity_step": _event_int(gate, "uwb_valid", 0),
                    "ekf_update_attempted_step": _event_int(gate, "ekf_update_attempted", 0),
                    "ekf_update_accepted_step": _event_int(gate, "ekf_update_accepted", 0),
                    "gate_failure_stage": str(gate.get("failure_stage", "not_selected")) if gate else "not_selected",
                    "accepted_update_count_step": int(len(target_accepted)),
                    "harmful_update_count_step": int(len(target_harmful)),
                    "accepted_update_count_all_targets_step": int(len(accepted_updates)),
                    "harmful_update_count_all_targets_step": int(len(harmful_updates)),
                    "j_loc": _float(objective_components.get("j_loc"), 0.0),
                    "j_com": _float(objective_components.get("j_com"), 0.0),
                    "joint_objective": _float(objective_components.get("joint_objective"), 0.0),
                    "team_reward": _float(next_info.get("team_reward"), 0.0),
                    "measurement_budget_enabled": int(bool(next_info.get("measurement_budget_enabled", False))),
                    "budget_limit": budget_limit,
                    "budget_remaining_before": budget_remaining_before,
                    "budget_remaining_after": budget_remaining_after,
                    "budget_used_before": budget_used_before,
                    "budget_used_after": budget_used_after,
                    "budget_consumed_step": budget_consumed,
                    "budget_exhausted_before": int(budget_remaining_before == 0.0),
                    "budget_overrun": int(budget_limit >= 0.0 and budget_used_after > budget_limit + 1e-9),
                }
                if not _finite_trace_value(row):
                    raise RuntimeError(f"non-finite policy trace row: {row}")
                _append_jsonl(trace_path, row)
                all_trace_rows.append(row)
                expected = expected_actions.get((step, uav_id))
                if expected is not None:
                    episode_action_rows += 1
                    if not _actions_match(decision.actions[uav_id], expected):
                        action_mismatches += 1
            observations, info = next_observations, next_info
            if terminated or truncated:
                break
        if episode_action_rows != len(expected_actions):
            action_mismatches += abs(episode_action_rows - len(expected_actions))
        if len(episode_replay_overhead) != len(expected_overhead):
            overhead_mismatches += abs(len(episode_replay_overhead) - len(expected_overhead))
        else:
            for replay, expected in zip(episode_replay_overhead, expected_overhead):
                for key in (
                    "source_uav_id", "target_uav_id", "payload_size_bytes",
                    "load_packets_requested", "load_packets_applied",
                    "potential_edge", "candidate_edge", "effective_edge",
                    "queue_before_packets", "queue_with_measurement_packets",
                    "queue_next_packets",
                ):
                    if key in {"source_uav_id", "target_uav_id"}:
                        if str(replay.get(key, "")) != str(expected.get(key, "")):
                            overhead_mismatches += 1
                    elif not math.isclose(
                        _float(replay.get(key, 0.0), 0.0),
                        _float(expected.get(key, 0.0), 0.0),
                        rel_tol=1e-6,
                        abs_tol=1e-7,
                    ):
                        overhead_mismatches += 1

        # ``info`` contains cumulative diagnostic snapshots.  Release the
        # temporary episode references before starting the next replay; the
        # compact audit trace is already serialized above.
        del environment
        del observations, info, next_observations, next_info
        gc.collect()

    return {
        "rows": all_trace_rows,
        "action_mismatches": int(action_mismatches),
        "reference_action_rows": int(reference_action_rows),
        "replay_overhead_rows": int(replay_overhead_rows),
        "reference_overhead_rows": int(reference_overhead_rows),
        "overhead_mismatches": int(overhead_mismatches),
        "config": config,
        "quality_model": model,
    }


def _quality_log_summary(reference_root: Path, seed: int, branch: str) -> list[dict[str, Any]]:
    scale_root, _manifest = _manifest_for(reference_root, seed, branch)
    rows = _read_jsonl(scale_root / "diagnostics" / "measurement_quality.jsonl")
    output: list[dict[str, Any]] = []
    for split in ("train", "validation"):
        selected = [row for row in rows if str(row.get("split")) == split]
        valid = [_float(row.get("quality_valid_count"), 0.0) for row in selected]
        risk = [
            _float(row.get("quality_mean_risk"), float("nan"))
            for row in selected
            if _float(row.get("quality_mean_risk"), float("nan")) >= 0.0
        ]
        output.append(
            {
                "base_seed": int(seed),
                "branch": branch,
                "split": split,
                "quality_feature_enabled": int(
                    branch in {"quality_aware_mappo", "budgeted_quality_aware_mappo"}
                ),
                "log_rows": len(selected),
                "mean_quality_valid_count": _safe_mean(valid),
                "valid_row_fraction": _safe_mean([value > 0.0 for value in valid]),
                "mean_logged_risk": _safe_mean(risk, default=-1.0),
                "min_logged_risk": float(np.min(risk)) if risk else -1.0,
                "max_logged_risk": float(np.max(risk)) if risk else -1.0,
                "quality_model_source": "warmup",
            }
        )
    return output


def _trace_action_summary(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[int, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault((int(row["base_seed"]), str(row["branch"])), []).append(row)
    output: list[dict[str, Any]] = []
    for (seed, branch), group in sorted(groups.items()):
        risk_values: list[float] = []
        probability_values: list[float] = []
        valid_values: list[float] = []
        none_probabilities = [_float(row.get("none_probability"), 0.0) for row in group]
        selected_non_none = [str(row.get("selected_measurement_uav_id")) != "none" for row in group]
        selected_risk = [_float(row.get("selected_quality_risk"), -1.0) for row in group if str(row.get("selected_measurement_uav_id")) != "none"]
        for row in group:
            risk = _array(row.get("quality_risk", []), dtype=float)
            probability = _array(row.get("measurement_probabilities", []), dtype=float)
            valid = _array(row.get("quality_valid", []), dtype=float)
            length = min(risk.size, probability.size, valid.size)
            risk_values.extend(risk[:length].tolist())
            probability_values.extend(probability[:length].tolist())
            valid_values.extend(valid[:length].tolist())
        valid_mask = np.asarray(valid_values, dtype=float) > 0.5
        valid_risk = np.asarray(risk_values, dtype=float)[valid_mask]
        valid_probability = np.asarray(probability_values, dtype=float)[valid_mask]
        output.append(
            {
                "base_seed": seed,
                "branch": branch,
                "validation_rows": len(group),
                "candidate_slot_rows": len(risk_values),
                "valid_candidate_slot_rows": int(valid_mask.sum()),
                "valid_candidate_slot_fraction": float(valid_mask.mean()) if valid_mask.size else 0.0,
                "risk_probability_pearson_r": _pearson(valid_risk, valid_probability),
                "none_selection_rate": 1.0 - _safe_mean(selected_non_none, default=0.0),
                "measurement_selection_rate": _safe_mean(selected_non_none, default=0.0),
                "mean_none_probability": _safe_mean(none_probabilities),
                "mean_selected_quality_risk": _safe_mean(selected_risk, default=-1.0),
                "mean_selected_probability": _safe_mean(
                    [
                        _float(row.get("selected_probability"), 0.0)
                        for row in group
                        if str(row.get("selected_measurement_uav_id")) != "none"
                    ],
                    default=0.0,
                ),
                "candidate_risk_mean": _safe_mean(
                    [
                        _float(value)
                        for row in group
                        for value in _array(row.get("candidate_quality_risk", []), dtype=float)
                    ],
                    default=-1.0,
                ),
                "quality_feature_in_actor": int(
                    branch in {"quality_aware_mappo", "budgeted_quality_aware_mappo"}
                ),
            }
        )
    return output


def _trace_overhead_summary(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[int, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault((int(row["base_seed"]), str(row["branch"])), []).append(row)
    output: list[dict[str, Any]] = []
    for (seed, branch), group in sorted(groups.items()):
        selected = [row for row in group if str(row.get("selected_measurement_uav_id")) != "none"]
        events = [row for row in selected if int(row.get("overhead_event_present", 0)) == 1]
        queue_rows = [row for row in events if _float(row.get("queue_before_packets"), -1.0) >= 0.0]
        effective = [row for row in group if int(row.get("effective_measurement", 0)) == 1]
        invalid = [row for row in group if int(row.get("invalid_measurement_action", 0)) == 1]
        transition = [row for row in group if int(row.get("measurement_transition_invalidated", 0)) == 1]
        output.append(
            {
                "base_seed": seed,
                "branch": branch,
                "decision_rows": len(group),
                "measurement_count": len(selected),
                "effective_measurement_count": len(effective),
                "invalid_measurement_action_count": len(invalid),
                "transition_invalidation_count": len(transition),
                "invalid_measurement_action_rate": len(invalid) / max(1, len(group)),
                "transition_invalidation_rate": len(transition) / max(1, len(group)),
                "overhead_event_count": len(events),
                "payload_size_bytes_ok": int(
                    all(
                        int(row.get("payload_size_bytes", 0)) == 512
                        and math.isclose(_float(row.get("load_packets_requested"), 0.0), 1.0)
                        and math.isclose(_float(row.get("load_packets_applied"), 0.0), 1.0)
                        for row in effective
                    )
                ),
                "load_packets_requested": _safe_mean(
                    [_float(row.get("load_packets_requested"), 0.0) for row in events], default=0.0
                ) * len(events),
                "load_packets_applied": _safe_mean(
                    [_float(row.get("load_packets_applied"), 0.0) for row in events], default=0.0
                ) * len(events),
                "mean_queue_before_packets": _safe_mean(
                    [_float(row.get("queue_before_packets"), float("nan")) for row in queue_rows]
                ),
                "mean_queue_with_measurement_packets": _safe_mean(
                    [_float(row.get("queue_with_measurement_packets"), float("nan")) for row in queue_rows]
                ),
                "mean_queue_next_packets": _safe_mean(
                    [_float(row.get("queue_next_packets"), float("nan")) for row in queue_rows]
                ),
                "mean_rate_bps": _safe_mean(
                    [_float(row.get("rate_bps"), float("nan")) for row in events]
                ),
                "mean_packet_loss": _safe_mean(
                    [_float(row.get("packet_loss"), float("nan")) for row in events]
                ),
                "mean_delay_s": _safe_mean(
                    [_float(row.get("delay_s"), float("nan")) for row in events]
                ),
                "mean_aoi_s": _safe_mean(
                    [_float(row.get("aoi_s"), float("nan")) for row in events]
                ),
                "queue_increase_fraction": _safe_mean(
                    [
                        _float(row.get("queue_with_measurement_packets"), -1.0)
                        >= _float(row.get("queue_before_packets"), -1.0) - 1e-9
                        for row in queue_rows
                    ],
                    default=1.0,
                ),
                "mean_j_com_on_events": _safe_mean(
                    [_float(row.get("j_com"), float("nan")) for row in events]
                ),
            }
        )
    return output


def _trace_transition_summary(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Summarize pre-action invalidity separately from post-motion invalidation."""

    groups: dict[tuple[int, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault((int(row["base_seed"]), str(row["branch"])), []).append(row)
    output: list[dict[str, Any]] = []
    for (seed, branch), group in sorted(groups.items()):
        output.append(_transition_summary_row(group, base_seed=seed, branch=branch, aggregation_level="seed_branch"))
    return output


def _transition_summary_row(
    group: Sequence[Mapping[str, Any]],
    *,
    base_seed: int | str | None,
    branch: str,
    aggregation_level: str,
) -> dict[str, Any]:
    """Summarize one transition-invalidation population with explicit denominators.

    A post-motion invalidation is a legitimate pre-motion action whose selected
    edge leaves the potential graph after the commanded motion.  It is only an
    actionable timing concern when the population has enough submitted actions
    and its aggregate rate exceeds the fixed threshold.
    """

    submitted = [
        row
        for row in group
        if str(row.get("submitted_measurement_uav_id", row.get("selected_measurement_uav_id", "none")))
        != "none"
    ]
    invalid = [row for row in group if int(row.get("invalid_measurement_action", 0)) == 1]
    transition = [
        row for row in group if int(row.get("measurement_transition_invalidated", 0)) == 1
    ]
    post_loss = [
        row
        for row in submitted
        if int(row.get("post_motion_potential_edge", 1)) == 0
        and int(row.get("invalid_measurement_action", 0)) == 0
    ]
    effective = [row for row in group if int(row.get("effective_measurement", 0)) == 1]
    denominator = len(submitted)
    rate = len(transition) / denominator if denominator else 0.0
    small_denominator = denominator < TRANSITION_MIN_DENOMINATOR
    actionable = (not small_denominator) and rate > TRANSITION_ACTIONABLE_RATE
    if denominator == 0:
        timing_status = "no_submitted_measurements"
    elif small_denominator:
        timing_status = "insufficient_denominator"
    elif actionable:
        timing_status = "actionable_post_motion_invalidation"
    else:
        timing_status = "low_rate_expected_link_breakage"
    return {
        "aggregation_level": aggregation_level,
        "base_seed": base_seed,
        "branch": branch,
        "decision_rows": len(group),
        "submitted_non_none_count": denominator,
        "invalid_measurement_action_count": len(invalid),
        "invalid_measurement_action_rate": len(invalid) / max(1, len(group)),
        "transition_invalidation_count": len(transition),
        "transition_invalidation_denominator": denominator,
        "transition_invalidation_rate": rate,
        "small_denominator": int(small_denominator),
        "actionable_transition_invalidation": int(actionable),
        "transition_timing_status": timing_status,
        "post_motion_potential_count": sum(
            int(row.get("post_motion_potential_edge", 0)) for row in submitted
        ),
        "post_motion_edge_loss_count": len(post_loss),
        "final_effective_measurement_count": len(effective),
        "final_effective_measurement_rate": len(effective) / max(1, len(group)),
    }


def _trace_transition_aggregates(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Return branch-, seed-, and global-level transition summaries."""

    by_branch_groups: dict[str, list[Mapping[str, Any]]] = {}
    by_seed_groups: dict[int, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_branch_groups.setdefault(str(row["branch"]), []).append(row)
        by_seed_groups.setdefault(int(row["base_seed"]), []).append(row)
    by_branch = [
        _transition_summary_row(
            group,
            base_seed=None,
            branch=branch,
            aggregation_level="branch",
        )
        for branch, group in sorted(by_branch_groups.items())
    ]
    by_seed = [
        _transition_summary_row(
            group,
            base_seed=seed,
            branch="all",
            aggregation_level="seed",
        )
        for seed, group in sorted(by_seed_groups.items())
    ]
    global_summary = _transition_summary_row(
        rows,
        base_seed=None,
        branch="all",
        aggregation_level="global",
    )
    return {"by_branch": by_branch, "by_seed": by_seed, "global": global_summary}


def _trace_budget_summary(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Summarize budget use without treating time windows as independent samples."""

    groups: dict[tuple[int, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault((int(row["base_seed"]), str(row["branch"])), []).append(row)
    output: list[dict[str, Any]] = []
    for (seed, branch), group in sorted(groups.items()):
        enabled = any(int(row.get("measurement_budget_enabled", 0)) == 1 for row in group)
        limits_by_uav: dict[str, float] = {}
        for row in group:
            limit = _float(row.get("budget_limit"), -1.0)
            if limit >= 0.0:
                limits_by_uav.setdefault(str(row.get("uav_id", "")), limit)
        limits = list(limits_by_uav.values())
        consumed = sum(_float(row.get("budget_consumed_step"), 0.0) for row in group)
        overruns = sum(int(row.get("budget_overrun", 0)) for row in group)
        exhausted = sum(int(row.get("budget_exhausted_before", 0)) for row in group)
        max_used = max((_float(row.get("budget_used_after"), 0.0) for row in group), default=0.0)
        remaining = min((_float(row.get("budget_remaining_after"), 0.0) for row in group), default=0.0)
        limit_total = sum(limits) if limits else -1.0
        output.append(
            {
                "base_seed": seed,
                "branch": branch,
                "budget_enabled": int(enabled),
                "budget_limit_min": min(limits) if limits else -1.0,
                "budget_limit_max": max(limits) if limits else -1.0,
                "budget_limit_total": limit_total,
                "budget_consumed_count": consumed,
                "budget_overrun_count": overruns,
                "max_budget_used": max_used,
                "max_budget_remaining": remaining,
                "budget_exhausted_rows": exhausted,
                "budget_usage_rate": consumed / limit_total if limit_total > 0.0 else 0.0,
            }
        )
    return output


def _reference_metrics(
    reference_root: Path,
    seeds: Sequence[int],
    branches: Sequence[str] = DEFAULT_BRANCHES,
) -> list[dict[str, str]]:
    root_manifest = _read_json(reference_root / "manifest.json")
    output_path = root_manifest.get("outputs", {}).get("episode_metrics")
    candidates = []
    if output_path:
        candidates.append(Path(str(output_path)))
    candidates.append(Path("logs/preliminary") / reference_root.name / "episode_metrics.csv")
    metrics_path = next((path for path in candidates if path.exists()), None)
    if metrics_path is None:
        raise FileNotFoundError(f"cannot locate episode metrics for {reference_root}")
    rows = _read_csv(metrics_path)
    seed_set = {str(int(seed)) for seed in seeds}
    return [
        row
        for row in rows
        if str(row.get("split")) == "validation"
        and str(row.get("base_seed", row.get("seed", ""))) in seed_set
        and str(row.get("treatment", "")) in set(branches)
    ]


def _seed_metric_rows(metrics: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[int, str], list[Mapping[str, Any]]] = {}
    for row in metrics:
        groups.setdefault((int(float(row["base_seed"])), str(row["treatment"])), []).append(row)
    numeric = (
        "mean_reward", "position_rmse_m", "position_error_p95_m", "j_loc", "j_com",
        "accepted_update_count", "beneficial_fraction", "harmful_fraction", "neutral_fraction",
        "packet_loss", "delay_s", "aoi_s", "connectivity", "energy_proxy", "collision_rate",
        "none_selection_rate",
    )
    output: list[dict[str, Any]] = []
    for (seed, treatment), rows in sorted(groups.items()):
        item: dict[str, Any] = {"base_seed": seed, "branch": treatment, "source": "reference_episode_metrics"}
        for key in numeric:
            item[key] = _safe_mean([_float(row.get(key)) for row in rows])
        output.append(item)
    return output


def _bootstrap_ci(values: Sequence[float], *, samples: int, seed: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    if values.size == 1:
        value = float(values[0])
        return value, value
    rng = np.random.default_rng(seed)
    sampled = rng.choice(values, size=(int(samples), values.size), replace=True).mean(axis=1)
    return float(np.quantile(sampled, 0.025)), float(np.quantile(sampled, 0.975))


def _write_plot(
    plot_path: Path,
    action_rows: Sequence[Mapping[str, Any]],
    overhead_rows: Sequence[Mapping[str, Any]],
    metric_rows: Sequence[Mapping[str, Any]],
) -> None:
    import matplotlib.pyplot as plt

    branches = _branch_order(list(action_rows) + list(overhead_rows) + list(metric_rows))
    labels = {
        "baseline_mappo": "baseline",
        "quality_aware_mappo": "quality-aware",
        "budgeted_quality_aware_mappo": "budgeted-quality",
    }
    colors = {
        "baseline_mappo": "#2f6de1",
        "quality_aware_mappo": "#d92d2d",
        "budgeted_quality_aware_mappo": "#1b9e5a",
    }
    seeds = sorted({int(row["base_seed"]) for row in action_rows})
    action_by_branch_seed = {
        (str(row.get("branch")), int(row.get("base_seed", -1))): row
        for row in action_rows
    }
    metric_by_branch_seed = {
        (str(row.get("branch")), int(row.get("base_seed", -1))): row
        for row in metric_rows
    }
    overhead_by_branch_seed = {
        (str(row.get("branch")), int(row.get("base_seed", -1))): row
        for row in overhead_rows
    }
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)

    def add_legend(axis: Any) -> None:
        handles, legend_labels = axis.get_legend_handles_labels()
        if handles:
            axis.legend()

    ax = axes[0, 0]
    for branch in branches:
        values = [
            _float(action_by_branch_seed.get((branch, seed), {}).get("none_selection_rate"), float("nan"))
            for seed in seeds
        ]
        if any(math.isfinite(value) for value in values):
            ax.plot(seeds, values, marker="o", label=labels[branch], color=colors[branch])
    ax.set_title("None selection rate")
    ax.set_xlabel("base seed")
    ax.set_ylabel("fraction")
    ax.set_ylim(0.0, 1.0)
    add_legend(ax)

    ax = axes[0, 1]
    for branch in branches:
        risk: list[float] = []
        probability: list[float] = []
        for row in action_rows:
            if row.get("branch") != branch:
                continue
            # The summary does not retain the raw slots; the trace scatter is
            # added by the caller through the temporary columns below when
            # available.  A summary point is still useful for both branches.
            risk.append(_float(row.get("mean_selected_quality_risk"), float("nan")))
            probability.append(_float(row.get("mean_selected_probability"), float("nan")))
        if risk:
            ax.scatter(risk, probability, label=labels[branch], color=colors[branch], alpha=0.8)
    ax.set_title("Selected risk vs probability (seed means)")
    ax.set_xlabel("selected quality risk")
    ax.set_ylabel("selected probability")
    add_legend(ax)

    ax = axes[0, 2]
    for branch in branches:
        xs = []
        ys = []
        for seed in seeds:
            row = overhead_by_branch_seed.get((branch, seed), {})
            xs.append(_float(row.get("measurement_count"), 0.0))
            ys.append(_float(row.get("load_packets_requested"), 0.0))
        if xs:
            ax.scatter(xs, ys, label=labels[branch], color=colors[branch])
    ax.set_title("Measurement count → packet load")
    ax.set_xlabel("measurement decisions")
    ax.set_ylabel("equivalent packets")
    add_legend(ax)

    ax = axes[1, 0]
    for branch in branches:
        xs = []
        ys = []
        for row in metric_rows:
            if row.get("branch") == branch:
                xs.append(_float(row.get("accepted_update_count"), 0.0))
                ys.append(_float(row.get("j_com"), float("nan")))
        if xs:
            ax.scatter(xs, ys, label=labels[branch], color=colors[branch])
    ax.set_title("Accepted updates vs communication cost")
    ax.set_xlabel("accepted EKF updates")
    ax.set_ylabel("J_com")
    add_legend(ax)

    ax = axes[1, 1]
    width = 0.75 / max(1, len(branches))
    x = np.arange(len(seeds))
    for offset, branch in enumerate(branches):
        values = [
            _float(overhead_by_branch_seed.get((branch, seed), {}).get("mean_packet_loss"), 0.0)
            for seed in seeds
        ]
        ax.bar(
            x + (offset - (len(branches) - 1) / 2) * width,
            values,
            width=width,
            label=labels[branch],
            color=colors[branch],
        )
    ax.set_title("Mean packet loss of selected measurement links")
    ax.set_xlabel("base seed")
    ax.set_ylabel("packet loss")
    ax.set_xticks(x, [str(seed) for seed in seeds])
    add_legend(ax)

    ax = axes[1, 2]
    for branch in branches:
        values = [
            _float(metric_by_branch_seed.get((branch, seed), {}).get("harmful_fraction"), float("nan"))
            for seed in seeds
        ]
        if any(math.isfinite(value) for value in values):
            ax.plot(seeds, values, marker="o", label=labels[branch], color=colors[branch])
    ax.set_title("Harmful update fraction")
    ax.set_xlabel("base seed")
    ax.set_ylabel("fraction")
    add_legend(ax)

    fig.suptitle("Policy mechanism audit — validation diagnostic only", fontsize=14)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)


def run_policy_mechanism_audit(config: PolicyMechanismAuditConfig) -> dict[str, Any]:
    """Run the deterministic, validation-only mechanism audit.

    The replay is deliberately diagnostic: it never writes episode data and it
    does not turn individual windows or EKF updates into independent samples.
    """

    reference_root = Path(config.reference_root).resolve()
    output_dir = Path(config.output_dir).resolve()
    plots_dir = Path(config.plots_dir).resolve() if config.plots_dir else (
        Path("plots/preliminary") / reference_root.name / "policy_mechanism_audit"
    ).resolve()
    _prepare_output(output_dir, force=config.force)
    plots_dir.mkdir(parents=True, exist_ok=True)
    trace_path = output_dir / "measurement_policy_trace.jsonl"
    if trace_path.exists() and config.force:
        trace_path.unlink()

    all_trace_rows: list[dict[str, Any]] = []
    replay_checks: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    for seed in config.seeds:
        for branch in config.branches:
            if config.progress:
                print(f"[policy-audit] start seed={seed} branch={branch}", flush=True)
            feature_rows.extend(_quality_log_summary(reference_root, int(seed), branch))
            replay = _replay_branch(
                reference_root=reference_root,
                base_seed=int(seed),
                branch=branch,
                split=config.split,
                trace_path=trace_path,
                progress=config.progress,
            )
            all_trace_rows.extend(replay["rows"])
            replay_checks.append(
                {
                    "base_seed": int(seed),
                    "branch": branch,
                    "action_mismatches": replay["action_mismatches"],
                    "reference_action_rows": replay["reference_action_rows"],
                    "reference_overhead_rows": replay["reference_overhead_rows"],
                    "replay_overhead_rows": replay["replay_overhead_rows"],
                    "overhead_mismatches": replay["overhead_mismatches"],
                    "quality_model_source": replay["quality_model"].get("source_split"),
                    "measurement_budget_enabled": bool(replay["config"].measurement_budget_enabled),
                    "measurement_budget_by_uav": dict(replay["config"].measurement_budget_by_uav or {}),
                }
            )

    action_summary = _trace_action_summary(all_trace_rows)
    overhead_summary = _trace_overhead_summary(all_trace_rows)
    transition_summary = _trace_transition_summary(all_trace_rows)
    transition_aggregates = _trace_transition_aggregates(all_trace_rows)
    budget_summary = _trace_budget_summary(all_trace_rows)
    reference_metrics = _reference_metrics(reference_root, config.seeds, config.branches)
    metric_summary = _seed_metric_rows(reference_metrics)
    combined: list[dict[str, Any]] = []
    action_by_key = {(int(row["base_seed"]), str(row["branch"])): row for row in action_summary}
    overhead_by_key = {(int(row["base_seed"]), str(row["branch"])): row for row in overhead_summary}
    transition_by_key = {(int(row["base_seed"]), str(row["branch"])): row for row in transition_summary}
    budget_by_key = {(int(row["base_seed"]), str(row["branch"])): row for row in budget_summary}
    metric_by_key = {(int(row["base_seed"]), str(row["branch"])): row for row in metric_summary}
    for seed in config.seeds:
        for branch in config.branches:
            row: dict[str, Any] = {"base_seed": int(seed), "branch": branch}
            row.update(action_by_key.get((int(seed), branch), {}))
            row.update(overhead_by_key.get((int(seed), branch), {}))
            row.update(transition_by_key.get((int(seed), branch), {}))
            row.update(budget_by_key.get((int(seed), branch), {}))
            row.update(metric_by_key.get((int(seed), branch), {}))
            combined.append(row)

    feature_columns = (
        "base_seed", "branch", "split", "quality_feature_enabled", "log_rows",
        "mean_quality_valid_count", "valid_row_fraction", "mean_logged_risk",
        "min_logged_risk", "max_logged_risk", "quality_model_source",
    )
    _write_csv(output_dir / "policy_feature_summary.csv", feature_columns, feature_rows)
    action_columns = (
        "base_seed", "branch", "validation_rows", "candidate_slot_rows",
        "valid_candidate_slot_rows", "valid_candidate_slot_fraction",
        "risk_probability_pearson_r", "none_selection_rate",
        "measurement_selection_rate", "mean_none_probability",
        "mean_selected_quality_risk", "mean_selected_probability",
        "candidate_risk_mean", "quality_feature_in_actor",
    )
    _write_csv(output_dir / "action_probability_summary.csv", action_columns, action_summary)
    overhead_columns = (
        "base_seed", "branch", "decision_rows", "measurement_count",
        "effective_measurement_count", "invalid_measurement_action_count",
        "transition_invalidation_count", "invalid_measurement_action_rate",
        "transition_invalidation_rate", "overhead_event_count",
        "payload_size_bytes_ok", "load_packets_requested", "load_packets_applied",
        "mean_queue_before_packets", "mean_queue_with_measurement_packets",
        "mean_queue_next_packets", "mean_rate_bps", "mean_packet_loss",
        "mean_delay_s", "mean_aoi_s", "queue_increase_fraction",
        "mean_j_com_on_events",
    )
    _write_csv(output_dir / "measurement_overhead_summary.csv", overhead_columns, overhead_summary)
    transition_columns = (
        "aggregation_level", "base_seed", "branch", "decision_rows", "submitted_non_none_count",
        "invalid_measurement_action_count", "invalid_measurement_action_rate",
        "transition_invalidation_count", "transition_invalidation_denominator",
        "transition_invalidation_rate", "small_denominator",
        "actionable_transition_invalidation", "transition_timing_status",
        "post_motion_potential_count", "post_motion_edge_loss_count",
        "final_effective_measurement_count", "final_effective_measurement_rate",
    )
    transition_csv_rows = (
        transition_summary
        + list(transition_aggregates["by_branch"])
        + list(transition_aggregates["by_seed"])
        + [transition_aggregates["global"]]
    )
    _write_csv(output_dir / "transition_invalidation_summary.csv", transition_columns, transition_csv_rows)
    budget_columns = (
        "base_seed", "branch", "budget_enabled", "budget_limit_min",
        "budget_limit_max", "budget_limit_total", "budget_consumed_count",
        "budget_overrun_count", "max_budget_used", "max_budget_remaining",
        "budget_exhausted_rows", "budget_usage_rate",
    )
    _write_csv(output_dir / "budget_usage_summary.csv", budget_columns, budget_summary)
    combined_columns = sorted({key for row in combined for key in row})
    _write_csv(output_dir / "seed_level_mechanism_summary.csv", combined_columns, combined)

    expected_quality_flag = {
        "baseline_mappo": 0,
        "quality_aware_mappo": 1,
        "budgeted_quality_aware_mappo": 1,
    }
    feature_injection_ok = bool(feature_rows) and all(
        int(row.get("quality_feature_enabled", -1)) == expected_quality_flag.get(str(row.get("branch")), -1)
        and row.get("quality_model_source") == "warmup"
        for row in feature_rows
        if row.get("split") in {"train", "validation"}
    )
    branch_keys = {(int(row["base_seed"]), str(row["branch"])) for row in all_trace_rows}
    expected_branch_keys = {(int(seed), branch) for seed in config.seeds for branch in config.branches}
    branch_complete = branch_keys == expected_branch_keys
    quality_branches = [
        branch for branch in config.branches
        if branch in {"quality_aware_mappo", "budgeted_quality_aware_mappo"}
    ]
    action_response_by_branch: dict[str, dict[str, Any]] = {}
    for branch in quality_branches:
        correlations = [
            _float(row.get("risk_probability_pearson_r"), float("nan"))
            for row in action_summary
            if row.get("branch") == branch
            and int(row.get("valid_candidate_slot_rows", 0)) >= 100
        ]
        finite_correlations = [value for value in correlations if math.isfinite(value)]
        negative_count = int(sum(value < -0.05 for value in finite_correlations))
        nonnegative_count = int(sum(value >= -0.05 for value in finite_correlations))
        if len(finite_correlations) >= 4 and negative_count >= 4:
            status = "responsive"
        elif len(finite_correlations) >= 4 and nonnegative_count >= 4:
            status = "not_responsive"
        else:
            status = "inconclusive"
        action_response_by_branch[branch] = {
            "status": status,
            "finite_seed_count": len(finite_correlations),
            "negative_r_count": negative_count,
            "nonnegative_r_count": nonnegative_count,
            "correlations": finite_correlations,
        }
    actor_response_status = action_response_by_branch.get(
        "quality_aware_mappo",
        next(iter(action_response_by_branch.values()), {"status": "not_applicable"}),
    ).get("status", "not_applicable")

    treatment_by_key = {(int(row["base_seed"]), str(row["branch"])): row for row in metric_summary}
    overhead_by_key = {(int(row["base_seed"]), str(row["branch"])): row for row in overhead_summary}
    overmeasurement_by_branch: dict[str, list[int]] = {}
    for branch in config.branches:
        if branch == "baseline_mappo":
            continue
        overmeasurement_by_branch[branch] = []
        for seed in config.seeds:
            baseline_overhead = overhead_by_key.get((int(seed), "baseline_mappo"), {})
            branch_overhead = overhead_by_key.get((int(seed), branch), {})
            baseline_metric = treatment_by_key.get((int(seed), "baseline_mappo"), {})
            branch_metric = treatment_by_key.get((int(seed), branch), {})
            more_measurements = _float(branch_overhead.get("measurement_count"), 0.0) > _float(
                baseline_overhead.get("measurement_count"), 0.0
            ) + 1e-9
            jcom_increased = _float(branch_metric.get("j_com"), 0.0) > _float(
                baseline_metric.get("j_com"), 0.0
            ) + 1e-9
            none_decreased = _float(branch_metric.get("none_selection_rate"), 1.0) < _float(
                baseline_metric.get("none_selection_rate"), 1.0
            )
            if more_measurements and jcom_increased and none_decreased:
                overmeasurement_by_branch[branch].append(int(seed))
    overmeasurement_seeds = overmeasurement_by_branch.get("quality_aware_mappo", [])

    queue_checks = [
        row for row in replay_checks
        if row["action_mismatches"] == 0
        and row["overhead_mismatches"] == 0
        and row["reference_action_rows"] > 0
        and row["reference_overhead_rows"] == row["replay_overhead_rows"]
    ]
    replay_ok = len(queue_checks) == len(replay_checks) and branch_complete
    trace_finite = all(_finite_trace_value(row) for row in all_trace_rows)
    forbidden_tokens = ("true_position", "true_error", "los_nlos", "future_state", "true_range")
    forbidden_trace_keys = sorted(
        {
            str(key)
            for row in all_trace_rows
            for key in row
            if any(token in str(key).lower() for token in forbidden_tokens)
        }
    )
    validation_only = bool(all_trace_rows) and all(row.get("split") == "validation" for row in all_trace_rows)
    pre_action_valid = all(
        int(row.get("pre_action_potential_edge", 1)) == 1
        and int(row.get("invalid_measurement_action", 0)) == 0
        for row in all_trace_rows
        if str(row.get("submitted_measurement_uav_id", row.get("selected_measurement_uav_id", "none"))) != "none"
    )
    payload_rows = [row for row in all_trace_rows if int(row.get("effective_measurement", 0)) == 1]
    payload_invariant_ok = all(
        int(row.get("overhead_event_present", 0)) == 1
        and int(row.get("payload_size_bytes", 0)) == 512
        and math.isclose(_float(row.get("load_packets_requested"), 0.0), 1.0)
        and math.isclose(_float(row.get("load_packets_applied"), 0.0), 1.0)
        and int(row.get("potential_edge", 0)) == 1
        and int(row.get("invalid_measurement_action", 0)) == 0
        and int(row.get("measurement_transition_invalidated", 0)) == 0
        for row in payload_rows
    ) and all(
        int(row.get("payload_size_bytes", 0)) == 0
        for row in all_trace_rows
        if int(row.get("overhead_event_present", 0)) == 0
    )
    payload_event_count_ok = all(
        int(summary.get("effective_measurement_count", 0)) == int(summary.get("overhead_event_count", 0))
        for summary in overhead_summary
    )
    budget_ok = all(
        int(row.get("branch") != "budgeted_quality_aware_mappo")
        or int(row.get("budget_overrun_count", 0)) == 0
        for row in budget_summary
    )
    quality_models_warmup_only = bool(replay_checks) and all(
        row.get("quality_model_source") == "warmup" for row in replay_checks
    )
    transition_rates = [
        _float(row.get("transition_invalidation_rate"), 0.0)
        for row in transition_summary
        if _float(row.get("submitted_non_none_count"), 0.0) > 0.0
    ]
    transition_rate = max(transition_rates, default=0.0)
    transition_global = transition_aggregates["global"]
    transition_actionable = bool(
        int(transition_global.get("actionable_transition_invalidation", 0)) == 1
    )
    harmful_by_branch: dict[str, dict[str, float]] = {}
    for row in metric_summary:
        harmful_by_branch.setdefault(str(row["branch"]), {})[str(row["base_seed"])] = _float(
            row.get("harmful_fraction"), 0.0
        )
    localization_effect = []
    for row in metric_summary:
        localization_effect.append(
            {
                "base_seed": int(row["base_seed"]),
                "branch": str(row["branch"]),
                "position_rmse_m": _float(row.get("position_rmse_m"), 0.0),
                "accepted_update_count": _float(row.get("accepted_update_count"), 0.0),
                "harmful_fraction": _float(row.get("harmful_fraction"), 0.0),
                "j_loc": _float(row.get("j_loc"), 0.0),
                "j_com": _float(row.get("j_com"), 0.0),
                "mean_reward": _float(row.get("mean_reward"), 0.0),
            }
        )
    audit_report: dict[str, Any] = {
        "audit": "quality_aware_mappo_action_communication_mechanism_v3",
        "schema_version": "diagnostic-policy-trace-v2",
        "reference_root": str(reference_root),
        "split": config.split,
        "branches": list(config.branches),
        "seeds": [int(seed) for seed in config.seeds],
        "independent_unit": "base training seed",
        "test_data_used": False,
        "episode_schema_modified": False,
        "checks": {
            "feature_injection_ok": bool(feature_injection_ok),
            "quality_models_warmup_only": bool(quality_models_warmup_only),
            "trace_rows_finite": bool(trace_finite),
            "trace_forbidden_keys": forbidden_trace_keys,
            "branch_complete": bool(branch_complete),
            "replay_actions_exact": bool(replay_ok),
            "replay_queue_events_exact": bool(replay_ok),
            "candidate_vector_records_present": bool(all_trace_rows),
            "all_validation_only": bool(validation_only),
            "pre_action_mask_validity": bool(pre_action_valid),
            "payload_invariant": bool(payload_invariant_ok and payload_event_count_ok),
            "budget_no_overrun": bool(budget_ok),
            "queue_increase_fraction_finite": all(
                math.isfinite(_float(row.get("queue_increase_fraction"), float("nan")))
                for row in overhead_summary
            ),
        },
        "action_mask_validity": {
            "invalid_measurement_action_count": int(sum(int(row.get("invalid_measurement_action", 0)) for row in all_trace_rows)),
            "submitted_non_none_count": int(sum(
                str(row.get("submitted_measurement_uav_id", row.get("selected_measurement_uav_id", "none"))) != "none"
                for row in all_trace_rows
            )),
            "all_submitted_non_none_are_pre_action_potential": bool(pre_action_valid),
        },
        "transition_invalidation": {
            "minimum_denominator": TRANSITION_MIN_DENOMINATOR,
            "actionable_rate_threshold": TRANSITION_ACTIONABLE_RATE,
            "max_rate_by_seed_branch": float(transition_rate),
            "summary": transition_summary,
            "by_branch": transition_aggregates["by_branch"],
            "by_seed": transition_aggregates["by_seed"],
            "global": transition_global,
            "transition_timing_status": transition_global["transition_timing_status"],
            "actionable_transition_invalidation": transition_actionable,
            "interpretation": (
                "post-motion topology invalidation is separate from submitted-action invalidity; "
                "only an aggregate rate above 5% with at least 30 submitted non-none actions "
                "is actionable, while smaller cells are reported without triggering a timing decision"
            ),
        },
        "queue_link": {
            "payload_rule_verified": bool(payload_invariant_ok),
            "payload_event_count_verified": bool(payload_event_count_ok),
            "measurement_overhead_status": (
                "verified"
                if replay_ok and payload_invariant_ok and payload_event_count_ok
                else "failed"
            ),
            "rule": (
                "only a final valid non-none measurement on a post-motion potential edge "
                "creates one 512-byte equivalent packet"
            ),
        },
        "policy_response": {
            "feature_actor_flags": {
                branch: expected_quality_flag[branch] for branch in config.branches
            },
            "actor_response_by_branch": action_response_by_branch,
            "actor_response_status": actor_response_status,
            "overmeasurement_by_branch": overmeasurement_by_branch,
            "budget_summary": budget_summary,
        },
        "localization_effect": {
            "seed_level_reference_metrics": localization_effect,
            "independent_unit": "base training seed; update events are mechanism diagnostics only",
        },
        "transition_timing_status": transition_global["transition_timing_status"],
        "replay_checks": replay_checks,
        "decision_rule_note": (
            "A post-motion invalidation is not counted as an invalid submitted action. "
            "This audit diagnoses mechanism consistency and is not a paper-level effectiveness test."
        ),
    }
    if not feature_injection_ok:
        decision = "feature_not_used"
    elif not replay_ok or not payload_invariant_ok or not payload_event_count_ok:
        decision = "queue_link_not_verified"
    elif transition_actionable:
        decision = "post_motion_transition_invalidation"
    elif any(
        len(seeds) >= max(1, math.ceil(len(config.seeds) / 2))
        for seeds in overmeasurement_by_branch.values()
    ):
        decision = "policy_overmeasurement"
    elif any(
        details.get("status") == "not_responsive"
        for details in action_response_by_branch.values()
    ):
        decision = "feature_response_inconclusive"
    else:
        decision = "statistical_evidence_insufficient"
    audit_report["decision"] = decision
    audit_report["audit_ok"] = bool(
        feature_injection_ok
        and quality_models_warmup_only
        and branch_complete
        and replay_ok
        and trace_finite
        and not forbidden_trace_keys
        and validation_only
        and pre_action_valid
        and payload_invariant_ok
        and payload_event_count_ok
        and budget_ok
    )
    audit_report["decision_supported"] = False
    (output_dir / "audit_report.json").write_text(
        json.dumps(_jsonable(audit_report), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    _write_plot(plots_dir / "policy_mechanism_audit.png", action_summary, overhead_summary, metric_summary)
    summary = {
        "output_dir": str(output_dir),
        "plots_dir": str(plots_dir),
        "trace_rows": len(all_trace_rows),
        "branches": list(config.branches),
        "seeds": [int(seed) for seed in config.seeds],
        "decision": decision,
        "transition_timing_status": transition_global["transition_timing_status"],
        "audit_ok": audit_report["audit_ok"],
        "decision_supported": False,
        "audit_report": str(output_dir / "audit_report.json"),
        "plot": str(plots_dir / "policy_mechanism_audit.png"),
    }
    return summary


__all__ = [
    "PolicyMechanismAuditConfig",
    "compute_risk_probability_correlation",
    "run_policy_mechanism_audit",
]
