"""Oracle demonstration, Actor-only behavior cloning, and frozen evaluation.

This module is intentionally separate from the MAPPO training path.  It uses
the already validated Actor-constrained estimated-state oracle as a teacher,
trains only the Actor mean head and its representation, and evaluates the
result without any PPO update.  The task/environment contract is inherited
from the baseline checkpoint and is never modified here.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import shutil
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ..schema import EPISODE_SCHEMA_VERSION, validate_episode_benchmark
from ..synthetic.episode_generator import _jsonable, _sha256, _write_json
from .formal import (
    FormalExperimentConfig,
    _copy_calibration_files,
    _make_environment,
    _scenario_for,
    _scale_seed,
    _write_scale_manifest,
)
from .pilot import (
    _EpisodeTraceWriter,
    _assert_action_feasible,
    _assert_observation_finite,
    _append_jsonl,
    _seed_everything,
)
from .single_uav_actor_constrained_oracle import (
    ACTOR_CONSTRAINED_ORACLE,
    ActorConstrainedEstimatedStateOracleController,
)
from .single_uav_checkpoint_evaluation import (
    _config_from_checkpoint,
    _load_frozen_trainer,
    _state_dict_hash,
)
from .single_uav_rl_equivalent_oracle_audit import (
    BASELINE_RL,
    ESTIMATED_ORACLE,
    FINE_RL,
    POLICY_LABELS as RL_POLICY_LABELS,
    _actor_step_row,
    _base_step_row,
    _complete_geometry_flags,
    _episode_summary,
    _extract_rl_visible_state,
    _mean,
    _runner_geometry,
    _unit,
    _write_rows_csv,
)


BC_POLICY = "bc_actor"
BC_POLICY_LABELS = {
    **RL_POLICY_LABELS,
    ACTOR_CONSTRAINED_ORACLE: "Actor-action-constrained estimated-state oracle",
    BC_POLICY: "Behavior-cloned Actor",
}
DEMO_DATASET_ID = "single_uav_actor_constrained_oracle_demonstrations"
BC_DATASET_ID = "single_uav_behavior_cloning"
BC_EVAL_DATASET_ID = "single_uav_bc_only_evaluation"
_NORMALIZED_FEATURE_KEYS = (
    "node_features",
    "effective_edge_features",
    "candidate_edge_features",
    "candidate_incoming_edge_features",
    "potential_edge_features",
)


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        _jsonable(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                columns.append(str(key))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            output: dict[str, Any] = {}
            for column in columns:
                value = row.get(column, "")
                if isinstance(value, (dict, list, tuple, np.ndarray)):
                    value = json.dumps(_jsonable(value), ensure_ascii=False, separators=(",", ":"))
                output[column] = value
            writer.writerow(output)


def _json_text(value: Any) -> str:
    return json.dumps(_jsonable(value), ensure_ascii=False, separators=(",", ":"))


def _clear_exact_root(path: Path, *, force: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not force:
            raise FileExistsError(f"output is not empty: {path}; use --force for this exact directory")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _copy_observation_for_json(observation: Mapping[str, Any]) -> dict[str, Any]:
    """Keep the exact normalized Actor input while making it JSON-safe."""

    result = copy.deepcopy(dict(observation))
    result["uav_id"] = str(result.get("uav_id", "uav_00"))
    return _jsonable(result)


def _restore_observation(value: Mapping[str, Any]) -> dict[str, Any]:
    """Restore numeric arrays from a JSONL Actor observation."""

    result = copy.deepcopy(dict(value))
    for key in _NORMALIZED_FEATURE_KEYS:
        if key in result:
            raw = result[key]
            if isinstance(raw, Mapping):
                result[key] = {
                    str(item_key): np.asarray(item_value, dtype=np.float32)
                    for item_key, item_value in raw.items()
                }
            else:
                result[key] = np.asarray(raw, dtype=np.float32)
    for key in (
        "edge_index",
        "effective_edge_index",
        "candidate_edge_index",
        "action_mask",
        "bandwidth_mask",
        "candidate_action_mask",
        "measurement_quality_valid",
        "measurement_quality_risk",
    ):
        if key in result and isinstance(result[key], list):
            try:
                result[key] = np.asarray(result[key], dtype=np.float32)
            except (TypeError, ValueError):
                pass
    return result


def actor_physical_mean(policy: Any, observation: Mapping[str, Any]):
    """Return the deterministic post-tanh, physically scaled Actor mean."""

    import torch

    _measurement, _bandwidth, delta_distribution, _mask = policy._actor_distributions(observation)
    mean = delta_distribution.base_dist.mean
    return torch.tanh(mean) * float(policy._delta_component_limit())


def actor_mean_trainable_parameter_names(policy: Any) -> tuple[str, ...]:
    """Freeze every head except the representation-to-delta-mean path."""

    trainable_prefixes = ("gat.", "mlp_flat_encoder.", "actor_body.", "delta_mean_head.")
    names: list[str] = []
    for name, parameter in policy.named_parameters():
        trainable = name.startswith(trainable_prefixes)
        parameter.requires_grad_(trainable)
        if trainable:
            names.append(name)
    if not names:
        raise RuntimeError("baseline Actor has no trainable mean-path parameters")
    for name, parameter in policy.named_parameters():
        if name == "delta_logstd" and parameter.requires_grad:
            raise AssertionError("delta_logstd must remain frozen during first BC stage")
    return tuple(names)


def _teacher_scenario(
    config: FormalExperimentConfig,
    *,
    scenario_seed: int,
    noise_seed: int,
    split: str,
    episode_index: int,
) -> dict[str, Any]:
    # Demonstrations deliberately use normal full-distance starts.  Setting
    # both curriculum fractions to zero prevents the teacher data from
    # reusing the prior terminal-policy test distribution.
    no_curriculum = replace(
        config,
        scenario_seed=int(scenario_seed),
        single_uav_braking_curriculum_fraction=0.0,
        single_uav_terminal_braking_curriculum_fraction=0.0,
        progress=False,
        force=True,
        command_line="oracle demonstration scenario generation; no curriculum",
    )
    scenario = _scenario_for(
        no_curriculum,
        int(noise_seed),
        split=split,
        episode_index=int(episode_index),
        phase="train" if split == "train" else "validation",
    )
    if "single_uav_braking_curriculum" in scenario:
        raise RuntimeError("demonstration scenario unexpectedly contains curriculum start")
    return copy.deepcopy(scenario)


def _normal_evaluation_scenarios(
    config: FormalExperimentConfig, split: str, episodes: int
) -> list[dict[str, Any]]:
    no_curriculum = replace(
        config,
        single_uav_braking_curriculum_fraction=0.0,
        single_uav_terminal_braking_curriculum_fraction=0.0,
        progress=False,
        force=True,
        command_line="BC-only paired normal evaluation scenario generation",
    )
    output: list[dict[str, Any]] = []
    for episode_index in range(int(episodes)):
        seed = _scale_seed(config.seed, config.uav_count, split, episode_index)
        scenario = _scenario_for(
            no_curriculum,
            seed,
            split=split,
            episode_index=episode_index,
            phase=split,
        )
        output.append(
            {
                "episode_index": episode_index + 1,
                "seed": int(seed),
                "scenario": copy.deepcopy(scenario),
                "scenario_hash": _canonical_hash(scenario),
                "curriculum_used": "single_uav_braking_curriculum" in scenario,
            }
        )
    if any(item["curriculum_used"] for item in output):
        raise RuntimeError(f"{split} evaluation scenario unexpectedly used curriculum")
    return output


def _teacher_rows_to_demo_records(
    rows: Sequence[Mapping[str, Any]],
    *,
    split: str,
    scenario_seed: int,
    noise_seed: int,
    episode_index: int,
    scenario_hash: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in rows:
        observation = row.get("actor_observation")
        action = row.get("teacher_action_delta_v_mps")
        if not isinstance(observation, Mapping) or action is None:
            raise ValueError("teacher row is missing Actor observation or constrained action")
        records.append(
            {
                "dataset_id": DEMO_DATASET_ID,
                "split": split,
                "episode_index": int(episode_index),
                "step_index": int(row.get("step_index", 0)),
                "scenario_seed": int(scenario_seed),
                "noise_seed": int(noise_seed),
                "scenario_hash": scenario_hash,
                "observation": _copy_observation_for_json(observation),
                "teacher_action_delta_v_mps": _jsonable(action),
                "goal_distance_m": float(row.get("goal_distance_m", float("nan"))),
                "estimated_goal_distance_m": float(row.get("estimated_goal_distance_m", float("nan"))),
                "speed_mps": float(row.get("speed_mps", float("nan"))),
                "radial_velocity_mps": float(row.get("radial_velocity_mps", float("nan"))),
                "tangential_velocity_mps": float(row.get("tangential_velocity_mps", float("nan"))),
                "goal_direction": _jsonable(row.get("goal_direction", (0.0, 0.0, 0.0))),
                "braking_component_mps": float(row.get("braking_component_mps", 0.0)),
                "action_norm_mps": float(row.get("action_norm_mps", float("nan"))),
                "radial_action_projection_mps": float(
                    row.get("action_projection_toward_goal_mps", float("nan"))
                ),
                "tangential_action_mps": float(row.get("tangential_action_mps", float("nan"))),
            }
        )
    return records


def _run_teacher_episode(
    *,
    trainer: Any,
    config: FormalExperimentConfig,
    scale_root: Path,
    split: str,
    phase: str,
    scenario: Mapping[str, Any],
    scenario_hash: str,
    episode_index: int,
    seed: int,
    serial: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    _seed_everything(seed)
    environment = _make_environment(config, seed, metric_scales=config.metric_scales)
    observations, info = environment.reset(seed=seed, scenario_config=copy.deepcopy(dict(scenario)))
    episode_id = f"episode_{serial:06d}"
    writer = _EpisodeTraceWriter(
        episode_root=scale_root / split / episode_id,
        episode_id=episode_id,
        scenario_id=f"scenario_{serial:06d}",
        split=split,
        seed=seed,
        phase=phase,
        environment=environment,
        normalization_fitted=bool(trainer.normalizer.fitted),
        diagnostics_path=scale_root / "diagnostics" / "attention_maps.jsonl",
        policy_source=ACTOR_CONSTRAINED_ORACLE,
    )
    controller = ActorConstrainedEstimatedStateOracleController(environment)
    goal = np.asarray(environment.goals["uav_00"], dtype=float).copy()
    rows: list[dict[str, Any]] = []
    for step in range(int(config.max_steps)):
        _assert_observation_finite(observations)
        actor_observation = trainer.normalizer.transform(observations)["uav_00"]
        actor_observation = copy.deepcopy(dict(actor_observation))
        actor_observation["uav_id"] = "uav_00"
        pre_state = environment.states["uav_00"].copy()
        action, control = controller.action(observations)
        control = dict(control)
        control["pre_velocity_mps"] = pre_state.velocity_mps.tolist()
        _assert_action_feasible(environment, action)
        next_observations, rewards, terminated, truncated, next_info = environment.step(action)
        execution = next_info.get("action_execution", {}).get("uav_00", {})
        commanded = np.asarray(action["uav_00"].get("delta_v", (0.0, 0.0, 0.0)), dtype=float)
        applied = np.asarray(execution.get("delta_v_applied", commanded), dtype=float)
        geometry = _runner_geometry(
            environment,
            goal=goal,
            pre_state=pre_state,
            post_state=environment.states["uav_00"],
            commanded=commanded,
            applied=applied,
        )
        estimated_before = _extract_rl_visible_state(observations["uav_00"])
        estimated_after = _extract_rl_visible_state(next_observations["uav_00"])
        row = _base_step_row(
            policy=ESTIMATED_ORACLE,
            episode_index=episode_index,
            seed=seed,
            step_index=step,
            timestamp_s=(step + 1) * environment.config.motion.dt_s,
            scenario_hash=scenario_hash,
            geometry=geometry,
            estimated_before=estimated_before,
            estimated_after=estimated_after,
            control=control,
            execution=execution,
            next_info=next_info,
            reward=next_info.get("learning_reward_components", {}),
        )
        row = _complete_geometry_flags(row)
        row.update(
            {
                "policy": ACTOR_CONSTRAINED_ORACLE,
                "policy_label": BC_POLICY_LABELS[ACTOR_CONSTRAINED_ORACLE],
                "actor_observation": actor_observation,
                "teacher_action_delta_v_mps": commanded.tolist(),
                "goal_distance_m": float(geometry["true_goal_distance_before_m"]),
                "speed_mps": float(geometry["total_speed_before_mps"]),
                "radial_velocity_mps": float(geometry["true_radial_velocity_before_mps"]),
                "tangential_velocity_mps": float(geometry["tangential_speed_before_mps"]),
                "goal_direction": _extract_rl_visible_state(observations["uav_00"])["goal_direction"].tolist(),
                "braking_component_mps": float(control.get("constrained_braking_projection_mps", 0.0)),
                "teacher_constraint_affected": int(control.get("actor_constraint_affected", 0)),
                "teacher_desired_action_norm_mps": float(control.get("desired_action_norm_mps", float("nan"))),
                "teacher_constrained_action_norm_mps": float(control.get("constrained_action_norm_mps", float("nan"))),
                "teacher_state_source": "actor_visible_node_features_only",
                "truth_state_read_for_control": False,
            }
        )
        rows.append(row)
        writer.record_step(
            step=step,
            observations_before=observations,
            observations_after=next_observations,
            actions=action,
            rewards=rewards,
            learning_rewards=next_info.get("learning_rewards", rewards),
            info=next_info,
            graph_traces=None,
            policy_source=ACTOR_CONSTRAINED_ORACLE,
            action_start=step * environment.config.motion.dt_s,
            timestamp=(step + 1) * environment.config.motion.dt_s,
            terminated=bool(terminated),
            truncated=bool(truncated),
        )
        observations, info = next_observations, next_info
        if terminated or truncated:
            break
    writer_summary = writer.finalize()
    summary = _episode_summary(rows, policy=ESTIMATED_ORACLE, writer_summary=writer_summary)
    summary.update(
        {
            "policy": ACTOR_CONSTRAINED_ORACLE,
            "policy_label": BC_POLICY_LABELS[ACTOR_CONSTRAINED_ORACLE],
            "evaluation_only": phase != "oracle_demonstration",
            "training_performed": False,
            "policy_updates_performed": False,
            "curriculum_used": False,
            "truth_state_read_for_control": False,
            "controller_state_source": "actor_visible_node_features_only",
            "writer_summary": writer_summary,
        }
    )
    return summary, rows


def generate_oracle_demonstrations(
    baseline_checkpoint: str | Path,
    *,
    output_dir: str | Path,
    logs_dir: str | Path | None = None,
    train_episodes: int = 48,
    validation_episodes: int = 20,
    force: bool = False,
    progress: bool = True,
) -> dict[str, Any]:
    """Generate independent train/validation normal-start teacher trajectories."""

    if train_episodes < 1 or validation_episodes < 1:
        raise ValueError("demonstration episode counts must be positive")
    checkpoint = Path(baseline_checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    output_root = Path(output_dir)
    _clear_exact_root(output_root, force=force)
    logs_root = Path(logs_dir) if logs_dir is not None else Path("logs/preliminary") / output_root.name
    _clear_exact_root(logs_root, force=force)
    scale_root = output_root / "scale_1"
    scale_root.mkdir(parents=True, exist_ok=True)
    (scale_root / "diagnostics").mkdir(parents=True, exist_ok=True)

    import torch

    checkpoint_sha = _sha256(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = _config_from_checkpoint(payload, output_dir=output_root, progress=False)
    if config.uav_count != 1:
        raise ValueError("oracle demonstrations require a 1-UAV baseline checkpoint")
    trainer, policy_hashes, _ = _load_frozen_trainer(payload, config, metric_scales=config.metric_scales)
    trainer.policy.eval()
    _copy_calibration_files(scale_root, config)
    _write_json(scale_root / "normalization_stats.json", _jsonable(payload["normalization_stats"]))

    all_summaries: list[dict[str, Any]] = []
    split_counts: dict[str, int] = {}
    coverage: dict[str, list[int]] = {"train_scenario_seeds": [], "validation_scenario_seeds": [], "train_noise_seeds": [], "validation_noise_seeds": []}
    for split, count, scenario_base, noise_offset in (
        ("train", int(train_episodes), 31001, 700_000),
        ("validation", int(validation_episodes), 91001, 900_000),
    ):
        demo_path = output_root / "demonstrations" / f"{split}.jsonl"
        if demo_path.exists():
            demo_path.unlink()
        for episode_index in range(count):
            scenario_seed = int(scenario_base + 37 * episode_index)
            noise_seed = int(config.seed + noise_offset + 1009 * episode_index)
            scenario = _teacher_scenario(
                config,
                scenario_seed=scenario_seed,
                noise_seed=noise_seed,
                split=split,
                episode_index=episode_index,
            )
            scenario_hash = _canonical_hash(scenario)
            summary, rows = _run_teacher_episode(
                trainer=trainer,
                config=config,
                scale_root=scale_root,
                split=split,
                phase="oracle_demonstration",
                scenario=scenario,
                scenario_hash=scenario_hash,
                episode_index=episode_index + 1,
                seed=noise_seed,
                serial=(1 if split == "train" else 100_001) + episode_index,
            )
            records = _teacher_rows_to_demo_records(
                rows,
                split=split,
                scenario_seed=scenario_seed,
                noise_seed=noise_seed,
                episode_index=episode_index + 1,
                scenario_hash=scenario_hash,
            )
            for record in records:
                _append_jsonl(demo_path, record)
            summary.update(
                {
                    "split": split,
                    "scenario_seed": scenario_seed,
                    "noise_seed": noise_seed,
                    "scenario_hash": scenario_hash,
                    "demo_records": len(records),
                }
            )
            all_summaries.append(summary)
            coverage[f"{split}_scenario_seeds"].append(scenario_seed)
            coverage[f"{split}_noise_seeds"].append(noise_seed)
            if progress and (episode_index == 0 or (episode_index + 1) % 10 == 0 or episode_index + 1 == count):
                print(f"[oracle-demo] {split} {episode_index + 1}/{count}", flush=True)
        split_counts[split] = count

    _write_csv(logs_root / "episode_metrics.csv", all_summaries)
    step_rows = []
    for summary in all_summaries:
        episode_path = Path(str(summary["writer_summary"]["path"]))
        # The custom rows are available only during generation; canonical
        # episode CSVs remain the source for the schema and the JSONL for BC.
        step_rows.append({
            "split": summary.get("split", ""),
            "episode_id": summary.get("episode_id", ""),
            "scenario_hash": summary.get("scenario_hash", ""),
            "path": str(episode_path),
            "entered_8m": summary.get("entered_8m", 0),
            "entered_5m": summary.get("entered_5m", 0),
            "entered_3m": summary.get("entered_3m", 0),
            "entered_1m": summary.get("entered_1m", 0),
            "stable_goal": summary.get("stable_goal", 0),
            "maximum_consecutive_goal_hold_steps": summary.get("maximum_consecutive_goal_hold_steps", 0),
        })
    _write_csv(logs_root / "episode_navigation_summary.csv", step_rows)
    scale_manifest = _write_scale_manifest(scale_root=scale_root, config=config, episodes=[s["writer_summary"] for s in all_summaries])
    scale_manifest.update(
        {
            "dataset_id": DEMO_DATASET_ID,
            "evaluation_only": False,
            "training_performed": False,
            "policy_updates_performed": False,
            "teacher": {
                "controller": ACTOR_CONSTRAINED_ORACLE,
                "state_source": "actor_visible_node_features_only",
                "truth_state_read_for_control": False,
                "action_target": "constrained physical delta-v after Actor componentwise tanh action set",
                "observation_source": "baseline frozen normalizer applied to raw environment observation",
            },
        }
    )
    _write_json(scale_root / "manifest.json", _jsonable(scale_manifest))
    schema_validation = validate_episode_benchmark(scale_root)
    manifest = {
        "dataset_id": DEMO_DATASET_ID,
        "schema_version": EPISODE_SCHEMA_VERSION,
        "source_checkpoint": {"path": str(checkpoint.resolve()), "sha256": checkpoint_sha},
        "policy_hash": policy_hashes,
        "splits": split_counts,
        "records": {
            "train_jsonl": str((output_root / "demonstrations" / "train.jsonl").resolve()),
            "validation_jsonl": str((output_root / "demonstrations" / "validation.jsonl").resolve()),
        },
        "coverage": {
            **coverage,
            "train_scenario_seed_count": len(set(coverage["train_scenario_seeds"])),
            "validation_scenario_seed_count": len(set(coverage["validation_scenario_seeds"])),
            "scenario_seed_sets_disjoint": set(coverage["train_scenario_seeds"]).isdisjoint(coverage["validation_scenario_seeds"]),
            "noise_seed_sets_disjoint": set(coverage["train_noise_seeds"]).isdisjoint(coverage["validation_noise_seeds"]),
        },
        "curriculum_contract": {"train": False, "validation": False, "test": False},
        "schema_validation": schema_validation,
        "outputs": {
            "scale_root": str(scale_root.resolve()),
            "episode_metrics": str((logs_root / "episode_metrics.csv").resolve()),
            "logs_root": str(logs_root.resolve()),
        },
    }
    _write_json(output_root / "manifest.json", _jsonable(manifest))
    _write_json(output_root / "summary.json", _jsonable(manifest))
    return manifest


def _read_demo_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid demonstration JSONL at {path}:{line_number}") from exc
            if not isinstance(row, Mapping):
                raise ValueError(f"demonstration row is not an object at {path}:{line_number}")
            if not isinstance(row.get("observation"), Mapping):
                raise ValueError(f"demonstration observation is missing at {path}:{line_number}")
            action = np.asarray(row.get("teacher_action_delta_v_mps"), dtype=np.float32).reshape(3)
            if not np.isfinite(action).all():
                raise ValueError(f"demonstration action is non-finite at {path}:{line_number}")
            item = dict(row)
            item["observation"] = _restore_observation(row["observation"])
            item["teacher_action_delta_v_mps"] = action
            rows.append(item)
    return rows


def _loss_on_samples(trainer: Any, samples: Sequence[Mapping[str, Any]]) -> tuple[float, float, float]:
    import torch

    if not samples:
        return float("nan"), float("nan"), float("nan")
    losses: list[float] = []
    errors: list[float] = []
    predicted_norms: list[float] = []
    with torch.no_grad():
        for sample in samples:
            prediction = actor_physical_mean(trainer.policy, sample["observation"])
            target = torch.as_tensor(sample["teacher_action_delta_v_mps"], dtype=torch.float32, device=prediction.device)
            errors.append(float(torch.mean((prediction - target) ** 2).item()))
            losses.append(errors[-1])
            predicted_norms.append(float(torch.linalg.vector_norm(prediction).item()))
    return float(np.mean(losses)), float(np.mean(errors)), float(np.mean(predicted_norms))


def train_behavior_cloning(
    baseline_checkpoint: str | Path,
    demonstrations_dir: str | Path,
    *,
    output_dir: str | Path,
    logs_dir: str | Path | None = None,
    plots_dir: str | Path | None = None,
    epochs: int = 12,
    batch_size: int = 32,
    learning_rate: float = 1.0e-4,
    seed: int = 2026,
    force: bool = False,
    progress: bool = True,
) -> dict[str, Any]:
    """Train only the Actor physical-mean path from the baseline checkpoint."""

    if epochs < 1 or batch_size < 1 or learning_rate <= 0.0:
        raise ValueError("BC epochs, batch_size, and learning_rate must be positive")
    checkpoint = Path(baseline_checkpoint)
    demo_root = Path(demonstrations_dir)
    output_root = Path(output_dir)
    _clear_exact_root(output_root, force=force)
    logs_root = Path(logs_dir) if logs_dir is not None else Path("logs/preliminary") / output_root.name
    plots_root = Path(plots_dir) if plots_dir is not None else Path("plots/preliminary") / output_root.name
    _clear_exact_root(logs_root, force=force)
    _clear_exact_root(plots_root, force=force)
    train_samples = _read_demo_jsonl(demo_root / "demonstrations" / "train.jsonl")
    validation_samples = _read_demo_jsonl(demo_root / "demonstrations" / "validation.jsonl")
    if not train_samples or not validation_samples:
        raise ValueError("BC requires non-empty train and validation demonstration sets")
    train_episode_ids = {(row.get("scenario_hash"), row.get("episode_index")) for row in train_samples}
    val_episode_ids = {(row.get("scenario_hash"), row.get("episode_index")) for row in validation_samples}
    if train_episode_ids & val_episode_ids:
        raise ValueError("demonstration train/validation trajectory leakage detected")

    import torch

    checkpoint_sha = _sha256(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = _config_from_checkpoint(payload, output_dir=output_root, progress=False)
    trainer, policy_hashes, _ = _load_frozen_trainer(payload, config, metric_scales=config.metric_scales)
    trainer.policy.eval()
    normalizer_hash_before = _canonical_hash(trainer.normalizer.state_dict())
    critic_hash_before = _state_dict_hash(trainer.policy.critic.state_dict())
    logstd_before = _state_dict_hash({"delta_logstd": trainer.policy.delta_logstd.detach().cpu()})
    trainable_names = actor_mean_trainable_parameter_names(trainer.policy)
    trainable_parameters = [p for p in trainer.policy.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_parameters, lr=float(learning_rate))
    rng = np.random.default_rng(int(seed))
    history: list[dict[str, Any]] = []
    for epoch in range(int(epochs)):
        trainer.policy.eval()
        order = rng.permutation(len(train_samples))
        batch_losses: list[float] = []
        for start in range(0, len(order), int(batch_size)):
            selected = [train_samples[int(index)] for index in order[start : start + int(batch_size)]]
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for sample in selected:
                prediction = actor_physical_mean(trainer.policy, sample["observation"])
                target = torch.as_tensor(sample["teacher_action_delta_v_mps"], dtype=torch.float32, device=prediction.device)
                losses.append(torch.mean((prediction - target) ** 2))
            batch_loss = torch.stack(losses).mean()
            batch_loss.backward()
            optimizer.step()
            batch_losses.append(float(batch_loss.detach().cpu().item()))
        train_loss, train_mse, train_action_norm = _loss_on_samples(trainer, train_samples)
        val_loss, val_mse, val_action_norm = _loss_on_samples(trainer, validation_samples)
        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_action_mse": train_mse,
            "validation_loss": val_loss,
            "validation_action_mse": val_mse,
            "train_mean_predicted_action_norm_mps": train_action_norm,
            "validation_mean_predicted_action_norm_mps": val_action_norm,
            "mean_batch_loss": float(np.mean(batch_losses)) if batch_losses else float("nan"),
            "samples_train": len(train_samples),
            "samples_validation": len(validation_samples),
        }
        history.append(row)
        if progress:
            print(
                f"[bc] epoch {epoch + 1}/{epochs} train_mse={train_mse:.6g} val_mse={val_mse:.6g}",
                flush=True,
            )

    bc_root = output_root / "scale_1"
    checkpoint_path = bc_root / "checkpoints" / "final.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    policy_hash_after = _state_dict_hash(trainer.policy.state_dict())
    normalizer_hash_after = _canonical_hash(trainer.normalizer.state_dict())
    critic_hash_after = _state_dict_hash(trainer.policy.critic.state_dict())
    logstd_after = _state_dict_hash({"delta_logstd": trainer.policy.delta_logstd.detach().cpu()})
    bc_payload = copy.deepcopy(dict(payload))
    bc_payload["policy_state_dict"] = trainer.policy.state_dict()
    bc_payload["bc_training"] = {
        "dataset_id": BC_DATASET_ID,
        "source_checkpoint": {"path": str(checkpoint.resolve()), "sha256": checkpoint_sha},
        "source_policy_hash": policy_hashes["loaded_policy_hash"],
        "policy_hash_after_training": policy_hash_after,
        "trainable_parameter_names": list(trainable_names),
        "critic_frozen": critic_hash_before == critic_hash_after,
        "normalizer_frozen": normalizer_hash_before == normalizer_hash_after,
        "logstd_frozen": logstd_before == logstd_after,
        "loss_space": "post_tanh_scaled_physical_delta_v_mse",
        "pre_tanh_target_used": False,
        "ppo_update_performed": False,
        "optimizer": "Adam_for_BC_only",
        "learning_rate": float(learning_rate),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "train_samples": len(train_samples),
        "validation_samples": len(validation_samples),
    }
    # Preserve the baseline optimizer payload for provenance, but do not use
    # or mutate it as a PPO update state.
    torch.save(bc_payload, checkpoint_path)
    _write_csv(logs_root / "bc_training_history.csv", history)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8.5, 5.2), constrained_layout=True)
    axis.plot([row["epoch"] for row in history], [row["train_loss"] for row in history], label="train MSE", linewidth=2)
    axis.plot([row["epoch"] for row in history], [row["validation_loss"] for row in history], label="validation MSE", linewidth=2)
    axis.set_xlabel("BC epoch")
    axis.set_ylabel("physical delta-v MSE")
    axis.set_title("Oracle behavior cloning loss")
    axis.grid(alpha=0.25)
    axis.legend()
    loss_plot = plots_root / "bc_loss_curves.png"
    figure.savefig(loss_plot, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)

    summary = {
        "dataset_id": BC_DATASET_ID,
        "source_checkpoint": {"path": str(checkpoint.resolve()), "sha256": checkpoint_sha},
        "demonstrations_dir": str(demo_root.resolve()),
        "checkpoint": str(checkpoint_path.resolve()),
        "history": history,
        "final_train_loss": history[-1]["train_loss"],
        "final_validation_loss": history[-1]["validation_loss"],
        "bc_contract": bc_payload["bc_training"],
        "files": {
            "history": str((logs_root / "bc_training_history.csv").resolve()),
            "loss_plot": str(loss_plot.resolve()),
            "checkpoint": str(checkpoint_path.resolve()),
        },
    }
    _write_json(output_root / "manifest.json", _jsonable(summary))
    _write_json(output_root / "summary.json", _jsonable(summary))
    return summary


def _add_action_diagnostics(summary: dict[str, Any], rows: Sequence[Mapping[str, Any]]) -> None:
    summary.update(
        {
            "mean_action_norm_mps": _mean(row.get("action_norm_mps") for row in rows),
            "mean_radial_action_projection_mps": _mean(row.get("action_projection_toward_goal_mps") for row in rows),
            "negative_radial_velocity_fraction": _mean(row.get("negative_radial_velocity") for row in rows),
            "mean_tangential_action_mps": _mean(row.get("tangential_action_mps") for row in rows),
            "mean_braking_magnitude_mps": _mean(row.get("braking_projection_mps") for row in rows),
            "mean_effective_braking_fraction": _mean(row.get("effective_braking") for row in rows),
        }
    )


def _aggregate_role(
    role: str, summaries: Sequence[Mapping[str, Any]], rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    selected = [row for row in summaries if str(row.get("policy")) == role]
    step_rows = [row for row in rows if str(row.get("policy")) == role]

    def count_field(name: str) -> int:
        return int(sum(int(row.get(name, 0) or 0) for row in selected))

    count = len(selected)
    result: dict[str, Any] = {
        "policy": role,
        "policy_label": BC_POLICY_LABELS[role],
        "episodes": count,
        "entered_8m_episodes": count_field("entered_8m"),
        "entered_8m_rate": count_field("entered_8m") / count if count else 0.0,
        "entered_5m_episodes": count_field("entered_5m"),
        "entered_5m_rate": count_field("entered_5m") / count if count else 0.0,
        "entered_3m_episodes": count_field("entered_3m"),
        "entered_3m_rate": count_field("entered_3m") / count if count else 0.0,
        "entered_1m_episodes": count_field("entered_1m"),
        "entered_1m_rate": count_field("entered_1m") / count if count else 0.0,
        "stable_goal_episodes": count_field("stable_goal"),
        "stable_goal_rate": count_field("stable_goal") / count if count else 0.0,
        "minimum_goal_distance_m": min((float(row.get("minimum_goal_distance_m", float("nan"))) for row in selected), default=float("nan")),
        "mean_minimum_goal_distance_m": _mean(row.get("minimum_goal_distance_m") for row in selected),
        "mean_speed_at_8m_entry_mps": _mean(row.get("mean_speed_at_8m_entry_mps") for row in selected),
        "mean_speed_at_5m_entry_mps": _mean(row.get("mean_speed_at_5m_entry_mps") for row in selected),
        "mean_speed_at_3m_entry_mps": _mean(row.get("mean_speed_at_3m_entry_mps") for row in selected),
        "mean_speed_at_1m_entry_mps": _mean(row.get("mean_speed_at_1m_entry_mps") for row in selected),
        "mean_speed_within_1m_mps": _mean(row.get("mean_speed_within_1m_mps") for row in selected),
        "mean_maximum_consecutive_goal_hold_steps": _mean(row.get("maximum_consecutive_goal_hold_steps") for row in selected),
        "maximum_consecutive_goal_hold_steps_max": max((int(row.get("maximum_consecutive_goal_hold_steps", 0) or 0) for row in selected), default=0),
        "episodes_with_hold_steps_ge_1": count_field("hold_steps_ge_1"),
        "episodes_with_hold_steps_ge_1_rate": count_field("hold_steps_ge_1") / count if count else 0.0,
        "episodes_with_hold_steps_ge_3": count_field("hold_steps_ge_3"),
        "episodes_with_hold_steps_ge_3_rate": count_field("hold_steps_ge_3") / count if count else 0.0,
        "episodes_with_hold_steps_ge_5": count_field("hold_steps_ge_5"),
        "episodes_with_hold_steps_ge_5_rate": count_field("hold_steps_ge_5") / count if count else 0.0,
    }
    _add_action_diagnostics(result, step_rows)
    return result


def _run_rl_or_bc_episode(
    *,
    role: str,
    trainer: Any,
    config: FormalExperimentConfig,
    scale_root: Path,
    split: str,
    scenario: Mapping[str, Any],
    scenario_hash: str,
    episode_index: int,
    seed: int,
    serial: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    import torch

    _seed_everything(seed)
    environment = _make_environment(config, seed, metric_scales=config.metric_scales)
    observations, info = environment.reset(seed=seed, scenario_config=copy.deepcopy(dict(scenario)))
    episode_id = f"episode_{serial:06d}"
    writer = _EpisodeTraceWriter(
        episode_root=scale_root / split / episode_id,
        episode_id=episode_id,
        scenario_id=f"scenario_{serial:06d}",
        split=split,
        seed=seed,
        phase="bc_only_evaluation",
        environment=environment,
        normalization_fitted=bool(trainer.normalizer.fitted),
        diagnostics_path=scale_root / "diagnostics" / "attention_maps.jsonl",
        policy_source=role,
    )
    goal = np.asarray(environment.goals["uav_00"], dtype=float).copy()
    rows: list[dict[str, Any]] = []
    for step in range(int(config.max_steps)):
        _assert_observation_finite(observations)
        pre_state = environment.states["uav_00"].copy()
        if role == BC_POLICY:
            # The BC Actor uses the identical frozen normalizer and action
            # transform.  Only the mean path differs from baseline weights.
            normalized = trainer.normalizer.transform(observations)
            local = copy.deepcopy(dict(normalized["uav_00"]))
            local["uav_id"] = "uav_00"
            with torch.no_grad():
                physical_mean = actor_physical_mean(trainer.policy, local)
            delta = physical_mean.detach().cpu().numpy().astype(float).tolist()
            action = {"uav_00": {"delta_v": delta, "measurement_uav_id": "none", "bandwidth": {}}}
            decision = None
            graph_traces = None
            actor_observation = local
        else:
            with torch.no_grad():
                decision = trainer.act(observations, info["central_state"], deterministic=True)
            action = decision.actions
            graph_traces = decision.graph_traces
            actor_observation = decision.observations["uav_00"]
        _assert_action_feasible(environment, action)
        next_observations, rewards, terminated, truncated, next_info = environment.step(action)
        execution = next_info.get("action_execution", {}).get("uav_00", {})
        commanded = np.asarray(action["uav_00"].get("delta_v", (0.0, 0.0, 0.0)), dtype=float)
        applied = np.asarray(execution.get("delta_v_applied", commanded), dtype=float)
        geometry = _runner_geometry(
            environment,
            goal=goal,
            pre_state=pre_state,
            post_state=environment.states["uav_00"],
            commanded=commanded,
            applied=applied,
        )
        if role == BC_POLICY:
            # Use the shared row schema without invoking a sampled PolicyDecision.
            estimated_before = _extract_rl_visible_state(observations["uav_00"])
            estimated_after = _extract_rl_visible_state(next_observations["uav_00"])
            control = {
                "legal_delta_v_mps": commanded.tolist(),
                "raw_delta_v_norm_mps": float(np.linalg.norm(commanded)),
                "pre_velocity_mps": pre_state.velocity_mps.tolist(),
                "demanded_braking_delta_v_mps": max(0.0, -float(np.dot(_unit(pre_state.velocity_mps), commanded))),
                "desired_speed_mps": float("nan"),
            }
            row = _base_step_row(
                policy=BASELINE_RL,
                episode_index=episode_index,
                seed=seed,
                step_index=step,
                timestamp_s=(step + 1) * environment.config.motion.dt_s,
                scenario_hash=scenario_hash,
                geometry=geometry,
                estimated_before=estimated_before,
                estimated_after=estimated_after,
                control=control,
                execution=execution,
                next_info=next_info,
                reward=next_info.get("learning_reward_components", {}),
            )
            row.update(
                {
                    "actor_observation": actor_observation,
                    "bc_predicted_physical_action_delta_v_mps": commanded.tolist(),
                    "bc_action_space": "post_tanh_scaled_physical_delta_v",
                }
            )
        else:
            row = _actor_step_row(
                policy=role,
                episode_index=episode_index,
                seed=seed,
                step_index=step,
                timestamp_s=(step + 1) * environment.config.motion.dt_s,
                scenario_hash=scenario_hash,
                raw_observation=observations["uav_00"],
                decision=decision,
                trainer=trainer,
                pre_position=pre_state.position_m.copy(),
                post_position=environment.states["uav_00"].position_m.copy(),
                pre_velocity=pre_state.velocity_mps.copy(),
                post_velocity=environment.states["uav_00"].velocity_mps.copy(),
                goal=goal,
                execution=execution,
                next_observations=next_observations,
                next_info=next_info,
                reward=next_info.get("learning_reward_components", {}),
            )
            row["actor_observation"] = actor_observation
        row.update(geometry)
        row = _complete_geometry_flags(row)
        row.update(
            {
                "policy": role,
                "policy_label": BC_POLICY_LABELS[role],
                "controller_state_source": "frozen_actor_network",
                "truth_state_read_for_control": False,
            }
        )
        rows.append(row)
        writer.record_step(
            step=step,
            observations_before=observations,
            observations_after=next_observations,
            actions=action,
            rewards=rewards,
            learning_rewards=next_info.get("learning_rewards", rewards),
            info=next_info,
            graph_traces=graph_traces,
            policy_source=role,
            action_start=step * environment.config.motion.dt_s,
            timestamp=(step + 1) * environment.config.motion.dt_s,
            terminated=bool(terminated),
            truncated=bool(truncated),
        )
        observations, info = next_observations, next_info
        if terminated or truncated:
            break
    writer_summary = writer.finalize()
    summary_policy = BASELINE_RL if role == BC_POLICY else role
    summary = _episode_summary(rows, policy=summary_policy, writer_summary=writer_summary)
    summary.update(
        {
            "policy": role,
            "policy_label": BC_POLICY_LABELS[role],
            "evaluation_only": True,
            "training_performed": False,
            "policy_updates_performed": False,
            "curriculum_used": False,
            "controller_state_source": "frozen_actor_network",
            "truth_state_read_for_control": False,
            "writer_summary": writer_summary,
        }
    )
    return summary, rows


def _role_root_name(role: str) -> str:
    return {
        BASELINE_RL: "baseline_rl",
        FINE_RL: "navigation_preserving_fine_tuned",
        BC_POLICY: "bc_actor",
        ACTOR_CONSTRAINED_ORACLE: "actor_constrained_oracle",
    }[role]


def _plot_bc_evaluation(aggregates: Mapping[str, Mapping[str, Any]], output_dir: Path) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    roles = [BASELINE_RL, FINE_RL, BC_POLICY, ACTOR_CONSTRAINED_ORACLE]
    labels = [BC_POLICY_LABELS[role] for role in roles]
    positions = np.arange(len(roles))
    paths: dict[str, str] = {}
    figure, axis = plt.subplots(figsize=(11, 5.8), constrained_layout=True)
    width = 0.2
    for offset, field, label, color in (
        (-1.5 * width, "entered_8m_rate", "8 m", "#0072B2"),
        (-0.5 * width, "entered_3m_rate", "3 m", "#E69F00"),
        (0.5 * width, "entered_1m_rate", "1 m", "#CC79A7"),
        (1.5 * width, "stable_goal_rate", "stable", "#009E73"),
    ):
        axis.bar(positions + offset, [100 * float(aggregates[role].get(field, 0.0)) for role in roles], width=width, label=label, color=color)
    axis.set_xticks(positions, labels, rotation=18, ha="right")
    axis.set_ylabel("Episodes (%)")
    axis.set_ylim(0, 100)
    axis.set_title("BC-only versus baseline, fine-tuned Actor, and Oracle")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    path = output_dir / "bc_evaluation_entry_stable.png"
    figure.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    paths["entry_stable"] = str(path.resolve())

    figure, axes = plt.subplots(1, 3, figsize=(13, 4.6), constrained_layout=True)
    for axis, field, title in (
        (axes[0], "mean_action_norm_mps", "Mean action norm"),
        (axes[1], "mean_radial_action_projection_mps", "Mean radial action"),
        (axes[2], "mean_braking_magnitude_mps", "Mean braking magnitude"),
    ):
        axis.bar(positions, [float(aggregates[role].get(field, float("nan"))) for role in roles], color=["#0072B2", "#D55E00", "#7B2CBF", "#CC79A7"])
        axis.set_title(title)
        axis.set_xticks(positions, labels, rotation=22, ha="right")
        axis.grid(axis="y", alpha=0.25)
    path = output_dir / "bc_action_diagnostics.png"
    figure.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    paths["action_diagnostics"] = str(path.resolve())
    return paths


def evaluate_bc_only(
    baseline_checkpoint: str | Path,
    fine_tuned_checkpoint: str | Path,
    bc_checkpoint: str | Path,
    *,
    output_dir: str | Path,
    logs_dir: str | Path | None = None,
    plots_dir: str | Path | None = None,
    validation_episodes: int = 20,
    test_episodes: int = 20,
    force: bool = False,
    progress: bool = True,
) -> dict[str, Any]:
    """Evaluate four frozen roles on paired normal validation/test scenarios."""

    if validation_episodes < 1 or test_episodes < 1:
        raise ValueError("evaluation episode counts must be positive")
    baseline_path = Path(baseline_checkpoint)
    fine_path = Path(fine_tuned_checkpoint)
    bc_path = Path(bc_checkpoint)
    for path in (baseline_path, fine_path, bc_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    output_root = Path(output_dir)
    _clear_exact_root(output_root, force=force)
    logs_root = Path(logs_dir) if logs_dir is not None else Path("logs/preliminary") / output_root.name
    plots_root = Path(plots_dir) if plots_dir is not None else Path("plots/preliminary") / output_root.name
    _clear_exact_root(logs_root, force=force)
    _clear_exact_root(plots_root, force=force)

    import torch

    payloads = {
        BASELINE_RL: torch.load(baseline_path, map_location="cpu", weights_only=False),
        FINE_RL: torch.load(fine_path, map_location="cpu", weights_only=False),
        BC_POLICY: torch.load(bc_path, map_location="cpu", weights_only=False),
    }
    configs = {
        role: _config_from_checkpoint(payload, output_dir=output_root / _role_root_name(role), progress=False)
        for role, payload in payloads.items()
    }
    trainers: dict[str, Any] = {}
    checkpoint_hashes: dict[str, str] = {}
    policy_hash_before: dict[str, str] = {}
    for role, payload in payloads.items():
        trainer, hashes, _ = _load_frozen_trainer(payload, configs[role], metric_scales=configs[role].metric_scales)
        trainer.policy.eval()
        trainers[role] = trainer
        checkpoint_hashes[role] = _sha256({BASELINE_RL: baseline_path, FINE_RL: fine_path, BC_POLICY: bc_path}[role])
        policy_hash_before[role] = hashes["loaded_policy_hash"]
    oracle_config = configs[BASELINE_RL]
    all_summaries: list[dict[str, Any]] = []
    all_step_rows: list[dict[str, Any]] = []
    role_reports: dict[str, Any] = {}
    aggregate_by_split: dict[str, dict[str, Any]] = {}
    role_order = (BASELINE_RL, FINE_RL, BC_POLICY, ACTOR_CONSTRAINED_ORACLE)
    for split, count in (("validation", int(validation_episodes)), ("test", int(test_episodes))):
        scenarios = _normal_evaluation_scenarios(oracle_config, split, count)
        split_summaries: list[dict[str, Any]] = []
        split_rows: list[dict[str, Any]] = []
        for role_index, role in enumerate(role_order):
            role_root = output_root / _role_root_name(role) / "scale_1"
            role_root.mkdir(parents=True, exist_ok=True)
            role_summaries: list[dict[str, Any]] = []
            role_rows: list[dict[str, Any]] = []
            for item in scenarios:
                index = int(item["episode_index"])
                serial = role_index * 100_000 + (0 if split == "validation" else 50_000) + index
                if role == ACTOR_CONSTRAINED_ORACLE:
                    summary, rows = _run_teacher_episode(
                        trainer=trainers[BASELINE_RL],
                        config=oracle_config,
                        scale_root=role_root,
                        split=split,
                        phase="bc_only_evaluation",
                        scenario=item["scenario"],
                        scenario_hash=item["scenario_hash"],
                        episode_index=index,
                        seed=int(item["seed"]),
                        serial=serial,
                    )
                else:
                    summary, rows = _run_rl_or_bc_episode(
                        role=role,
                        trainer=trainers[role],
                        config=configs[role],
                        scale_root=role_root,
                        split=split,
                        scenario=item["scenario"],
                        scenario_hash=item["scenario_hash"],
                        episode_index=index,
                        seed=int(item["seed"]),
                        serial=serial,
                    )
                summary.update({"split": split, "scenario_hash": item["scenario_hash"], "paired_episode_index": index})
                role_summaries.append(summary)
                role_rows.extend(rows)
                if progress and (index == 1 or index % 10 == 0 or index == count):
                    print(f"[bc-eval] {split} {role} {index}/{count}", flush=True)
            _copy_calibration_files(role_root, oracle_config)
            manifest = _write_scale_manifest(scale_root=role_root, config=oracle_config, episodes=[s["writer_summary"] for s in role_summaries])
            manifest.update({
                "dataset_id": BC_EVAL_DATASET_ID,
                "evaluation_only": True,
                "training_performed": False,
                "policy_updates_performed": False,
                "policy_source": role,
                "policy_label": BC_POLICY_LABELS[role],
                "curriculum_used": False,
                "stable_criterion_reused": True,
            })
            _write_json(role_root / "manifest.json", _jsonable(manifest))
            role_validation = validate_episode_benchmark(role_root)
            role_reports.setdefault(role, {})[split] = {
                "schema_validation": role_validation,
                "episodes": len(role_summaries),
            }
            split_summaries.extend(role_summaries)
            split_rows.extend(role_rows)
            all_summaries.extend(role_summaries)
            all_step_rows.extend(role_rows)
        aggregate_by_split[split] = {
            role: _aggregate_role(role, split_summaries, split_rows) for role in role_order
        }
        _write_csv(logs_root / f"episode_metrics_{split}.csv", split_summaries)
        _write_rows_csv(logs_root / f"step_metrics_{split}.csv", split_rows)

    # Consolidated metrics and plots use only the current frozen evaluation.
    _write_csv(logs_root / "episode_metrics.csv", all_summaries)
    _write_rows_csv(logs_root / "step_metrics.csv", all_step_rows)
    overall_aggregates = {
        role: _aggregate_role(role, all_summaries, all_step_rows) for role in role_order
    }
    plots = _plot_bc_evaluation(overall_aggregates, plots_root)
    unchanged: dict[str, bool] = {}
    for role, path in ((BASELINE_RL, baseline_path), (FINE_RL, fine_path), (BC_POLICY, bc_path)):
        unchanged[role] = _sha256(path) == checkpoint_hashes[role]
    summary = {
        "dataset_id": BC_EVAL_DATASET_ID,
        "evaluation_only": True,
        "training_performed": False,
        "policy_updates_performed": False,
        "roles": role_order,
        "checkpoint_hashes": checkpoint_hashes,
        "policy_hash_before": policy_hash_before,
        "checkpoint_files_unchanged": unchanged,
        "curriculum_contract": {
            "validation": False,
            "test": False,
            "scenario_generation": "paired normal far-distance starts",
        },
        "validation": aggregate_by_split["validation"],
        "test": aggregate_by_split["test"],
        "overall": overall_aggregates,
        "role_reports": role_reports,
        "plots": plots,
        "outputs": {
            "logs_root": str(logs_root.resolve()),
            "plots_root": str(plots_root.resolve()),
            "episode_metrics": str((logs_root / "episode_metrics.csv").resolve()),
            "step_metrics": str((logs_root / "step_metrics.csv").resolve()),
        },
        "bc_evaluation_contract": {
            "bc_loss_space": "post_tanh_scaled_physical_delta_v",
            "critic_frozen": True,
            "normalizer_frozen": True,
            "logstd_unchanged": True,
            "ppo_update_performed": False,
            "reward_or_environment_modified": False,
        },
    }
    _write_json(output_root / "manifest.json", _jsonable(summary))
    _write_json(output_root / "summary.json", _jsonable(summary))
    return summary


__all__ = [
    "BC_POLICY",
    "ACTOR_CONSTRAINED_ORACLE",
    "actor_physical_mean",
    "actor_mean_trainable_parameter_names",
    "generate_oracle_demonstrations",
    "train_behavior_cloning",
    "evaluate_bc_only",
]
