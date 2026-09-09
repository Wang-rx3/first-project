"""Read-only three-link diagnostic for the 3-UAV communication-localization line.

This module does not train, select a checkpoint, or read the test split.  It
combines the existing equal-budget validation intervention with two bounded
diagnostics:

* a measurement-to-EKF conversion funnel; and
* a first-window same-state action probe using the frozen Proposed checkpoint.

The probe is deliberately descriptive.  It keeps the policy's deterministic
bandwidth and velocity actions fixed, changes one UAV's legal measurement
source at a time, and replays the same validation scenario from its initial
state.  It therefore tests whether candidate actions produce distinguishable
immediate outcomes; it is not a causal performance claim for the full episode.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ..synthetic.episode_generator import _write_csv, _write_json
from .baseline import _read_csv, _read_json
from .cooperative_event_logging import accepted_counts
from .formal import _make_environment
from .mainline_baseline_comparison import _checkpoint_trainer
from .measurement_density_intervention import actor_visible_measurement_candidates


FUNNEL_STAGES: tuple[tuple[str, str], ...] = (
    ("candidate_available", "candidate_available"),
    ("selected_non_none", "selected_non_none"),
    ("uwb_valid", "uwb_valid"),
    ("effective_edge", "effective_edge"),
    ("message_arrived", "message_arrived"),
    ("update_period_passed", "update_period_passed"),
    ("new_packet", "new_packet"),
    ("ekf_update_attempted", "ekf_update_attempted"),
    ("ekf_update_accepted", "ekf_update_accepted"),
)
TREATMENTS: tuple[str, ...] = (
    "recorded_mappo",
    "equal_budget_round_robin",
    "equal_budget_quality_aware",
)


def _number(value: Any, default: float = float("nan")) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _bool(value: Any) -> int:
    return int(str(value).strip().lower() in {"1", "true", "yes"})


def _mean(values: Iterable[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.mean(finite)) if finite else float("nan")


def _prepare_output(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"diagnostic output is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)


def _write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write an empty diagnostic table: {path}")
    _write_csv(path, tuple(rows[0].keys()), [dict(row) for row in rows])


def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    return _read_csv(path)


def _conversion_funnel(equal_budget_logs: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = _load_csv_rows(equal_budget_logs / "window_metrics.csv")
    funnel: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for treatment in TREATMENTS:
        selected = [row for row in rows if str(row.get("treatment")) == treatment]
        previous = len(selected)
        for order, (stage, column) in enumerate(FUNNEL_STAGES, start=1):
            passed = sum(_bool(row.get(column, 0)) for row in selected)
            funnel.append(
                {
                    "treatment": treatment,
                    "stage_order": order,
                    "stage": stage,
                    "actor_rows": len(selected),
                    "passed_count": passed,
                    "fraction_of_all_actor_rows": passed / max(1, len(selected)),
                    "fraction_of_previous_stage": passed / max(1, previous),
                }
            )
            previous = passed
        selected_rows = [row for row in selected if _bool(row.get("selected_non_none", 0))]
        by_failure: dict[str, int] = defaultdict(int)
        for row in selected_rows:
            by_failure[str(row.get("failure_stage", "unknown"))] += 1
        for failure_stage, count in sorted(by_failure.items()):
            failures.append(
                {
                    "treatment": treatment,
                    "failure_stage": failure_stage,
                    "selected_rows": len(selected_rows),
                    "count": count,
                    "fraction_of_selected": count / max(1, len(selected_rows)),
                }
            )
    return funnel, failures


def _action_reward_alignment(reference_scale: Path) -> list[dict[str, Any]]:
    manifest = _read_json(reference_scale / "manifest.json")
    episodes = sorted(
        str(item["episode_id"])
        for item in manifest.get("episodes", [])
        if isinstance(item, Mapping) and str(item.get("split")) == "validation"
    )
    accum: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for episode_id in episodes:
        episode_root = reference_scale / "validation" / episode_id
        actions = _load_csv_rows(episode_root / "actions.csv")
        rewards = _load_csv_rows(episode_root / "rewards.csv")
        reward_map = {
            (str(row["timestamp"]), str(row["uav_id"])): row
            for row in rewards
        }
        for action in actions:
            key = (str(action["timestamp"]), str(action["uav_id"]))
            reward = reward_map.get(key)
            if reward is None:
                raise ValueError(f"missing reward for action {episode_id}/{key}")
            group = "selected_non_none" if str(action.get("measurement_uav_id", "none")) != "none" else "none"
            target = accum[group]
            target["count"].append(1.0)
            for field in (
                "reward_total",
                "joint_objective",
                "reward_localization",
                "reward_packet_loss",
                "reward_aoi",
                "reward_delay",
                "reward_connectivity",
                "reward_collision",
                "reward_energy",
                "actual_measurement_count",
                "actual_measurement_bytes",
            ):
                target[field].append(_number(reward.get(field)))
    output: list[dict[str, Any]] = []
    for group in ("none", "selected_non_none"):
        values = accum[group]
        output.append(
            {
                "action_group": group,
                "row_count": int(sum(values["count"])),
                "mean_report_reward": _mean(values["reward_total"]),
                "mean_joint_objective": _mean(values["joint_objective"]),
                "mean_localization_component": _mean(values["reward_localization"]),
                "mean_packet_loss_component": _mean(values["reward_packet_loss"]),
                "mean_aoi_component": _mean(values["reward_aoi"]),
                "mean_delay_component": _mean(values["reward_delay"]),
                "mean_connectivity_component": _mean(values["reward_connectivity"]),
                "mean_collision_component": _mean(values["reward_collision"]),
                "mean_energy_component": _mean(values["reward_energy"]),
                "mean_logged_measurement_count": _mean(values["actual_measurement_count"]),
                "mean_logged_measurement_bytes": _mean(values["actual_measurement_bytes"]),
                "interpretation": "descriptive_same_window_association_not_causal",
            }
        )
    return output


def _legal_sources(observation: Mapping[str, Any], uav_ids: Sequence[str]) -> list[str]:
    mask = np.asarray(observation.get("action_mask", ()), dtype=bool).reshape(-1)
    output: list[str] = []
    for index, enabled in enumerate(mask[: len(uav_ids) + 1]):
        if not bool(enabled):
            continue
        output.append("none" if index >= len(uav_ids) else str(uav_ids[index]))
    return output


def _candidate_link_spread(observation: Mapping[str, Any], candidates: Sequence[str]) -> float:
    links = observation.get("candidate_incoming_links", {})
    if not isinstance(links, Mapping):
        return 0.0
    values: list[float] = []
    for source in candidates:
        link = links.get(source, {})
        if isinstance(link, Mapping):
            packet_loss = _number(link.get("packet_loss"), float("nan"))
            if math.isfinite(packet_loss):
                values.append(packet_loss)
    return float(max(values) - min(values)) if values else 0.0


def _same_state_action_probe(reference_scale: Path, output_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    manifest = _read_json(reference_scale / "manifest.json")
    raw_config = dict(manifest.get("config", {}))
    from .baseline import _reference_formal_config

    formal = _reference_formal_config(manifest, output_root, force=False, progress=False)
    reference_environment = _make_environment(
        formal,
        int(manifest.get("seed", 42)),
        metric_scales=formal.metric_scales,
    )
    trainer = _checkpoint_trainer(reference_scale, formal, reference_environment)
    del raw_config
    probe_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    episodes = [
        dict(item)
        for item in manifest.get("episodes", [])
        if isinstance(item, Mapping) and str(item.get("split")) == "validation"
    ]
    episodes.sort(key=lambda item: str(item["episode_id"]))
    for episode in episodes:
        episode_id = str(episode["episode_id"])
        seed = int(episode["seed"])
        source_root = reference_scale / "validation" / episode_id
        scenario_payload = _read_json(source_root / "scenario.json")
        scenario = scenario_payload["scenario_generation"]
        environment = _make_environment(formal, seed, metric_scales=formal.metric_scales)
        observations, info = environment.reset(seed=seed, scenario_config=scenario)
        decision = trainer.act(observations, info["central_state"], deterministic=True)
        base_actions = copy.deepcopy(decision.actions)
        for target in environment.uav_ids:
            legal = _legal_sources(observations[target], environment.uav_ids)
            if not legal:
                continue
            diagnostics = decision.measurement_diagnostics[target]
            probabilities = np.asarray(diagnostics.get("measurement_probabilities", []), dtype=float)
            logits = np.asarray(diagnostics.get("measurement_logits_adjusted", []), dtype=float)
            source_to_index = {str(source): index for index, source in enumerate(environment.uav_ids)}
            source_to_index["none"] = len(environment.uav_ids)
            legal_indices = [source_to_index[source] for source in legal]
            legal_probs = [probabilities[index] for index in legal_indices if index < probabilities.size]
            legal_logits = [logits[index] for index in legal_indices if index < logits.size]
            probability_spread = float(max(legal_probs) - min(legal_probs)) if legal_probs else float("nan")
            logit_spread = float(max(legal_logits) - min(legal_logits)) if legal_logits else float("nan")
            selected_by_policy = str(base_actions[target].get("measurement_uav_id", "none"))
            candidate_count = len([source for source in legal if source != "none"])
            link_spread = _candidate_link_spread(observations[target], legal)
            target_rows: list[dict[str, Any]] = []
            for source in legal:
                action = copy.deepcopy(base_actions)
                action[target]["measurement_uav_id"] = source
                replay = _make_environment(formal, seed, metric_scales=formal.metric_scales)
                replay.reset(seed=seed, scenario_config=scenario)
                before = accepted_counts(replay)
                _next_obs, _rewards, _terminated, _truncated, next_info = replay.step(action)
                after = accepted_counts(replay)
                accepted = sum(int(after[key] - before[key]) for key in after)
                gate = next(
                    (
                        row for row in next_info.get("measurement_gate_events", [])
                        if str(row.get("target_uav_id")) == target
                        and str(row.get("source_uav_id")) == source
                    ),
                    {},
                )
                error = float(
                    np.linalg.norm(
                        np.asarray(replay.states[target].position_m, dtype=float)
                        - np.asarray(replay.ekfs[target].position, dtype=float)
                    )
                )
                report_rewards = next_info.get("reporting_rewards", {})
                learning_rewards = next_info.get("learning_rewards", {})
                row = {
                    "episode_id": episode_id,
                    "scene_seed": seed,
                    "step": 0,
                    "target_uav_id": target,
                    "candidate_source": source,
                    "policy_selected_source": selected_by_policy,
                    "candidate_count_non_none": candidate_count,
                    "candidate_link_packet_loss_spread": link_spread,
                    "policy_probability": float(probabilities[source_to_index[source]]) if source_to_index[source] < probabilities.size else float("nan"),
                    "policy_adjusted_logit": float(logits[source_to_index[source]]) if source_to_index[source] < logits.size else float("nan"),
                    "legal_probability_spread": probability_spread,
                    "legal_logit_spread": logit_spread,
                    "selection_is_policy_choice": int(source == selected_by_policy),
                    "same_initial_state_probe": 1,
                    "same_bandwidth_and_velocity": 1,
                    "accepted_update_count": accepted,
                    "gate_uwb_valid": _bool(gate.get("uwb_valid", 0)),
                    "gate_effective_edge": _bool(gate.get("effective_edge", 0)),
                    "gate_message_arrived": _bool(gate.get("message_arrived", 0)),
                    "gate_new_packet": _bool(gate.get("new_packet", 0)),
                    "gate_ekf_update_attempted": _bool(gate.get("ekf_update_attempted", 0)),
                    "gate_ekf_update_accepted": _bool(gate.get("ekf_update_accepted", 0)),
                    "gate_failure_stage": str(gate.get("failure_stage", "none" if source == "none" else "not_selected")),
                    "target_position_error_after_m": error,
                    "report_team_reward": _number(next_info.get("team_reward")),
                    "report_j_loc": _number(dict(next_info.get("objective_components", {})).get("j_loc")),
                    "report_j_com": _number(dict(next_info.get("objective_components", {})).get("j_com")),
                    "learning_team_reward": _mean(_number(value) for value in dict(learning_rewards).values()),
                    "actual_measurement_count": _number(next_info.get("actual_measurement_count")),
                    "actual_measurement_bytes": _number(next_info.get("actual_measurement_bytes")),
                    "termination_reason": str(next_info.get("termination_reason", "")),
                }
                probe_rows.append(row)
                target_rows.append(row)
            for field in (
                "policy_probability",
                "policy_adjusted_logit",
                "accepted_update_count",
                "target_position_error_after_m",
                "report_team_reward",
                "report_j_loc",
                "report_j_com",
            ):
                values = [_number(row[field]) for row in target_rows]
                summary_rows.append(
                    {
                        "episode_id": episode_id,
                        "scene_seed": seed,
                        "target_uav_id": target,
                        "candidate_count": len(target_rows),
                        "metric": field,
                        "candidate_min": min(values) if values else float("nan"),
                        "candidate_max": max(values) if values else float("nan"),
                        "candidate_range": (max(values) - min(values)) if values else float("nan"),
                        "policy_choice_value": next(
                            (_number(row[field]) for row in target_rows if int(row["selection_is_policy_choice"])),
                            float("nan"),
                        ),
                        "descriptive_scope": "first_window_same_initial_state_only",
                    }
                )
    return probe_rows, summary_rows


def run(reference_scale: Path, equal_budget_logs: Path, output: Path) -> dict[str, Any]:
    _prepare_output(output)
    funnel, failures = _conversion_funnel(equal_budget_logs)
    _write_rows(output / "measurement_conversion_funnel.csv", funnel)
    _write_rows(output / "measurement_conversion_failures.csv", failures)
    _write_rows(output / "action_reward_alignment.csv", _action_reward_alignment(reference_scale))
    probe_rows, probe_summary = _same_state_action_probe(reference_scale, output)
    _write_rows(output / "same_state_action_probe.csv", probe_rows)
    _write_rows(output / "same_state_action_sensitivity_summary.csv", probe_summary)
    report = {
        "experiment": "three_link_diagnostic",
        "reference_scale": str(reference_scale),
        "equal_budget_logs": str(equal_budget_logs),
        "validation_only": True,
        "test_data_used": False,
        "training_update_called": False,
        "probe_scope": "first decision window; one legal measurement source changed at a time; bandwidth and velocity held fixed",
        "files": {
            "measurement_conversion_funnel": str(output / "measurement_conversion_funnel.csv"),
            "measurement_conversion_failures": str(output / "measurement_conversion_failures.csv"),
            "action_reward_alignment": str(output / "action_reward_alignment.csv"),
            "same_state_action_probe": str(output / "same_state_action_probe.csv"),
            "same_state_action_sensitivity_summary": str(output / "same_state_action_sensitivity_summary.csv"),
        },
    }
    _write_json(output / "audit_report.json", report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the read-only three-link validation diagnostic")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--equal-budget-logs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run(args.reference, args.equal_budget_logs, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["run"]
