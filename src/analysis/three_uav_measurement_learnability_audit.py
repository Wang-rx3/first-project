"""Bounded, offline learnability audit for the frozen v3 measurement run.

The audit only reads the calibration observation bank and CSV event logs.  It
never constructs an environment, policy, optimiser, or training transition.
The small counterfactual probe uses a cloned :class:`LocalEKF` for one scalar
range update, with the already logged post-transition state as its prior.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.simulation.ekf import LocalEKF


INPUT_STEM = "three_uav_measurement_selection_v3_seed2026_20261012"
OUTPUT_STEM = "three_uav_measurement_learnability_v4_seed2026_20261012"
UIDS = ("uav_00", "uav_01", "uav_02")
DT_S = 0.2
PERIOD_S = 0.8
STALE_S = 0.6
RANGE_VARIANCE_M2 = 0.08**2
NOISE_SCORES = (-1.0, 0.0, 1.0)
FROZEN = (
    {"scene_id": "train_compact_x", "step": 2, "uav_id": "uav_00"},
    {"scene_id": "train_bridge_x", "step": 49, "uav_id": "uav_00"},
    {"scene_id": "train_bridge_y", "step": 52, "uav_id": "uav_00"},
)


def _json(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(str(key))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in fields} for row in rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _input_hash_audit(root: Path) -> dict[str, Any]:
    protected_path = root / "logs" / "preliminary" / OUTPUT_STEM / "protected_inputs_before.json"
    protected = json.loads(protected_path.read_text(encoding="utf-8"))
    relative_paths = [
        f"data/configs/{INPUT_STEM}.json",
        f"data/synthetic/{INPUT_STEM}/calibration_s2026/calibration.json",
        f"data/synthetic/{INPUT_STEM}/calibration_s2026/configuration.json",
        f"data/synthetic/{INPUT_STEM}/calibration_s2026/manifest.json",
        f"data/synthetic/{INPUT_STEM}/calibration_s2026/observation_bank.json",
        f"logs/preliminary/{INPUT_STEM}/calibration_s2026/actions.csv",
        f"logs/preliminary/{INPUT_STEM}/calibration_s2026/candidate_values.csv",
        f"logs/preliminary/{INPUT_STEM}/calibration_s2026/fusion_events.csv",
        f"logs/preliminary/{INPUT_STEM}/calibration_s2026/localization_diagnostics.csv",
        f"logs/preliminary/{INPUT_STEM}/calibration_s2026/measurement_budget.csv",
        f"logs/preliminary/{INPUT_STEM}/calibration_s2026/measurement_gates.csv",
        f"logs/preliminary/{INPUT_STEM}/calibration_s2026/measurement_overhead.csv",
        f"logs/preliminary/{INPUT_STEM}/calibration_s2026/calibration_readiness.json",
    ]
    checks: list[dict[str, Any]] = []
    for relative in relative_paths:
        path = root / Path(relative)
        expected = protected.get(relative)
        if expected is None:
            raise ValueError(f"protected input hash missing: {relative}")
        actual = _sha256(path)
        checks.append({"path": relative, "expected_sha256": expected, "actual_sha256": actual, "match": actual == expected})
    if not all(item["match"] for item in checks):
        raise ValueError("protected input hash mismatch")
    return {"checked_count": len(checks), "all_match": True, "checks": checks}


def native_measurement_mask(observation: Mapping[str, Any]) -> list[bool]:
    """Return the four native action-mask entries (three peers then ``none``)."""

    mask = [bool(v) for v in observation.get("action_mask", ())]
    if len(mask) != 4:
        raise ValueError("observation action_mask must have three peers plus none")
    eligibility = [bool(v) for v in observation.get("measurement_update_eligibility", ())]
    if len(eligibility) != 3:
        raise ValueError("measurement_update_eligibility must contain three entries")
    expected = [False] * 4
    own = str(observation.get("uav_id", ""))
    budget_positive = _float(observation.get("measurement_budget_remaining")) > 0.0
    for index, peer in enumerate(UIDS):
        if peer != own and budget_positive and eligibility[index]:
            messages = observation.get("neighbor_messages", {})
            message = messages.get(peer) if isinstance(messages, Mapping) else None
            valid_message = False
            if isinstance(message, Mapping):
                try:
                    estimate = np.asarray(message.get("estimate"), dtype=float).reshape(15)
                    covariance = np.asarray(message.get("covariance"), dtype=float).reshape(15, 15)
                    valid_message = bool(np.isfinite(estimate).all() and np.isfinite(covariance).all() and np.allclose(covariance, covariance.T, atol=1e-10) and np.linalg.eigvalsh(covariance).min() >= -1e-9)
                except (TypeError, ValueError, np.linalg.LinAlgError):
                    valid_message = False
            physical = peer in set(observation.get("candidate_measurement_sources", ())) and peer in set(observation.get("effective_sources", ())) and bool(observation.get("uwb_available", {}).get(peer, False))
            expected[index] = bool(physical and valid_message)
    expected[3] = True
    return expected


def _physical_sources(observation: Mapping[str, Any]) -> list[str]:
    candidate = set(observation.get("candidate_measurement_sources", ()))
    effective = set(observation.get("effective_sources", ()))
    uwb = observation.get("uwb_available", {})
    messages = observation.get("neighbor_messages", {})
    valid: list[str] = []
    for source in sorted(candidate & effective):
        if not bool(uwb.get(source, False)) or not isinstance(messages.get(source), Mapping):
            continue
        try:
            estimate = np.asarray(messages[source].get("estimate"), dtype=float).reshape(15)
            covariance = np.asarray(messages[source].get("covariance"), dtype=float).reshape(15, 15)
            if np.isfinite(estimate).all() and np.isfinite(covariance).all() and np.allclose(covariance, covariance.T, atol=1e-10) and np.linalg.eigvalsh(covariance).min() >= -1e-9:
                valid.append(source)
        except (TypeError, ValueError, np.linalg.LinAlgError):
            continue
    return valid


def expected_linear_credit(observation: Mapping[str, Any], source: str) -> float:
    """Reproduce v3's documented accepted-linear-update proxy formula."""

    try:
        own = np.asarray(observation["estimate"], dtype=float)
        message = observation["neighbor_messages"][source]
        neighbor = np.asarray(message["estimate"], dtype=float)
        own_cov = np.asarray(observation["covariance"], dtype=float)[:3, :3]
        neighbor_cov = np.asarray(message["covariance"], dtype=float)[:3, :3]
        direction = neighbor[:3] - own[:3]
        direction /= np.linalg.norm(direction)
        local_var = float(direction @ own_cov @ direction)
        neighbor_var = float(direction @ neighbor_cov @ direction)
        trace = float(np.trace(own_cov))
        denominator = local_var + RANGE_VARIANCE_M2 + neighbor_var + 2.0 * math.sqrt(
            max(local_var, 0.0) * max(neighbor_var, 0.0)
        )
        numerator = float(direction @ own_cov @ own_cov @ direction)
        return float(np.clip((numerator / denominator) / trace, 0.0, 1.0))
    except (KeyError, TypeError, ValueError, FloatingPointError, ZeroDivisionError):
        return 0.0


def mask_loop_audit(bank: list[Mapping[str, Any]]) -> dict[str, Any]:
    checked = 0
    mismatches: list[dict[str, Any]] = []
    for frame in bank:
        if int(frame.get("step", -1)) >= 120:
            continue
        for uid, observation in frame["observations"].items():
            checked += 1
            try:
                expected = native_measurement_mask(observation)
            except ValueError as exc:
                mismatches.append({"scene_id": frame.get("scene_id"), "step": frame.get("step"), "uav_id": uid, "reason": str(exc)})
                continue
            actual = [bool(v) for v in observation.get("action_mask", ())]
            if expected != actual:
                mismatches.append({"scene_id": frame.get("scene_id"), "step": frame.get("step"), "uav_id": uid, "expected": expected, "actual": actual})
    return {"checked": checked, "mismatch_count": len(mismatches), "mismatches": mismatches}


def decision_audit(bank: list[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    stages = {"decision_frames": 0, "budget_positive": 0, "two_peer_physical": 0, "after_eligibility_and_base_mask": 0, "spread_gt_1e-6": 0}
    for frame in bank:
        step = int(frame.get("step", -1))
        if step >= 120:
            continue
        for uid, observation in frame["observations"].items():
            stages["decision_frames"] += 1
            budget_positive = _float(observation.get("measurement_budget_remaining")) > 0.0
            if budget_positive:
                stages["budget_positive"] += 1
            physical = _physical_sources(observation)
            if budget_positive and len(physical) >= 2:
                stages["two_peer_physical"] += 1
            legal = [source for source in physical if native_measurement_mask(observation)[UIDS.index(source)]]
            if budget_positive and len(legal) >= 2:
                stages["after_eligibility_and_base_mask"] += 1
            values = {source: expected_linear_credit(observation, source) for source in legal}
            spread = max(values.values(), default=0.0) - min(values.values(), default=0.0) if values else 0.0
            if budget_positive and len(legal) >= 2 and spread > 1e-6:
                stages["spread_gt_1e-6"] += 1
            rows.append({"scene_id": frame["scene_id"], "step": step, "uav_id": uid, "budget_positive": int(budget_positive), "physical_sources": ";".join(physical), "legal_sources": ";".join(legal), "legal_peer_count": len(legal), "expected_credit_spread": spread, "source_credits": json.dumps(values, sort_keys=True)})
    return rows, stages


def _episode_rows(rows: list[dict[str, str]], scene: str, seed: str | int | None = None) -> list[dict[str, str]]:
    return [row for row in rows if row.get("scene_id") == scene and (seed is None or str(row.get("seed")) == str(seed))]


def _matching_event(events: list[dict[str, str]], bank_frame: Mapping[str, Any], target: str) -> dict[str, str]:
    scene = str(bank_frame["scene_id"])
    step = int(bank_frame["step"])
    observation = bank_frame["observations"][target]
    estimate = np.asarray(observation["estimate"], dtype=float)
    candidates = [row for row in events if row.get("scene_id") == scene and int(_float(row.get("step"), -1)) == step and row.get("target_uav_id") == target]
    if not candidates:
        raise ValueError(f"no historical fusion event for {scene} step {step} target {target}")
    return min(candidates, key=lambda row: float(np.linalg.norm(np.asarray(_json(row["state_after"]), dtype=float) - estimate)))


def _truth_row(rows: list[dict[str, str]], scene: str, step: int, uid: str, seed: str) -> dict[str, str]:
    matches = [r for r in rows if r.get("scene_id") == scene and int(_float(r.get("step"), -1)) == step and r.get("uav_id") == uid and str(r.get("seed")) == str(seed)]
    if not matches:
        raise ValueError(f"missing diagnostic truth row for {scene} step {step} {uid}")
    return matches[0]


def _fresh_source(events: list[dict[str, str]], gates: list[dict[str, str]], scene: str, step: int, source: str, target: str, seed: str, *, post_message_available: bool = False, current_event: Mapping[str, str] | None = None) -> dict[str, Any]:
    previous = [r for r in events if r.get("scene_id") == scene and r.get("source_uav_id") == source and r.get("target_uav_id") == target and str(r.get("seed")) == str(seed) and int(_float(r.get("step"), -1)) < step]
    previous.sort(key=lambda r: int(_float(r.get("step"), -1)))
    last = previous[-1] if previous else None
    decision_time = step * DT_S
    # The bank frame is a fresh-arrived packet snapshot by the environment's
    # observation contract, but it does not retain packet timing.  Preserve
    # that uncertainty for alternate sources.  Selected historical events do
    # carry the measured age/send time and are reported directly.
    if current_event is not None:
        age: float | None = _float(current_event.get("message_age_s"), math.nan)
        send_time: float | None = _float(current_event.get("message_send_time_s"), math.nan)
    else:
        age = None
        send_time = None
    attempts = [r for r in gates if r.get("scene_id") == scene and r.get("source_uav_id") == source and r.get("target_uav_id") == target and str(r.get("seed")) == str(seed) and int(_float(r.get("step"), -1)) < step and str(r.get("ekf_update_attempted", "")).lower() == "true"]
    last_attempt_step = max((int(_float(r.get("step"), -1)) for r in attempts), default=-10**9)
    period_passed = step - last_attempt_step >= int(round(PERIOD_S / DT_S))
    age_ok = bool(current_event is not None and age is not None and math.isfinite(age) and age <= STALE_S)
    fresh = bool(post_message_available and period_passed and (age_ok if current_event is not None else True))
    return {"fresh": fresh, "message_age_s": age, "last_update_step": last_attempt_step, "new_send_time": send_time, "attempt_count_before": len(attempts), "period_passed": bool(period_passed), "post_message_available": bool(post_message_available), "freshness_source": "selected_event_timestamp+past_gate_attempts" if current_event is not None else "bank_fresh_arrived_message_contract+past_gate_attempts", "timestamp_known": bool(current_event is not None)}


def _replay_selected(event: Mapping[str, str], post_observation: Mapping[str, Any], target_truth: np.ndarray, *, boundary_projected: bool = False) -> dict[str, Any]:
    source = str(event["source_uav_id"])
    message = post_observation["neighbor_messages"][source]
    before = np.asarray(_json(event["state_before"]), dtype=float)
    prior_cov = np.asarray(post_observation["covariance"], dtype=float)
    ekf = LocalEKF(before)
    ekf.covariance = prior_cov.copy()
    accepted = ekf.update_relative_range(float(event["measured_range_m"]), np.asarray(message["estimate"], dtype=float)[:3], np.asarray(message["covariance"], dtype=float), measurement_variance=_float(event["measurement_variance_m2"], RANGE_VARIANCE_M2), nlos_bias_m=_float(event["nlos_bias_m"]), timestamp=_float(event["measurement_time_s"]))
    logged_after = np.asarray(_json(event["state_after"]), dtype=float)
    trace_after = float(np.trace(ekf.covariance[:3, :3]))
    logged_trace = _float(event["position_covariance_trace_after_final_m2"])
    return {"accepted": bool(accepted), "source": source, "state_max_abs_error": float(np.max(np.abs(ekf.state - logged_after))), "final_cov_trace_abs_error": abs(trace_after - logged_trace), "covariance_restored": bool(_json(event.get("covariance_restored", False))), "ci_weight": _float(event.get("ci_weight"), math.nan), "boundary_projected": bool(boundary_projected), "position_error_before_m": float(np.linalg.norm(before[:3] - target_truth)), "position_error_after_m": float(np.linalg.norm(ekf.state[:3] - target_truth)), "logged_final_cov_trace_m2": logged_trace, "replayed_final_cov_trace_m2": trace_after}


def _counterfactual_row(post: Mapping[str, Any], event: Mapping[str, str], target_truth: np.ndarray, source_truth: np.ndarray, choice: str, z: float) -> dict[str, Any]:
    source = str(event["source_uav_id"])
    message = post["neighbor_messages"].get(source)
    before = np.asarray(_json(event["state_before"]), dtype=float)
    prior_cov = np.asarray(post["covariance"], dtype=float)
    before_error = float(np.linalg.norm(before[:3] - target_truth))
    row: dict[str, Any] = {"scene_id": post.get("_scene_id"), "step": int(post.get("_step", -1)), "uav_id": post.get("_uav_id"), "choice": choice, "noise_z": z, "mean_before_15": before.tolist(), "full_P_before_15x15": prior_cov.tolist(), "trace_before_m2": float(np.trace(prior_cov[:3, :3])), "position_error_before_m": before_error, "accepted": 0, "mean_after_15": before.tolist(), "full_P_after_15x15": prior_cov.tolist(), "trace_after_m2": float(np.trace(prior_cov[:3, :3])), "position_error_after_m": before_error, "improvement_m": 0.0, "linear_proxy": 0.0, "final_cov_gain_m2": 0.0, "position_correction_norm_m": 0.0, "velocity_correction_norm_mps": 0.0, "cost_packet": 0, "bytes": 0, "budget_used": 0, "truth_actor_available": False, "reward_used": False}
    if choice == "none":
        return row
    if not isinstance(message, Mapping):
        row["unavailable_reason"] = "post_message_missing"
        return row
    true_range = float(np.linalg.norm(target_truth - source_truth))
    measured = true_range + float(z) * 0.08
    ekf = LocalEKF(before)
    ekf.covariance = prior_cov.copy()
    accepted = ekf.update_relative_range(measured, np.asarray(message["estimate"], dtype=float)[:3], np.asarray(message["covariance"], dtype=float), measurement_variance=RANGE_VARIANCE_M2, nlos_bias_m=0.0, timestamp=_float(event.get("measurement_time_s")))
    trace_linear = _float(ekf.last_relative_range_update.get("position_covariance_trace_after_linear_m2"), row["trace_before_m2"])
    correction = ekf.state - before
    row.update({"accepted": int(bool(accepted)), "mean_after_15": ekf.state.tolist(), "full_P_after_15x15": ekf.covariance.tolist(), "trace_after_m2": float(np.trace(ekf.covariance[:3, :3])), "position_error_after_m": float(np.linalg.norm(ekf.state[:3] - target_truth)), "improvement_m": before_error - float(np.linalg.norm(ekf.state[:3] - target_truth)), "linear_proxy": (row["trace_before_m2"] - trace_linear) / max(row["trace_before_m2"], 1e-12) if accepted else 0.0, "final_cov_gain_m2": row["trace_before_m2"] - float(np.trace(ekf.covariance[:3, :3])) if accepted else 0.0, "position_correction_norm_m": float(np.linalg.norm(correction[:3])), "velocity_correction_norm_mps": float(np.linalg.norm(correction[3:6])), "cost_packet": 1, "bytes": 512, "budget_used": 1, "true_range_m": true_range, "measured_range_m": measured})
    return row


def scalar_covariance_counterexample() -> dict[str, Any]:
    prior = 1.0
    # The scalar witness uses the same 9 + R innovation denominator as the
    # range update (R=0.0064), hence k=1/(9.0064).
    gain = 1.0 / 9.0064
    noise = 0.0064
    posterior = (1.0 - gain + 2.0 * gain) ** 2 * prior + gain**2 * noise
    joint = np.asarray([[1.0, 2.0], [2.0, 4.0]])
    return {"prior_variance": prior, "source_variance": 4.0, "measurement_variance": noise, "gain": gain, "true_posterior_variance": posterior, "restored_prior_variance": prior, "joint_covariance": joint.tolist(), "joint_covariance_eigenvalues": np.linalg.eigvalsh(joint).tolist(), "restored_is_smaller": bool(prior < posterior), "interpretation": "The scalar witness has a true posterior above the restored prior under unknown cross-correlation; the restored prior is a diagnostic baseline, not a guaranteed conservative posterior bound."}


def run_audit(project_root: str | Path, run_id: str, source_run: str = "calibration_s2026") -> dict[str, Any]:
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', str(run_id)):
        raise ValueError('Unsafe run-id')
    if source_run != 'calibration_s2026':
        raise ValueError('This audit is locked to the v3 calibration_s2026 source')
    root = Path(project_root)
    input_data = root / "data" / "synthetic" / INPUT_STEM / source_run
    input_logs = root / "logs" / "preliminary" / INPUT_STEM / source_run
    output_data = root / "data" / "synthetic" / OUTPUT_STEM / run_id
    output_logs = root / "logs" / "preliminary" / OUTPUT_STEM / run_id
    if output_data.exists() or output_logs.exists():
        raise FileExistsError(f"refusing to reuse existing offline audit run: {run_id}")
    with (input_data / "observation_bank.json").open("r", encoding="utf-8") as handle:
        bank = json.load(handle)
    lock_path = root / "data" / "synthetic" / OUTPUT_STEM / "diagnostic_design.lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get('frozen_counterfactuals') != list(FROZEN) or tuple(lock.get('noise_standard_scores', ())) != NOISE_SCORES:
        raise ValueError('Frozen counterfactual design changed')
    source = {name: _read_csv(input_logs / name) for name in ("actions.csv", "fusion_events.csv", "measurement_budget.csv", "measurement_gates.csv", "candidate_values.csv", "localization_diagnostics.csv")}
    input_hashes = _input_hash_audit(root)
    readiness = json.loads((input_logs / "calibration_readiness.json").read_text(encoding="utf-8"))
    mask_report = mask_loop_audit(bank)
    decision_rows, stage_counts = decision_audit(bank)
    accepted = [row for row in source["fusion_events.csv"] if str(row.get("accepted", "")).lower() == "true"]
    readiness_aggregate = readiness.get("aggregate", {})
    csv_readiness_checks = {
        "candidate_values_rows": len(source["candidate_values.csv"]) == stage_counts["decision_frames"],
        "actions_rows": len(source["actions.csv"]) == stage_counts["decision_frames"],
        "measurement_budget_rows": len(source["measurement_budget.csv"]) == stage_counts["decision_frames"],
        "localization_rows": len(source["localization_diagnostics.csv"]) == stage_counts["decision_frames"],
        "readiness_decision_count": int(readiness_aggregate.get("decision_count", -1)) == stage_counts["decision_frames"],
        "readiness_positive_credit_count": int(readiness_aggregate.get("positive_credit_count", -1)) == len(accepted),
        "readiness_multi_peer_count": int(readiness_aggregate.get("multi_peer_decision_count", -1)) == stage_counts["after_eligibility_and_base_mask"],
        "readiness_separated_count": int(readiness_aggregate.get("separated_decision_count", -1)) == stage_counts["spread_gt_1e-6"],
        "readiness_max_spread": abs(_float(readiness_aggregate.get("max_expected_credit_spread"), math.nan) - max((row["expected_credit_spread"] for row in decision_rows if row["legal_peer_count"] >= 2), default=0.0)) <= 1e-10,
    }
    if not all(csv_readiness_checks.values()):
        raise ValueError(f"offline stage/readiness mismatch: {csv_readiness_checks}")
    fusion = source["fusion_events.csv"]
    diagnostics = source["localization_diagnostics.csv"]
    gates = source["measurement_gates.csv"]
    frozen_rows: list[dict[str, Any]] = []
    cf_rows: list[dict[str, Any]] = []
    for frozen in FROZEN:
        scene, step, target = frozen["scene_id"], frozen["step"], frozen["uav_id"]
        post = next(frame for frame in bank if frame["scene_id"] == scene and int(frame["step"]) == step)
        pre = next(frame for frame in bank if frame["scene_id"] == scene and int(frame["step"]) == step - 1)
        event = _matching_event(fusion, post, target)
        seed = event["seed"]
        target_diag = _truth_row(diagnostics, scene, step, target, seed)
        target_truth = np.asarray(_json(target_diag["true_position_m_diagnostic"]), dtype=float)
        source_truths = {peer: np.asarray(_json(_truth_row(diagnostics, scene, step, peer, seed)["true_position_m_diagnostic"]), dtype=float) for peer in UIDS if peer != target}
        replay = _replay_selected(event, post["observations"][target], target_truth, boundary_projected=str(target_diag.get("boundary_projected", "False")).lower() == "true")
        replay["scene_id"], replay["step"], replay["uav_id"] = scene, step, target
        replay["selected_event_source"] = event["source_uav_id"]
        replay["selected_event_seed"] = seed
        replay["pre_legal_sources"] = [peer for peer in UIDS if peer != target and native_measurement_mask(pre["observations"][target])[UIDS.index(peer)]]
        replay["post_actual_sources"] = _physical_sources(post["observations"][target])
        replay["replay_verified"] = bool(replay["accepted"] and replay["state_max_abs_error"] <= 1e-8 and replay["final_cov_trace_abs_error"] <= 1e-8 and replay["covariance_restored"] and abs(replay["ci_weight"] - 1.0) <= 1e-12 and not replay["boundary_projected"])
        replay["source_freshness"] = {peer: _fresh_source(fusion, gates, scene, step, peer, target, seed, post_message_available=peer in post["observations"][target].get("neighbor_messages", {}), current_event=event if peer == event["source_uav_id"] else None) for peer in UIDS if peer != target}
        replay["post_boundary_projected"] = bool(replay["boundary_projected"])
        replay["freshness_all_sources"] = all(item["fresh"] for item in replay["source_freshness"].values())
        frozen_rows.append(replay)
        post_sources = set(replay["post_actual_sources"])
        pre_sources = sorted(replay["pre_legal_sources"])
        unavailable: list[str] = []
        if not replay["replay_verified"]:
            unavailable.append("historical_replay_not_verified")
        if len(pre_sources) != 2:
            unavailable.append("pre_legal_source_count_not_two")
        if not set(pre_sources).issubset(post_sources):
            unavailable.append("post_physical_source_missing")
        if not replay["freshness_all_sources"]:
            unavailable.append("post_message_freshness_not_established")
        if unavailable:
            replay["counterfactual_available"] = False
            replay["counterfactual_unavailable_reason"] = unavailable
            continue
        replay["counterfactual_available"] = True
        for source_name, choice in zip(pre_sources, ("B", "C")):
            for z in NOISE_SCORES:
                post_obs = dict(post["observations"][target]); post_obs.update({"_scene_id": scene, "_step": step, "_uav_id": target})
                cf_rows.append(_counterfactual_row(post_obs, {**event, "source_uav_id": source_name}, target_truth, source_truths[source_name], choice, z))
        for z in NOISE_SCORES:
            post_obs = dict(post["observations"][target]); post_obs.update({"_scene_id": scene, "_step": step, "_uav_id": target})
            cf_rows.append(_counterfactual_row(post_obs, event, target_truth, source_truths[event["source_uav_id"]], "none", z))
    improvements = [_float(row.get("true_position_error_before_m")) - _float(row.get("true_position_error_after_m")) for row in accepted]
    proxy = [_float(row.get("linear_relative_credit_proxy")) for row in accepted]
    distances = {"accepted": len(accepted), "improve": sum(value > 0 for value in improvements), "worsen": sum(value < 0 for value in improvements), "mean_distance_improvement_m": float(np.mean(improvements)), "mean_positive_distance_improvement_m": float(np.mean([value for value in improvements if value > 0])), "proxy_min": min(proxy), "proxy_max": max(proxy), "proxy_distance_correlation": float(np.corrcoef(proxy, improvements)[0, 1])}
    square_errors = [_float(row.get("true_position_error_before_m")) ** 2 - _float(row.get("true_position_error_after_m")) ** 2 for row in accepted]
    reward_audit = {"linear_proxy_is_final_covariance": False, "distance_improvement_count": distances["improve"], "distance_worsen_count": distances["worsen"], "mean_squared_error_improvement_m2": float(np.mean(square_errors)), "final_covariance_trace_gain_m2": 0.0, "true_j_loc_unchanged": all(abs(_float(row.get("j_loc_before")) - _float(row.get("j_loc_after"))) <= 1e-15 for row in accepted), "accepted_count": len(accepted), "error_improvement_definition": "true_position_error_before_m - true_position_error_after_m", "warning": "true-position diagnostics are audit-only and never used as a training reward"}
    counterfactual_summary: dict[str, Any] = {}
    for choice in ("B", "C", "none"):
        choice_rows = [row for row in cf_rows if row.get("choice") == choice]
        improvements_cf = [_float(row.get("improvement_m")) for row in choice_rows]
        counterfactual_summary[choice] = {"rows": len(choice_rows), "accepted": sum(int(_float(row.get("accepted"))) for row in choice_rows), "positive_improvement": sum(value > 0.0 for value in improvements_cf), "mean_improvement_m": float(np.mean(improvements_cf)) if improvements_cf else None, "min_improvement_m": min(improvements_cf) if improvements_cf else None, "max_improvement_m": max(improvements_cf) if improvements_cf else None, "mean_linear_proxy": float(np.mean([_float(row.get("linear_proxy")) for row in choice_rows])) if choice_rows else None, "final_covariance_gain_values_m2": sorted(set(_float(row.get("final_cov_gain_m2")) for row in choice_rows)), "ci_or_pvalue": None}
    budget_end_map: dict[tuple[str, str], dict[str, Any]] = {}
    for row in source["measurement_budget.csv"]:
        if _float(row.get("budget_consumed")) > 0:
            key = (str(row.get("episode_id")), str(row.get("uav_id")))
            candidate = {"episode_id": row.get("episode_id"), "scene_id": row.get("scene_id"), "uav_id": row.get("uav_id"), "last_consumption_step": int(_float(row.get("step"), -1))}
            if key not in budget_end_map or candidate["last_consumption_step"] > budget_end_map[key]["last_consumption_step"]:
                budget_end_map[key] = candidate
    budget_end_rows = sorted(budget_end_map.values(), key=lambda row: (str(row["episode_id"]), str(row["uav_id"])))
    gates_count = {"accepted": sum(row.get("failure_stage") == "accepted" for row in gates), "uwb_invalid": sum(row.get("failure_stage") == "uwb_invalid" for row in gates), "total_gate_rows": len(gates), "budget_units": len(accepted) + sum(row.get("failure_stage") == "uwb_invalid" for row in gates)}
    report = {"schema": "bounded_v4_measurement_learnability_audit", "run_id": run_id, "source_run": source_run, "input_stem": INPUT_STEM, "output_stem": OUTPUT_STEM, "offline_only": True, "optimizer_updates": 0, "environment_steps": 0, "design_lock": lock, "protected_input_hashes": input_hashes, "mask_loop": mask_report, "stage_counts": stage_counts, "csv_readiness_checks": csv_readiness_checks, "gate_counts": gates_count, "distances_and_proxy": distances, "reward_audit": reward_audit, "frozen_replay": frozen_rows, "counterfactual_rows": len(cf_rows), "counterfactual_summary": counterfactual_summary, "scalar_counterexample": scalar_covariance_counterexample(), "claim_limitation": "This is a bounded frozen-state and historical-log diagnostic; it does not establish policy learnability or full-horizon counterfactual performance."}
    _write_json(output_data / "audit_report.json", report)
    _write_json(output_data / "frozen_replay.json", frozen_rows)
    _write_json(output_data / "counterfactual_rows.json", cf_rows)
    _write_json(output_data / "scalar_counterexample.json", report["scalar_counterexample"])
    _write_csv(output_logs / "stage_counts.csv", [stage_counts])
    _write_csv(output_logs / "decision_rows.csv", decision_rows)
    _write_csv(output_logs / "budget_end.csv", budget_end_rows)
    _write_csv(output_logs / "reward_audit.csv", [reward_audit])
    _write_csv(output_logs / "counterfactual_rows.csv", cf_rows)
    _write_json(output_logs / "audit_report.json", report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source-run", default="calibration_s2026")
    parser.add_argument("--project-root", default=None)
    args = parser.parse_args(argv)
    root = Path(args.project_root) if args.project_root else Path(__file__).resolve().parents[2]
    report = run_audit(root, args.run_id, args.source_run)
    print(json.dumps({"run_id": report["run_id"], "stage_counts": report["stage_counts"], "frozen_replay": report["frozen_replay"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

