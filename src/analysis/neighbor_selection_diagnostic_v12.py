"""Read-only Stage 1 diagnostic for the v12 neighbour learnability study.

The module consumes the frozen v11 counterfactual JSON files and (when
available) v12 counterfactual records.  It never imports a simulator, a
filter implementation, a PPO policy, or a replay runner.  All quantities in
this file are derived from JSON that already exists on disk.

``collect_points`` is intentionally a small data boundary.  A physical case
record is loaded once and cached by ``(phase, geometry, seed)`` while its six
counterfactual checkpoints are flattened.  ``summarize_points`` is pure and
does not write files.  ``stage1`` is the only function that creates the new
v12 evidence directory.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .neighbor_selection_gate_v11 import correlation as _v11_correlation
from ..simulation.random_streams import indexed_rng


ROOT = Path(__file__).resolve().parents[2]
V11_ROOT = ROOT / "data" / "synthetic" / "three_uav_neighbor_value_v11_20261013"
V11_BATCH = V11_ROOT / "batches" / "research_001"
V12_ROOT = ROOT / "data" / "synthetic" / "three_uav_neighbor_learnability_v12_20261013"
V12_STAGE1 = V12_ROOT / "stage1"
V12_LOG = ROOT / "logs" / "preliminary" / "three_uav_neighbor_learnability_v12_20261013" / "stage1"
DEFAULT_CF_DIR = V11_BATCH / "counterfactual"
DEFAULT_PHYSICAL_BASE = V11_BATCH

TARGET = "uav_02"
PEERS = ("uav_00", "uav_01")
CF_STEPS = (6, 10, 30, 62, 94, 114)
COST_M2 = 0.01
MEANINGFUL_M2 = 0.01
SENSITIVITY_M2 = 1.0e-12
TIME_TOLERANCE_S = 1.0e-12


# Re-exporting the v11 helper keeps tied-rank behavior identical to the
# frozen report.  In particular, a constant vector returns None.
correlation = _v11_correlation


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _same_time(left: Any, right: Any) -> bool:
    a, b = _finite(left), _finite(right)
    return a is not None and b is not None and abs(a - b) <= TIME_TOLERANCE_S


def _mapping(value: Any) -> Mapping[str, Any] | None:
    return value if isinstance(value, Mapping) else None


def _json_load(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _as_position(value: Any) -> list[float] | None:
    if isinstance(value, Mapping):
        for key in ("position", "estimate", "state", "P3", "p3", "arrived_peer_P3"):
            if key in value:
                parsed = _as_position(value[key])
                if parsed is not None:
                    return parsed
        return None
    try:
        values = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    if len(values) >= 3 and all(math.isfinite(item) for item in values[:3]):
        return values[:3]
    return None


def _as_matrix(value: Any, size: int = 3) -> list[list[float]] | None:
    try:
        rows = [[float(item) for item in row] for row in value]
    except (TypeError, ValueError):
        return None
    if len(rows) != size or any(len(row) != size for row in rows):
        return None
    if not all(math.isfinite(item) for row in rows for item in row):
        return None
    return rows


def _trace(value: Any, size: int = 3) -> float | None:
    matrix = _as_matrix(value, size)
    if matrix is None:
        return None
    return float(sum(matrix[i][i] for i in range(size)))


def _position_trace(value: Any) -> float | None:
    """Trace the leading 3x3 block of a full covariance payload."""

    try:
        matrix = [[float(item) for item in row] for row in value]
    except (TypeError, ValueError):
        return None
    if len(matrix) < 3 or any(len(row) < 3 for row in matrix[:3]):
        return None
    diagonal = [matrix[i][i] for i in range(3)]
    return float(sum(diagonal)) if all(math.isfinite(item) for item in diagonal) else None


def _sse(branch: Mapping[str, Any] | None) -> list[float] | None:
    if branch is None:
        return None
    value = branch.get("sse_m2", branch.get("sse"))
    try:
        result = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    if not result or not all(math.isfinite(item) and item >= 0.0 for item in result):
        return None
    return result


def _branches(point: Mapping[str, Any]) -> Mapping[str, Any]:
    for key in ("branches", "v12_branches", "counterfactual_branches", "labels"):
        value = point.get(key)
        if isinstance(value, Mapping):
            return value
    nested = point.get("counterfactual")
    if isinstance(nested, Mapping) and isinstance(nested.get("branches"), Mapping):
        return nested["branches"]
    return {}


def _branch(point: Mapping[str, Any], action: str) -> Mapping[str, Any] | None:
    value = _branches(point).get(action)
    return value if isinstance(value, Mapping) else None


def _mean_gain(none_sse: Sequence[float] | None, peer_sse: Sequence[float] | None) -> float | None:
    if none_sse is None or peer_sse is None or len(none_sse) != len(peer_sse):
        return None
    return float(sum(a - b for a, b in zip(none_sse, peer_sse)) / len(none_sse))


def _current_gain(none_sse: Sequence[float] | None, peer_sse: Sequence[float] | None) -> tuple[float | None, float | None]:
    if not none_sse or not peer_sse:
        return None, None
    current_sse = float(none_sse[0] - peer_sse[0])
    norm = math.sqrt(none_sse[0]) - math.sqrt(peer_sse[0])
    return float(norm), current_sse


def _action_cost(branch: Mapping[str, Any] | None, action: str) -> float:
    # v12 fixes the accounting charge to one 512-byte-equivalent request,
    # including an attempted or rejected peer update.  Do not infer a cost
    # from a truth-dependent acceptance flag.
    if action == "none":
        return 0.0
    return COST_M2


def _score_from_scoring(scoring: Mapping[str, Any] | None, peer: str) -> float | None:
    if scoring is None:
        return None
    candidates = scoring.get("candidates", scoring.get("peer_candidates", {}))
    candidate = candidates.get(peer) if isinstance(candidates, Mapping) else None
    if not isinstance(candidate, Mapping):
        return None
    for key in ("score", "proxy_score", "old_score"):
        number = _finite(candidate.get(key))
        if number is not None:
            return number
    return None


def _score_difference(point: Mapping[str, Any], scoring: Mapping[str, Any] | None = None) -> float | None:
    direct = point.get("score_difference(old)")
    if direct is None:
        direct = point.get("old_score_difference")
    if direct is None:
        direct = point.get("score_difference")
    number = _finite(direct)
    if number is not None:
        return number
    scores = [_score_from_scoring(scoring, peer) for peer in PEERS]
    return None if any(item is None for item in scores) else float(scores[0] - scores[1])


def _minimal_score_difference(point: Mapping[str, Any]) -> float | None:
    for key in ("minimal_score_difference", "minimal_proxy_score_difference", "v12_score_difference"):
        number = _finite(point.get(key))
        if number is not None:
            return number
    scoring = point.get("minimal_scoring", point.get("v12_scoring"))
    if not isinstance(scoring, Mapping):
        scoring = point.get("old_scoring")
    if isinstance(scoring, Mapping):
        direct = _finite(scoring.get("score_difference"))
        if direct is not None:
            return direct
        candidates = scoring.get("candidates", {})
        if isinstance(candidates, Mapping):
            gains = [_finite(candidates.get(peer, {}).get("trace_gain_m2")) if isinstance(candidates.get(peer), Mapping) else None for peer in PEERS]
            if all(gain is not None for gain in gains):
                # Both candidates pay the same fixed request cost.  Keeping
                # it in the scalar preserves the offline R definition while
                # the difference remains invariant to that common charge.
                return float((gains[0] - COST_M2) - (gains[1] - COST_M2))
        values = [_score_from_scoring(scoring, peer) for peer in PEERS]
        if all(item is not None for item in values):
            return float(values[0] - values[1])
    return None


def _deterministic_tie_choice(seed: Any, step: Any, tied: Sequence[str], *, minimal: bool = False) -> str:
    if not tied:
        return "none"
    stream = "v12_minimal_tie" if minimal else "v11_greedy_tie"
    return str(indexed_rng(int(_finite(seed) or 0), int(_finite(step) or 0), stream, TARGET).choice(list(tied)))


def _choice_from_scoring(scoring: Mapping[str, Any] | None, point: Mapping[str, Any], *, minimal: bool = False) -> str | None:
    if not isinstance(scoring, Mapping):
        return None
    context = point.get("pre_context") if isinstance(point.get("pre_context"), Mapping) else {}
    budget = _finite(context.get("measurement_budget_remaining"))
    if budget is not None and budget <= 0.0:
        return "none"
    candidates = scoring.get("candidates", scoring.get("peer_candidates", {}))
    if not isinstance(candidates, Mapping):
        return "none"
    eligible: list[tuple[str, float]] = []
    for peer in PEERS:
        candidate = candidates.get(peer)
        if not isinstance(candidate, Mapping) or candidate.get("valid") is False:
            continue
        if minimal:
            value = _finite(candidate.get("trace_gain_m2", candidate.get("predicted_trace_gain_m2")))
            value = None if value is None else value - COST_M2
        else:
            value = _finite(candidate.get("score", candidate.get("old_score")))
        if value is not None and value > 0.0:
            eligible.append((peer, value))
    if not eligible:
        return "none"
    best = max(value for _, value in eligible)
    tied = [peer for peer, value in eligible if abs(value - best) <= 1.0e-12]
    return tied[0] if len(tied) == 1 else _deterministic_tie_choice(
        point.get("seed"), point.get("step"), tied, minimal=minimal
    )


def label_point(point: Mapping[str, Any]) -> dict[str, Any]:
    """Return a flattened point with v12's fixed labels and observable choices.

    The function is pure.  ``meaningful`` follows the v11 four-state
    threshold, while ``sensitivity_no_effect`` records the much tighter
    numerical no-effect check.  Rows without the three branch SSE traces are
    retained but marked non-deployable.
    """

    row = dict(point)
    branches = _branches(point)
    branch_values: dict[str, dict[str, Any]] = {}
    for action in ("none", *PEERS):
        branch = _branch(point, action)
        sse = _sse(branch)
        norm, current_sse = _current_gain(_sse(_branch(point, "none")), sse)
        four = _mean_gain(_sse(_branch(point, "none")), sse)
        cost = _action_cost(branch, action)
        branch_values[action] = {
            "current_norm_gain": norm,
            "current_sse_gain": current_sse,
            "four_state_sse_gain": four,
            "offline_reward_current": None if current_sse is None else current_sse - cost,
            "offline_reward_fourstate": None if four is None else four - cost,
            "cost_m2": cost,
            "sse": sse,
        }

    for action, values in branch_values.items():
        suffix = action
        for name, value in values.items():
            if name == "sse":
                row[f"{name}_{suffix}"] = value
            else:
                row[f"{name}_{suffix}"] = value

    none = branch_values["none"]
    for peer in PEERS:
        values = branch_values[peer]
        row[f"current_norm_gain_{peer}"] = values["current_norm_gain"]
        row[f"current_sse_gain_{peer}"] = values["current_sse_gain"]
        row[f"four_state_sse_gain_{peer}"] = values["four_state_sse_gain"]
        row[f"offline_R_current_{peer}"] = values["offline_reward_current"]
        row[f"offline_R_fourstate_{peer}"] = values["offline_reward_fourstate"]
    row["current_norm_gain"] = {peer: branch_values[peer]["current_norm_gain"] for peer in PEERS}
    row["current_sse_gain"] = {peer: branch_values[peer]["current_sse_gain"] for peer in PEERS}
    row["mean4state_sse_gain"] = {peer: branch_values[peer]["four_state_sse_gain"] for peer in PEERS}
    row["offline_Rcurrent"] = {peer: branch_values[peer]["offline_reward_current"] for peer in PEERS}
    row["offline_Rfour"] = {peer: branch_values[peer]["offline_reward_fourstate"] for peer in PEERS}

    score_difference = _score_difference(point, point.get("old_scoring") if isinstance(point.get("old_scoring"), Mapping) else None)
    minimal_difference = _minimal_score_difference(point)
    row["score_difference(old)"] = score_difference
    row["old_score_difference"] = score_difference
    row["score_difference"] = score_difference
    row["minimal_score_difference"] = minimal_difference
    old_scoring = point.get("old_scoring") if isinstance(point.get("old_scoring"), Mapping) else None
    minimal_scoring = point.get("minimal_scoring") if isinstance(point.get("minimal_scoring"), Mapping) else point.get("v12_scoring")
    old_choice = _choice_from_scoring(old_scoring, point, minimal=False)
    minimal_choice = _choice_from_scoring(minimal_scoring if isinstance(minimal_scoring, Mapping) else old_scoring, point, minimal=True)
    row["minimal_pred_choice"] = minimal_choice
    row["minimal_choice"] = row["minimal_pred_choice"]
    row["old_choice"] = old_choice

    true_differences: dict[str, float | None] = {}
    for metric in ("current_norm_gain", "current_sse_gain", "four_state_sse_gain", "offline_reward_current", "offline_reward_fourstate"):
        left, right = branch_values[PEERS[0]][metric], branch_values[PEERS[1]][metric]
        true_differences[metric] = None if left is None or right is None else float(left - right)
    row["true_value_peer_difference"] = true_differences
    row["true_peer_difference_current_norm"] = true_differences["current_norm_gain"]
    row["true_peer_difference_current_sse"] = true_differences["current_sse_gain"]
    row["true_peer_difference_fourstate"] = true_differences["four_state_sse_gain"]
    row["utility_difference(fourstate)"] = _finite(point.get("utility_difference(fourstate)", point.get("utility_difference")))
    if row["utility_difference(fourstate)"] is None:
        utility = []
        for peer in PEERS:
            value = _finite(_branch(point, peer).get("utility")) if _branch(point, peer) is not None else None
            utility.append(value)
        if all(value is not None for value in utility):
            row["utility_difference(fourstate)"] = float(utility[0] - utility[1])
    if row["utility_difference(fourstate)"] is None:
        row["utility_difference(fourstate)"] = true_differences["four_state_sse_gain"]
    row["utility_difference"] = row["utility_difference(fourstate)"]

    four_difference = true_differences["four_state_sse_gain"]
    row["sensitivity_no_effect"] = bool(four_difference is not None and abs(four_difference) <= SENSITIVITY_M2)
    row["meaningful"] = bool(four_difference is not None and abs(four_difference) >= MEANINGFUL_M2)
    row["label_complete"] = bool(all(branch_values[action]["sse"] is not None for action in ("none", *PEERS)))
    # True-value labels are evaluation-only.  No row is treated as a
    # deployable selector input, even when all three branch SSE traces exist.
    row["label_deployable"] = False
    row["rankmeaningful"] = bool(row["meaningful"] and score_difference is not None)
    if row["meaningful"] and score_difference is not None and four_difference is not None:
        if abs(score_difference) <= 1.0e-12 or abs(four_difference) <= SENSITIVITY_M2:
            row["ranking_correct"] = 0.5
        else:
            row["ranking_correct"] = float((score_difference * four_difference) > 0.0)
    else:
        row["ranking_correct"] = None
    if row["meaningful"] and minimal_difference is not None and four_difference is not None:
        if abs(minimal_difference) <= 1.0e-12 or abs(four_difference) <= SENSITIVITY_M2:
            row["minimal_ranking_correct"] = 0.5
        else:
            row["minimal_ranking_correct"] = float((minimal_difference * four_difference) > 0.0)
    else:
        row["minimal_ranking_correct"] = None

    # V11's report uses branch utility (negative mean SSE), whereas v12's
    # Available offline labels use gain minus a fixed request charge.  Keep
    # both explicit so no sign convention is silently changed.
    regrets: dict[str, Any] = {}
    for reward_name in ("offline_reward_current", "offline_reward_fourstate"):
        action_rewards = {action: branch_values[action][reward_name] for action in ("none", *PEERS)}
        usable = [float(value) for value in action_rewards.values() if value is not None]
        ceiling = max(usable) if len(usable) == 3 else None
        old_choice = row["old_choice"]
        minimal_choice = row["minimal_pred_choice"]
        regrets[reward_name] = {
            "label_ceiling": ceiling,
            "oldchoice": None if ceiling is None or old_choice not in action_rewards or action_rewards[old_choice] is None else ceiling - action_rewards[old_choice],
            "minimalpredchoice": None if ceiling is None or minimal_choice not in action_rewards or action_rewards[minimal_choice] is None else ceiling - action_rewards[minimal_choice],
        }
    row["regret_all3actions"] = regrets
    return row


def _candidate_context(decision: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    if not isinstance(decision, Mapping):
        return None
    context = decision.get("context", decision.get("pre_context"))
    return context if isinstance(context, Mapping) else None


def _decision_index(record: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    result: dict[int, Mapping[str, Any]] = {}
    for key in ("v11_decisions", "v12_decisions", "decisions"):
        values = record.get(key)
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            continue
        for item in values:
            if isinstance(item, Mapping) and _finite(item.get("step")) is not None:
                result[int(item["step"])] = item
    return result


def _record_filename(physical_base: Path, phase: str, scene: str, seed: int, policy: str = "random_peer") -> Path | None:
    roots = [physical_base / phase, physical_base, physical_base / "records" / phase]
    names = (
        f"{scene}_s{seed}_{policy}_record.json",
        f"{phase}_{scene}_s{seed}_{policy}_record.json",
        f"{scene}_s{seed}_{policy}.json",
    )
    for root in roots:
        for name in names:
            candidate = root / name
            if candidate.exists():
                return candidate
    return None


def _prior_queue_message(record: Mapping[str, Any], context: Mapping[str, Any], peer: str, target: str = TARGET) -> Mapping[str, Any] | None:
    """Read exactly the previous frame's arrived queue entry.

    A current or future frame is deliberately never consulted.  Matching the
    source, target, send timestamp and propagated state timestamp prevents a
    nearby packet from being mistaken for the selected pre-decision packet.
    """

    step = int(_finite(context.get("step")) or 0)
    frames = record.get("frames")
    if step <= 1 or not isinstance(frames, Sequence) or step - 2 >= len(frames):
        return None
    previous = frames[step - 2]
    if not isinstance(previous, Mapping):
        return None
    audit = previous.get("audit")
    live_queue = audit.get("live_queue") if isinstance(audit, Mapping) else None
    arrived = live_queue.get("arrived_messages_after") if isinstance(live_queue, Mapping) else None
    if not isinstance(arrived, Mapping):
        return None
    peer_payload = context.get("peers", {}).get(peer) if isinstance(context.get("peers"), Mapping) else None
    if not isinstance(peer_payload, Mapping):
        return None
    expected_send = peer_payload.get("send_time_s")
    expected_state = peer_payload.get("state_time_s", context.get("time_s"))
    for key, value in arrived.items():
        item = value if isinstance(value, Mapping) else None
        if item is None:
            continue
        pair: tuple[str, str] | None = None
        try:
            parsed = ast.literal_eval(str(key))
            if isinstance(parsed, tuple) and len(parsed) == 2:
                pair = (str(parsed[0]), str(parsed[1]))
        except (ValueError, SyntaxError):
            pair = None
        if pair is not None and pair != (peer, target):
            continue
        if pair is None and str(key).replace(" ", "") != f"('{peer}','{target}')":
            continue
        if str(item.get("source")) != peer or str(item.get("target")) != target:
            continue
        if not _same_time(item.get("send_time_s"), expected_send):
            continue
        propagated = item.get("last_propagated_time_s", item.get("state_time_s"))
        if not _same_time(propagated, expected_state):
            continue
        return item
    return None


def recover_prior_frame_delay(record: Mapping[str, Any], context: Mapping[str, Any], peer: str, target: str = TARGET) -> float | None:
    """Return a pre-decision packet delay, or ``None`` when it cannot match.

    ``prelink_delay_s`` is a separate pre-observable feature and is never
    substituted for an actual packet delay here.  A packet delay is accepted
    directly only from ``packet_delay_s`` or from an exact prior-frame queue
    match.
    """

    peer_payload = context.get("peers", {}).get(peer) if isinstance(context.get("peers"), Mapping) else None
    for key in ("packet_delay_s",):
        direct = _finite(context.get(key))
        if direct is not None and direct >= 0.0:
            return direct
        if isinstance(peer_payload, Mapping):
            direct = _finite(peer_payload.get(key))
            if direct is not None and direct >= 0.0:
                return direct
    message = _prior_queue_message(record, context, peer, target)
    delay = _finite(message.get("delay_s")) if message is not None else None
    return delay if delay is not None and delay >= 0.0 else None


def derive_directional_features(context: Mapping[str, Any] | None) -> dict[str, Any]:
    """Derive minimal directional observables from the pre-decision context."""

    if not isinstance(context, Mapping):
        return {}
    own = context.get("own_state", context.get("own_estimate"))
    own_position = _as_position(own)
    if own_position is None:
        own_position = _as_position(context.get("own_P3"))
    if own_position is None:
        return {}
    own_cov_raw = context.get("own_covariance", context.get("own_cov"))
    own_cov = _as_matrix(own_cov_raw, 3)
    if own_cov is None and isinstance(own_cov_raw, Sequence):
        try:
            full = [[float(v) for v in row] for row in own_cov_raw]
            if len(full) >= 3 and all(len(row) >= 3 for row in full[:3]):
                own_cov = [row[:3] for row in full[:3]]
        except (TypeError, ValueError):
            own_cov = None
    result: dict[str, Any] = {
        "self_position_trace_m2": _trace(own_cov),
        "shared_angle_rad": None,
        "world_unit_direction_diff": None,
        # These are descriptive pre-observables.  ``recover_prior_frame_delay``
        # deliberately does not use the prelink value as packet-arrival data.
        "packet_delay_s": _finite(context.get("packet_delay_s")),
        "prelink_delay_s": _finite(context.get("prelink_delay_s")),
    }
    directions: dict[str, list[float] | None] = {}
    self_variances: dict[str, float | None] = {}
    peer_variances: dict[str, float | None] = {}
    positions: dict[str, list[float] | None] = {}
    for peer in PEERS:
        payload = context.get("peers", {}).get(peer) if isinstance(context.get("peers"), Mapping) else None
        if not isinstance(payload, Mapping):
            payload = {}
        result[f"packet_delay_{peer}_s"] = _finite(payload.get("packet_delay_s"))
        result[f"prelink_delay_{peer}_s"] = _finite(payload.get("prelink_delay_s"))
        position = _as_position(
            payload.get(
                "state",
                payload.get(
                    "estimate",
                    payload.get("position", payload.get("arrived_peer_P3", payload.get("peer_P3"))),
                ),
            )
        )
        positions[peer] = position
        if position is None:
            directions[peer] = None
            self_variances[peer] = None
            peer_variances[peer] = None
            continue
        delta = [position[i] - own_position[i] for i in range(3)]
        norm = math.sqrt(sum(item * item for item in delta))
        if norm <= 1.0e-12:
            directions[peer] = None
            self_variances[peer] = None
            peer_variances[peer] = None
            continue
        direction = [item / norm for item in delta]
        directions[peer] = direction
        self_variances[peer] = None if own_cov is None else float(sum(direction[i] * own_cov[i][j] * direction[j] for i in range(3) for j in range(3)))
        peer_cov_raw = payload.get("covariance", payload.get("P"))
        peer_cov = _as_matrix(peer_cov_raw, 3)
        if peer_cov is None and isinstance(peer_cov_raw, Sequence):
            try:
                full = [[float(v) for v in row] for row in peer_cov_raw]
                if len(full) >= 3 and all(len(row) >= 3 for row in full[:3]):
                    peer_cov = [row[:3] for row in full[:3]]
            except (TypeError, ValueError):
                peer_cov = None
        peer_variances[peer] = None if peer_cov is None else float(sum(direction[i] * peer_cov[i][j] * direction[j] for i in range(3) for j in range(3)))
        result[f"{peer}_position_trace_m2"] = _position_trace(peer_cov_raw)
        result[f"{peer}_distance_m"] = norm
        result[f"{peer}_rssi_dbm"] = _finite(payload.get("rssi_dbm"))
        send_time = _finite(payload.get("send_time_s"))
        now = _finite(context.get("time_s"))
        result[f"{peer}_aoi_s"] = None if send_time is None or now is None else float(now - send_time)
    if all(directions[peer] is not None for peer in PEERS):
        d0, d1 = directions[PEERS[0]], directions[PEERS[1]]
        assert d0 is not None and d1 is not None
        dot = max(-1.0, min(1.0, sum(d0[i] * d1[i] for i in range(3))))
        result["shared_angle_rad"] = float(math.acos(dot))
        result["world_unit_direction_diff"] = [float(d0[i] - d1[i]) for i in range(3)]
        for index, value in enumerate(result["world_unit_direction_diff"]):
            result[f"world_unit_direction_diff_{index}"] = value
    result.update({f"{peer}_direction": directions[peer] for peer in PEERS})
    result.update({f"h_{peer}": directions[peer] for peer in PEERS})
    result.update({f"directional_self_variance_{peer}_m2": self_variances[peer] for peer in PEERS})
    result.update({f"directional_peer_variance_{peer}_m2": peer_variances[peer] for peer in PEERS})
    result["directional_self_variance_difference_m2"] = None if any(self_variances[p] is None for p in PEERS) else float(self_variances[PEERS[0]] - self_variances[PEERS[1]])
    result["directional_peer_variance_difference_m2"] = None if any(peer_variances[p] is None for p in PEERS) else float(peer_variances[PEERS[0]] - peer_variances[PEERS[1]])
    result["distance_difference_m"] = None if any(positions[p] is None for p in PEERS) else float(math.dist(own_position, positions[PEERS[0]]) - math.dist(own_position, positions[PEERS[1]]))  # type: ignore[arg-type]
    result["peer_trace_difference_m2"] = None if any(_finite(result.get(f"{p}_position_trace_m2")) is None for p in PEERS) else float(result[f"{PEERS[0]}_position_trace_m2"] - result[f"{PEERS[1]}_position_trace_m2"])
    result["rssi_difference_dbm"] = None if any(_finite(result.get(f"{p}_rssi_dbm")) is None for p in PEERS) else float(result[f"{PEERS[0]}_rssi_dbm"] - result[f"{PEERS[1]}_rssi_dbm"])
    result["aoi_difference_s"] = None if any(_finite(result.get(f"{p}_aoi_s")) is None for p in PEERS) else float(result[f"{PEERS[0]}_aoi_s"] - result[f"{PEERS[1]}_aoi_s"])
    return result


# Friendly aliases used by small pure tests and downstream notebooks.
directional_features = derive_directional_features
recover_packet_delay = recover_prior_frame_delay


def _old_feature_values(context: Mapping[str, Any] | None, scoring: Mapping[str, Any] | None) -> dict[str, Any]:
    values: dict[str, Any] = {}
    features = scoring.get("features") if isinstance(scoring, Mapping) else None
    if isinstance(features, Mapping):
        values.update(features)
    if not values and isinstance(context, Mapping):
        values.update(derive_directional_features(context))
    for name, keys in {
        "peer_trace_difference_m2": ("peer0_position_trace_m2", "peer1_position_trace_m2"),
        "distance_difference_m": ("distance20_m", "distance21_m"),
        "rssi_difference_dbm": ("rssi0_dbm", "rssi1_dbm"),
        "aoi_difference_s": ("aoi0_s", "aoi1_s"),
    }.items():
        if name not in values:
            left, right = (_finite(values.get(key)) for key in keys)
            values[name] = None if left is None or right is None else float(left - right)
    if "self_position_trace_m2" not in values and isinstance(context, Mapping):
        values.update({k: v for k, v in derive_directional_features(context).items() if k not in values})
    if "shared_angle_rad" not in values and "angle01_rad" in values:
        values["shared_angle_rad"] = values["angle01_rad"]
    return values


def _flatten_point(point: Mapping[str, Any], record: Mapping[str, Any] | None = None) -> dict[str, Any]:
    row = dict(point)
    phase, scene, seed, step = row.get("study_phase"), row.get("scene_id"), row.get("seed"), row.get("step")
    if record is not None:
        decisions = _decision_index(record)
        decision = decisions.get(int(step)) if _finite(step) is not None else None
        context = _candidate_context(decision)
        if context is not None:
            row["pre_context"] = context
            row["old_scoring"] = decision.get("scoring") if isinstance(decision, Mapping) else None
            row["v12_scoring"] = decision.get("v12_scoring") if isinstance(decision, Mapping) else None
            row["minimal_scoring"] = decision.get("minimal_scoring") if isinstance(decision, Mapping) else None
            row["old_selected"] = decision.get("selected") if isinstance(decision, Mapping) else None
            row["delay_uav_00_s"] = recover_prior_frame_delay(record, context, PEERS[0])
            row["delay_uav_01_s"] = recover_prior_frame_delay(record, context, PEERS[1])
            row["old_feature_values"] = _old_feature_values(context, row.get("old_scoring"))
            row["directional_features"] = derive_directional_features(context)
    if "v12_scoring" in point and isinstance(point["v12_scoring"], Mapping):
        row["minimal_scoring"] = point["v12_scoring"]
    if not isinstance(row.get("minimal_scoring"), Mapping) and isinstance(row.get("v12_scoring"), Mapping):
        row["minimal_scoring"] = row["v12_scoring"]
    labelled = label_point(row)
    labelled["study_phase"] = phase
    labelled["scene_id"] = scene
    labelled["seed"] = seed
    labelled["step"] = step
    labelled["geometry"] = scene
    labelled["score_difference(old)"] = _score_difference(labelled, labelled.get("old_scoring") if isinstance(labelled.get("old_scoring"), Mapping) else None)
    labelled["old_score_difference"] = labelled["score_difference(old)"]
    labelled["score_difference"] = labelled["score_difference(old)"]
    labelled["minimal_score_difference"] = _minimal_score_difference(labelled)
    old_features = labelled.get("old_feature_values")
    if isinstance(old_features, Mapping):
        for key, value in old_features.items():
            labelled[f"feature_{key}"] = value
    directional = labelled.get("directional_features")
    if isinstance(directional, Mapping):
        for key, value in directional.items():
            labelled[f"minimal_{key}"] = value
    _annotate_covariance_and_acceptance(labelled)
    # Preserve old scalar aliases used by the v11 report while exposing the
    # new names required by the v12 CSV contract.
    labelled["identity_matches"] = point.get("identity_matches")
    labelled["identity_error_max"] = point.get("identity_error_max")
    return labelled


def _annotate_covariance_and_acceptance(row: dict[str, Any]) -> None:
    """Attach post-CF covariance and prediction/acceptance audit scalars."""

    scoring = row.get("v12_scoring") if isinstance(row.get("v12_scoring"), Mapping) else row.get("old_scoring")
    candidates = scoring.get("candidates", {}) if isinstance(scoring, Mapping) else {}
    branches = _branches(row)
    zero_prediction_updates = 0
    actual_update_count = 0
    agreements = []
    for peer in PEERS:
        branch = branches.get(peer) if isinstance(branches, Mapping) else None
        branch = branch if isinstance(branch, Mapping) else {}
        event = branch.get("event") if isinstance(branch.get("event"), Mapping) else {}
        candidate = candidates.get(peer) if isinstance(candidates, Mapping) and isinstance(candidates.get(peer), Mapping) else {}
        before = _position_trace(event.get("P_initial", event.get("covariance_initial")))
        after = _position_trace(event.get("P_post", event.get("covariance_post")))
        actual_gain = None if before is None or after is None else float(before - after)
        predicted_gain = _finite(candidate.get("trace_gain_m2", candidate.get("predicted_trace_gain_m2")))
        actual_accepted = branch.get("accepted")
        predicted_accepted = candidate.get("predicted_accepted", candidate.get("accepted"))
        if predicted_gain is not None and abs(predicted_gain) <= SENSITIVITY_M2 and bool(actual_accepted):
            zero_prediction_updates += 1
        if bool(actual_accepted):
            actual_update_count += 1
        if predicted_accepted is not None and actual_accepted is not None:
            agreements.append(bool(predicted_accepted) == bool(actual_accepted))
        suffix = str(peer)
        row[f"actual_post_cf_cov_before_trace_m2_{suffix}"] = before
        row[f"actual_post_cf_cov_after_trace_m2_{suffix}"] = after
        row[f"actual_post_cf_cov_trace_gain_m2_{suffix}"] = actual_gain
        row[f"predecision_clone_pred_trace_gain_m2_{suffix}"] = predicted_gain
        row[f"pre_post_gain_acceptance_agreement_{suffix}"] = (bool(predicted_accepted) == bool(actual_accepted)) if predicted_accepted is not None and actual_accepted is not None else None
        row[f"actual_weight_{suffix}"] = _finite(branch.get("ci_weight", event.get("ci_weight")))
        row[f"actual_accepted_{suffix}"] = actual_accepted
        row[f"actual_reason_{suffix}"] = branch.get("reason", event.get("reason"))
        true_gain = row.get(f"current_sse_gain_{suffix}")
        row[f"true_help_{suffix}"] = bool(_finite(true_gain) is not None and float(true_gain) > SENSITIVITY_M2)
        row[f"true_harm_{suffix}"] = bool(_finite(true_gain) is not None and float(true_gain) < -SENSITIVITY_M2)
        row[f"true_effect_zero_{suffix}"] = bool(_finite(true_gain) is not None and abs(float(true_gain)) <= SENSITIVITY_M2)
    row["pre_post_gain_acceptance_agreement"] = bool(all(agreements)) if agreements else None
    row["zero_prediction_actual_update_count"] = zero_prediction_updates
    row["actual_update_count"] = actual_update_count
    row["actual_gain_true_available_for_evaluation"] = bool(row.get("label_complete"))


def collect_points(cf_dir: str | Path, physical_base: str | Path) -> list[dict[str, Any]]:
    """Collect and label one flat row per counterfactual checkpoint.

    Record JSON is cached by case key, so the six checkpoints of a case never
    cause six repeated reads of the same 17 MB physical record.
    """

    cf_root, physical_root = Path(cf_dir), Path(physical_base)
    if not (physical_root / "develop").exists() and (physical_root / "batches" / "research_001").exists():
        physical_root = physical_root / "batches" / "research_001"
    files = sorted(path for path in cf_root.glob("*.json") if path.name != "manifest.json")
    record_cache: dict[tuple[str, str, int], Mapping[str, Any] | None] = {}
    rows: list[dict[str, Any]] = []
    for path in files:
        point = _json_load(path)
        if not isinstance(point, Mapping) or "step" not in point:
            continue
        phase = str(point.get("study_phase", point.get("phase", "")))
        scene = str(point.get("scene_id", point.get("geometry", "")))
        seed = int(_finite(point.get("seed")) or 0)
        key = (phase, scene, seed)
        if key not in record_cache:
            record_path = _record_filename(physical_root, phase, scene, seed)
            record_cache[key] = _json_load(record_path) if record_path is not None else None
        rows.append(_flatten_point(point, record_cache[key]))
    rows.sort(key=lambda row: (str(row.get("study_phase")), str(row.get("scene_id")), int(_finite(row.get("seed")) or 0), int(_finite(row.get("step")) or 0)))
    return rows


def _rows_group(rows: Iterable[Mapping[str, Any]], *, phase: Any = None, scene: Any = None, seed: Any = None) -> list[Mapping[str, Any]]:
    return [row for row in rows if (phase is None or row.get("study_phase") == phase) and (scene is None or row.get("scene_id", row.get("geometry")) == scene) and (seed is None or row.get("seed") == seed)]


def _ranking_stats(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    meaningful = [row for row in rows if row.get("meaningful")]
    old_values = [row.get("ranking_correct") for row in meaningful if row.get("ranking_correct") is not None]
    minimal_values = [row.get("minimal_ranking_correct") for row in meaningful if row.get("minimal_ranking_correct") is not None]
    return {
        "meaningful_n": len(meaningful),
        "old_n": len(old_values),
        "minimal_n": len(minimal_values),
        "rankmeaningful_old": float(sum(float(v) for v in old_values) / len(old_values)) if old_values else None,
        "rankmeaningful_minimal": float(sum(float(v) for v in minimal_values) / len(minimal_values)) if minimal_values else None,
        "ranking_accuracy_old": float(sum(float(v) for v in old_values) / len(old_values)) if old_values else None,
        "minimal_ranking_accuracy": float(sum(float(v) for v in minimal_values) / len(minimal_values)) if minimal_values else None,
    }


def _group_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    score_rows = [row for row in rows if row.get("dual_eligible") and row.get("score_difference(old)") is not None and row.get("utility_difference(fourstate)") is not None]
    meaningful = [row for row in score_rows if row.get("meaningful")]
    minimal_rows = [row for row in meaningful if row.get("minimal_score_difference") is not None]
    old_pairs = [(float(row["score_difference(old)"]), float(row["true_peer_difference_fourstate"])) for row in meaningful if row.get("true_peer_difference_fourstate") is not None]
    minimal_pairs = [(float(row["minimal_score_difference"]), float(row["true_peer_difference_fourstate"])) for row in minimal_rows if row.get("true_peer_difference_fourstate") is not None]
    def pairs_for(score_key: str, target_key: str) -> list[tuple[float, float]]:
        return [(float(row[score_key]), float(row[target_key])) for row in meaningful if _finite(row.get(score_key)) is not None and _finite(row.get(target_key)) is not None]

    def pairs_for_all(score_key: str, target_key: str) -> list[tuple[float, float]]:
        return [(float(row[score_key]), float(row[target_key])) for row in score_rows if _finite(row.get(score_key)) is not None and _finite(row.get(target_key)) is not None]

    old_norm = pairs_for("score_difference(old)", "true_peer_difference_current_norm")
    old_sse = pairs_for("score_difference(old)", "true_peer_difference_current_sse")
    old_four = pairs_for("score_difference(old)", "true_peer_difference_fourstate")
    min_norm = pairs_for("minimal_score_difference", "true_peer_difference_current_norm")
    min_sse = pairs_for("minimal_score_difference", "true_peer_difference_current_sse")
    min_four = pairs_for("minimal_score_difference", "true_peer_difference_fourstate")
    old_norm_all = pairs_for_all("score_difference(old)", "true_peer_difference_current_norm")
    old_sse_all = pairs_for_all("score_difference(old)", "true_peer_difference_current_sse")
    old_four_all = pairs_for_all("score_difference(old)", "true_peer_difference_fourstate")
    min_norm_all = pairs_for_all("minimal_score_difference", "true_peer_difference_current_norm")
    min_sse_all = pairs_for_all("minimal_score_difference", "true_peer_difference_current_sse")
    min_four_all = pairs_for_all("minimal_score_difference", "true_peer_difference_fourstate")
    instant_four_norm = [(float(row["true_peer_difference_current_norm"]), float(row["true_peer_difference_fourstate"])) for row in meaningful if _finite(row.get("true_peer_difference_current_norm")) is not None and _finite(row.get("true_peer_difference_fourstate")) is not None]
    instant_four_sse = [(float(row["true_peer_difference_current_sse"]), float(row["true_peer_difference_fourstate"])) for row in meaningful if _finite(row.get("true_peer_difference_current_sse")) is not None and _finite(row.get("true_peer_difference_fourstate")) is not None]
    instant_four_norm_all = [(float(row["true_peer_difference_current_norm"]), float(row["true_peer_difference_fourstate"])) for row in score_rows if _finite(row.get("true_peer_difference_current_norm")) is not None and _finite(row.get("true_peer_difference_fourstate")) is not None]
    instant_four_sse_all = [(float(row["true_peer_difference_current_sse"]), float(row["true_peer_difference_fourstate"])) for row in score_rows if _finite(row.get("true_peer_difference_current_sse")) is not None and _finite(row.get("true_peer_difference_fourstate")) is not None]

    def corr(pair_values: Sequence[tuple[float, float]], rank: bool = False) -> float | None:
        return correlation([item[0] for item in pair_values], [item[1] for item in pair_values], rank)

    result: dict[str, Any] = {
        "n": len(rows),
        "dual_points": len(score_rows),
        "meaningful_points": len(meaningful),
        "label_complete_n": sum(bool(row.get("label_complete")) for row in rows),
        "label_deployable_n": 0,
        "label_nondeployable_n": len(rows),
        "old_pearson": correlation([pair[0] for pair in old_pairs], [pair[1] for pair in old_pairs]),
        "old_spearman": correlation([pair[0] for pair in old_pairs], [pair[1] for pair in old_pairs], True),
        "minimal_pearson": correlation([pair[0] for pair in minimal_pairs], [pair[1] for pair in minimal_pairs]),
        "minimal_spearman": correlation([pair[0] for pair in minimal_pairs], [pair[1] for pair in minimal_pairs], True),
        "old_score_vs_current_norm_pearson": corr(old_norm),
        "old_score_vs_current_norm_spearman": corr(old_norm, True),
        "old_score_vs_current_sse_pearson": corr(old_sse),
        "old_score_vs_current_sse_spearman": corr(old_sse, True),
        "old_score_vs_fourstate_pearson": corr(old_four),
        "old_score_vs_fourstate_spearman": corr(old_four, True),
        "minimal_score_vs_current_norm_pearson": corr(min_norm),
        "minimal_score_vs_current_norm_spearman": corr(min_norm, True),
        "minimal_score_vs_current_sse_pearson": corr(min_sse),
        "minimal_score_vs_current_sse_spearman": corr(min_sse, True),
        "minimal_score_vs_fourstate_pearson": corr(min_four),
        "minimal_score_vs_fourstate_spearman": corr(min_four, True),
        "instant_norm_vs_fourstate_pearson": corr(instant_four_norm),
        "instant_norm_vs_fourstate_spearman": corr(instant_four_norm, True),
        "instant_sse_vs_fourstate_pearson": corr(instant_four_sse),
        "instant_sse_vs_fourstate_spearman": corr(instant_four_sse, True),
        "old_score_vs_current_norm_pearson_all": corr(old_norm_all),
        "old_score_vs_current_norm_spearman_all": corr(old_norm_all, True),
        "old_score_vs_current_sse_pearson_all": corr(old_sse_all),
        "old_score_vs_current_sse_spearman_all": corr(old_sse_all, True),
        "old_score_vs_fourstate_pearson_all": corr(old_four_all),
        "old_score_vs_fourstate_spearman_all": corr(old_four_all, True),
        "minimal_score_vs_current_norm_pearson_all": corr(min_norm_all),
        "minimal_score_vs_current_norm_spearman_all": corr(min_norm_all, True),
        "minimal_score_vs_current_sse_pearson_all": corr(min_sse_all),
        "minimal_score_vs_current_sse_spearman_all": corr(min_sse_all, True),
        "minimal_score_vs_fourstate_pearson_all": corr(min_four_all),
        "minimal_score_vs_fourstate_spearman_all": corr(min_four_all, True),
        "instant_norm_vs_fourstate_pearson_all": corr(instant_four_norm_all),
        "instant_norm_vs_fourstate_spearman_all": corr(instant_four_norm_all, True),
        "instant_sse_vs_fourstate_pearson_all": corr(instant_four_sse_all),
        "instant_sse_vs_fourstate_spearman_all": corr(instant_four_sse_all, True),
    }
    for prefix in (
        "old_score_vs_current_norm", "old_score_vs_current_sse", "old_score_vs_fourstate",
        "minimal_score_vs_current_norm", "minimal_score_vs_current_sse", "minimal_score_vs_fourstate",
        "instant_norm_vs_fourstate", "instant_sse_vs_fourstate",
    ):
        result[f"{prefix}_pearson_meaningful"] = result[f"{prefix}_pearson"]
        result[f"{prefix}_spearman_meaningful"] = result[f"{prefix}_spearman"]
    result.update(_ranking_stats(rows))
    result["pearson"] = result["old_pearson"]
    result["spearman"] = result["old_spearman"]
    return result


FEATURE_SPECS = (
    ("peer_trace_difference", ("feature_peer_trace_difference_m2", "feature_peer_trace_difference"), False),
    ("distance_difference", ("feature_distance_difference_m", "feature_distance_difference"), False),
    ("rssi_difference", ("feature_rssi_difference_dbm", "feature_rssi_difference"), False),
    ("aoi_difference", ("feature_aoi_difference_s", "feature_aoi_difference"), False),
    ("self_trace", ("feature_self_position_trace_m2", "feature_self_trace"), True),
    ("shared_angle", ("feature_shared_angle_rad", "feature_shared_angle", "feature_angle01_rad"), True),
    ("directional_self_variance_difference", ("minimal_directional_self_variance_difference_m2",), False),
    ("directional_peer_variance_difference", ("minimal_directional_peer_variance_difference_m2",), False),
    ("world_unit_direction_diff_0", ("minimal_world_unit_direction_diff_0", "minimal_world_unit_direction_diff"), False),
)


def _feature_value(row: Mapping[str, Any], keys: Sequence[str], component: int | None = None) -> float | None:
    for key in keys:
        value = row.get(key)
        if component is not None and isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) > component:
            value = value[component]
        number = _finite(value)
        if number is not None:
            return number
    return None


def _feature_correlations(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    contexts = [(None, None, None, "pooled")]
    phases = sorted({row.get("study_phase") for row in rows if row.get("study_phase") is not None})
    for phase in phases:
        contexts.append((phase, None, None, "study_phase"))
        scenes = sorted({row.get("scene_id", row.get("geometry")) for row in rows if row.get("study_phase") == phase})
        for scene in scenes:
            contexts.append((phase, scene, None, "geometry"))
            seeds = sorted({row.get("seed") for row in rows if row.get("study_phase") == phase and row.get("scene_id", row.get("geometry")) == scene})
            for seed in seeds:
                contexts.append((phase, scene, seed, "seed"))
    for phase, scene, seed, scope in contexts:
        group = _rows_group(rows, phase=phase, scene=scene, seed=seed)
        for feature, keys, absolute in FEATURE_SPECS:
            # Keep the signed current-SSE relationship used by the original
            # report, while also exposing norm and four-state outcomes so a
            # feature is never presented as explaining only one label.
            for target_metric, target_key in (
                ("current_norm", "true_peer_difference_current_norm"),
                ("current_sse", "true_peer_difference_current_sse"),
                ("fourstate", "true_peer_difference_fourstate"),
            ):
                pairs: list[tuple[float, float]] = []
                for row in group:
                    target = row.get(target_key)
                    if absolute:
                        target = None if target is None else abs(float(target))
                    value = _feature_value(row, keys, 0 if feature == "world_unit_direction_diff_0" else None)
                    target_number = _finite(target)
                    if value is not None and target_number is not None:
                        pairs.append((value, target_number))
                output.append({
                    "scope": scope,
                    "study_phase": phase,
                    "scene_id": scene,
                    "geometry": scene,
                    "seed": seed,
                    "feature": feature,
                    "target_metric": target_metric,
                    "outcome": f"{'absolute' if absolute else 'signed'}_peer_difference_{target_metric}",
                    "predicts_peer_identity": not absolute,
                    "n": len(pairs),
                    "pearson": correlation([pair[0] for pair in pairs], [pair[1] for pair in pairs]),
                    "spearman": correlation([pair[0] for pair in pairs], [pair[1] for pair in pairs], True),
                })
    return output


def _regret_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for metric in ("offline_reward_current", "offline_reward_fourstate"):
        values: dict[str, list[float]] = defaultdict(list)
        for row in rows:
            regrets = row.get("regret_all3actions")
            item = regrets.get(metric) if isinstance(regrets, Mapping) else None
            if not isinstance(item, Mapping):
                continue
            for key in ("label_ceiling", "oldchoice", "minimalpredchoice"):
                number = _finite(item.get(key))
                if number is not None:
                    values[key].append(number)
        result[metric] = {key: {"n": len(value), "mean": float(sum(value) / len(value)) if value else None} for key, value in values.items()}
    return result


def summarize_points(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize rows by phase, geometry, seed, pooled, and feature scope."""

    labelled = [label_point(row) if "label_deployable" not in row else dict(row) for row in rows]
    phases = sorted({row.get("study_phase") for row in labelled if row.get("study_phase") is not None})
    scenes = sorted({row.get("scene_id", row.get("geometry")) for row in labelled if row.get("scene_id", row.get("geometry")) is not None})
    groups: list[dict[str, Any]] = []
    hierarchy: dict[str, Any] = {}
    for phase in phases:
        phase_rows = _rows_group(labelled, phase=phase)
        phase_entry = {"summary": _group_summary(phase_rows), "by_geometry": {}}
        groups.append({"study_phase": phase, "scene_id": "all", "seed": "all", **phase_entry["summary"]})
        for scene in [item for item in scenes if item in {row.get("scene_id", row.get("geometry")) for row in phase_rows}]:
            scene_rows = _rows_group(phase_rows, scene=scene)
            scene_entry = {"summary": _group_summary(scene_rows), "by_seed": {}}
            groups.append({"study_phase": phase, "scene_id": scene, "seed": "all", **scene_entry["summary"]})
            for seed in sorted({row.get("seed") for row in scene_rows}):
                seed_rows = _rows_group(scene_rows, seed=seed)
                seed_entry = _group_summary(seed_rows)
                scene_entry["by_seed"][str(seed)] = seed_entry
                groups.append({"study_phase": phase, "scene_id": scene, "seed": seed, **seed_entry})
            phase_entry["by_geometry"][str(scene)] = scene_entry
        hierarchy[str(phase)] = phase_entry
    pooled = _group_summary(labelled)
    rankings = [{"study_phase": row["study_phase"], "scene_id": row["scene_id"], "seed": row["seed"], **_ranking_stats(_rows_group(labelled, phase=row["study_phase"], scene=row["scene_id"], seed=row["seed"]))} for row in groups if row.get("seed") != "all"]
    return {
        "groups": groups,
        "by_study_phase": hierarchy,
        "pooled": pooled,
        "correlations": _feature_correlations(labelled),
        "rankings": rankings,
        "regret_all3actions": _regret_summary(labelled),
        "n": len(labelled),
        "meaningful_n": sum(bool(row.get("meaningful")) for row in labelled),
        "label_deployable": all(bool(row.get("label_deployable")) for row in labelled) if labelled else False,
        "label_complete": all(bool(row.get("label_complete")) for row in labelled) if labelled else False,
        "label_nondeployable_n": sum(not bool(row.get("label_deployable")) for row in labelled),
    }


def _flatten_csv_value(value: Any) -> Any:
    if isinstance(value, (Mapping, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return value


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key.startswith("_"):
                continue
            if key not in seen:
                seen.add(key)
                keys.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _flatten_csv_value(row.get(key)) for key in keys})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_recorded_path(name: str) -> Path:
    candidate = Path(name)
    if candidate.exists():
        return candidate
    # Stored manifests use Windows absolute paths; retaining this fallback
    # makes the diagnostic portable when a checkout root is relocated.
    text = name.replace("\\", "/")
    marker = "/research project/"
    if marker in text:
        return ROOT / text.split(marker, 1)[1]
    return ROOT / text.lstrip("/")


def _canonical_path(name: str) -> str:
    return str(_resolve_recorded_path(name).resolve())


def _protected_hashes() -> dict[str, str]:
    hashes: dict[str, str] = {}
    current_readme = str((ROOT / "README.md").resolve())
    for name in ("protected_inputs_before.json", "final_artifact_manifest.json"):
        path = V11_ROOT / name
        payload = _json_load(path)
        if isinstance(payload, Mapping):
            source = payload.get("artifact_sha256") if isinstance(payload.get("artifact_sha256"), Mapping) else payload
            if isinstance(source, Mapping):
                for key, value in source.items():
                    if not isinstance(value, str):
                        continue
                    canonical = _canonical_path(str(key))
                    # Only the live project README is deliberately excluded;
                    # historical README-named archives remain immutable inputs.
                    if canonical == current_readme:
                        continue
                    hashes[canonical] = value
    explicit = [
        V11_ROOT / "final_artifact_manifest.json",
        V11_ROOT / "final_verification.lock.json",
        Path(__file__).resolve(),
    ]
    explicit.extend(sorted(ROOT.glob("tests/test_neighbor_selection_*_v11.py")))
    v12_test = ROOT / "tests" / "test_neighbor_selection_diagnostic_v12.py"
    if v12_test.exists():
        explicit.append(v12_test)
    explicit.extend(sorted((ROOT / "data" / "configs").glob("*neighbor*value*v11*.json")))
    for path in explicit:
        if path.exists() and str(path.resolve()) != current_readme:
            hashes[str(path.resolve())] = _sha256(path)
    return hashes


def _validate_hashes(hashes: Mapping[str, str]) -> list[str]:
    failures: list[str] = []
    for name, expected in hashes.items():
        path = _resolve_recorded_path(str(name))
        if not path.exists():
            failures.append(f"missing:{name}")
            continue
        actual = _sha256(path)
        if actual != str(expected):
            failures.append(f"changed:{name}")
    return failures


def _archive_readme_prefix(target: Path, size: int = 4096) -> dict[str, Any]:
    readme = ROOT / "README.md"
    data = readme.read_bytes() if readme.exists() else b""
    prefix = data[:size]
    path = target / "README_current_prefix_before.bin"
    path.write_bytes(prefix)
    (target / "README_current_before.bin").write_bytes(data)
    metadata = {"path": str(readme), "prefix_bytes": len(prefix), "full_bytes": len(data), "prefix_sha256": hashlib.sha256(prefix).hexdigest(), "full_sha256": hashlib.sha256(data).hexdigest()}
    (target / "README_current_prefix_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return metadata


def _legacy_records_count(physical_base: Path) -> dict[str, int]:
    total_requests = 0
    records = 0
    total_windows = 0
    total_pred_no_update = 0
    by_phase: dict[str, dict[str, int]] = {}
    for phase in ("develop", "confirm"):
        phase_requests = 0
        phase_records = 0
        phase_windows = 0
        phase_pred_no_update = 0
        for path in sorted((physical_base / phase).glob("*_greedy_record.json")):
            records += 1
            phase_records += 1
            payload = _json_load(path)
            decisions = payload.get("v11_decisions", []) if isinstance(payload, Mapping) else []
            items = [item for item in decisions if isinstance(item, Mapping)]
            chosen = sum(1 for item in items if str(item.get("selected", "none")) in PEERS)
            both_no_update = 0
            for item in items:
                scoring = item.get("scoring")
                candidates = scoring.get("candidates", {}) if isinstance(scoring, Mapping) else {}
                values = [candidates.get(peer) for peer in PEERS] if isinstance(candidates, Mapping) else []
                if len(values) == 2 and all(isinstance(value, Mapping) for value in values) and all(_finite(value.get("score")) is not None for value in values) and all(value.get("predicted_accepted") is False for value in values):
                    both_no_update += 1
            total_requests += chosen
            total_windows += len(items)
            total_pred_no_update += both_no_update
            phase_requests += chosen
            phase_windows += len(items)
            phase_pred_no_update += both_no_update
        by_phase[phase] = {"greedy_records": phase_records, "windows": phase_windows, "pred_no_update": phase_pred_no_update, "chosen_requests": phase_requests}
    return {"greedy_records": records, "windows": total_windows, "pred_no_update": total_pred_no_update, "greedy_requests": total_requests, "by_phase": by_phase}


def _old_score_no_update_count(rows: Sequence[Mapping[str, Any]]) -> int:
    count = 0
    for row in rows:
        scoring = row.get("old_scoring")
        candidates = scoring.get("candidates", {}) if isinstance(scoring, Mapping) else {}
        values = [candidates.get(peer) for peer in PEERS] if isinstance(candidates, Mapping) else []
        if len(values) != 2 or any(not isinstance(item, Mapping) for item in values):
            continue
        if all(_finite(item.get("score")) is not None for item in values) and all(item.get("predicted_accepted") is False for item in values):
            count += 1
    return count


def stage1(*, cf_dir: str | Path = DEFAULT_CF_DIR, physical_base: str | Path = DEFAULT_PHYSICAL_BASE, output_root: str | Path = V12_ROOT) -> dict[str, Any]:
    """Run the read-only Stage 1 aggregation and create new v12 artifacts."""

    output = Path(output_root)
    physical_path = Path(physical_base)
    if not (physical_path / "develop").exists() and (physical_path / "batches" / "research_001").exists():
        physical_path = physical_path / "batches" / "research_001"
    stage_dir, log_dir = output / "stage1", (ROOT / "logs" / "preliminary" / "three_uav_neighbor_learnability_v12_20261013" / "stage1") if output == V12_ROOT else output / "logs" / "stage1"
    if stage_dir.exists() or log_dir.exists():
        raise FileExistsError("v12 Stage1 output already exists; refusing overwrite")
    protected = _protected_hashes()
    before_failures = _validate_hashes(protected)
    if before_failures:
        raise RuntimeError("protected v11 inputs changed before collection: " + ";".join(before_failures[:5]))
    stage_dir.mkdir(parents=True)
    log_dir.mkdir(parents=True)
    readme_info = _archive_readme_prefix(stage_dir)
    protected_path = stage_dir / "protected_input_sha256.json"
    # The lock is intentionally a flat absolute-path -> digest mapping.  The
    # README prefix is archived in its own files because README is excluded
    # from the immutable input set.
    protected_path.write_text(json.dumps(protected, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    rows = collect_points(cf_dir, physical_path)
    summary = summarize_points(rows)
    # Keep giant state/covariance arrays out of CSV while retaining all labels
    # and the branch-level JSON record for later audit.
    full_rows = []
    for row in rows:
        full = {key: value for key, value in row.items() if key not in {"pre_context", "old_scoring", "directional_features", "old_feature_values"}}
        full_rows.append(full)
    _write_csv(log_dir / "stage1_points.csv", full_rows)
    _write_csv(log_dir / "correlations.csv", summary["correlations"])
    _write_csv(log_dir / "rankings.csv", summary["groups"])
    _write_csv(stage_dir / "stage1_points.csv", full_rows)
    _write_csv(stage_dir / "correlations.csv", summary["correlations"])
    _write_csv(stage_dir / "rankings.csv", summary["groups"])
    (stage_dir / "stage1_points_full.json").write_text(json.dumps(full_rows, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    greedy_counts = _legacy_records_count(physical_path)
    diagnosis = {
        "summary": summary,
        "old96_points": len(rows),
        "old_steps": list(CF_STEPS),
        "greedy_counts": greedy_counts,
        "counts": {"old_score_both_pred_no_update": greedy_counts["pred_no_update"], "all_greedy_requests": greedy_counts["greedy_requests"], "all_greedy_windows": greedy_counts["windows"]},
        "readme_archive": readme_info,
        "structural_encoding": {
            "minimal_state_features": [
                "minimal_directional_self_variance_uav_00_m2",
                "minimal_directional_self_variance_uav_01_m2",
                "minimal_directional_peer_variance_uav_00_m2",
                "minimal_directional_peer_variance_uav_01_m2",
            ],
            "geometry_audit_features": ["minimal_h_uav_00", "minimal_h_uav_01", "minimal_world_unit_direction_diff"],
            "no_deployable_mapping": {"none": "baseline_action", "uav_00": "peer_request_uav_00", "uav_01": "peer_request_uav_01"},
            "fitted_model": False,
        },
        "constraints": {"read_only": True, "ppo_calls": 0, "filter_calls": 0, "environment_calls": 0, "new_physical_data": False, "filter_replay": False},
    }
    (stage_dir / "diagnosis.json").write_text(json.dumps(diagnosis, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (log_dir / "diagnosis.json").write_text(json.dumps(diagnosis, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    legacy = physical_path.parents[2] / "logs" / "preliminary" / "three_uav_neighbor_value_v11_20261013" / "research_001" / "report" / "legacy_proxy_counts.csv"
    if not legacy.exists():
        legacy = ROOT / "logs" / "preliminary" / "three_uav_neighbor_value_v11_20261013" / "research_001" / "report" / "legacy_proxy_counts.csv"
    if legacy.exists():
        shutil.copyfile(legacy, log_dir / "baseline_legacy_proxy_counts.csv")
        shutil.copyfile(legacy, stage_dir / "baseline_legacy_proxy_counts.csv")
    after_failures = _validate_hashes(protected)
    if after_failures:
        raise RuntimeError("protected v11 inputs changed after collection: " + ";".join(after_failures[:5]))
    return {"stage1_dir": str(stage_dir), "log_dir": str(log_dir), "points": len(rows), "summary": summary, "protected_failures_before": before_failures, "protected_failures_after": after_failures}


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1", action="store_true", help="write the new v12 Stage 1 read-only diagnostic")
    args = parser.parse_args(argv)
    if not args.stage1:
        parser.error("pass --stage1")
    print(json.dumps(stage1(), ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()


__all__ = [
    "CF_STEPS",
    "COST_M2",
    "correlation",
    "collect_points",
    "derive_directional_features",
    "directional_features",
    "label_point",
    "recover_packet_delay",
    "recover_prior_frame_delay",
    "stage1",
    "summarize_points",
]
