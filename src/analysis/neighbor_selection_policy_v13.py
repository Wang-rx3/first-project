"""Frozen, non-learning root-trace proxy; truth is never a selector input."""
from __future__ import annotations
import math
from time import perf_counter
import numpy as np
from . import neighbor_selection_policy_v11 as legacy

PEERS = legacy.PEERS
PENALTY_M = .01
DT_S = .2
_CONTEXT = {"step", "seed", "time_s", "own_state", "own_covariance", "legal_sources", "peers", "measurement_budget_remaining"}
_PEER = {"state", "covariance", "send_time_s", "state_time_s", "rssi_dbm", "link_aoi_s", "cache_mean", "cache_covariance", "cache_time_s", "cache_message_send_s", "last_update_time_s", "packet_delay_s", "prelink_delay_s", "gnss_std_m", "gnss_report_time_s", "quality_packet_id"}


def request_cost(delay_s):
    if delay_s is None or not math.isfinite(float(delay_s)) or float(delay_s) < 0:
        return None
    return 1. + float(delay_s) / DT_S


def extend(context, baseline):
    """Reuse the original CI scoring probe, without an additional filter call."""
    if set(context) - _CONTEXT or any(set(p) - _PEER for p in context.get("peers", {}).values()):
        raise ValueError("unregistered/non-observable selector input")
    own_p = np.asarray(context["own_covariance"], dtype=float)[:3, :3]
    before = float(np.trace(own_p))
    if not math.isfinite(before) or before <= 0:
        raise ValueError("invalid receiver covariance")
    candidates, features = {}, dict(baseline["features"])
    eig = np.linalg.eigvalsh(own_p)
    features.update({f"self_eigen{i}_m2": float(v) for i, v in enumerate(eig)})
    for i, uid in enumerate(PEERS):
        c, p = baseline["candidates"][uid], context.get("peers", {}).get(uid, {})
        after, cost = c.get("predicted_position_trace_m2"), request_cost(p.get("prelink_delay_s"))
        valid = bool(c.get("valid") and cost is not None and after is not None and math.isfinite(float(after)) and after >= 0)
        gain = math.sqrt(before) - math.sqrt(after) if valid else None
        candidates[uid] = {"valid": valid, "score": gain - PENALTY_M * cost if valid else None,
                           "root_trace_gain_m": gain, "cost_unit": cost, "predicted_accepted": c.get("predicted_accepted"),
                           "predicted_ci_weight": c.get("predicted_ci_weight"), "reason": c.get("reason")}
        for key, val in (("gnss_std_m", p.get("gnss_std_m")), ("prelink_delay_s", p.get("prelink_delay_s")),
                         ("packet_delay_s", p.get("packet_delay_s"))):
            features[f"peer{i}_{key}"] = val
        report_time = p.get("gnss_report_time_s")
        features[f"peer{i}_quality_age_s"] = None if report_time is None else context["time_s"] - report_time
        direction = c.get("direction")
        if direction is not None and p.get("covariance") is not None:
            h = np.asarray(direction)
            pp = np.asarray(p["covariance"])[:3, :3]
            features[f"peer{i}_directional_variance_m2"] = float(h @ pp @ h)
            features[f"self_towards_peer{i}_variance_m2"] = float(h @ own_p @ h)
            for k, v in enumerate(h):
                features[f"peer{i}_direction{k}"] = float(v)
    # Signed differences preserve the ordered UAV0/UAV1 comparison.
    for a, b in (("peer0_position_trace_m2", "peer1_position_trace_m2"), ("distance20_m", "distance21_m"),
                 ("rssi0_dbm", "rssi1_dbm"), ("aoi0_s", "aoi1_s"),
                 ("peer0_gnss_std_m", "peer1_gnss_std_m"), ("peer0_prelink_delay_s", "peer1_prelink_delay_s"),
                 ("peer0_directional_variance_m2", "peer1_directional_variance_m2"),
                 ("self_towards_peer0_variance_m2", "self_towards_peer1_variance_m2")):
        features[a + "_minus_" + b] = None if features.get(a) is None or features.get(b) is None else features[a] - features[b]
    a, b = (candidates[u]["score"] for u in PEERS)
    return {"step": context["step"], "seed": context["seed"], "none_score": 0., "candidates": candidates,
            "score_difference": None if a is None or b is None else a - b, "features": features,
            "penalty_m": PENALTY_M, "semantics": "root-trace uncertainty proxy in meters, not expected true MSE"}


def score_context(context, sim):
    started = perf_counter()
    baseline = legacy.score_context(context, sim)
    result = extend(context, baseline)
    result["compute_ms"] = (perf_counter() - started) * 1000
    return baseline, result


def choose_greedy(scoring):
    valid = [(u, scoring["candidates"][u]["score"]) for u in PEERS
             if scoring["candidates"][u]["valid"] and scoring["candidates"][u]["score"] > 0]
    return max(valid, key=lambda x: x[1])[0] if valid else "none"
