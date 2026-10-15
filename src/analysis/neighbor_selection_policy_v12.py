"""One execution-only proxy hypothesis; no modification to online EKF/reward."""
from __future__ import annotations

from time import perf_counter
import math

from . import neighbor_selection_policy_v11 as legacy
from .neighbor_selection_diagnostic_v12 import derive_directional_features
from ..simulation.random_streams import indexed_rng
from ..experiments.partial_gnss_baseline_v7 import old

PEERS = legacy.PEERS
TARGET = legacy.TARGET
REQUEST_PENALTY_M2 = .01
_CONTEXT_KEYS = {"step", "seed", "time_s", "own_state", "own_covariance", "legal_sources", "peers", "measurement_budget_remaining"}
_PEER_KEYS = {"state", "covariance", "send_time_s", "state_time_s", "rssi_dbm", "link_aoi_s",
              "cache_mean", "cache_covariance", "cache_time_s", "cache_message_send_s", "last_update_time_s",
              "packet_delay_s", "prelink_delay_s", "aoi_s"}


def score_context(context, sim):
    started = perf_counter()
    if set(context) - _CONTEXT_KEYS:
        raise ValueError("non-observable/unregistered selector input")
    if any(set(peer) - _PEER_KEYS for peer in context.get("peers", {}).values()):
        raise ValueError("non-observable/unregistered peer input")
    baseline = legacy.score_context(context, sim)
    candidates = {}
    for uid in PEERS:
        c = baseline["candidates"][uid]
        gain = c.get("trace_gain_m2")
        valid = bool(c["valid"] and gain is not None and math.isfinite(float(gain)))
        candidates[uid] = {"valid": valid, "score": float(gain)-REQUEST_PENALTY_M2 if valid else None,
                           "trace_gain_m2": gain, "predicted_accepted": c.get("predicted_accepted"),
                           "predicted_ci_weight": c.get("predicted_ci_weight"), "reason": c["reason"]}
    extension = {"schema": "execution_only_neighbor_proxy_v12", "step": context["step"], "seed": context["seed"],
                 "time_s": context["time_s"], "none_score": 0., "candidates": candidates,
                 "request_penalty_m2": REQUEST_PENALTY_M2,
                 "directional_encoding": derive_directional_features(old.normalize(context)),
                 "interpretation": "predicted conservative position covariance trace objective, not true error gain"}
    a, b = (candidates[uid]["score"] for uid in PEERS)
    extension["score_difference"] = a-b if a is not None and b is not None else None
    extension["compute_ms"] = (perf_counter()-started)*1000
    return baseline, extension


def choose_greedy(scoring):
    choices = [(uid, scoring["candidates"][uid]["score"]) for uid in PEERS
               if scoring["candidates"][uid]["valid"] and scoring["candidates"][uid]["score"] > 0.]
    if not choices:
        return "none"
    maximum = max(score for uid, score in choices)
    tied = [uid for uid, score in choices if abs(score-maximum) <= 1e-12]
    if len(tied) == 1:
        return tied[0]
    return str(indexed_rng(scoring["seed"], scoring["step"], "v12_minimal_tie", TARGET).choice(tied))
