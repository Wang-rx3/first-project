"""Scoped v12 selector, keeping native v10 physical/queue/filter feedback."""
from __future__ import annotations

import copy
from unittest.mock import patch
import numpy as np

from . import neighbor_selection_execution_v11 as prior
from ..analysis import neighbor_selection_policy_v12 as policy_module

native, v7 = prior.native, prior.v7
POLICIES = ("none", "random_peer", "greedy", "minimal_greedy")
NeighborAuditEnv = prior.NeighborAuditEnv


def visible_context(env, legal, seed, step):
    context = prior.visible_context(env, legal, seed, step)
    links = env._observations()[v7.TARGET].get("candidate_incoming_links", {})
    for uid, payload in context["peers"].items():
        message = env.arrived_messages[(uid, v7.TARGET)]
        payload["packet_delay_s"] = getattr(message, "delay_s", None)
        payload["prelink_delay_s"] = links.get(uid, {}).get("delay_s")
    return context


def physical(case, spec, policy):
    native._main_thread_guard()
    if policy not in POLICIES:
        raise ValueError(policy)
    if spec["proxy"]["request_penalty_m2"] != policy_module.REQUEST_PENALTY_M2:
        raise RuntimeError("candidate penalty changed")
    def choose(name, legal, seed, step):
        env = native._CAPTURED_ENVS[-1]
        env.v11_active_policy = policy
        if step not in spec["decision_steps"]:
            return "none"
        context = visible_context(env, legal, seed, step)
        try:
            baseline, extension = policy_module.score_context(context, env.config)
            if name == "minimal_greedy":
                chosen = policy_module.choose_greedy(extension)
            elif name == "greedy":
                chosen = prior.policy_module.choose_greedy(baseline)
            else:
                chosen = prior._ORIGINAL_CHOOSE(name, legal, seed, step)
        except (RuntimeError, ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
            env.v11_selection_failure = str(exc)
            baseline = {"failure": str(exc), "candidates": {}, "clone_calls": 0, "score_compute_ms": 0}
            extension = {"failure": str(exc), "compute_ms": 0}
            chosen = "none"  # Native step guard aborts before a physical transition.
        env.v11_decisions.append({"step": step, "context": v7.old.normalize(context), "scoring": v7.old.normalize(baseline),
                                  "v12_scoring": v7.old.normalize(extension), "selected": chosen,
                                  "request_id": None if chosen == "none" else f"{case['scene_id']}:{seed}:{policy}:{step}:{chosen}->{v7.TARGET}",
                                  "exogenous_stream_scope": "native v7 indexed streams; same seed and step, physical geometry may differ"})
        return chosen
    with patch.object(native, "AuditEnv", NeighborAuditEnv), patch.object(native, "POLICIES", POLICIES), patch.object(v7, "choose", choose):
        record = native.physical(case, spec, "original_ci", policy)
        env = native._CAPTURED_ENVS[-1]
        record["v11_decisions"] = copy.deepcopy(env.v11_decisions)
        record["v11_snapshots"] = copy.deepcopy(env.v11_snapshots)
        record["v12_schema"] = "actual_live_queue_original_ci_execution_only_selector"
    return v7.old.normalize(record)
