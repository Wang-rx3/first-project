"""Dynamic-quality adapter over the unchanged V10 closed-loop execution."""
from __future__ import annotations
import copy
from unittest.mock import patch
import numpy as np
from . import neighbor_selection_execution_v11 as prior
from . import neighbor_selection_environment_v13 as environment
from ..analysis import neighbor_selection_policy_v13 as chooser

native, v7 = prior.native, prior.v7
POLICIES = ("none", "random_peer", "greedy", "v13_greedy")


def visible_context(env, legal, seed, step):
    context = prior.visible_context(env, legal, seed, step)
    links = env._observations()[v7.TARGET].get("candidate_incoming_links", {})
    for uid, payload in context["peers"].items():
        msg = env.arrived_messages[(uid, v7.TARGET)]
        quality = env.received_quality(msg)
        payload.update(packet_delay_s=msg.delay_s, prelink_delay_s=links.get(uid, {}).get("delay_s"),
                       gnss_std_m=None if quality is None else quality["gnss_std_m"],
                       gnss_report_time_s=None if quality is None else quality["gnss_report_time_s"],
                       quality_packet_id=None if quality is None else quality["packet_id"])
    return context


def physical(case, spec, policy):
    native._main_thread_guard()
    if policy not in POLICIES:
        raise ValueError(policy)
    def choose(name, legal, seed, step):
        env = native._CAPTURED_ENVS[-1]
        env.v11_active_policy = policy
        if step not in spec["decision_steps"]:
            return "none"
        context = visible_context(env, legal, seed, step)
        try:
            legacy, scoring = chooser.score_context(context, env.config)
            chosen = (chooser.choose_greedy(scoring) if name == "v13_greedy" else
                      prior.policy_module.choose_greedy(legacy) if name == "greedy" else
                      prior._ORIGINAL_CHOOSE(name, legal, seed, step))
        except (RuntimeError, ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
            env.v11_selection_failure = str(exc)
            legacy = {"failure": str(exc), "candidates": {}, "clone_calls": 0}
            scoring = {"failure": str(exc), "compute_ms": 0}
            chosen = "none"  # Native guard aborts before executing a transition.
        env.v11_decisions.append({"step": step, "context": v7.old.normalize(context), "scoring": v7.old.normalize(legacy),
                                 "v13_scoring": v7.old.normalize(scoring), "selected": chosen})
        return chosen
    with environment.adapters(case, spec) as env_class, patch.object(native, "AuditEnv", env_class), \
         patch.object(native, "POLICIES", POLICIES), patch.object(v7, "choose", choose):
        record = native.physical(case, spec, "original_ci", policy)
        env = native._CAPTURED_ENVS[-1]
        for name in ("v11_decisions", "v11_snapshots", "v13_quality_log", "v13_radio_log", "v13_metadata_events", "v13_failure_probes"):
            record[name] = copy.deepcopy(getattr(env, name))
        record["v13_schema"] = "live_native_queue_dynamic_quality_original_ci"
        record["quality_offset"] = case["quality_offset"]  # Offline design record, never context.
    return v7.old.normalize(record)
