"""Isolated neighbor-choice adapter; the v10 physics/filter/control stay native."""
from __future__ import annotations
import copy
from time import perf_counter
from unittest.mock import patch
import numpy as np
from . import partial_gnss_execution_v10 as native
from . import partial_gnss_baseline_v7 as v7
from ..analysis import neighbor_selection_policy_v11 as policy_module
from ..analysis import neighbor_selection_counterfactual_v11 as counterfactual

POLICIES = ("none", "random_peer", "greedy", "fixed0", "fixed1")
CF_STEPS = (6, 10, 30, 62, 94, 114)
_OriginalAuditEnv = native.AuditEnv
_ORIGINAL_CHOOSE = v7.choose


def visible_context(env, legal, seed, step):
    """Read only receiver-local information, never a reference's private EKF."""
    now = env.timestep * env.config.motion.dt_s
    observation = env._observations()[v7.TARGET]
    messages = {}
    for peer in v7.PEERS:
        msg = env.arrived_messages.get((peer, v7.TARGET))
        if msg is None:
            continue
        key = (v7.TARGET, peer)
        cached = env.neighbor_beliefs.get(key)
        link = observation.get("candidate_incoming_links", {}).get(peer, {})
        messages[peer] = {
            "state": msg.state.copy(), "covariance": msg.covariance.copy(),
            "send_time_s": msg.send_time_s, "state_time_s": msg.last_propagated_time_s,
            "rssi_dbm": link.get("rssi_dbm"), "link_aoi_s": link.get("aoi_s"),
            "cache_mean": None if cached is None else cached[0].copy(),
            "cache_covariance": None if cached is None else cached[1].copy(),
            "cache_time_s": env.neighbor_belief_timestamps.get(key),
            "cache_message_send_s": env.neighbor_belief_message_times.get(key),
            "last_update_time_s": env.uwb_update_times.get((peer, v7.TARGET))}
    return {"step": step, "seed": seed, "time_s": now,
            "own_state": observation["estimate"].copy(),
            "own_covariance": observation["covariance"].copy(),
            "legal_sources": list(legal), "peers": messages,
            "measurement_budget_remaining": env.measurement_budget_remaining[v7.TARGET]}


class NeighborAuditEnv(_OriginalAuditEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.v11_decisions = []
        self.v11_snapshots = []
        self.v11_active_policy = None
        self.v11_selection_failure = None

    def step(self, actions):
        if self.v11_selection_failure is not None:
            # v9 catches this inside its transition guard and retains the prefix.
            raise RuntimeError("v11 selection integrity failure: "+self.v11_selection_failure)
        return super().step(actions)

    def _release_messages(self, next_timestep):
        # Called after all local sensor updates, before native cooperative CI.
        super()._release_messages(next_timestep)
        if self.v11_active_policy != "random_peer" or next_timestep not in CF_STEPS:
            return
        effective = self._visible_effective_edges()
        now = next_timestep * self.config.motion.dt_s
        peers, ranges = {}, {}
        for peer in v7.PEERS:
            msg = self.arrived_messages.get((peer, v7.TARGET))
            key = (v7.TARGET, peer)
            cache = self.neighbor_beliefs.get(key)
            if msg is not None:
                peers[peer] = {"state": msg.state.copy(), "covariance": msg.covariance.copy(),
                    "send_time_s": msg.send_time_s, "state_time_s": msg.last_propagated_time_s,
                    "cache_mean": None if cache is None else cache[0].copy(),
                    "cache_covariance": None if cache is None else cache[1].copy(),
                    "cache_time_s": self.neighbor_belief_timestamps.get(key),
                    "cache_message_send_s": self.neighbor_belief_message_times.get(key),
                    "last_update_time_s": self.uwb_update_times.get((peer, v7.TARGET))}
            raw = self.snapshot.uwb_measurements.get((peer, v7.TARGET))
            ranges[peer] = {"present": raw is not None, "valid": bool(raw and raw.get("valid", False)),
                "measurement": None if raw is None else raw.get("measured_range"),
                "variance": None if raw is None else float(raw["measurement_std"])**2 * (1 if raw["los"] else self.config.sensor.uwb_nlos_variance_multiplier),
                "nlos_bias": None if raw is None else (0 if raw["los"] else self.config.sensor.uwb_nlos_bias_m),
                "effective_edge": (peer, v7.TARGET) in effective}
        decision = self.v11_decisions[-1]
        self.v11_snapshots.append({"step": next_timestep, "time_s": now,
            "own_filter": counterfactual.freeze_filter(self.ekfs[v7.TARGET]),
            "budget_remaining_before": decision["context"]["measurement_budget_remaining"],
            "peers": v7.old.normalize(peers), "ranges": v7.old.normalize(ranges),
            "scope": "offline_only_post_local_pre_cooperation; never an online chooser input"})


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
            scored = policy_module.score_context(context, env.config)
            chosen = policy_module.choose_greedy(scored) if name == "greedy" else _ORIGINAL_CHOOSE(name, legal, seed, step)
        except (RuntimeError, ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
            env.v11_selection_failure = str(exc)
            scored = {"step":step,"time_s":context["time_s"],"seed":seed,
                      "failure":str(exc),"candidates":{},"clone_calls":0,"score_compute_ms":0}
            chosen = "none"  # Aborted action: step() raises before any execution.
        env.v11_decisions.append({"step": step, "context": v7.old.normalize(context),
                                  "scoring": v7.old.normalize(scored), "selected": chosen})
        return chosen
    # Scoped runtime adapters only; byte-identical historical source remains.
    with patch.object(native, "AuditEnv", NeighborAuditEnv), patch.object(native, "POLICIES", POLICIES), patch.object(v7, "choose", choose):
        record = native.physical(case, spec, "original_ci", policy)
        env = native._CAPTURED_ENVS[-1]
        record["v11_decisions"] = copy.deepcopy(env.v11_decisions)
        record["v11_snapshots"] = copy.deepcopy(env.v11_snapshots)
    return v7.old.normalize(record)
