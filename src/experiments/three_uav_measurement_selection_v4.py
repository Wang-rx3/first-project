"""Diagnostic Stage-3 measurement scheduler (v4).

The v4 entry point keeps the v3 simulator and credit proxy available for a
human calibration run, while making the decision schedule an explicit,
serializable boundary.  Importing this module is pure: no environment,
trainer, checkpoint, or optimizer is constructed.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import re
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from . import three_uav_measurement_selection as v3


ROOT = Path(__file__).resolve().parents[2]
STEM = "three_uav_measurement_learnability_v4_seed2026_20261012"
CONFIG = ROOT / "data/configs" / (STEM + ".json")
SCHEMA = "three_uav_measurement_only_v4_diagnostic"
UIDS = tuple(v3.UIDS)
SOURCE_FILES = tuple(dict.fromkeys((*v3.SOURCE_FILES, "src/experiments/three_uav_measurement_selection_v4.py")))
DEFAULT_DECISION_SCHEDULE = {
    "first_step": 2,
    "period_steps": 4,
    "last_step": 118,
    "physical_pair_period_steps": 4,
}


def native(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def encode(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=native, allow_nan=False)


def content_hash(value: Any) -> str:
    return hashlib.sha256(encode(value).encode("utf-8")).hexdigest()


def file_hash(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path: str | Path, value: Any) -> None:
    with Path(path).open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2, default=native, allow_nan=False) + "\n")


class WindowRoundRobin:
    """Persistent legal-peer round robin evaluated only in decision windows."""

    schema = "window_round_robin_v4"

    def __init__(self, decision_schedule: Any = None, uids: tuple[str, ...] = UIDS):
        self.uids = tuple(str(uid) for uid in uids)
        if len(self.uids) != len(set(self.uids)) or not self.uids:
            raise ValueError("uids must be nonempty and unique")
        self.decision_steps = self._parse_schedule(decision_schedule)
        self._decision_set = set(self.decision_steps)
        self._cursors = {uid: 0 for uid in self.uids}

    @staticmethod
    def _parse_schedule(value: Any) -> tuple[int, ...]:
        if value is None:
            value = DEFAULT_DECISION_SCHEDULE
        if isinstance(value, Mapping):
            explicit = value.get("window_steps", value.get("decision_steps", value.get("steps")))
            if explicit is not None:
                value = explicit
            else:
                first = int(value.get("first_step", 2))
                period = int(value.get("period_steps", value.get("period", 4)))
                last = int(value.get("last_step", 118))
                if first < 1 or period < 1 or last < first:
                    raise ValueError("invalid decision schedule")
                value = list(range(first, last + 1, period))
        if isinstance(value, (str, bytes)):
            raise ValueError("decision schedule must be a sequence or mapping")
        try:
            steps = tuple(int(step) for step in value)
        except (TypeError, ValueError):
            raise ValueError("decision schedule must be a sequence or mapping") from None
        if not steps or any(step < 1 for step in steps) or list(steps) != sorted(set(steps)):
            raise ValueError("decision schedule must contain increasing positive steps")
        return steps

    def reset(self) -> None:
        self._cursors = {uid: 0 for uid in self.uids}

    def in_window(self, step: int) -> bool:
        return isinstance(step, (int, np.integer)) and not isinstance(step, bool) and int(step) in self._decision_set

    def mask_observations(self, observations: Mapping[str, Mapping[str, Any]], next_step: int) -> dict[str, dict[str, Any]]:
        return mask_observations(observations, next_step, self.decision_steps)

    scheduling_observations = mask_observations

    def _receiver(self, local: Mapping[str, Any]) -> str:
        nested = local.get("local") if isinstance(local, Mapping) else None
        view = nested if isinstance(nested, Mapping) else local
        return str(view.get("uav_id", ""))

    def _current_mask(self, local: Mapping[str, Any], receiver: str) -> np.ndarray:
        nested = local.get("local") if isinstance(local, Mapping) else None
        view = nested if isinstance(nested, Mapping) else local
        try:
            current = np.asarray(view.get("action_mask"), dtype=bool).reshape(-1)
        except (TypeError, ValueError):
            current = np.zeros(0, dtype=bool)
        if current.size >= len(self.uids) + 1:
            return current[: len(self.uids) + 1]
        # Synthetic callers need not provide a precomputed mask.
        return np.asarray(_BASE_CONTRACT.admissible_mask(view, uid=receiver), dtype=bool).reshape(-1)

    def select(self, local: Mapping[str, Any], step: int) -> str:
        """Return a legal peer and advance only after a legal request.

        ``local`` should come from :meth:`mask_observations`; the native
        contract is intersected again here so a caller cannot widen a mask by
        supplying a hand-written ``action_mask``.
        """
        receiver = self._receiver(local)
        if receiver not in self._cursors or not self.in_window(step):
            return "none"
        mask = self._current_mask(local, receiver)
        nested = local.get("local") if isinstance(local, Mapping) else None
        view = nested if isinstance(nested, Mapping) else local
        physical_legal = set(_BASE_CONTRACT._legal_sources(view, apply_base_mask=True))
        peers = [uid for uid in self.uids if uid != receiver]
        cursor = self._cursors[receiver] % len(peers)
        legal = {uid for uid in peers if uid in self.uids and len(mask) > self.uids.index(uid)
                 and bool(mask[self.uids.index(uid)]) and uid in physical_legal}
        selected = next((peers[(cursor + offset) % len(peers)] for offset in range(len(peers))
                         if peers[(cursor + offset) % len(peers)] in legal), None)
        if selected is None:
            return "none"  # blocked: cursor is deliberately unchanged
        self._cursors[receiver] = (peers.index(selected) + 1) % len(peers)
        return selected

    choose = select

    def state_dict(self) -> dict[str, Any]:
        return {"schema": self.schema, "uids": list(self.uids),
                "decision_steps": list(self.decision_steps),
                "cursors": {uid: int(self._cursors[uid]) for uid in self.uids}}

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "WindowRoundRobin":
        if not isinstance(state, Mapping) or state.get("schema") != cls.schema:
            raise ValueError("invalid WindowRoundRobin state schema")
        scheduler = cls(state.get("decision_steps"), tuple(state.get("uids", UIDS)))
        cursors = state.get("cursors", {})
        if not isinstance(cursors, Mapping) or set(cursors) != set(scheduler.uids):
            raise ValueError("invalid WindowRoundRobin cursor state")
        peers = len(scheduler.uids) - 1
        if peers < 1 or any(int(cursors[uid]) < 0 or int(cursors[uid]) >= peers for uid in scheduler.uids):
            raise ValueError("invalid WindowRoundRobin cursor")
        scheduler._cursors = {uid: int(cursors[uid]) for uid in scheduler.uids}
        return scheduler


def mask_observations(observations: Mapping[str, Mapping[str, Any]], next_step: int,
                      decision_schedule: Any = None) -> dict[str, dict[str, Any]]:
    """Copy native observations and apply only the v4 decision-window mask."""
    scheduler = WindowRoundRobin(decision_schedule)
    window = scheduler.in_window(next_step)
    native_obs = _BASE_CONTRACT.admissible_observations(observations)
    result: dict[str, dict[str, Any]] = {}
    for uid in UIDS:
        local = copy.deepcopy(native_obs[uid])
        native_mask = np.asarray(local.get("action_mask", np.zeros(4, dtype=bool)), dtype=bool).reshape(-1)
        if native_mask.size < len(UIDS) + 1:
            padded = np.zeros(len(UIDS) + 1, dtype=bool)
            padded[:native_mask.size] = native_mask
            native_mask = padded
        native_base_mask = np.asarray(_BASE_CONTRACT._base_mask(local), dtype=bool).reshape(-1)
        if native_base_mask.size < len(UIDS) + 1:
            padded = np.zeros(len(UIDS) + 1, dtype=bool)
            padded[:native_base_mask.size] = native_base_mask
            native_base_mask = padded
        native_mask = native_mask & native_base_mask
        local["native_action_mask"] = native_mask.copy()
        local["native_admissible_mask"] = native_mask.copy()
        local["native_base_action_mask"] = native_base_mask.copy()
        # This diagnostic deliberately excludes budget/cooldown/eligibility;
        # those native fields remain in ``local`` and are applied by the
        # admissibility contract.  It is therefore a candidate-source audit,
        # not a fabricated physical-action mask.
        candidates = set(local.get("candidate_measurement_sources", ()))
        effective = set(local.get("effective_sources", ()))
        available = local.get("uwb_available", {})
        local["physical_candidate_sources"] = [source for source in UIDS
            if source in candidates and source in effective
            and isinstance(available, Mapping) and bool(available.get(source, False))
            and bool(_BASE_CONTRACT._valid_message(local, source))]
        local["decision_window"] = bool(window)
        local["decision_step"] = int(next_step)
        restricted = native_mask.copy()
        if not window:
            restricted[: len(UIDS)] = False
            restricted[len(UIDS)] = True
        local["action_mask"] = restricted
        local["base_action_mask"] = restricted.copy()
        result[uid] = local
    return result


scheduling_observations = mask_observations


_BASE_CONTRACT = v3.contract()


class _SchedulerContractProxy:
    """v3 contract adapter with per-episode v4 schedule state."""
    def __init__(self, scheduler: WindowRoundRobin):
        self.scheduler = scheduler
        self.next_step = 0
        self._base_contract = _BASE_CONTRACT
        self.last_observations: dict[str, dict[str, Any]] = {}
        self.observations_by_step: dict[int, dict[str, dict[str, Any]]] = {}

    def reset_episode(self) -> None:
        self.scheduler.reset()
        self.next_step = 0
        self.last_observations = {}
        self.observations_by_step = {}

    def admissible_observations(self, observations: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
        self.next_step += 1
        self.last_observations = mask_observations(observations, self.next_step, self.scheduler.decision_steps)
        self.observations_by_step[self.next_step] = copy.deepcopy(self.last_observations)
        return self.last_observations

    def round_robin_selection(self, local: Mapping[str, Any], step: int) -> str:
        return self.scheduler.select(local, int(step))

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base_contract, name)


@contextmanager
def scheduler_context(scheduler: WindowRoundRobin):
    """Temporarily bind v3 globals; every binding is restored on exit."""
    proxy = _SchedulerContractProxy(scheduler)
    saved = {name: getattr(v3, name) for name in ("STEM", "SCHEMA", "SOURCE_FILES", "CONFIG", "contract")}
    v3.STEM, v3.SCHEMA, v3.SOURCE_FILES, v3.CONFIG = STEM, SCHEMA, SOURCE_FILES, CONFIG
    v3.contract = lambda: proxy
    try:
        yield proxy
    finally:
        for name, value in saved.items():
            setattr(v3, name, value)


class RunOutput:
    """Exclusive v4 data/log/plot roots, preserving partial artifacts on failure."""
    def __init__(self, run_id: str, root: str | Path = ROOT):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", str(run_id)):
            raise ValueError("run-id must contain only letters, digits, underscore or hyphen")
        self.run_id = str(run_id)
        self.data = Path(root) / "data/synthetic" / STEM / self.run_id
        self.logs = Path(root) / "logs/preliminary" / STEM / self.run_id
        self.plots = Path(root) / "plots/preliminary" / STEM / self.run_id
        occupied = [str(path) for path in (self.data, self.logs, self.plots) if path.exists()]
        if occupied:
            raise FileExistsError("Refusing to reuse output directories: " + ", ".join(occupied))
        for path in (self.data, self.logs, self.plots):
            path.mkdir(parents=True, exist_ok=False)
        self.tables: dict[str, list[dict[str, Any]]] = {}

    def row(self, table: str, values: Mapping[str, Any]) -> None:
        self.tables.setdefault(table, []).append(copy.deepcopy(dict(values)))

    def flush(self) -> None:
        for table, rows in self.tables.items():
            if not rows:
                continue
            keys = list(dict.fromkeys(key for row in rows for key in row))
            with (self.logs / (table + ".csv")).open("x", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=keys)
                writer.writeheader()
                for row in rows:
                    writer.writerow({key: encode(value) if isinstance(value, (dict, tuple, list, np.ndarray)) else value
                                     for key, value in row.items()})


def source_manifest() -> dict[str, str]:
    return {name: file_hash(ROOT / name) for name in SOURCE_FILES}


def validate_frozen_config(spec: Mapping[str, Any]) -> None:
    if spec.get("schema") != SCHEMA:
        raise ValueError("v4 diagnostic schema is required")
    baseline = read_json(ROOT / "data/configs/three_uav_measurement_selection_v3_seed2026_20261012.json")
    for key, value in baseline.items():
        if key in ("schema", "experiment_name"):
            continue
        if spec.get(key) != value:
            raise ValueError("Frozen v3 configuration mismatch: " + str(key))
    schedule = spec.get("decision_schedule")
    if WindowRoundRobin(schedule).decision_steps != tuple(range(2, 119, 4)):
        raise ValueError("decision schedule must be steps 2, 6, ..., 118")
    status = spec.get("learnability_status", {})
    if not isinstance(status, Mapping) or status.get("ready") is not False or status.get("training_ready") is not False:
        raise ValueError("v4 diagnostic status must remain not ready")


def effective_simulation(spec: Mapping[str, Any]):
    validate_frozen_config(spec)
    return v3.effective_simulation(spec)


def policy_config(spec: Mapping[str, Any]):
    validate_frozen_config(spec)
    return v3.policy_config(spec)


def _episode(output: RunOutput, case: Mapping[str, Any], sim: Any, cfg: Any, spec: Mapping[str, Any],
             counters: dict[str, int], scheduler: WindowRoundRobin, bank: list[dict[str, Any]]):
    with scheduler_context(scheduler) as proxy:
        proxy.reset_episode()
        class AuditOutput:
            def row(self, table: str, values: Mapping[str, Any]) -> None:
                enriched = dict(values)
                if table in ("actions", "candidate_values"):
                    uid = str(enriched.get("uav_id", ""))
                    try:
                        decision_step = int(enriched.get("step", proxy.next_step))
                    except (TypeError, ValueError):
                        decision_step = proxy.next_step
                    if decision_step not in proxy.observations_by_step:
                        raise RuntimeError('Missing predecision observation for audit row')
                    frame = proxy.observations_by_step[decision_step]
                    local = frame.get(uid, {})
                    enriched["native_action_mask"] = local.get("native_action_mask")
                    enriched["native_admissible_mask"] = local.get("native_admissible_mask")
                    enriched["native_base_action_mask"] = local.get("native_base_action_mask")
                    enriched["physical_candidate_sources"] = local.get("physical_candidate_sources")
                    enriched["decision_window"] = bool(local.get("decision_window", False))
                    enriched["decision_step"] = local.get("decision_step", decision_step)
                output.row(table, enriched)

            def __getattr__(self, name: str) -> Any:
                return getattr(output, name)

        return v3.episode(AuditOutput(), case, sim, cfg, spec, counters, "calibration", bank=bank)


@contextmanager
def _forbid_training_runtime():
    """Install and restore fail-fast guards around legacy training entry points."""
    patches: list[tuple[Any, str, Any]] = []
    def forbid(target: Any, name: str) -> None:
        original = getattr(target, name)
        patches.append((target, name, original))
        setattr(target, name, lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("Forbidden v4 calibration operation: " + name)))
    for name in ("__init__", "update", "add_transition", "collect_rollout"):
        forbid(v3.MAPPOTrainer, name)
    for name in ("__init__", "act", "act_with_trace"):
        forbid(v3.MAPPOPolicy, name)
    forbid(v3.torch, "load")
    for optimizer in (v3.torch.optim.Adam, v3.torch.optim.AdamW, v3.torch.optim.SGD):
        forbid(optimizer, "step")
    try:
        yield
    finally:
        for target, name, original in reversed(patches):
            setattr(target, name, original)


def calibration_workflow(output: RunOutput, spec: Mapping[str, Any], sim: Any, cfg: Any,
                         counters: dict[str, int]) -> dict[str, Any]:
    """Run TRAIN-only proxy calibration and always export scientifically blocked readiness."""
    bank: list[dict[str, Any]] = []
    metrics, decisions, credits = [], [], []
    with _forbid_training_runtime():
        for case in v3.contract().make_scene_splits()["train"]:
            scheduler = WindowRoundRobin(spec["decision_schedule"])
            result = _episode(output, case, sim, cfg, spec, counters, scheduler, bank)
            metrics.append({key: value for key, value in result.items() if key not in ("decision_rows", "positive_credits")})
            decisions.extend(result["decision_rows"])
            credits.extend(result["positive_credits"])
    actor = v3.ObservationNormalizer().fit([frame["observations"] for frame in bank])
    critic = v3.FeatureNormalizer().fit([v3.critic_vector(frame["central_state"], cfg) for frame in bank])
    counters["normalization_fits"] = 2
    scale = float(np.quantile(credits, spec["reward"]["credit_scale_quantile"])) if credits else 0.0
    # Keep the original proxy gate semantics: greedy positivity is evaluated
    # only after the TRAIN-only credit scale has been fitted.
    if scale > 0:
        decisions = []
        for row in output.tables.get("candidate_values", []):
            maximum = max(row.get("unscaled_expected_relative_credit", {}).values(), default=0.0)
            positive = min(maximum / scale, 1.0) > spec["reward"]["communication_cost"]
            row["greedy_positive_after_train_scale"] = bool(positive)
            decisions.append({**row, "greedy_positive": bool(positive)})
    for statistic in [*actor.normalizers.values(), critic]:
        if (not np.isfinite(statistic.mean).all() or not np.isfinite(statistic.scale).all()
                or not np.all(np.asarray(statistic.scale) > 0)):
            raise ValueError("Calibration produced nonfinite normalization statistics")
    original_readiness = v3.contract().calibration_readiness(metrics, decisions, credits, scale, spec["calibration_gates"])
    scientific = {"ready": False, "training_ready": False,
                  "blocked_reason": "proxy_real_quality_and_fusion_consistency_unresolved",
                  "original_proxy_gate_passed": bool(original_readiness.get("ready", False))}
    artifact = {
        "schema": SCHEMA, "ready": False, "training_ready": False,
        "learnability_status": scientific, "original_readiness": original_readiness,
        "original_proxy_gate_passed": bool(original_readiness.get("ready", False)),
        "fit_split": "train", "fit_scene_ids": [case["scene_id"] for case in v3.contract().make_scene_splits()["train"]],
        "config_hash": content_hash(spec), "scene_hash": content_hash(v3.contract().make_scene_splits()),
        "sources": source_manifest(), "actor_normalizer": actor.state_dict(), "critic_normalizer": critic.state_dict(),
        "credit_scale": scale, "effective_sensor_config": asdict(sim.sensor),
        "range_variance_m2": sim.sensor.uwb_los_std_m ** 2, "seeds": spec["seeds"],
        "decision_schedule": WindowRoundRobin(spec["decision_schedule"]).state_dict(),
        "decision_window_count": len(WindowRoundRobin(spec["decision_schedule"]).decision_steps),
        "full_horizon_frames": len(bank), "joint_environment_steps": counters["joint_environment_steps"],
        "max_joint_steps": 480, "budget_contract": copy.deepcopy(spec["simulation"]["measurement_budget_by_uav"]),
        "safety_projection_enabled": bool(spec["simulation"]["safety_projection_enabled"]),
        "native_action_masks_logged": True,
        "credit_label": "accepted_relative_linear_update_proxy_diagnostic_gain_not_final_covariance_reduction",
        "final_covariance_diagnostic_gain_available": True,
        "final_covariance_diagnostic_gain": sum(
            float(event['position_covariance_trace_before_m2'])
            - float(event['position_covariance_trace_after_final_m2'])
            for event in output.tables.get('fusion_events', [])),
        "fusion_consistency_status": "unresolved",
        "reward_active_final_covariance_change": False,
        "no_fake_precision_claim": True,
        "diagnostic_only": True,
    }
    write_json(output.data / "observation_bank.json", bank)
    write_json(output.data / "calibration.json", artifact)
    write_json(output.logs / "calibration_readiness.json", {"scientific": scientific, "original_proxy": original_readiness})
    return {"passed": False, "ready": False, "training_ready": False,
            "original_readiness": original_readiness, "scientific_readiness": scientific}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage-3 v4 diagnostic calibration; never trains or loads checkpoints.")
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--run-id", required=True, help="New exclusive run identifier")
    return parser


def main(workflow: str = "calibrate") -> None:
    if workflow != "calibrate":
        raise ValueError("v4 exposes calibration only; training and evaluation are incompatible")
    args = build_parser().parse_args()
    spec = read_json(args.config)
    validate_frozen_config(spec)
    output = RunOutput(args.run_id)
    write_json(output.data / "configuration.json", spec)
    write_json(output.data / "manifest.json", {"schema": SCHEMA, "workflow": workflow,
        "sources": source_manifest(), "scene_splits": v3.contract().make_scene_splits(),
        "config_hash": content_hash(spec), "decision_schedule": spec["decision_schedule"],
        "decision_window_count": len(WindowRoundRobin(spec["decision_schedule"]).decision_steps),
        "max_joint_steps": 480, "measurement_budget_by_uav": spec["simulation"]["measurement_budget_by_uav"],
        "safety_projection_enabled": spec["simulation"]["safety_projection_enabled"],
        "native_action_masks_logged": True, "diagnostic_only": True})
    counters = {"joint_environment_steps": 0, "normalization_fits": 0, "ppo_updates": 0,
                "optimizer_steps": 0, "rollout_additions": 0, "checkpoint_loads": 0,
                "neural_actions_executed": 0}
    status, failure, result = "not_ready", None, None
    try:
        sim, cfg = effective_simulation(spec), policy_config(spec)
        result = calibration_workflow(output, spec, sim, cfg, counters)
    except Exception as exc:
        status = "failed"
        failure = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        output.flush()
        write_json(output.logs / "run_status.json", {"workflow": workflow, "status": status,
            "failure": failure, "counts": counters, "run_id": args.run_id})
    print(encode({"workflow": workflow, "status": status, "counts": counters, "result": result}))
    raise SystemExit(2)


__all__ = ["CONFIG", "DEFAULT_DECISION_SCHEDULE", "RunOutput", "SCHEMA", "STEM", "UIDS",
           "WindowRoundRobin", "calibration_workflow", "content_hash", "effective_simulation",
           "file_hash", "main", "mask_observations", "policy_config", "read_json", "scheduling_observations",
           "source_manifest", "validate_frozen_config"]

