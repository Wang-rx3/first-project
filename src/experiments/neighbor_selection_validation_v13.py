"""Pre-registered V13: bounded physical comparison, no optimization updates."""
from __future__ import annotations
import argparse
import copy
import re
import subprocess
import xml.etree.ElementTree as ET
from collections import Counter
from time import perf_counter
import numpy as np
from . import neighbor_selection_validation_v12 as previous
from . import neighbor_selection_execution_v11 as prior_execution
from ..analysis import neighbor_selection_counterfactual_v11 as cf
from ..analysis.partial_gnss_replay_v9 import physical_rows
from ..analysis.partial_gnss_validation_gate_v10 import summarize, physical_safety
from ..analysis.partial_gnss_report_v7 import write_csv

ROOT, old = previous.ROOT, previous.old
STEM = "three_uav_dynamic_neighbor_value_v13_20261013"
BASE = ROOT / "data/synthetic" / STEM
LOGS = ROOT / "logs/preliminary" / STEM
CONFIG = ROOT / "data/configs" / (STEM + ".json")
SOURCES = ("src/experiments/neighbor_selection_environment_v13.py", "src/experiments/neighbor_selection_execution_v13.py",
           "src/experiments/neighbor_selection_validation_v13.py", "src/analysis/neighbor_selection_policy_v13.py",
           "src/analysis/neighbor_selection_report_v13.py")
TESTS = ("tests/test_neighbor_selection_environment_v13.py", "tests/test_neighbor_selection_policy_v13.py",
         "tests/test_neighbor_selection_report_v13.py")


def validate_batch(batch):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", batch):
        raise ValueError("unsafe batch id")
    return BASE / "batches" / batch


def freeze():
    if BASE.exists() or CONFIG.exists():
        raise FileExistsError("V13 already frozen; no overwrite")
    prior = previous.verify()
    seal = old.read_json(previous.BASE / "final_artifact_manifest.json")
    protected = previous.verify_inputs().copy()
    for p, digest in seal["artifact_sha256"].items():
        if p != "README.md":
            if old.file_hash(ROOT / p) != digest:
                raise RuntimeError("V12 seal changed: " + p)
            protected[str(ROOT / p)] = digest
    for name in ("final_artifact_manifest.json", "final_verification.lock.json"):
        protected[str(previous.BASE / name)] = old.file_hash(previous.BASE / name)
    dev, confirm = list(range(2026101301, 2026101305)), list(range(2026101341, 2026101345))
    errors = {str(s): {u: (prior_execution.v7.indexed_rng(s, 0, "v7_initial_error", u).normal(size=15)
                          * np.sqrt([4] * 3 + [1] * 12)).tolist() for u in prior_execution.v7.UIDS}
              for s in (*dev, *confirm, 2026101324)}
    def cases(seeds, phase):
        return [{**copy.deepcopy(next(c for c in prior["development_cases"] if c["scene_id"] == scene)),
                 "seed": seed, "split": phase, "quality_offset": 0 if i % 2 == 0 else 10}
                for scene in prior["geometry_ids"] for i, seed in enumerate(seeds)]
    spec = {k: copy.deepcopy(prior[k]) for k in ("startup_steps", "motion_vo_chi_square_limit", "dt_s", "horizon_steps",
            "decision_steps", "measurement_budget", "message_ttl_s", "pair_period_s", "geometry_ids", "safety", "legacy_proxy_gate")}
    spec.update(schema="dynamic_neighbor_value_v13_r3", development_seeds=dev, confirmation_seeds=confirm,
                development_cases=cases(dev, "development"), confirmation_cases=cases(confirm, "independent_confirmation"),
                initial_errors_by_seed=errors, policies=["none", "random_peer", "greedy", "v13_greedy"],
                cf_steps=[6, 10, 30, 62, 94, 114], dynamic_gnss=True, dynamic_radio=True,
                phase_wall_cap_s=900, physical_step_cap=7680, precheck_attempt_cap=18,
                cf_local_state_cap=1152, cf_range_call_cap=192, future_predict_cap=864, online_clone_cap=3712,
                phase_caps={"precheck": 18, "develop": 3840, "confirm": 3840, "counterfactual": 0},
                field={"phase": "(x-40+.1*(y-125)+.15*(z-28)+offset)%20", "bands": [[0,5,.5,0,.02],[5,10,1.5,6,.22],[10,15,3,12,.42],[15,20,1.5,6,.22]],
                       "clear_above_m": 50, "radio_midpoint_offset": 5, "uid_specific": False,
                       "semantics": "joint phenomenological virtual GNSS/RF field, not calibrated city raycast"},
                metadata={"only_actual_native_packets": True, "payload_time": "prestep", "header_bytes": 16,
                          "on_abstract_payload_bytes": 496, "off_abstract_payload_bytes": 512, "abstract_envelope_bytes": 512,
                          "no_wire_codec_or_covariance_serialization_claim": True, "missing_never_alters_native_fusion": True},
                proxy={"penalty_m": .01, "cost": "1+native_prelink_delay/.2", "formula": "sqrt(trPbefore)-sqrt(trPafter_original_CI_zero_innovation)-.01*cost",
                       "none_score": 0, "tie": "native peer order", "fitted": False, "normalized": False},
                gates={"ranking_accuracy": .65, "spearman": .2, "positive_rank_seeds": 3,
                       "meaningful_overall": 36, "meaningful_confirmation": 18, "utility_resolution_m": .01,
                       "trajectory_fraction": .05, "maximum_case_harm": .05, "positive_joint_seeds": 3,
                       "all_seeds_both_geometries_both_labels": True, "score_p95_ms": 200},
                prechecks={"off_seed": 2026101301, "scene_id": "train_collinear", "prefix_steps": 6,
                           "retained_known_failure_seed": 2026101324, "known_failure_not_main_seed_filter": True},
                confirmation_execution_gate="development numerical/physical/interface safety only, not performance or ranking",
                cf_scope="Random postlocal/precooperation frozen actual range/messages; three future recorded target local packets; not queue/nav intervention",
                count_table={"main_episodes": 64, "planned_cf_points": 96, "physical_attempts_max": 7680,
                             "precheck_physical_attempts_max": 18, "cf_state_evaluations_max": 1152, "relative_CI_calls_max": 192},
                training_allowed=False, optimizer_updates=0,
                independent_unit="4 noise seeds per split, 2 offset blocks with 2 replicates; geometry/window repeated")
    BASE.mkdir(parents=True, exist_ok=False)
    (BASE / "README_before.bin").write_bytes((ROOT / "README.md").read_bytes())
    old.write_json(CONFIG, spec)
    old.write_json(BASE / "design.lock.json", spec)
    old.write_json(BASE / "protected_inputs_before.json", protected)
    old.write_json(BASE / "design.sha256.json", {"config_sha256": old.file_hash(CONFIG),
                   "source_sha256": {p: old.file_hash(ROOT / p) for p in (*SOURCES, *TESTS, "knowledge/neighbor_selection_v13_protocol.md")},
                   "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()})
    print("V13 frozen: 64 episodes, 96 checkpoints, 18 precheck attempts; protected", len(protected), flush=True)


def verify(require_release=True):
    lock = old.read_json(BASE / "design.sha256.json")
    if old.file_hash(CONFIG) != lock["config_sha256"]:
        raise RuntimeError("V13 config changed")
    for p, digest in lock["source_sha256"].items():
        if old.file_hash(ROOT / p) != digest:
            raise RuntimeError("V13 frozen source changed: " + p)
    for p, digest in old.read_json(BASE / "protected_inputs_before.json").items():
        if old.file_hash(p) != digest:
            raise RuntimeError("historical input changed: " + p)
    if require_release:
        release = old.read_json(BASE / "unit_release.lock.json")
        if not release["passed"] or release["source_sha256"] != lock["source_sha256"]:
            raise RuntimeError("pure unit release missing")
    return old.read_json(CONFIG)


def release(junit):
    verify(False)
    xml = ET.parse(junit).getroot()
    suites = list(xml.iter("testsuite"))
    total = sum(int(s.get("tests", 0)) for s in suites)
    failed = sum(int(s.get(k, 0)) for s in suites for k in ("errors", "failures"))
    if not total or failed:
        raise RuntimeError("pure unit tests failed/missing")
    with (BASE / "unit_release.lock.json").open("x", encoding="utf-8") as stream:
        stream.write(old.encode({"passed": True, "tests": total, "junit_sha256": old.file_hash(junit),
                                "source_sha256": old.read_json(BASE / "design.sha256.json")["source_sha256"]}))


def write_manifest(data, logs, payload):
    payload["artifact_sha256"] = {str(p.relative_to(ROOT)): old.file_hash(p) for folder in (data, logs)
                                  for p in folder.rglob("*") if p.is_file()}
    old.write_json(data / "manifest.json", payload)


def checked_manifest(path):
    value = old.read_json(path)
    for p, digest in value["artifact_sha256"].items():
        if old.file_hash(ROOT / p) != digest:
            raise RuntimeError("phase artifact changed: " + p)
    return value


def start(batch, phase, spec):
    reserved = BASE / "reservations"
    reserved.mkdir(exist_ok=True)
    prior = [old.read_json(p) for p in reserved.glob("*.json")]
    amount = spec["phase_caps"][phase]
    cap = spec["physical_step_cap"] + spec["precheck_attempt_cap"]
    if sum(p["attempts"] for p in prior) + amount > cap:
        raise RuntimeError("global bounded attempt reservation exhausted")
    with (reserved / (batch + "_" + phase + ".json")).open("x", encoding="utf-8") as stream:
        stream.write(old.encode({"phase": phase, "attempts": amount}))
    data, logs = validate_batch(batch) / phase, LOGS / batch / phase
    data.mkdir(parents=True, exist_ok=False)
    logs.mkdir(parents=True, exist_ok=False)
    return data, logs, perf_counter() + spec["phase_wall_cap_s"]


def precheck(batch):
    from unittest.mock import patch
    from . import neighbor_selection_execution_v13 as execution
    spec = verify()
    data, logs, deadline = start(batch, "precheck", spec)
    case = copy.deepcopy(spec["development_cases"][0])
    local = {**spec, "prefix_steps": 6, "dynamic_gnss": False, "dynamic_radio": False, "phase_deadline": deadline}
    class PrefixEnv(prior_execution.NeighborAuditEnv):
        def step(self, actions):
            if self.timestep >= 6:
                raise RuntimeError("V13 planned prefix stop before transition 7")
            return super().step(actions)
    with old._forbid_training_runtime():
        with patch.object(prior_execution.native, "AuditEnv", PrefixEnv):
            baseline = prior_execution.native.physical(case, local, "original_ci", "none")
        off = execution.physical(case, local, "none")
        # Ignore reporting/timing extras; compare physical measurements, filter and real queues exactly.
        keys = ("observable", "evaluation_truth", "audit")
        def core(r):
            return [{k: f.get(k) for k in keys} for f in r["frames"] if "states" in f.get("audit", {})]
        equal = old.encode(core(baseline)) == old.encode(core(off))
        for name, record in (("native_off", baseline), ("dynamic_off", off)):
            old.write_json(data / (name + "_record.json"), record)
        known = {**case, "seed": 2026101324}
        bad = execution.physical(known, {**spec, "prefix_steps": 6, "phase_deadline": deadline}, "none")
        old.write_json(data / "retained_known_failure_record.json", bad)
    count = sum(sum("states" in f.get("audit", {}) or f.get("incomplete_transition", False) for f in r["frames"][:6]) for r in (baseline, off, bad))
    result = {"off_equivalence": equal, "off_completed_steps": [len(core(r)) for r in (baseline, off)],
              "known_seed_failure": bad["failure"], "known_seed_frames": len(bad["frames"]),
              "known_seed_is_retained_diagnostic": True, "physical_attempts": count,
              "passed": bool(equal and len(core(baseline)) == len(core(off)) == 6 and count <= 18), "optimizer_updates": 0}
    old.write_json(data / "checks.json", result)
    write_manifest(data, logs, result)
    print(old.encode(result), flush=True)


def run_physical(batch, phase):
    from . import neighbor_selection_execution_v13 as execution
    spec = verify()
    if not checked_manifest(validate_batch(batch) / "precheck/manifest.json")["passed"]:
        raise RuntimeError("OFF/interface precheck blocks execution")
    data, logs, deadline = start(batch, phase, spec)
    blocked = phase == "confirm" and not checked_manifest(validate_batch(batch) / "develop/manifest.json")["safety"]["passed"]
    summaries, failures, timing = [], [], []
    steps = attempts = clones = 0
    cases = spec["development_cases"] if phase == "develop" else spec["confirmation_cases"]
    with old._forbid_training_runtime():
        for case in cases:
            for policy in spec["policies"]:
                label = f"{case['scene_id']}_s{case['seed']}_{policy}"
                if blocked or perf_counter() >= deadline:
                    failures.append({"label": label, "reason": "development safety blocks confirmation" if blocked else "wall budget: not executed"})
                    continue
                try:
                    record = execution.physical(case, {**spec, "phase_deadline": deadline}, policy)
                    old.write_json(data / (label + "_record.json"), record)
                    steps += sum("states" in f.get("audit", {}) for f in record["frames"])
                    attempts += len(record["frames"])
                    rows, events, audit = physical_rows(record, spec)
                    result = summarize(record, rows, events, audit, "original_ci", policy, "v13_physical")
                    result.update(study_phase=phase, quality_offset=case["quality_offset"],
                                  request_windows=[f["step"] for f in record["frames"] if f.get("audit", {}).get("selection", "none") != "none"],
                                  selection_counts=dict(Counter(d["selected"] for d in record["v11_decisions"])),
                                  uav2_final_localization_error_m=float(np.linalg.norm(rows[-1]["target_position_error_m"])) if rows else None,
                                  decision_points_logged=len(record["v11_decisions"]),
                                  metadata_valid_arrivals=sum(e.get("event") == "visible_arrival" and e.get("valid_quality", False) for e in record["v13_metadata_events"]),
                                  metadata_missing_arrivals=sum(e.get("event") == "visible_arrival" and not e.get("valid_quality", False) for e in record["v13_metadata_events"]),
                                  gnss_regime_counts=dict(Counter(str(q.get("regime", q.get("level"))) for q in record["v13_quality_log"])),
                                  radio_regime_counts=dict(Counter(str(q.get("regime", q.get("level"))) for q in record["v13_radio_log"])))
                    samples = [d["v13_scoring"]["compute_ms"] for d in record["v11_decisions"]]
                    clones += sum(d["scoring"]["clone_calls"] for d in record["v11_decisions"])
                    timing.extend(samples)
                    summaries.append(result)
                    old.write_json(data / (label + "_estimator.json"), {"rows": rows, "events": events, "audit": audit})
                    for name, rr in (("steps", rows), ("events", events), ("decisions", record["v11_decisions"]),
                                     ("gnss_quality", record["v13_quality_log"]), ("radio_quality", record["v13_radio_log"]), ("metadata", record["v13_metadata_events"])):
                        write_csv(logs / (label + "_" + name + ".csv"), rr)
                    print(phase, label, record["status"], len(record["frames"]), record["failure"], flush=True)
                except Exception as exc:
                    failures.append({"label": label, "reason": type(exc).__name__ + ": " + str(exc),
                                     "raw_record_retained": (data / (label + "_record.json")).exists()})
                    old.write_json(data / (label + "_failure.json"), failures[-1])
    safety = physical_safety(summaries, expected=32)
    p95 = float(np.percentile(timing, 95)) if timing else None
    if failures or clones > 1856 or attempts > 3840 or any(r["decision_points_logged"] != 29 for r in summaries) or p95 is None or p95 >= 200:
        safety["passed"] = False
        safety["failures"].append("V13 completeness/clone/attempt/computation bound")
    write_csv(logs / "summary.csv", summaries)
    write_manifest(data, logs, {"study_phase": phase, "summaries": summaries, "failures": failures, "safety": safety,
                               "physical_steps": steps, "attempted_steps": attempts, "score_clone_calls": clones,
                               "score_p95_ms": p95, "optimizer_updates": 0, "blocked": blocked})
    verify()
    print(old.encode({"phase": phase, "safety": safety, "physical_steps": steps}), flush=True)


def norm_labels(result, decision):
    from ..analysis.neighbor_selection_policy_v13 import request_cost, PENALTY_M
    branches = result["branches"]
    none = np.sqrt(np.asarray(branches["none"]["sse_m2"]))
    labels = {}
    for uid in prior_execution.v7.PEERS:
        errors = np.sqrt(np.asarray(branches[uid]["sse_m2"]))
        cost = request_cost(decision["context"].get("peers", {}).get(uid, {}).get("prelink_delay_s"))
        gain = none - errors
        labels[uid] = {"current_norm_gain_m": float(gain[0]), "four_norm_gain_m": float(np.mean(gain)), "cost_unit": cost,
                       "norm_utility_m": None if cost is None else float(np.mean(gain)) - PENALTY_M * cost}
    a, b = (labels[uid]["norm_utility_m"] for uid in prior_execution.v7.PEERS)
    delta = None if a is None or b is None else a - b
    meaningful = bool(result["dual_eligible"] and delta is not None and abs(delta) >= .01)
    def correct(score):
        return None if not meaningful or score is None else .5 if abs(score) <= 1e-12 else float(score * delta > 0)
    return {"norm_labels": labels, "norm_utility_difference": delta, "meaningful_norm": meaningful,
            "v13_score_difference": decision["v13_scoring"].get("score_difference"),
            "v13_ranking_correct": correct(decision["v13_scoring"].get("score_difference")),
            "legacy_norm_ranking_correct": correct(result["score_difference"]),
            "v13_features": decision["v13_scoring"].get("features", {})}


def run_counterfactual(batch):
    spec = verify()
    data, logs, deadline = start(batch, "counterfactual", spec)
    results, failures = [], []
    states = calls = 0
    with old._forbid_training_runtime():
        for phase in ("develop", "confirm"):
            checked_manifest(validate_batch(batch) / phase / "manifest.json")
            cases = spec["development_cases"] if phase == "develop" else spec["confirmation_cases"]
            for case in cases:
                label = f"{case['scene_id']}_s{case['seed']}_random_peer"
                path = validate_batch(batch) / phase / (label + "_record.json")
                record = old.read_json(path) if path.exists() else None
                for step in spec["cf_steps"]:
                    point = {"study_phase": phase, "scene_id": case["scene_id"], "seed": case["seed"], "step": step}
                    try:
                        if perf_counter() >= deadline or record is None:
                            raise RuntimeError("frozen wall cap or physical record unavailable")
                        snap = next(s for s in record["v11_snapshots"] if s["step"] == step)
                        dec = next(d for d in record["v11_decisions"] if d["step"] == step)
                        result = cf.evaluate_point(record, snap, dec["scoring"], spec)
                        states += result["local_states"]
                        calls += result["relative_ci_calls"]
                        if states > 1152 or calls > 192:
                            raise RuntimeError("frozen counterfactual quota")
                        result.update(norm_labels(result, dec))
                        old.write_json(data / (phase + "_" + label + f"_step{step}.json"), {**point, **result})
                        results.append({**point, **{k: v for k, v in result.items() if k != "branches"}})
                    except Exception as exc:
                        failure = {**point, "reason": type(exc).__name__ + ": " + str(exc), "dual_eligible": False, "meaningful_norm": False}
                        results.append(failure)
                        failures.append(failure)
                        old.write_json(data / (phase + "_" + label + f"_step{step}_failure.json"), failure)
                print(phase, label, "CF finished", flush=True)
    ids = [(p["study_phase"], p["scene_id"], p["seed"], p["step"]) for p in results]
    if len(ids) != len(set(ids)) or len(ids) != 96:
        raise RuntimeError("counterfactual exhaustive membership failed")
    write_csv(logs / "points.csv", results)
    write_manifest(data, logs, {"results": results, "failures": failures, "local_states": states, "relative_ci_calls": calls,
                               "physical_steps": 0, "optimizer_updates": 0, "planned_points": 96, "valid_points": 96 - len(failures)})
    verify()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=("freeze", "release", "precheck", "develop", "confirm", "counterfactual", "report"))
    parser.add_argument("--batch-id", default="research_001")
    parser.add_argument("--junit")
    args = parser.parse_args()
    if args.phase == "freeze": freeze()
    elif args.phase == "release": release(args.junit)
    elif args.phase == "precheck": precheck(args.batch_id)
    elif args.phase in ("develop", "confirm"): run_physical(args.batch_id, args.phase)
    elif args.phase == "counterfactual": run_counterfactual(args.batch_id)
    else:
        from ..analysis.neighbor_selection_report_v13 import report
        report(args.batch_id)


if __name__ == "__main__":
    main()
