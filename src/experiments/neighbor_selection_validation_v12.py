"""Isolated, pre-registered v12 diagnostic; never performs learning updates."""
from __future__ import annotations

import argparse
import copy
import re
from collections import Counter
from time import perf_counter

import numpy as np

from . import neighbor_selection_validation_v11 as previous
from ..analysis import neighbor_selection_counterfactual_v11 as cf
from ..analysis.partial_gnss_replay_v9 import physical_rows
from ..analysis.partial_gnss_validation_gate_v10 import summarize, physical_safety
from ..analysis.partial_gnss_report_v7 import write_csv

ROOT = previous.ROOT
STEM = "three_uav_neighbor_learnability_v12_20261013"
BASE = ROOT / "data/synthetic" / STEM
LOGS = ROOT / "logs/preliminary" / STEM
CONFIG = ROOT / "data/configs" / (STEM + ".json")
old = previous.old
SOURCES = (
    "src/analysis/neighbor_selection_diagnostic_v12.py",
    "src/analysis/neighbor_selection_policy_v12.py",
    "src/analysis/neighbor_selection_report_v12.py",
    "src/experiments/neighbor_selection_execution_v12.py",
    "src/experiments/neighbor_selection_validation_v12.py",
)


def validate_batch(batch):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", batch):
        raise ValueError("unsafe batch id")
    return BASE / "batches" / batch


def verify_inputs():
    inputs = old.read_json(BASE / "stage1/protected_input_sha256.json")
    for path, digest in inputs.items():
        if old.file_hash(path) != digest:
            raise RuntimeError("historical input changed: " + path)
    return inputs


def freeze():
    if CONFIG.exists() or (BASE / "design.lock.json").exists():
        raise FileExistsError("v12 design already frozen")
    verify_inputs()
    if not (BASE / "stage1/diagnosis.json").is_file():
        raise RuntimeError("complete read-only stage1 before freezing candidate")
    prior = previous.verify()
    seeds = [2026101321, 2026101322, 2026101323, 2026101324]
    cases = [{**copy.deepcopy(next(c for c in prior["development_cases"] if c["scene_id"] == scene)),
              "seed": seed, "split": "independent_v12_confirmation"}
             for scene in prior["geometry_ids"] for seed in seeds]
    errors = {str(s): {uid: (previous.previous.v9.v7.indexed_rng(s, 0, "v7_initial_error", uid).normal(size=15)
                      * np.sqrt([4]*3 + [1]*12)).tolist() for uid in previous.previous.v9.v7.UIDS} for s in seeds}
    spec = {key: copy.deepcopy(prior[key]) for key in
            ("startup_steps", "motion_vo_chi_square_limit", "dt_s", "horizon_steps", "decision_steps",
             "measurement_budget", "message_ttl_s", "pair_period_s", "geometry_ids", "safety", "gates", "legacy_proxy_gate")}
    spec.update(schema="neighbor_learnability_v12", development_cases=prior["development_cases"],
                confirmation_cases=cases, development_seeds=prior["development_seeds"], confirmation_seeds=seeds,
                initial_errors_by_seed={**prior["initial_errors_by_seed"], **errors},
                policies=["none", "random_peer", "greedy", "minimal_greedy"], cf_steps=list(previous.execution.CF_STEPS),
                physical_step_cap=7680, cf_local_state_cap=1152, cf_range_call_cap=192,
                online_clone_cap=3712, phase_wall_cap_s=900,
                phase_caps={"develop": {"physical": 3840}, "confirm": {"physical": 3840}, "counterfactual": {"physical": 0}},
                proxy={"request_penalty_m2": .01, "none_score": 0,
                       "formula": "current original-CI cloned position trace decrease in m2 minus .01 per request",
                       "predecision_time": "(step-1)*dt; no post-local states or actual new range",
                       "fixed_kernel": "original v7 20 weights .05,.10,...,1",
                       "tie_stream": "v12_minimal_tie", "fit_or_tuning": False},
                state_extension="receiver-local self and received-peer directional position variances and unit directions; diagnostic encoding only",
                cost={"unit": "one unique measurement request, 512-byte-equivalent accounting only",
                      "counterfactual": "one hypothetical peer request unit even invalid/rejected; none zero; not a queue intervention",
                      "lambda_m2_per_unit": .01},
                cf_scope="post-local/pre-cooperation same frozen actual range/message/cache; three future recorded local packets; no queue/navigation replay",
                main_endpoint="full120step physical absolute and cluster-relative RMSE; stopped prefixes separate",
                safety_transition="unchanged partial_gnss_validation_gate_v10.physical_safety(expected=32) plus native V10 watchdog; no accuracy prerequisite",
                training_allowed=False, training_ready=False,
                independent_unit="four independent confirmation seeds; two geometries and all decision windows repeated blocks",
                count_table={"develop": {"episodes": 32, "steps": 3840, "random_records": 8, "cf_points": 48},
                             "confirm": {"episodes": 32, "steps": 3840, "random_records": 8, "cf_points": 48},
                             "total": {"episodes": 64, "steps": 7680, "cf_points": 96, "state_evaluations": 1152,
                                       "future_predicts": 864, "range_calls_max": 192}})
    old.write_json(CONFIG, spec)
    old.write_json(BASE / "design.lock.json", spec)
    old.write_json(BASE / "design.sha256.json", {"config_sha256": old.file_hash(CONFIG),
                   "source_sha256": {p: old.file_hash(ROOT / p) for p in SOURCES},
                   "stage1_sha256": {str(p.relative_to(ROOT)): old.file_hash(p) for p in (BASE / "stage1").rglob("*") if p.is_file()}})
    print("v12 frozen: 64 episodes / 7680 steps; 96 conditional checkpoints", flush=True)


def verify():
    verify_inputs()
    lock = old.read_json(BASE / "design.sha256.json")
    if old.file_hash(CONFIG) != lock["config_sha256"]:
        raise RuntimeError("v12 configuration changed")
    for path, digest in {**lock["source_sha256"], **lock["stage1_sha256"]}.items():
        if old.file_hash(ROOT / path) != digest:
            raise RuntimeError("v12 frozen source/evidence changed: " + path)
    release = old.read_json(BASE / "unit_release.lock.json")
    if not release["passed"] or release["source_sha256"] != lock["source_sha256"]:
        raise RuntimeError("v12 pure-unit release missing or outdated")
    return old.read_json(CONFIG)


def write_manifest(data, logs, content):
    content["artifact_sha256"] = {str(p.relative_to(ROOT)): old.file_hash(p)
                                 for folder in (data, logs) for p in folder.rglob("*") if p.is_file()}
    old.write_json(data / "manifest.json", content)


def checked_manifest(path):
    value = old.read_json(path)
    for name, digest in value["artifact_sha256"].items():
        if old.file_hash(ROOT / name) != digest:
            raise RuntimeError("phase evidence changed: " + name)
    return value


def start_phase(batch, phase, spec):
    base = validate_batch(batch)
    reservations = BASE / "reservations"
    reservations.mkdir(parents=True, exist_ok=True)
    allocated = [old.read_json(p) for p in reservations.glob("*.json")]
    amount = spec["phase_caps"][phase]["physical"]
    if sum(p["physical"] for p in allocated) + amount > spec["physical_step_cap"]:
        raise RuntimeError("global v12 physical step cap")
    with (reservations / (batch + "_" + phase + ".json")).open("x", encoding="utf-8") as stream:
        stream.write(old.encode({"batch": batch, "phase": phase, "physical": amount}) + "\n")
    data, logs = base / phase, LOGS / batch / phase
    data.mkdir(parents=True, exist_ok=False)
    logs.mkdir(parents=True, exist_ok=False)
    return data, logs, perf_counter() + spec["phase_wall_cap_s"]


def run_physical(batch, phase):
    from . import neighbor_selection_execution_v12 as execution
    spec = verify()
    if phase == "confirm" and not checked_manifest(validate_batch(batch) / "develop/manifest.json")["safety"]["passed"]:
        raise RuntimeError("frozen development safety predicate blocks confirmation")
    data, logs, deadline = start_phase(batch, phase, spec)
    summaries, failures, timing = [], [], []
    steps = clones = 0
    cases = spec["development_cases"] if phase == "develop" else spec["confirmation_cases"]
    with old._forbid_training_runtime():
        for case in cases:
            for policy in spec["policies"]:
                label = f"{case['scene_id']}_s{case['seed']}_{policy}"
                if perf_counter() >= deadline:
                    failures.append({"label": label, "reason": "frozen wall cap; not executed"})
                    continue
                try:
                    record = execution.physical(case, {**spec, "phase_deadline": deadline}, policy)
                    old.write_json(data / (label + "_record.json"), record)
                    steps += sum("states" in f.get("audit", {}) for f in record["frames"])
                    rows, events, audit = physical_rows(record, spec)
                    result = summarize(record, rows, events, audit, "original_ci", policy, "v12_physical")
                    result.update(study_phase=phase,
                                  request_windows=[f["step"] for f in record["frames"] if f.get("audit", {}).get("selection", "none") != "none"],
                                  selection_counts=dict(Counter(d["selected"] for d in record["v11_decisions"])),
                                  uav2_final_localization_error_m=float(np.linalg.norm(rows[-1]["target_position_error_m"])) if rows else None,
                                  decision_points_logged=len(record["v11_decisions"]))
                    result["score_clone_calls"] = sum(d["scoring"]["clone_calls"] for d in record["v11_decisions"])
                    samples = [d["v12_scoring"]["compute_ms"] for d in record["v11_decisions"]]
                    clones += result["score_clone_calls"]
                    timing.extend(samples)
                    summaries.append(result)
                    old.write_json(data / (label + "_estimator.json"), {"rows": rows, "events": events, "audit": audit})
                    write_csv(logs / (label + "_steps.csv"), rows)
                    write_csv(logs / (label + "_events.csv"), events)
                    write_csv(logs / (label + "_decisions.csv"), record["v11_decisions"])
                    print(phase, label, record["status"], len(record["frames"]), record["failure"], flush=True)
                except Exception as exc:
                    failures.append({"label": label, "reason": type(exc).__name__ + ": " + str(exc),
                                     "raw_record_retained": (data / (label + "_record.json")).exists()})
                    old.write_json(data / (label + "_failure.json"), failures[-1])
    safety = physical_safety(summaries, expected=32)
    p95 = float(np.percentile(timing, 95)) if timing else None
    if failures or clones > 1856 or steps > 3840 or any(r["decision_points_logged"] != 29 for r in summaries) or p95 is None or p95 >= 200:
        safety["passed"] = False
        safety["failures"].append("v12 completeness/clone/step/computation bound")
    write_csv(logs / "summary.csv", summaries)
    write_manifest(data, logs, {"study_phase": phase, "summaries": summaries, "failures": failures,
                               "safety": safety, "physical_steps": steps, "score_clone_calls": clones,
                               "score_p95_ms": p95, "optimizer_updates": 0})
    verify()
    print(old.encode({"phase": phase, "safety": safety, "physical_steps": steps}), flush=True)


def run_counterfactual(batch):
    spec = verify()
    data, logs, deadline = start_phase(batch, "counterfactual", spec)
    results, failures = [], []
    states = calls = 0
    with old._forbid_training_runtime():
        for phase in ("develop", "confirm"):
            manifest_path = validate_batch(batch) / phase / "manifest.json"
            if not manifest_path.exists():
                failures.append({"phase": phase, "reason": "physical phase missing"})
                continue
            checked_manifest(manifest_path)
            cases = spec["development_cases"] if phase == "develop" else spec["confirmation_cases"]
            for case in cases:
                label = f"{case['scene_id']}_s{case['seed']}_random_peer"
                record = old.read_json(validate_batch(batch) / phase / (label + "_record.json"))
                for step in spec["cf_steps"]:
                    point = {"study_phase": phase, "scene_id": case["scene_id"], "seed": case["seed"], "step": step}
                    try:
                        if perf_counter() >= deadline:
                            raise RuntimeError("frozen counterfactual wall cap")
                        snapshot = next(s for s in record["v11_snapshots"] if s["step"] == step)
                        decision = next(d for d in record["v11_decisions"] if d["step"] == step)
                        result = cf.evaluate_point(record, snapshot, decision["scoring"], spec)
                        states += result["local_states"]
                        calls += result["relative_ci_calls"]
                        if states > 1152 or calls > 192:
                            raise RuntimeError("frozen counterfactual quota")
                        old.write_json(data / (phase + "_" + label + f"_step{step}.json"), {**point, **result})
                        results.append({**point, **{k: v for k, v in result.items() if k != "branches"}})
                    except Exception as exc:
                        failure = {**point, "reason": type(exc).__name__ + ": " + str(exc), "dual_eligible": False}
                        results.append(failure)
                        failures.append(failure)
                        old.write_json(data / (phase + "_" + label + f"_step{step}_failure.json"), failure)
                print(phase, label, "conditional counterfactual complete", flush=True)
    write_csv(logs / "points.csv", results)
    write_manifest(data, logs, {"results": results, "failures": failures, "local_states": states,
                               "relative_ci_calls": calls, "physical_steps": 0, "optimizer_updates": 0, "planned_points": 96})
    verify()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=("stage1", "freeze", "develop", "confirm", "counterfactual", "report"))
    parser.add_argument("--batch-id", default="research_001")
    args = parser.parse_args()
    if args.phase == "stage1":
        from ..analysis.neighbor_selection_diagnostic_v12 import stage1
        stage1()
    elif args.phase == "freeze":
        freeze()
    elif args.phase in ("develop", "confirm"):
        run_physical(args.batch_id, args.phase)
    elif args.phase == "counterfactual":
        run_counterfactual(args.batch_id)
    else:
        from ..analysis.neighbor_selection_report_v12 import report
        report(args.batch_id)


if __name__ == "__main__":
    main()
