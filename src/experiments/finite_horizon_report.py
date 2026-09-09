"""Validation-only, paired finite-horizon comparison; no test data access."""
from pathlib import Path
import argparse
import json
import numpy as np
from .baseline import _read_json, _read_csv
from .mainline_baseline_comparison import _write_table
from .mainline_figures_v2 import SourceGuard
from .mainline_corrected_stages import common_prefix_sensitivity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="mainline_finite_horizon_seed42/verified")
    args = parser.parse_args()
    root = Path("logs/preliminary")/args.run_name
    out = root/"paired_comparison"
    if out.exists():
        raise FileExistsError(out)
    reports, rows, checks, training = {}, {}, {}, {}
    for arm in ("existing_reward", "report_direct"):
        folder = root/f"finite_horizon_{arm}"/"development/loc-progress-v2"
        report = _read_json(folder/"audit_report.json")
        if not report["audit_ok"]:
            raise ValueError(f"failed development: {arm}")
        reports[arm] = report
        rows[arm] = [r for r in _read_csv(folder/"validation_episode_metrics.csv") if r["method"] == "baseline_mappo"]
        scale = Path(report["method_paths"]["baseline_mappo"])
        training[arm] = [json.loads(s) for s in (scale/"diagnostics/training_episodes.jsonl").read_text().splitlines() if s.strip()]
    sources = [Path(p) for r in reports.values() for p in r["method_paths"].values()]
    out.mkdir(parents=True)
    with SourceGuard(sources, [out]) as guard:
        metrics = ("position_rmse_norm_m", "position_p95_norm_m", "actual_measurement_bytes", "j_com",
                   "mean_reward", "harmful_fraction", "accepted_updates_per_window", "update_coverage_any", "windows")
        paired = []
        for left, right in zip(rows["existing_reward"], rows["report_direct"]):
            if left["episode_id"] != right["episode_id"] or left["scenario_generation_hash"] != right["scenario_generation_hash"]:
                raise ValueError("unpaired validation scenes")
            row = {"episode_id": left["episode_id"]}
            for field in metrics:
                try:
                    row[field] = float(right[field])-float(left[field])
                except ValueError:
                    row[field] = "NA"
            paired.append(row)
        _write_table(out/"paired_episode_differences.csv", paired)
        averages = {}
        for arm in rows:
            averages[arm] = {}
            for field in metrics:
                values = [float(r[field]) for r in rows[arm] if r[field] not in ("NA", "", "nan")]
                averages[arm][field] = float(np.mean(values)) if values else None
            averages[arm]["collision_episodes"] = sum(str(r["collision_occurred"]).lower() in ("true", "1", "1.0") for r in rows[arm])
        a, b = averages["existing_reward"], averages["report_direct"]
        checks.update(rmse_lower=b[metrics[0]] < a[metrics[0]], p95_not_higher=b[metrics[1]] <= a[metrics[1]],
            payload_within_five_percent=b[metrics[2]] <= 1.05*a[metrics[2]], jcom_not_higher=b["j_com"] <= a["j_com"],
            collision_not_higher=b["collision_episodes"] <= a["collision_episodes"],
            harmful_not_higher=(b["harmful_fraction"] is not None and a["harmful_fraction"] is not None and b["harmful_fraction"] <= a["harmful_fraction"]))
        scales = {arm: Path(r["method_paths"]["baseline_mappo"]) for arm,r in reports.items()}
        manifests = {arm:_read_json(p/"manifest.json") for arm,p in scales.items()}
        checks["initial_parameters_equal"] = manifests["existing_reward"]["parameter_initialization"] == manifests["report_direct"]["parameter_initialization"]
        norms = {arm:_read_json(p/"normalization_stats.json") for arm,p in scales.items()}
        checks["shared_scales_equal"] = all(norms["existing_reward"][k] == norms["report_direct"][k] for k in ("metric_scales", "normalizers", "central_normalizer"))
        monitors = {}
        for arm,p in scales.items():
            history = [json.loads(s) for s in (p/"diagnostics/development_monitor.jsonl").read_text().splitlines() if s.strip()]
            monitors[arm] = {str(t):float(np.mean([r["position_rmse_norm_m"] for r in history if r["completed_training_episodes"]==t])) for t in (0,100)}
        checks["monitor_rmse_not_worse"] = monitors["report_direct"]["100"] <= monitors["report_direct"]["0"]
        checks["direct_reward_closure"] = all(abs(r["cumulative_return"]-r["learning_cumulative_return"]) <= 1e-6 for r in training["report_direct"])
        for arm, report in reports.items():
            prefix = common_prefix_sensitivity(report["method_paths"], report["active_methods"])
            ids = sorted({r["episode_id"] for r in prefix})[:5]
            _write_table(out/f"{arm}_first_five_common_prefix.csv", [r for r in prefix if r["episode_id"] in ids])
        between = common_prefix_sensitivity(
            {"baseline_mappo": str(scales["report_direct"]), "existing_reward": str(scales["existing_reward"])},
            ("baseline_mappo", "existing_reward"))
        first_ids = sorted({r["episode_id"] for r in between})[:5]
        _write_table(out/"between_arms_first_five_common_prefix.csv", [r for r in between if r["episode_id"] in first_ids])
        hashes = guard.verify()
    effective = {arm: {key:m["config"][key] for key in (
        "episode_objective", "include_remaining_time", "bootstrap_time_limits", "learning_reward_mode",
        "learning_feedback_mode", "learning_potential_coef", "loc_progress_coef", "ppo_action_logprob_mode")}
        for arm,m in manifests.items()}
    result = dict(seed=42, stage="development", averages=averages, checks=checks, monitor_rmse=monitors,
        effective_configurations=effective, rule_training_reference=str(scales["report_direct"] / "train"),
        conclusion="observed_comprehensive_improvement" if all(checks.values()) else "not_all_improvement_criteria_met",
        test_access_attempts=guard.test_attempts, sources_unchanged=all(h["unchanged"] for h in hashes))
    (out/"comparison_report.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    (out/"README.md").write_text("# seed=42有限时域对照\n\n差值为直接奖励组减现有奖励组；每个点为配对validation场景，仅作单种子开发描述。\n\n"+json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
