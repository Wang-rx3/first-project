"""Run the predeclared seed-42 finite-horizon reward comparison."""
from pathlib import Path
import argparse
import json
from .mainline_baseline_comparison import MainlineBaselineComparisonConfig
from .mainline_corrected_stages import run_stage


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["smoke", "development"], required=True)
    parser.add_argument("--arm", choices=["report_direct", "existing"], required=True)
    parser.add_argument("--run-name", default="mainline_finite_horizon_seed42")
    parser.add_argument("--recover-plots", action="store_true")
    args = parser.parse_args()
    arm = "finite_horizon_report_direct" if args.arm == "report_direct" else "finite_horizon_existing_reward"
    name = args.run_name
    relative = Path(name) / arm / args.stage / "loc-progress-v2"
    reference = Path("data/synthetic") / name / "finite_horizon_report_direct" / args.stage / "loc-progress-v2/seed_42/baseline_mappo/scale_3"
    config = MainlineBaselineComparisonConfig(
        seed=42, protocol="core-framework-v4-delta-scale", experiment_arm="loc-progress-v2",
        stage=args.stage, finite_horizon_reward=args.arm,
        rule_reference_scale=str(reference) if args.arm == "existing" else None,
        output_dir=Path("data/synthetic") / relative,
        logs_dir=Path("logs/preliminary") / relative,
        plots_dir=Path("plots/preliminary") / relative,
    )
    if args.recover_plots:
        from .mainline_corrected_stages import summarize, audit_training, configuration_signature
        from .mainline_compact_figures import generate_compact_figures
        logs, plots = config.resolved_logs_dir, config.resolved_plots_dir
        path = logs / "audit_report.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("error", {}).get("type") != "KeyError" or report.get("error", {}).get("message") != "'delay'":
            raise ValueError("only the identified delay plot failure may be recovered")
        if report["configuration_signature"] != configuration_signature(config):
            raise ValueError("training implementation changed; recovery refused")
        if not all(s["ok"] for s in report["schema_reports"].values()):
            raise ValueError("schema failure")
        audit_training(report["method_paths"], config.counts, 4, learners=("baseline_mappo",))
        rows, isolation = summarize(report["method_paths"], config.counts[2], report["active_methods"])
        figures = generate_compact_figures(report["method_paths"], rows, logs/"figures_recovered", plots/"recovered",
            arm=arm, figure_profile="compact-proposed-v4")
        (logs/"plot_failure_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        report.update(audit_ok=True, status="complete", figures=figures, validation_isolation=isolation)
        report.pop("error")
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        (config.resolved_output_dir/"comparison_manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    else:
        run_stage(config)


if __name__ == "__main__":
    main()
