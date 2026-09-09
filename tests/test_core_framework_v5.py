"""Regression checks for the staged core-framework-v5 protocol."""
import json

from src.experiments.mainline_baseline_comparison import MainlineBaselineComparisonConfig
from src.experiments.mainline_corrected_stages import (
    _template,
    configuration_signature,
    find_compatible_smoke_report,
)
from src.experiments.formal import _make_environment


def test_v5_clean_reward_survives_collision_without_clipping(tmp_path):
    config = MainlineBaselineComparisonConfig(
        protocol="core-framework-v5", experiment_arm="report-clean", stage="smoke"
    )
    formal = _template(config, "baseline_mappo", tmp_path)
    env = _make_environment(formal, 42)
    env.reset(seed=42)
    actions = {u: {"measurement_uav_id": "none", "bandwidth": {},
                   "delta_v": [0, 0, 0]} for u in env.uav_ids}
    execution = {u: {"delta_v_applied": [0, 0, 0]} for u in env.uav_ids}
    report, info = env._rewards(1, actions, execution)
    assert min(report.values()) < -1.0
    assert report == info["learning_rewards"]
    assert formal.learning_potential_coef == formal.loc_progress_coef == 0.0


def test_v5_development_gate_accepts_a_current_suffixed_smoke_retry(tmp_path):
    root = tmp_path / "logs" / "v5"
    logs = root / "development" / "actual-load-rerun"
    config = MainlineBaselineComparisonConfig(
        protocol="core-framework-v5",
        experiment_arm="actual-load",
        figure_profile="compact-proposed-v4",
        stage="development",
        output_dir=tmp_path / "data" / "v5" / "development" / "actual-load-rerun",
        logs_dir=logs,
        plots_dir=tmp_path / "plots" / "v5" / "development" / "actual-load-rerun",
        progress=False,
    )
    report_path = root / "smoke" / "actual-load-rerun" / "audit_report.json"
    report_path.parent.mkdir(parents=True)
    report_path.write_text(
        json.dumps(
            {
                "audit_ok": True,
                "stage": "smoke",
                "arm": "actual-load",
                "configuration_signature": configuration_signature(config),
            }
        ),
        encoding="utf-8",
    )
    match = find_compatible_smoke_report(config)
    assert match is not None
    assert match[0] == report_path
