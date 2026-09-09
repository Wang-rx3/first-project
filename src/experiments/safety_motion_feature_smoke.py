"""Single-seed smoke experiment for dynamic-obstacle motion observations.

The smoke is intentionally small and diagnostic.  It trains two fresh
three-UAV policies on one frozen safety scene with the extended 41-dimensional
node observation, then checks that the episode contract, budget wiring, and
finite-value invariants remain intact.  It is not a performance evaluation.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from ..schema import EPISODE_SCHEMA_VERSION, validate_episode_benchmark
from ..synthetic.episode_generator import _write_csv, _write_json
from .formal import FormalExperimentConfig, run_preliminary


BRANCHES = ("baseline_mappo", "budgeted_quality_aware_mappo")


@dataclass(frozen=True)
class SafetyMotionFeatureSmokeConfig:
    """Configuration for the explicit-scene 41-dimensional observation smoke."""

    scenario_path: Path
    output_dir: Path = Path("data/synthetic/safety_motion_feature_smoke_seed2026")
    logs_output: Path = Path("logs/preliminary/safety_motion_feature_smoke_seed2026")
    plots_output: Path = Path("plots/preliminary/safety_motion_feature_smoke_seed2026")
    uav_count: int = 3
    seed: int = 2026
    max_steps: int = 50
    warmup_episodes: int = 2
    train_episodes: int = 5
    validation_episodes: int = 5
    test_episodes: int = 5
    measurement_overhead_enabled: bool = True
    quality_logit_penalty: float = 1.0
    dynamic_obstacle_swept_projection_enabled: bool = False
    device: str = "cpu"
    force: bool = False
    progress: bool = True
    command_line: str = ""

    def __post_init__(self) -> None:
        if self.uav_count != 3:
            raise ValueError("safety motion-feature smoke is fixed to three UAVs")
        if self.device != "cpu":
            raise ValueError("safety motion-feature smoke currently supports device='cpu' only")
        if not Path(self.scenario_path).is_file():
            raise ValueError(f"scenario_path does not exist or is not a file: {self.scenario_path}")
        if min(
            self.max_steps,
            self.warmup_episodes,
            self.train_episodes,
            self.validation_episodes,
            self.test_episodes,
        ) < 1:
            raise ValueError("steps and all episode counts must be positive")
        if not math.isfinite(float(self.quality_logit_penalty)) or self.quality_logit_penalty < 0.0:
            raise ValueError("quality_logit_penalty must be finite and non-negative")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _number(row: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = float(row.get(key, default))
    except (TypeError, ValueError):
        return float(default)
    return value if math.isfinite(value) else float(default)


def _episode_directories(scale_root: Path, split: str) -> list[Path]:
    return sorted(
        path for path in (scale_root / split).glob("episode_*") if path.is_dir()
    )


def _measurement_budgets(scale_root: Path, uav_count: int) -> dict[str, int]:
    """Fit one per-UAV budget from baseline training episodes only."""

    episode_counts: list[dict[str, int]] = []
    for episode in _episode_directories(scale_root, "train"):
        counts = {f"uav_{index:02d}": 0 for index in range(uav_count)}
        for row in _read_csv(episode / "actions.csv"):
            uav_id = str(row.get("uav_id", ""))
            if uav_id in counts and str(row.get("measurement_uav_id", "none")) != "none":
                counts[uav_id] += 1
        episode_counts.append(counts)
    if not episode_counts:
        raise RuntimeError(f"baseline training episodes are missing under {scale_root}")
    return {
        uav_id: int(round(float(np.mean([counts[uav_id] for counts in episode_counts]))))
        for uav_id in episode_counts[0]
    }


def _finite_csv_fields(rows: Iterable[Mapping[str, Any]], fields: Iterable[str]) -> bool:
    for row in rows:
        for field in fields:
            try:
                value = float(row[field])
            except (KeyError, TypeError, ValueError):
                return False
            if not math.isfinite(value):
                return False
    return True


def _graph_feature_check(scale_root: Path) -> dict[str, Any]:
    """Check that serialized graph nodes use the new eight-value aggregate."""

    rows: list[dict[str, str]] = []
    for episode in _episode_directories(scale_root, "train") + _episode_directories(
        scale_root, "validation"
    ) + _episode_directories(scale_root, "test"):
        path = episode / "graph_features.csv"
        if path.exists():
            rows.extend(_read_csv(path))
    dimensions: set[int] = set()
    dynamic_valid_values: list[float] = []
    finite = True
    for row in rows:
        try:
            node_features = np.asarray(json.loads(row.get("node_features_json", "[]")), dtype=float)
        except (TypeError, ValueError, json.JSONDecodeError):
            finite = False
            continue
        if node_features.ndim != 2 or node_features.shape[1] < 1:
            finite = False
            continue
        dimensions.add(int(node_features.shape[1]))
        finite = finite and bool(np.isfinite(node_features).all())
        if node_features.shape[1] >= 41:
            dynamic_valid_values.extend(node_features[:, 40].tolist())
    return {
        "graph_feature_rows": len(rows),
        "node_feature_dimensions": sorted(dimensions),
        "node_feature_dim_41": dimensions == {41},
        "dynamic_motion_valid_values_finite": bool(
            dynamic_valid_values and np.isfinite(np.asarray(dynamic_valid_values, dtype=float)).all()
        ),
        "dynamic_motion_valid_observed": bool(
            dynamic_valid_values
            and np.any(np.asarray(dynamic_valid_values, dtype=float) > 0.5)
        ),
        "all_graph_features_finite": finite,
    }


def _branch_validation(
    branch_root: Path,
    *,
    expected_counts: Mapping[str, int],
) -> dict[str, Any]:
    scale_root = branch_root / "scale_3"
    report = validate_episode_benchmark(scale_root)
    integration_path = scale_root / "checks" / "integration_summary.json"
    integration: dict[str, Any] = {}
    if integration_path.exists():
        integration = json.loads(integration_path.read_text(encoding="utf-8"))
    counts = {split: len(_episode_directories(scale_root, split)) for split in ("train", "validation", "test")}
    metrics_path = scale_root / "diagnostics" / "episode_metrics.csv"
    metric_rows = _read_csv(metrics_path) if metrics_path.exists() else []
    finite = _finite_csv_fields(
        metric_rows,
        ("mean_reward", "j_loc", "j_com", "position_rmse_m", "covariance_position_trace"),
    )
    validation_rows = [row for row in metric_rows if row.get("split") == "validation"]
    graph_check = _graph_feature_check(scale_root)
    return {
        "branch": branch_root.name,
        "scale_root": str(scale_root),
        "episode_validation_ok": bool(report.get("ok", False)),
        "episode_validation_report": report,
        "episode_counts": counts,
        "expected_episode_counts": dict(expected_counts),
        "episode_counts_match": counts == dict(expected_counts),
        "integration_node_feature_dim": integration.get("node_feature_dim"),
        "integration_dynamic_motion_features": bool(
            integration.get("dynamic_obstacle_motion_features_enabled", False)
        ),
        "integration_measurement_overhead_enabled": bool(
            integration.get("measurement_overhead_enabled", False)
        ),
        "all_metric_fields_finite": finite,
        "validation_rows": validation_rows,
        "graph_feature_check": graph_check,
    }


def _write_validation_outputs(
    logs_root: Path,
    branch_reports: Mapping[str, Mapping[str, Any]],
) -> None:
    validation_rows: list[dict[str, Any]] = []
    branch_rows: list[dict[str, Any]] = []
    for branch, report in branch_reports.items():
        for row in report.get("validation_rows", []):
            validation_rows.append({"branch": branch, **row})
        rows = report.get("validation_rows", [])
        branch_rows.append(
            {
                "branch": branch,
                "node_feature_dim": report.get("integration_node_feature_dim", 0),
                "dynamic_motion_features_enabled": int(
                    report.get("integration_dynamic_motion_features", False)
                ),
                "episode_validation_ok": int(report.get("episode_validation_ok", False)),
                "episode_counts_match": int(report.get("episode_counts_match", False)),
                "all_metric_fields_finite": int(report.get("all_metric_fields_finite", False)),
                "graph_feature_dim_41": int(
                    report.get("graph_feature_check", {}).get("node_feature_dim_41", False)
                ),
                "dynamic_motion_valid_observed": int(
                    report.get("graph_feature_check", {}).get(
                        "dynamic_motion_valid_observed", False
                    )
                ),
                "validation_mean_reward": float(
                    np.mean([_number(row, "mean_reward") for row in rows]) if rows else 0.0
                ),
                "validation_position_rmse_m": float(
                    np.mean([_number(row, "position_rmse_m") for row in rows]) if rows else 0.0
                ),
                "validation_j_loc": float(
                    np.mean([_number(row, "j_loc") for row in rows]) if rows else 0.0
                ),
            }
        )
    _write_csv(
        logs_root / "validation_summary.csv",
        [
            "branch",
            "episode_id",
            "split",
            "phase",
            "seed",
            "mean_reward",
            "j_loc",
            "j_com",
            "position_rmse_m",
            "covariance_position_trace",
            "packet_loss",
            "delay_s",
            "aoi_s",
            "connectivity",
            "energy_proxy",
            "collision_rate",
        ],
        validation_rows,
    )
    _write_csv(
        logs_root / "branch_summary.csv",
        list(branch_rows[0]) if branch_rows else ["branch"],
        branch_rows,
    )


def _plot_smoke(plots_root: Path, branch_reports: Mapping[str, Mapping[str, Any]]) -> str:
    import matplotlib.pyplot as plt

    labels = list(branch_reports)
    x = np.arange(len(labels), dtype=float)
    rmse = [
        float(np.mean([_number(row, "position_rmse_m") for row in branch_reports[label]["validation_rows"]]))
        if branch_reports[label]["validation_rows"]
        else 0.0
        for label in labels
    ]
    rewards = [
        float(np.mean([_number(row, "mean_reward") for row in branch_reports[label]["validation_rows"]]))
        if branch_reports[label]["validation_rows"]
        else 0.0
        for label in labels
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
    axes[0].bar(x, rmse, color=("#3776d2", "#d62728"))
    axes[0].set_title("Validation position RMSE")
    axes[0].set_ylabel("RMSE (m)")
    axes[1].bar(x, rewards, color=("#3776d2", "#d62728"))
    axes[1].set_title("Validation mean team reward")
    axes[1].set_ylabel("Reward")
    for axis in axes:
        axis.set_xticks(x, [label.replace("_", "\n") for label in labels])
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("Safety motion-feature smoke (interface check; not a performance claim)")
    path = plots_root / "safety_motion_feature_smoke.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return str(path)


def run_safety_motion_feature_smoke(
    config: SafetyMotionFeatureSmokeConfig,
) -> dict[str, Any]:
    """Train and validate the two fresh 41-dimensional smoke branches."""

    output_root = Path(config.output_dir)
    logs_root = Path(config.logs_output)
    plots_root = Path(config.plots_output)
    if output_root.exists() and any(output_root.iterdir()) and not config.force:
        raise FileExistsError(f"smoke output is not empty: {output_root}; use --force")
    if logs_root.exists() and any(logs_root.iterdir()) and not config.force:
        raise FileExistsError(f"smoke logs are not empty: {logs_root}; use --force")
    if plots_root.exists() and any(plots_root.iterdir()) and not config.force:
        raise FileExistsError(f"smoke plots are not empty: {plots_root}; use --force")

    common = dict(
        uav_count=config.uav_count,
        seed=config.seed,
        gnss_mode="denied",
        max_steps=config.max_steps,
        warmup_episodes=config.warmup_episodes,
        train_episodes=config.train_episodes,
        validation_episodes=config.validation_episodes,
        test_episodes=config.test_episodes,
        device=config.device,
        force=config.force,
        scenario_path=Path(config.scenario_path),
        dynamic_obstacle_motion_features_enabled=True,
        dynamic_obstacle_swept_projection_enabled=(
            config.dynamic_obstacle_swept_projection_enabled
        ),
        measurement_overhead_enabled=config.measurement_overhead_enabled,
        command_line=config.command_line,
        progress=config.progress,
    )

    branch_reports: dict[str, dict[str, Any]] = {}
    baseline_root = output_root / "baseline_mappo"
    _progress(config, "training baseline_mappo from scratch")
    run_preliminary(
        FormalExperimentConfig(
            output_dir=baseline_root,
            mappo_variant="baseline_mappo",
            measurement_quality_logit_penalty=0.0,
            measurement_budget_enabled=False,
            **common,
        )
    )
    budgets = _measurement_budgets(baseline_root / "scale_3", config.uav_count)
    _progress(config, f"baseline train-derived measurement budgets={budgets}")

    budgeted_root = output_root / "budgeted_quality_aware_mappo"
    _progress(config, "training budgeted_quality_aware_mappo from scratch")
    run_preliminary(
        FormalExperimentConfig(
            output_dir=budgeted_root,
            mappo_variant="budgeted_quality_aware_mappo",
            measurement_quality_logit_penalty=config.quality_logit_penalty,
            measurement_budget_enabled=True,
            measurement_budget_by_uav=budgets,
            **common,
        )
    )

    expected_counts = {
        "train": config.train_episodes,
        "validation": config.validation_episodes,
        "test": config.test_episodes,
    }
    for branch in BRANCHES:
        branch_reports[branch] = _branch_validation(
            output_root / branch,
            expected_counts=expected_counts,
        )
    _write_validation_outputs(logs_root, branch_reports)
    plot_path = _plot_smoke(plots_root, branch_reports)

    finite = all(
        bool(report["all_metric_fields_finite"])
        and bool(report["graph_feature_check"]["all_graph_features_finite"])
        and bool(report["graph_feature_check"]["dynamic_motion_valid_values_finite"])
        for report in branch_reports.values()
    )
    structural = all(
        bool(report["episode_validation_ok"])
        and bool(report["episode_counts_match"])
        and report["integration_node_feature_dim"] == 41
        and bool(report["integration_dynamic_motion_features"])
        and bool(report["graph_feature_check"]["node_feature_dim_41"])
        and bool(report["graph_feature_check"]["dynamic_motion_valid_observed"])
        for report in branch_reports.values()
    )
    smoke_summary = {
        "experiment": "safety_motion_feature_smoke",
        "schema_version": EPISODE_SCHEMA_VERSION,
        "interface_only": True,
        "scenario_path": str(Path(config.scenario_path).resolve()),
        "output_dir": str(output_root.resolve()),
        "logs_output": str(logs_root.resolve()),
        "plots_output": str(plots_root.resolve()),
        "branches": list(BRANCHES),
        "measurement_overhead_enabled": bool(config.measurement_overhead_enabled),
        "quality_logit_penalty": float(config.quality_logit_penalty),
        "dynamic_obstacle_swept_projection_enabled": bool(
            config.dynamic_obstacle_swept_projection_enabled
        ),
        "measurement_budgets_from_baseline_train": budgets,
        "branch_reports": branch_reports,
        "plot": plot_path,
    }
    _write_json(logs_root / "smoke_summary.json", smoke_summary)
    audit_report = {
        "audit_ok": bool(structural and finite),
        "structural_checks": {
            "both_branches_trained_from_scratch": True,
            "node_feature_dim_41": structural,
            "dynamic_motion_feature_observed": structural,
            "episode_v1_3": all(
                bool(report["episode_validation_ok"]) for report in branch_reports.values()
            ),
            "expected_phase_counts": all(
                bool(report["episode_counts_match"]) for report in branch_reports.values()
            ),
        },
        "finite_checks": {
            "episode_metrics": finite,
            "graph_features": all(
                bool(report["graph_feature_check"]["all_graph_features_finite"])
                for report in branch_reports.values()
            ),
            "dynamic_motion_valid": all(
                bool(report["graph_feature_check"]["dynamic_motion_valid_values_finite"])
                for report in branch_reports.values()
            ),
        },
        "budget_source": "baseline_mappo/train only",
        "test_role": "generated and schema-validated; not used for fitting or performance claims",
        "physical_model_unchanged": True,
        "claim_boundary": "Smoke validates the 41-dimensional observation interface only; it does not establish safety performance improvement.",
        "branches": branch_reports,
    }
    _write_json(logs_root / "audit_report.json", audit_report)
    _write_json(
        output_root / "manifest.json",
        {
            "experiment": "safety_motion_feature_smoke",
            "schema_version": EPISODE_SCHEMA_VERSION,
            "uav_count": config.uav_count,
            "seed": config.seed,
            "scenario_path": str(Path(config.scenario_path).resolve()),
            "node_feature_dim": 41,
            "dynamic_obstacle_motion_features_enabled": True,
            "branches": list(BRANCHES),
            "measurement_budgets_from_baseline_train": budgets,
            "logs_output": str(logs_root.resolve()),
            "plots_output": str(plots_root.resolve()),
            "interface_only": True,
        },
    )
    return {
        "audit_ok": bool(audit_report["audit_ok"]),
        "output_dir": str(output_root),
        "logs_output": str(logs_root),
        "plots_output": str(plots_root),
        "audit_report": str(logs_root / "audit_report.json"),
        "plot": plot_path,
        "measurement_budgets": budgets,
    }


def _progress(config: SafetyMotionFeatureSmokeConfig, message: str) -> None:
    if config.progress:
        print(f"[safety-motion-smoke] {message}", flush=True)


__all__ = [
    "SafetyMotionFeatureSmokeConfig",
    "run_safety_motion_feature_smoke",
]
