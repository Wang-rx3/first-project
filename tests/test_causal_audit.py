"""Tests for the episode-level action-to-KPI diagnostic audit."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from src.experiments.causal_audit import CausalAuditConfig, run_causal_audit


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_episode(scale: Path, *, seed: int, controller: str, loss_offset: float) -> None:
    episode = scale / "test" / f"episode_{seed}"
    episode.mkdir(parents=True, exist_ok=True)
    scenario = {
        "episode_id": f"episode_{seed}",
        "scenario_id": f"scenario_{seed}",
        "seed": seed,
        "split": "test",
        "reward_config": {
            "weights": {
                "uncertainty": 0.35,
                "loss": 0.20,
                "aoi": 0.15,
                "delay": 0.15,
                "connectivity": 0.15,
            }
        },
    }
    (episode / "scenario.json").write_text(json.dumps(scenario), encoding="utf-8")

    actions: list[dict[str, object]] = []
    links: list[dict[str, object]] = []
    rewards: list[dict[str, object]] = []
    for index, timestamp in enumerate((0.2, 0.4, 0.6), start=1):
        for uav_index in range(3):
            source = f"uav_{uav_index:02d}"
            target = f"uav_{(uav_index + 1) % 3:02d}"
            actions.append(
                {
                    "timestamp": timestamp,
                    "uav_id": source,
                    "measurement_uav_id": "none",
                    "bandwidth_allocations_json": json.dumps({target: 1.0}),
                    "delta_vx": 0.01 * index,
                    "delta_vy": 0.0,
                    "delta_vz": 0.0,
                    "action_clipped": 0,
                    "safety_projected": 0,
                }
            )
            # Deliberately leave the baseline's final window without any link
            # row.  The audit must retain that difficult window and apply the
            # configured worst-case communication values.
            if not (controller == "baseline" and index == 3):
                links.append(
                    {
                        "timestamp": timestamp,
                        "src_uav_id": source,
                        "dst_uav_id": target,
                        "distance": 10.0 + index,
                        "los": 1,
                        "rate_bps": 1_000_000.0,
                        "packet_loss": loss_offset + 0.01 * index,
                        "delay": 0.001 * index,
                        "aoi": 0.002 * index,
                        "queue_length": 0,
                        "potential_edge": 1,
                        "candidate_edge": 1,
                        "effective_edge": 1,
                    }
                )
        norm_unc = 0.2
        norm_loss = loss_offset + 0.01 * index
        norm_aoi = 0.02 * index
        norm_delay = 0.03 * index
        norm_connectivity = 1.0
        j_loc = 0.35 * norm_unc
        j_com = 0.20 * norm_loss + 0.15 * norm_aoi + 0.15 * norm_delay - 0.15
        joint = j_loc + j_com
        energy_penalty = 0.001
        for uav_index in range(3):
            rewards.append(
                {
                    "timestamp": timestamp,
                    "uav_id": f"uav_{uav_index:02d}",
                    "reward_total": -(joint + energy_penalty),
                    "reward_collision": 0.0,
                    "reward_energy": -energy_penalty,
                    "joint_objective": joint,
                    "normalized_uncertainty": norm_unc,
                    "normalized_packet_loss": norm_loss,
                    "normalized_aoi": norm_aoi,
                    "normalized_delay": norm_delay,
                    "normalized_connectivity": norm_connectivity,
                }
            )
    _write_csv(episode / "actions.csv", actions)
    _write_csv(episode / "links.csv", links)
    _write_csv(episode / "rewards.csv", rewards)


class CausalAuditTests(unittest.TestCase):
    def test_audit_closes_objective_and_uses_episode_level_pairs(self) -> None:
        with tempfile.TemporaryDirectory(prefix="causal_audit_") as temporary:
            root = Path(temporary)
            reference = root / "reference" / "scale_3"
            baseline = root / "baseline" / "scale_3"
            for scale in (reference, baseline):
                scale.mkdir(parents=True, exist_ok=True)
                (scale / "manifest.json").write_text(
                    json.dumps(
                        {"communication_parameters": {"max_delay_s": 1.0, "aoi_cap_s": 60.0}}
                    ),
                    encoding="utf-8",
                )
            for seed in (502026, 503035):
                _write_episode(reference, seed=seed, controller="mappo", loss_offset=0.01)
                _write_episode(baseline, seed=seed, controller="baseline", loss_offset=0.03)
            logs = root / "logs"
            plots = root / "plots"
            report = run_causal_audit(
                CausalAuditConfig(
                    reference_scale=reference,
                    baseline_scale=baseline,
                    logs_dir=logs,
                    plots_dir=plots,
                    bootstrap_samples=200,
                )
            )
            self.assertEqual(report["paired_episodes"], 2)
            self.assertEqual(report["window_rows"], 12)
            self.assertLess(report["objective_identity_max_abs_error"], 1e-10)
            self.assertLess(report["reward_identity_max_abs_error"], 1e-10)
            self.assertTrue((logs / "paired_episode_summary.csv").exists())
            self.assertTrue((plots / "action_kpi_chain.png").exists())
            with (logs / "paired_episode_summary.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                summary = {row["metric"]: row for row in csv.DictReader(handle)}
            self.assertEqual(int(summary["packet_loss"]["paired_episodes"]), 2)
            self.assertGreater(float(summary["packet_loss"]["mean_baseline_minus_mappo"]), 0.0)


if __name__ == "__main__":
    unittest.main()
