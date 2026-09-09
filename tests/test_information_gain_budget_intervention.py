"""Unit tests for the validation-only information-gain intervention."""

from __future__ import annotations

from copy import deepcopy

import numpy as np

from src.experiments.execution_uncertainty_proxy_audit import ProxyScales
from src.experiments.information_gain_budget_intervention import (
    InformationGainBudgetInterventionConfig,
    _online_candidate_rows,
    _resolve_base_seeds,
    intervene_information_gain_selection,
)


UAV_IDS = ("uav_00", "uav_01", "uav_02")


def _scales() -> ProxyScales:
    return ProxyScales(
        measurement_std_median=0.1,
        neighbor_covariance_trace_q95_m2=3.0,
        delay_q75_s=0.01,
        age_q75_s=1.0,
        queue_q75_packets=5.0,
        rssi_q05_dbm=-90.0,
        rssi_q95_dbm=-40.0,
        snr_q05_db=0.0,
        snr_q95_db=20.0,
        ig_q05=0.0,
        ig_q95=1.0,
    )


def _observation() -> dict[str, object]:
    estimate = np.zeros(15, dtype=float)
    covariance = np.eye(15, dtype=float)
    return {
        "estimate": estimate,
        "covariance": covariance,
        "action_mask": np.asarray([False, True, True], dtype=bool),
        "candidate_measurement_sources": ["uav_01", "uav_02"],
        "candidate_incoming_links": {
            "uav_01": {
                "rssi_dbm": -45.0,
                "snr_db": 18.0,
                "packet_loss": 0.01,
                "delay_s": 0.001,
                "aoi_s": 0.05,
                "queue_length_packets": 0.0,
                "link_state": "connected",
                "candidate_edge": 1.0,
            },
            "uav_02": {
                "rssi_dbm": -85.0,
                "snr_db": 2.0,
                "packet_loss": 0.4,
                "delay_s": 0.02,
                "aoi_s": 2.0,
                "queue_length_packets": 4.0,
                "link_state": "connected",
                "candidate_edge": 1.0,
            },
        },
        "neighbor_messages": {
            "uav_01": {
                "estimate": np.asarray([1.0, 0.0, 0.0] + [0.0] * 12),
                "covariance": np.eye(15, dtype=float) * 0.1,
            },
            "uav_02": {
                "estimate": np.asarray([0.0, 1.0, 0.0] + [0.0] * 12),
                "covariance": np.eye(15, dtype=float) * 0.1,
            },
        },
    }


def _recorded_actions() -> dict[str, dict[str, object]]:
    return {
        uav_id: {
            "measurement_uav_id": "none",
            "bandwidth": {"uav_01": 0.5, "uav_02": 0.5},
            "delta_v": np.zeros(3, dtype=float),
        }
        for uav_id in UAV_IDS
    }


def _context() -> dict[str, object]:
    return {
        "proxy_scales": _scales(),
        "measurement_std_m": 0.1,
        "age_variance_coefficient": 1.0,
        "budgets": {uav_id: 1 for uav_id in UAV_IDS},
    }


def test_online_proxy_is_finite_and_uses_visible_candidates() -> None:
    rows = _online_candidate_rows(_observation(), UAV_IDS, _scales(), 0.1, 1.0)

    assert [row["source_uav_id"] for row in rows] == ["uav_01", "uav_02"]
    assert all(np.isfinite(float(row["expected_information_gain_raw"])) for row in rows)
    assert all(float(row["expected_information_gain_raw"]) >= 0.0 for row in rows)
    assert float(rows[0]["expected_information_gain_raw"]) > float(rows[1]["expected_information_gain_raw"])


def test_unconstrained_selection_changes_only_measurement_choice() -> None:
    actions, metadata = intervene_information_gain_selection(
        _recorded_actions(),
        {uav_id: _observation() for uav_id in UAV_IDS},
        UAV_IDS,
        "unconstrained_information_gain",
        _context(),
        {uav_id: 0 for uav_id in UAV_IDS},
    )

    assert actions["uav_00"]["measurement_uav_id"] == "uav_01"
    assert actions["uav_00"]["bandwidth"] == {"uav_01": 0.5, "uav_02": 0.5}
    np.testing.assert_array_equal(actions["uav_00"]["delta_v"], np.zeros(3))
    assert metadata["uav_00"]["actor_visible_only"] is True
    assert metadata["uav_00"]["uses_truth_error"] is False


def test_budgeted_selection_stops_after_one_measurement() -> None:
    cursors = {uav_id: 0 for uav_id in UAV_IDS}
    observation_map = {uav_id: _observation() for uav_id in UAV_IDS}
    first, _ = intervene_information_gain_selection(
        _recorded_actions(), observation_map, UAV_IDS, "budgeted_information_gain", _context(), cursors
    )
    second, metadata = intervene_information_gain_selection(
        _recorded_actions(), observation_map, UAV_IDS, "budgeted_information_gain", _context(), cursors
    )

    assert first["uav_00"]["measurement_uav_id"] == "uav_01"
    assert second["uav_00"]["measurement_uav_id"] == "none"
    assert metadata["uav_00"]["budget_exhausted"] == 1
    assert cursors["uav_00"] == 1


def test_truth_fields_do_not_change_selection() -> None:
    clean = _observation()
    contaminated = deepcopy(clean)
    contaminated["true_position"] = np.asarray([999.0, 999.0, 999.0])
    contaminated["true_range"] = 999.0
    contaminated["los"] = 0
    clean_actions, _ = intervene_information_gain_selection(
        _recorded_actions(), {uav_id: clean for uav_id in UAV_IDS}, UAV_IDS,
        "unconstrained_information_gain", _context(), {uav_id: 0 for uav_id in UAV_IDS}
    )
    contaminated_actions, _ = intervene_information_gain_selection(
        _recorded_actions(), {uav_id: contaminated for uav_id in UAV_IDS}, UAV_IDS,
        "unconstrained_information_gain", _context(), {uav_id: 0 for uav_id in UAV_IDS}
    )

    assert clean_actions["uav_00"]["measurement_uav_id"] == contaminated_actions["uav_00"]["measurement_uav_id"]


def test_explicit_seed_set_resolves_without_legacy_hardcoding(tmp_path) -> None:
    seeds = (7071, 8080, 9099, 10118, 11127)
    for seed in seeds:
        (tmp_path / f"seed_{seed}" / "baseline_mappo" / "scale_3").mkdir(parents=True)

    resolved, roots = _resolve_base_seeds(tmp_path, "baseline_mappo", seeds)

    assert resolved == seeds
    assert set(roots) == set(seeds)


def test_seed_config_requires_five_unique_nonnegative_values(tmp_path) -> None:
    config = InformationGainBudgetInterventionConfig(
        reference_root=tmp_path,
        seeds=(11127, 7071, 10118, 8080, 9099),
        measurement_overhead_enabled=True,
    )

    assert config.seeds == (7071, 8080, 9099, 10118, 11127)
    assert config.measurement_overhead_enabled is True


def test_seed_config_rejects_duplicate_or_incomplete_values(tmp_path) -> None:
    for seeds in ((1, 2, 3, 4), (1, 2, 3, 4, 4), (-1, 2, 3, 4, 5)):
        try:
            InformationGainBudgetInterventionConfig(reference_root=tmp_path, seeds=seeds)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid seed list accepted: {seeds}")
