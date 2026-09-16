import numpy as np

from src.experiments.single_uav_bc_v2_dagger import (
    _distance_bin,
    _select_correction_records,
)


def test_distance_bins_match_terminal_diagnostic_boundaries():
    assert _distance_bin(8.01) == ">8m"
    assert _distance_bin(8.0) == "8-5m"
    assert _distance_bin(5.0) == "5-3m"
    assert _distance_bin(3.0) == "3-1m"
    assert _distance_bin(1.0) == "<=1m"


def test_dagger_selection_is_capped_and_retains_distance_bins():
    records = [
        {"distance_bin": label, "index": index}
        for index, label in enumerate((">8m", "8-5m", "5-3m", "3-1m", "<=1m") * 4)
    ]
    selected = _select_correction_records(records, 10)
    assert len(selected) == 10
    assert {row["distance_bin"] for row in selected} == {">8m", "8-5m", "5-3m", "3-1m", "<=1m"}
    assert len({row["index"] for row in selected}) == 10


def test_dagger_selection_keeps_all_records_when_under_cap():
    records = [{"distance_bin": "3-1m", "index": index} for index in range(3)]
    selected = _select_correction_records(records, 10)
    assert selected == records
