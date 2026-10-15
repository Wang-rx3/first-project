"""Pure cost-shortcut decomposition tests, no filter or physical update."""
from src.analysis.neighbor_selection_conclusions_v13 import stratify


def point(gain0=0., gain1=0., cost0=1., cost1=3.):
    return {"study_phase": "confirm", "seed": 1, "dual_eligible": True,
            "v13_score_difference": gain0 - gain1 - .01 * (cost0 - cost1),
            "norm_labels": {"uav_00": {"current_norm_gain_m": gain0, "four_norm_gain_m": gain0, "cost_unit": cost0, "norm_utility_m": gain0 - .01 * cost0},
                            "uav_01": {"current_norm_gain_m": gain1, "four_norm_gain_m": gain1, "cost_unit": cost1, "norm_utility_m": gain1 - .01 * cost1}}}


def test_pure_cost_differences_do_not_create_raw_localization_samples():
    result = stratify([point(), point(cost0=2, cost1=1), point(cost0=1, cost1=2)])
    raw = next(x for x in result if x["seed"] == "all" and x["study_phase"] == "confirm" and x["label"] == "four_norm_gain_m" and x["stratum"] != "all_dual")
    net = next(x for x in result if x["seed"] == "all" and x["study_phase"] == "confirm" and x["label"] == "norm_utility_m" and x["stratum"] != "all_dual")
    assert raw["n"] == 0 and raw["rank_accuracy"] is None
    assert net["n"] == 3 and net["rank_accuracy"] == 1
    assert net["zero_gain_both_neighbors"] == 3


def test_remove_cost_before_evaluating_raw_trace_score():
    result = stratify([point(.2, .1), point(.1, .2), point(.5, 0)])
    raw = next(x for x in result if x["seed"] == "all" and x["study_phase"] == "confirm" and x["label"] == "four_norm_gain_m" and x["stratum"] != "all_dual")
    assert raw["n"] == 3 and raw["root_trace_only_rank_accuracy"] == 1
