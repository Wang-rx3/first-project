from __future__ import annotations

import csv

import pytest

from src.experiments.ordered_v4_diagnostic import (
    _assert_allowed,
    _event_class,
    _episode_action_stats,
    _parse_ids,
)


def test_parse_update_episode_ids_accepts_json_and_python_literals() -> None:
    assert _parse_ids('["episode_000001", "episode_000002"]') == [
        "episode_000001",
        "episode_000002",
    ]
    assert _parse_ids("['episode_000003']") == ["episode_000003"]


def test_event_quality_classification_uses_signed_position_error_change() -> None:
    base = {"accepted": True, "true_position_error_before_m": 1.0}
    assert _event_class({**base, "true_position_error_after_m": 0.99}) == "beneficial"
    assert _event_class({**base, "true_position_error_after_m": 1.01}) == "harmful"
    assert _event_class({**base, "true_position_error_after_m": 1.0005}) == "neutral"
    assert _event_class({**base, "accepted": False, "true_position_error_after_m": 0.5}) == "rejected"


def test_allowed_path_rejects_test_split(tmp_path) -> None:
    root = tmp_path / "scale_3"
    path = root / "test" / "episode_500001" / "rewards.csv"
    with pytest.raises(RuntimeError, match="test data"):
        _assert_allowed(path, root=root)


def test_episode_measurement_count_comes_from_actions_not_per_window_reward_field(tmp_path) -> None:
    root = tmp_path / "scale_3"
    episode = root / "train" / "episode_000001"
    episode.mkdir(parents=True)
    with (episode / "actions.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["timestamp", "measurement_uav_id"])
        writer.writeheader()
        writer.writerows(
            [
                {"timestamp": "0.2", "measurement_uav_id": "uav_01"},
                {"timestamp": "0.2", "measurement_uav_id": "none"},
                {"timestamp": "0.2", "measurement_uav_id": "uav_02"},
                {"timestamp": "0.4", "measurement_uav_id": "none"},
            ]
        )
    stats = _episode_action_stats(root, "train")["episode_000001"]
    assert stats["actual_measurement_count"] == 2.0
    assert stats["actual_measurement_bytes"] == 1024.0
    assert stats["actual_measurement_rate"] == pytest.approx(2.0 / 6.0)
