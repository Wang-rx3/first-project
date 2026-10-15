from __future__ import annotations

import random
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from src.experiments import neighbor_selection_environment_v13 as v13
from src.simulation.environment import MessageRecord, SimulationConfig
from src.synthetic.channel_model import ChannelConfig, compute_link_metrics


def test_quality_field_boundaries_wrapping_and_sky_clear() -> None:
    # At the reference point phase is zero; x advances the phase directly.
    assert v13.quality_at([40.0, 125.0, 28.0])["level"] == "clear"
    assert v13.quality_at([45.0, 125.0, 28.0])["level"] == "mild"
    assert v13.quality_at([50.0, 125.0, 28.0])["level"] == "severe"
    assert v13.quality_at([55.0, 125.0, 28.0])["level"] == "mild"
    assert v13.quality_at([60.0, 125.0, 28.0])["phase"] == pytest.approx(0.0)
    assert v13.quality_at([40.0, 125.0, 51.0])["level"] == "clear"


def test_quality_offset_changes_only_field_phase() -> None:
    base = v13.quality_at([40.0, 125.0, 28.0], 0)
    shifted = v13.quality_at([40.0, 125.0, 28.0], 10)
    assert base["level"] == "clear"
    assert shifted["level"] == "severe"


def _channel_kwargs(rng: random.Random) -> dict[str, object]:
    return {
        "distance_m": 20.0,
        "los": True,
        "bandwidth_ratio": 0.5,
        "queue_length_packets": 0.0,
        "previous_aoi_s": 0.0,
        "dt_s": 0.1,
        "rng": rng,
        "config": ChannelConfig(),
        "max_comm_range_m": 75.0,
    }


def test_channel_disabled_is_native_and_preserves_rng_state() -> None:
    rng = random.Random(73)
    before = rng.getstate()
    native = compute_link_metrics(**_channel_kwargs(rng))
    native_state = rng.getstate()
    rng.setstate(before)
    adapted = v13.channel_metrics(_channel_kwargs(rng), 12.0, 0.42, enabled=False)
    assert adapted == native
    assert rng.getstate() == native_state


def test_channel_enabled_shifts_rssi_and_adds_wait_once() -> None:
    native = compute_link_metrics(**_channel_kwargs(random.Random(8)))
    degraded_base = v13.channel_metrics(_channel_kwargs(random.Random(8)), 6.0, 0.0)
    adapted = v13.channel_metrics(_channel_kwargs(random.Random(8)), 6.0, 0.22)
    assert adapted.rssi_dbm == pytest.approx(native.rssi_dbm - 6.0)
    assert adapted.raw_delay_s == pytest.approx(degraded_base.raw_delay_s + 0.22)
    assert adapted.delay_s == pytest.approx(adapted.raw_delay_s)


def test_channel_delay_overflow_keeps_candidate_but_drops_packet() -> None:
    config = ChannelConfig(max_delay_s=0.1, loss_beta0=-100.0)
    kwargs = _channel_kwargs(random.Random(4))
    kwargs["config"] = config
    kwargs["dt_s"] = 0.1
    metrics = v13.channel_metrics(kwargs, 0.0, 0.42)
    assert metrics.raw_delay_s > config.max_delay_s
    assert metrics.delay_s == config.max_delay_s
    assert metrics.packet_success == 0
    assert metrics.effective_edge == 0
    assert metrics.drop_reason == "v13_delay_limit"


def _bare_env(dynamic: bool = True) -> v13.DynamicEnv:
    env = v13.DynamicEnv(
        SimulationConfig(max_steps=20),
        dynamic_gnss=dynamic,
        dynamic_radio=dynamic,
        precheck_guard=False,
    )
    env.pending_messages = []
    env.arrived_messages = {}
    env.v13_packet_registry = {}
    env.v13_metadata_events = []
    env._v13_pre_gnss = {}
    env._v13_pre_payloads = {}
    return env


def test_metadata_ids_and_received_quality_require_actual_arrival_identity() -> None:
    env = _bare_env()
    env.timestep = 1
    env.snapshot.metrics = {
        ("uav_00", "uav_01"): SimpleNamespace(delay_s=0.2)
    }
    payload = {
        "uav_00": (np.zeros(15), np.eye(15)),
    }
    env._v13_pre_gnss = {"uav_00": {"gnss_std_m": 1.5, "report_time_s": 0.2}}
    env._enqueue_messages({("uav_00", "uav_01")}, payload)
    message = env.pending_messages[0]
    assert message.attr_v13_packet_id
    assert v13.received_quality(message) == {"gnss_std_m": None, "gnss_report_time_s": None, "packet_id": None}
    env.arrived_messages[(message.source, message.target)] = message
    quality = env.received_quality(message)
    assert quality["gnss_std_m"] == 1.5
    assert quality["gnss_report_time_s"] == 0.2
    duplicate = replace(message)
    assert env.received_quality(duplicate)["packet_id"] is None
    assert env.v13_metadata_events[0]["payload_bytes"] == 496
    assert env.v13_metadata_events[0]["quality_header_bytes"] == 16


def test_off_metadata_budget_has_no_quality_header() -> None:
    env = _bare_env(dynamic=False)
    env.timestep = 1
    env.snapshot.metrics = {("uav_00", "uav_01"): SimpleNamespace(delay_s=0.2)}
    env._enqueue_messages(
        {("uav_00", "uav_01")},
        {"uav_00": (np.zeros(15), np.eye(15))},
    )
    row = env.v13_metadata_events[0]
    assert row["payload_bytes"] == 512
    assert row["quality_header_bytes"] == 0
    assert row["payload_bytes"] + row["quality_header_bytes"] == 512


def test_ttl_boundary_uses_native_release_and_registry_cleanup() -> None:
    env = _bare_env()
    env.timestep = 0
    message = MessageRecord(
        source="uav_00",
        target="uav_01",
        state=np.zeros(15),
        covariance=np.eye(15),
        send_time_s=0.0,
        arrival_step=1,
        delay_s=0.2,
    )
    message.attr_v13_packet_id = ("uav_00", "uav_01", 0.0, 1, 1)
    message.v13_packet_envelope = tuple(message.attr_v13_packet_id)
    message._v13_owner = env
    env.pending_messages = [message]
    env.v13_packet_registry = {
        message.attr_v13_packet_id: {
            "message": message,
            "envelope": message.v13_packet_envelope,
            "gnss_std_m": 0.5,
            "gnss_report_time_s": 0.0,
        }
    }
    env._release_messages(1)
    assert env.arrived_messages[("uav_00", "uav_01")] is message
    assert v13.received_quality(message)["packet_id"] == message.attr_v13_packet_id
    env._release_messages(5)  # 1.0 s, beyond the native 0.6 s freshness limit.
    assert not env.arrived_messages
    assert not env.v13_packet_registry


def test_adapters_restore_original_method_after_exception_and_preserve_mro() -> None:
    original = v13.ORIGINALCLASS.step
    assert issubclass(v13.DynamicEnv, v13.prior.NeighborAuditEnv)
    with pytest.raises(RuntimeError, match="synthetic"):
        with v13.adapters({"quality_offset": 0}, {"dynamic_gnss": True}) as env_class:
            assert v13.ORIGINALCLASS.step is not original
            assert issubclass(env_class, v13.DynamicEnv)
            raise RuntimeError("synthetic")
    assert v13.ORIGINALCLASS.step is original


def test_prefix_guard_is_opt_in_to_explicit_prefix_spec() -> None:
    with v13.adapters({}, {"dynamic_gnss": False, "dynamic_radio": False}) as env_class:
        env = env_class(SimulationConfig(max_steps=10))
        assert env.v13_precheck_guard is False
    with v13.adapters({}, {"prefix_steps": 6}) as env_class:
        env = env_class(SimulationConfig(max_steps=10))
        assert env.v13_precheck_guard is True


def test_prefix_guard_allows_six_transitions_then_marks_seventh_call() -> None:
    env = _bare_env()
    env.v13_precheck_guard = True
    env.timestep = 6
    with pytest.raises(RuntimeError, match="v13_prefix_limit"):
        env.step({})
    assert env.v13_prefix_guard_calls == [6]
    assert env.v13_prefix_limit_marker["reason"] == "v13_prefix_limit"
