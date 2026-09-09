"""Behavioral tests for the pure-Python multi-UAV environment."""

from __future__ import annotations

import unittest

import numpy as np

from src.simulation.ekf import EKFConfig, LocalEKF, covariance_intersection
from src.simulation.environment import MessageRecord, MultiUAVJointEnv, SimulationConfig
from src.simulation.motion_model import MotionConfig, UAVKinematicState, apply_velocity_increment
from src.simulation.obstacle_model import (
    BoxObstacle,
    SphereObstacle,
    collision_event,
    line_of_sight,
    moving_obstacle_swept_segment_intersects,
)
from src.simulation.sensor_model import SensorConfig, sample_visual_odometry
from src.simulation.topology import TopologyConfig, build_topology, normalize_bandwidth_allocations
from src.synthetic.channel_model import ChannelConfig


def _state(x: float, y: float, z: float = 5.0) -> UAVKinematicState:
    return UAVKinematicState(
        position_m=np.array([x, y, z], dtype=float),
        velocity_mps=np.zeros(3),
        attitude_rad=np.zeros(3),
        acceleration_mps2=np.zeros(3),
        angular_velocity_rps=np.zeros(3),
    )


class SimulationEnvironmentTests(unittest.TestCase):
    def test_visual_odometry_uses_actual_window_displacement(self) -> None:
        state = UAVKinematicState(
            position_m=np.array([3.0, 5.0, 7.0]),
            velocity_mps=np.array([2.0, 0.0, 0.0]),
            attitude_rad=np.zeros(3),
            acceleration_mps2=np.array([1.0, 0.0, 0.0]),
            angular_velocity_rps=np.zeros(3),
        )
        measurement = sample_visual_odometry(
            state,
            0.2,
            np.random.default_rng(0),
            SensorConfig(vo_position_std_m=0.0),
            reference_position=np.array([2.0, 4.0, 6.0]),
        )
        self.assertEqual(measurement["valid"], 1)
        np.testing.assert_allclose(measurement["delta_position"], np.ones(3))

    def test_visual_odometry_updates_position_not_velocity(self) -> None:
        initial = np.zeros(15, dtype=float)
        initial[:3] = [2.0, 4.0, 6.0]
        initial[3:6] = [1.0, 0.0, 0.0]
        ekf = LocalEKF(initial, EKFConfig(vo_position_std_m=0.01))
        reference = ekf.position.copy()
        ekf.predict({"accel": np.zeros(3), "gyro": np.zeros(3)}, 0.2)
        ekf.update_visual_delta(
            np.array([0.2, 0.0, 0.0]),
            reference_position=reference,
            covariance=np.eye(3) * 1e-6,
            dt_s=0.2,
        )
        np.testing.assert_allclose(ekf.position, [2.2, 4.0, 6.0], atol=1e-8)
        self.assertTrue(np.all(np.isfinite(ekf.state)))
        self.assertTrue(np.all(np.isfinite(ekf.covariance)))

    def test_relative_visual_update_keeps_cross_covariance_uncertainty(self) -> None:
        initial = np.zeros(15, dtype=float)
        initial[:3] = [2.0, 4.0, 6.0]
        initial[3:6] = [1.0, 0.0, 0.0]
        ekf = LocalEKF(initial, EKFConfig(vo_position_std_m=0.01))
        previous_state = ekf.state.copy()
        previous_covariance = ekf.covariance.copy()
        transition = ekf.predict({"accel": np.zeros(3), "gyro": np.zeros(3)}, 0.2)
        self.assertTrue(
            ekf.update_visual_delta(
                np.array([0.2, 0.0, 0.0]),
                covariance=np.eye(3) * 1e-6,
                previous_state=previous_state,
                previous_covariance=previous_covariance,
                transition_matrix=transition,
                dt_s=0.2,
            )
        )
        self.assertGreater(float(np.trace(ekf.covariance[:3, :3])), 1.0)
        self.assertEqual(ekf.diagnostics_snapshot()["vo_update_accepted"], 1)
        self.assertTrue(np.all(np.linalg.eigvalsh(ekf.covariance) >= -1e-10))

    def test_relative_vo_constant_velocity_regression(self) -> None:
        initial = np.zeros(15, dtype=float)
        initial[:3] = [1.0, 2.0, 3.0]
        initial[3:6] = [1.0, -0.5, 0.25]
        config = EKFConfig(
            process_position_std_m=1e-6,
            process_velocity_std_mps=1e-6,
            vo_position_std_m=1e-6,
        )
        ekf = LocalEKF(initial, config)
        previous_state = ekf.state.copy()
        previous_covariance = ekf.covariance.copy()
        transition = ekf.predict({"accel": np.zeros(3), "gyro": np.zeros(3)}, 0.2)
        accepted = ekf.update_visual_delta(
            np.array([0.2, -0.1, 0.05]),
            covariance=np.eye(3) * 1e-12,
            previous_state=previous_state,
            previous_covariance=previous_covariance,
            transition_matrix=transition,
            dt_s=0.2,
            previous_time_s=0.0,
            current_time_s=0.2,
        )
        self.assertTrue(accepted)
        np.testing.assert_allclose(ekf.position, [1.2, 1.9, 3.05], atol=1e-8)
        self.assertTrue(np.all(np.linalg.eigvalsh(ekf.covariance) >= -1e-10))

    def test_relative_vo_acceleration_regression(self) -> None:
        initial = np.zeros(15, dtype=float)
        initial[:3] = [4.0, 5.0, 6.0]
        config = EKFConfig(
            process_position_std_m=1e-6,
            process_velocity_std_mps=1e-6,
            vo_position_std_m=1e-6,
        )
        ekf = LocalEKF(initial, config)
        previous_state = ekf.state.copy()
        previous_covariance = ekf.covariance.copy()
        transition = ekf.predict({"accel": np.array([1.0, 0.0, 0.0]), "gyro": np.zeros(3)}, 0.2)
        accepted = ekf.update_visual_delta(
            np.array([0.02, 0.0, 0.0]),
            covariance=np.eye(3) * 1e-12,
            previous_state=previous_state,
            previous_covariance=previous_covariance,
            transition_matrix=transition,
            dt_s=0.2,
            previous_time_s=0.0,
            current_time_s=0.2,
        )
        self.assertTrue(accepted)
        np.testing.assert_allclose(ekf.position, [4.02, 5.0, 6.0], atol=1e-8)
        np.testing.assert_allclose(ekf.velocity, [0.2, 0.0, 0.0], atol=1e-8)

    def test_relative_range_unknown_correlation_does_not_reduce_local_covariance(self) -> None:
        initial = np.zeros(15, dtype=float)
        initial[:3] = [0.0, 0.0, 5.0]
        ekf = LocalEKF(
            initial,
            EKFConfig(
                process_position_std_m=0.03,
                process_velocity_std_mps=0.08,
                uwb_range_std_m=0.1,
            ),
        )
        before = ekf.covariance.copy()
        accepted = ekf.update_relative_range(
            measured_range_m=10.02,
            neighbor_position=np.array([10.0, 0.0, 5.0]),
            neighbor_covariance=np.eye(3, dtype=float),
            measurement_variance=0.1**2,
        )
        self.assertTrue(accepted)
        self.assertGreaterEqual(
            float(np.trace(ekf.covariance)),
            float(np.trace(before)) - 1e-10,
        )
        self.assertTrue(np.all(np.isfinite(ekf.covariance)))
        self.assertTrue(np.all(np.linalg.eigvalsh(ekf.covariance) >= -1e-10))

    def test_uwb_update_period_must_be_positive(self) -> None:
        with self.assertRaises(ValueError):
            SensorConfig(uwb_update_period_s=0.0)

    def test_relative_vo_rejects_mismatched_time_window(self) -> None:
        ekf = LocalEKF(np.zeros(15, dtype=float))
        previous_state = ekf.state.copy()
        previous_covariance = ekf.covariance.copy()
        transition = ekf.predict({"accel": np.zeros(3), "gyro": np.zeros(3)}, 0.2)
        with self.assertRaises(ValueError):
            ekf.update_visual_delta(
                np.zeros(3),
                previous_state=previous_state,
                previous_covariance=previous_covariance,
                transition_matrix=transition,
                dt_s=0.2,
                previous_time_s=0.0,
                current_time_s=0.3,
            )

    def test_relative_vo_context_cannot_use_reference_truth(self) -> None:
        ekf = LocalEKF(np.zeros(15, dtype=float))
        previous_state = ekf.state.copy()
        previous_covariance = ekf.covariance.copy()
        transition = ekf.predict({"accel": np.zeros(3), "gyro": np.zeros(3)}, 0.2)
        with self.assertRaises(ValueError):
            ekf.update_visual_delta(
                np.zeros(3),
                previous_state=previous_state,
                previous_covariance=previous_covariance,
                transition_matrix=transition,
                reference_position=np.zeros(3),
                dt_s=0.2,
            )

    def test_constraint_projection_inflates_covariance_without_state_leak(self) -> None:
        initial = np.zeros(15, dtype=float)
        initial[:3] = [2.0, 4.0, 6.0]
        ekf = LocalEKF(initial, EKFConfig(projection_position_std_m=1.5, projection_velocity_std_mps=0.5))
        state_before = ekf.state.copy()
        covariance_before = ekf.covariance.copy()
        ekf.inflate_motion_uncertainty()
        np.testing.assert_allclose(ekf.state, state_before)
        self.assertGreater(
            float(np.trace(ekf.covariance[:3, :3])),
            float(np.trace(covariance_before[:3, :3])),
        )
        self.assertGreater(
            float(np.trace(ekf.covariance[3:6, 3:6])),
            float(np.trace(covariance_before[3:6, 3:6])),
        )
        self.assertTrue(np.all(np.linalg.eigvalsh(ekf.covariance) >= -1e-10))

    def test_safety_projection_keeps_position_velocity_and_imu_consistent(self) -> None:
        state = UAVKinematicState(
            position_m=np.array([10.0, 10.0, 10.0]),
            velocity_mps=np.array([4.0, 0.0, 0.0]),
            attitude_rad=np.zeros(3),
            acceleration_mps2=np.zeros(3),
            angular_velocity_rps=np.zeros(3),
        )
        config = MotionConfig(dt_s=0.2)
        next_state, info = apply_velocity_increment(
            state,
            [0.0, 0.0, 0.0],
            config,
            safety_projector=lambda _proposed: state.position_m.copy(),
        )
        self.assertTrue(info["safety_projected"])
        np.testing.assert_allclose(next_state.position_m, state.position_m)
        np.testing.assert_allclose(next_state.velocity_mps, np.zeros(3))
        np.testing.assert_allclose(
            next_state.acceleration_mps2,
            (next_state.velocity_mps - state.velocity_mps) / config.dt_s,
        )

    def test_one_step_moving_obstacle_swept_envelope_catches_post_update_risk(self) -> None:
        obstacle = SphereObstacle(
            "crossing_vehicle",
            np.array([0.0, -4.0, 10.0]),
            1.0,
            np.array([0.0, 20.0, 0.0]),
        )
        start = np.array([0.0, 0.0, 10.0])
        end = np.array([1.0, 0.0, 10.0])
        bounds_min = np.array([0.0, 0.0, 2.0])
        bounds_max = np.array([200.0, 200.0, 60.0])
        self.assertTrue(
            moving_obstacle_swept_segment_intersects(
                start,
                end,
                obstacle,
                0.2,
                bounds_min,
                bounds_max,
                2.0,
            )
        )
        # The pure geometry audit must not mutate the live obstacle velocity or
        # center while predicting the one-window envelope.
        np.testing.assert_allclose(obstacle.center_m, [0.0, -4.0, 10.0])
        np.testing.assert_allclose(obstacle.velocity_mps, [0.0, 20.0, 0.0])

    def test_swept_projection_is_opt_in_and_blocks_moving_obstacle(self) -> None:
        common = dict(uav_count=3, max_steps=5, gnss_mode="denied")
        legacy = MultiUAVJointEnv(SimulationConfig(**common))
        corrected = MultiUAVJointEnv(
            SimulationConfig(**common, dynamic_obstacle_swept_projection_enabled=True)
        )
        for env in (legacy, corrected):
            env.reset(seed=321)
            env.states["uav_00"].position_m[:] = [0.0, 0.0, 10.0]
            env.obstacles = [
                SphereObstacle(
                    "crossing_vehicle",
                    np.array([0.0, -4.0, 10.0]),
                    1.0,
                    np.array([0.0, 20.0, 0.0]),
                )
            ]
        proposed = np.array([1.0, 0.0, 10.0])
        legacy_position = legacy._safety_project("uav_00", proposed)
        corrected_position = corrected._safety_project("uav_00", proposed)
        np.testing.assert_allclose(legacy_position, proposed)
        np.testing.assert_allclose(corrected_position, [0.0, 0.0, 10.0])

    def test_ekf_position_projection_enforces_flight_volume(self) -> None:
        ekf = LocalEKF(np.zeros(15, dtype=float))
        ekf.state[:3] = [-2.0, 5.0, 100.0]
        changed = ekf.project_position_bounds(
            np.array([0.0, 0.0, 2.0]),
            np.array([20.0, 20.0, 60.0]),
        )
        self.assertTrue(changed)
        np.testing.assert_allclose(ekf.position, [0.0, 5.0, 60.0])
        self.assertTrue(np.all(np.linalg.eigvalsh(ekf.covariance) >= -1e-10))

    def test_denied_gnss_position_estimate_remains_close_to_motion(self) -> None:
        env = MultiUAVJointEnv(
            SimulationConfig(uav_count=3, max_steps=25, gnss_mode="denied")
        )
        env.reset(seed=123)
        action = {
            uav_id: {
                "measurement_uav_id": "none",
                "bandwidth": {
                    target: 1.0 / (len(env.uav_ids) - 1)
                    for target in env.uav_ids
                    if target != uav_id
                },
                "delta_v": [0.0, 0.0, 0.0],
            }
            for uav_id in env.uav_ids
        }
        maximum_error = 0.0
        for _ in range(25):
            env.step(action)
            for uav_id in env.uav_ids:
                estimate = env.ekfs[uav_id].position
                truth = env.states[uav_id].position_m
                self.assertTrue(np.all(np.isfinite(estimate)))
                maximum_error = max(maximum_error, float(np.linalg.norm(estimate - truth)))
        self.assertLess(maximum_error, 2.0)

    def test_collision_terminates_episode_once(self) -> None:
        env = MultiUAVJointEnv(
            SimulationConfig(uav_count=3, max_steps=10, gnss_mode="denied")
        )
        env.reset(seed=124)
        env.obstacles = [
            SphereObstacle(
                "forced_collision",
                env.states["uav_00"].position_m.copy(),
                1.0,
                np.zeros(3),
            )
        ]
        action = {
            uav_id: {
                "measurement_uav_id": "none",
                "bandwidth": {},
                "delta_v": [0.0, 0.0, 0.0],
            }
            for uav_id in env.uav_ids
        }
        _obs, rewards, terminated, truncated, info = env.step(action)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["termination_reason"], "collision")
        self.assertFalse(info["task_complete"])
        self.assertTrue(all(value <= -env.config.reward.collision_penalty for value in rewards.values()))

    def test_time_limit_is_truncation_not_task_completion(self) -> None:
        env = MultiUAVJointEnv(
            SimulationConfig(uav_count=3, max_steps=1, gnss_mode="denied")
        )
        env.reset(seed=125)
        action = {
            uav_id: {
                "measurement_uav_id": "none",
                "bandwidth": {},
                "delta_v": [0.0, 0.0, 0.0],
            }
            for uav_id in env.uav_ids
        }
        _obs, _rewards, terminated, truncated, info = env.step(action)
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertEqual(info["termination_reason"], "time_limit")
        self.assertFalse(info["task_complete"])

    def test_bandwidth_vector_is_feasible_and_zero_is_not_promoted(self) -> None:
        allocation = normalize_bandwidth_allocations(
            "uav_00", {"uav_01": 0.8, "uav_02": 0.8}, ["uav_01", "uav_02"]
        )
        self.assertLessEqual(sum(allocation.values()), 1.0 + 1e-12)
        zero = normalize_bandwidth_allocations(
            "uav_00", {}, ["uav_01", "uav_02"]
        )
        self.assertEqual(zero, {"uav_01": 0.0, "uav_02": 0.0})

    def test_potential_candidate_effective_order(self) -> None:
        states = {"uav_00": _state(0.0, 0.0), "uav_01": _state(10.0, 0.0)}
        channel = ChannelConfig(
            shadowing_std_los_db=0.0,
            rssi_measurement_std_db=0.0,
            loss_beta0=-100.0,
            min_snr_db=-100.0,
            max_packet_loss=1.0,
        )
        common = dict(
            states=states,
            obstacles=[],
            queue_state={},
            aoi_state={},
            rng=np.random.default_rng(4),
            dt_s=0.2,
            channel_config=channel,
            topology_config=TopologyConfig(max_comm_range_m=20.0, uwb_range_m=20.0),
            sensor_config=SensorConfig(uwb_max_range_m=20.0),
        )
        no_bandwidth = build_topology(
            bandwidth_allocations={"uav_00": {"uav_01": 0.0}}, **common
        )
        self.assertIn(("uav_00", "uav_01"), no_bandwidth.potential_edges)
        self.assertNotIn(("uav_00", "uav_01"), no_bandwidth.candidate_edges)
        self.assertNotIn(("uav_00", "uav_01"), no_bandwidth.effective_edges)

        with_bandwidth = build_topology(
            bandwidth_allocations={"uav_00": {"uav_01": 1.0}},
            **{**common, "rng": np.random.default_rng(4)},
        )
        self.assertIn(("uav_00", "uav_01"), with_bandwidth.candidate_edges)
        self.assertIn(("uav_00", "uav_01"), with_bandwidth.effective_edges)

    def test_three_dimensional_geometry_and_dynamic_obstacle(self) -> None:
        box = BoxObstacle("building", np.array([5.0, 0.0, 5.0]), np.array([2.0, 4.0, 4.0]))
        self.assertFalse(line_of_sight(np.array([0.0, 0.0, 5.0]), np.array([10.0, 0.0, 5.0]), [box]))
        collision, _ = collision_event({"uav_00": np.array([5.0, 0.0, 5.0])}, [box], 1.0)
        self.assertEqual(collision, 1)

        moving = SphereObstacle(
            "vehicle", np.array([5.0, 3.0, 5.0]), 1.0, np.array([0.0, -2.0, 0.0])
        )
        start = np.array([0.0, 0.0, 5.0])
        end = np.array([10.0, 0.0, 5.0])
        self.assertTrue(line_of_sight(start, end, [moving]))
        moving.update(1.0, np.array([-10.0, -10.0, -10.0]), np.array([20.0, 20.0, 20.0]))
        self.assertFalse(line_of_sight(start, end, [moving]))

    def test_uwb_can_exist_without_a_radio_edge(self) -> None:
        states = {"uav_00": _state(0.0, 0.0), "uav_01": _state(10.0, 0.0)}
        snapshot = build_topology(
            states,
            [],
            {"uav_00": {}, "uav_01": {}},
            {},
            {},
            rng=np.random.default_rng(5),
            dt_s=0.2,
            channel_config=ChannelConfig(),
            topology_config=TopologyConfig(max_comm_range_m=5.0, uwb_range_m=20.0),
            sensor_config=SensorConfig(uwb_max_range_m=20.0),
        )
        self.assertNotIn(("uav_00", "uav_01"), snapshot.potential_edges)
        self.assertIn(("uav_00", "uav_01"), snapshot.uwb_measurements)

    def test_action_changes_the_next_window_and_zero_bandwidth_blocks_messages(self) -> None:
        config = SimulationConfig(max_steps=2, gnss_mode="denied")
        moving = MultiUAVJointEnv(config)
        still = MultiUAVJointEnv(config)
        moving.reset(seed=22)
        still.reset(seed=22)
        zero_actions = {
            uav_id: {
                "measurement_uav_id": "none",
                "bandwidth": {},
                "delta_v": [0.0, 0.0, 0.0],
            }
            for uav_id in moving.uav_ids
        }
        move_actions = {
            uav_id: {
                "measurement_uav_id": "none",
                "bandwidth": {},
                "delta_v": [1.0, 0.0, 0.0],
            }
            for uav_id in moving.uav_ids
        }
        still.step(zero_actions)
        moving.step(move_actions)
        self.assertFalse(np.allclose(still.states["uav_00"].position_m, moving.states["uav_00"].position_m))
        self.assertEqual(len(moving.snapshot.effective_edges), 0)
        self.assertTrue(all(ekf.valid_neighbor_count == 0 for ekf in moving.ekfs.values()))

    def test_delayed_messages_arrive_on_a_later_decision_window(self) -> None:
        channel = ChannelConfig(
            packet_size_bytes=3_000_000,
            max_delay_s=1.0,
            shadowing_std_los_db=0.0,
            rssi_measurement_std_db=0.0,
            loss_beta0=-100.0,
        )
        env = MultiUAVJointEnv(
            SimulationConfig(max_steps=4, gnss_mode="denied", channel=channel)
        )
        env.reset(seed=21)

        def actions() -> dict[str, dict[str, object]]:
            return {
                uav_id: {
                    "measurement_uav_id": "none",
                    "bandwidth": {
                        target: 1.0 / (len(env.uav_ids) - 1)
                        for target in env.uav_ids
                        if target != uav_id
                    },
                    "delta_v": [0.0, 0.0, 0.0],
                }
                for uav_id in env.uav_ids
            }

        _obs, _rewards, _terminated, _truncated, _info = env.step(actions())
        self.assertEqual(len(env.snapshot.effective_edges), 0)
        self.assertGreater(len(env.pending_messages), 0)
        _obs, _rewards, _terminated, _truncated, _info = env.step(actions())
        self.assertGreater(len(env.snapshot.effective_edges), 0)
        self.assertGreater(len(env.arrived_messages), 0)

    def test_arrived_message_propagates_without_duplicate_confidence(self) -> None:
        env = MultiUAVJointEnv(SimulationConfig(max_steps=3, gnss_mode="denied"))
        env.reset(seed=23)
        covariance = np.eye(15, dtype=float)
        message = MessageRecord(
            source="uav_00",
            target="uav_01",
            state=np.array([1.0, 2.0, 3.0, 2.0, 0.0, 0.0] + [0.0] * 9),
            covariance=covariance.copy(),
            send_time_s=0.0,
            arrival_step=1,
            delay_s=0.2,
            arrival_time_s=0.2,
            last_propagated_time_s=0.2,
        )
        env.arrived_messages[(message.source, message.target)] = message
        env._advance_arrived_messages(0.4)
        state_at_04 = message.state.copy()
        trace_at_04 = float(np.trace(message.covariance[:3, :3]))
        env._advance_arrived_messages(0.6)
        np.testing.assert_allclose(message.state[:3], state_at_04[:3] + state_at_04[3:6] * 0.2)
        self.assertGreater(float(np.trace(message.covariance[:3, :3])), trace_at_04)
        self.assertEqual(message.last_propagated_time_s, 0.6)

    def test_local_graph_uses_source_to_target_message_direction(self) -> None:
        channel = ChannelConfig(
            shadowing_std_los_db=0.0,
            rssi_measurement_std_db=0.0,
            loss_beta0=-100.0,
            min_snr_db=-100.0,
            max_packet_loss=1.0,
        )
        env = MultiUAVJointEnv(SimulationConfig(max_steps=1, channel=channel))
        env.reset(seed=24)
        actions = {
            uav_id: {
                "measurement_uav_id": "none",
                "bandwidth": {
                    target: 1.0 / (len(env.uav_ids) - 1)
                    for target in env.uav_ids
                    if target != uav_id
                },
                "delta_v": [0.0, 0.0, 0.0],
            }
            for uav_id in env.uav_ids
        }
        env.step(actions)
        self.assertTrue(env.arrived_messages)
        source, target = sorted(env.arrived_messages)[0]
        message = env.arrived_messages[(source, target)]
        neighbor = env.graph_batch.local_observations[target]["neighbor_messages"][source]
        np.testing.assert_allclose(neighbor["estimate"], message.state)
        np.testing.assert_allclose(neighbor["covariance"], message.covariance)

    def test_ci_and_seed_reproducibility(self) -> None:
        mean, covariance, weight = covariance_intersection(
            np.zeros(3), np.eye(3), np.ones(3), np.eye(3) * 2.0
        )
        self.assertEqual(mean.shape, (3,))
        self.assertEqual(covariance.shape, (3, 3))
        self.assertGreaterEqual(weight, 0.0)
        self.assertLessEqual(weight, 1.0)

        config = SimulationConfig(max_steps=1)
        first = MultiUAVJointEnv(config)
        second = MultiUAVJointEnv(config)
        obs_first, _ = first.reset(seed=99)
        obs_second, _ = second.reset(seed=99)
        for uav_id in first.uav_ids:
            np.testing.assert_allclose(
                obs_first[uav_id]["node_features"], obs_second[uav_id]["node_features"]
            )

    def test_team_reward_and_core_objective_are_shared_scalars(self) -> None:
        env = MultiUAVJointEnv(SimulationConfig(max_steps=1, gnss_mode="denied"))
        env.reset(seed=17)
        actions = {
            uav_id: {
                "measurement_uav_id": "none",
                "bandwidth": {
                    target: 1.0 / (len(env.uav_ids) - 1)
                    for target in env.uav_ids
                    if target != uav_id
                },
                "delta_v": [0.0, 0.0, 0.0],
            }
            for uav_id in env.uav_ids
        }
        _obs, rewards, _terminated, _truncated, info = env.step(actions)
        self.assertEqual(len(set(rewards.values())), 1)
        self.assertIsInstance(info["joint_objective"], float)
        self.assertIn("j_loc", info["objective_components"])
        self.assertIn("j_com", info["objective_components"])

    def test_actor_observation_has_no_ground_truth_field(self) -> None:
        env = MultiUAVJointEnv(SimulationConfig(max_steps=1))
        obs, _ = env.reset(seed=8)
        for local in obs.values():
            self.assertNotIn("ground_truth", local)
            self.assertNotIn("true_position", local)
            self.assertIn("action_mask", local)
            self.assertIn("candidate_links", local)
            self.assertIn("effective_links", local)
            for features in local["potential_edge_features"].values():
                self.assertEqual(float(features[4]), -1.0)

    def test_measurement_mask_uses_incoming_uwb_sources_and_none_disables_update(self) -> None:
        channel = ChannelConfig(
            shadowing_std_los_db=0.0,
            rssi_measurement_std_db=0.0,
            loss_beta0=-100.0,
            min_snr_db=-100.0,
            max_packet_loss=1.0,
        )
        env = MultiUAVJointEnv(
            SimulationConfig(max_steps=1, gnss_mode="denied", channel=channel)
        )
        obs, _ = env.reset(seed=31)
        for uav_id, local in obs.items():
            measurement_mask = local["action_mask"][: len(env.uav_ids) + 1]
            expected = set(local["candidate_measurement_sources"])
            for source_index, source_id in enumerate(env.uav_ids):
                self.assertEqual(bool(measurement_mask[source_index]), source_id in expected)
            self.assertTrue(measurement_mask[-1])

        actions = {}
        for uav_id, local in obs.items():
            outgoing = {
                target: 1.0 / max(1, len(local["potential_targets"]))
                for target in local["potential_targets"]
            }
            actions[uav_id] = {
                "measurement_uav_id": "none",
                "bandwidth": outgoing,
                "delta_v": [0.0, 0.0, 0.0],
            }
        _obs, _reward, _terminated, _truncated, _info = env.step(actions)
        self.assertTrue(all(ekf.valid_neighbor_count == 0 for ekf in env.ekfs.values()))


if __name__ == "__main__":
    unittest.main()
