"""Read-only motion/stream telemetry over the unchanged foundation dynamics."""
from __future__ import annotations

import copy
import re

import numpy as np

from .single_uav_ppo_a_environment import FoundationSingleUAVPPOEnv
from .single_uav_ppo_baseline import build_single_uav_simulation_config
from .single_uav_ppo_smoke import canonical_hash, CSVLog, write_json
from ..simulation.motion_model import clip_vector_norm


def motion_candidate(env, delta):
    state = env.states[env.uav_ids[0]]
    motion = env.config.motion
    bounded = clip_vector_norm(np.asarray(delta), motion.max_acceleration_mps2*motion.dt_s)
    velocity = clip_vector_norm(state.velocity_mps+bounded, motion.max_speed_mps)
    position = state.position_m+(state.velocity_mps+velocity)*(.5*motion.dt_s)
    lower, upper = np.asarray(motion.bounds_min_m), np.asarray(motion.bounds_max_m)
    axes = np.logical_or(position < lower, position > upper)
    return {'pre_position_m': state.position_m.tolist(), 'pre_velocity_mps': state.velocity_mps.tolist(),
            'limited_command_delta_v': bounded.tolist(), 'candidate_velocity_mps': velocity.tolist(),
            'candidate_position_m': position.tolist(), 'boundary_attempt_axes': axes.tolist(),
            'nominal_acceleration_mps2': float(np.linalg.norm(velocity-state.velocity_mps)/motion.dt_s)}


class NormTelemetryEnv(FoundationSingleUAVPPOEnv):
    def reset(self, *, seed=None, scenario_config=None):
        config = scenario_config or {}
        episode_id = str(config.get('episode_id', ''))
        match = re.search(r'_r(\d+)$', episode_id) if '_online_' in episode_id else None
        ordinal = int(match[1]) if match else 0
        actual_seed = int(seed if seed is not None else self.baseline.seed)+1_000_000*ordinal
        obs, info = super().reset(seed=actual_seed, scenario_config=scenario_config)
        self.norm_noise_seed = actual_seed
        self.norm_reset_ordinal = ordinal
        return obs, info

    def step(self, action):
        uid = self.uav_ids[0]
        telemetry = motion_candidate(self, action[uid].get('delta_v', [0., 0., 0.]))
        telemetry.update(noise_seed=self.norm_noise_seed, reset_ordinal=self.norm_reset_ordinal,
                         env_rng_before=canonical_hash(self.rng.bit_generator.state))
        obs, rewards, terminated, truncated, info = super().step(action)
        telemetry.update(env_rng_after=canonical_hash(self.rng.bit_generator.state),
            post_position_m=self.states[uid].position_m.tolist(), post_velocity_mps=self.states[uid].velocity_mps.tolist(),
            sensor_source_names=sorted(info.get('sensor_packets', {}).keys()) if isinstance(info.get('sensor_packets'), dict) else [])
        candidate = np.asarray(telemetry['candidate_position_m'])
        actual = self.states[uid].position_m
        telemetry['boundary_projected_axes'] = (np.abs(actual-candidate) > 1e-9).tolist()
        info['norm_motion_telemetry'] = telemetry
        self.norm_last_telemetry = telemetry
        return obs, rewards, terminated, truncated, info


def create_norm_environment(base):
    return NormTelemetryEnv(build_single_uav_simulation_config(base), base)


def boundary_probes(base, log_root):
    env = create_norm_environment(base)
    logger = CSVLog(log_root / 'boundary_probes.csv')
    rows = []
    lo, hi = np.array(base.bounds_min_m), np.array(base.bounds_max_m)
    for axis in range(3):
        for side in (-1, 1):
            for outward in (True, False):
                p = np.array([100., 100., 30.])
                p[axis] = (lo[axis]+.05) if side == -1 else (hi[axis]-.05)
                direction = side*(1 if outward else -1)
                v = np.array([.15, .15, .15])
                v[axis] = 1.2*direction
                goal = p.copy()
                goal[axis] -= side*10
                scene = {'initial_positions': {'uav_00': p.tolist()}, 'initial_velocities': {'uav_00': v.tolist()},
                         'initial_attitudes': {'uav_00': [0., 0., 0.]}, 'goals': {'uav_00': goal.tolist()},
                         'obstacles': [], 'obstacle_specs': [], 'foundation_task_kind': 'normal',
                         'episode_id': f'boundary_{axis}_{side}_{outward}'}
                env.reset(seed=2026117100+len(rows), scenario_config=scene)
                delta = np.zeros(3)
                delta[axis] = .2*direction
                _, _, terminated, truncated, info = env.step({'uav_00': {'delta_v': delta.tolist(), 'measurement_uav_id': 'none', 'bandwidth': {}}})
                telemetry = info['norm_motion_telemetry']
                candidate_v = np.asarray(telemetry['candidate_velocity_mps'])
                candidate_p = np.asarray(telemetry['candidate_position_m'])
                projected = np.logical_or(candidate_p < lo, candidate_p > hi)
                expected_p = np.clip(candidate_p, lo, hi)
                # Legacy projection explicitly reconciles all velocity components
                # to executed displacement/dt, rather than zeroing just one axis.
                expected_v = clip_vector_norm((expected_p-p)/.2, base.max_speed_mps) if np.any(projected) else candidate_v
                match = bool(np.allclose(env.states['uav_00'].position_m, expected_p, atol=1e-8)
                             and np.allclose(env.states['uav_00'].velocity_mps, expected_v, atol=1e-8))
                row = {'axis': axis, 'side': side, 'outward': outward, **telemetry,
                       'actual_acceleration_mps2': float(np.linalg.norm(env.states['uav_00'].velocity_mps-v)/.2),
                       'expected_boundary_behavior': match, 'boundary_projected': info['action_execution']['uav_00']['boundary_projected'],
                       'learning_reward': info['learning_rewards']['uav_00'], 'reward_components': info['learning_reward_components'],
                       'terminated': terminated, 'truncated': truncated}
                logger.append(row)
                rows.append(row)
                assert match and telemetry['nominal_acceleration_mps2'] <= 3+1e-8
    result = {'env_steps': len(rows), 'passed': True, 'reward_or_dynamics_modified': False,
              'position_clips_only_out_of_bounds_axes': True,
              'velocity_reconciliation': 'executed displacement / dt on all axes after projection',
              'maximum_actual_acceleration_mps2': max(r['actual_acceleration_mps2'] for r in rows)}
    write_json(log_root / 'boundary_probe_result.json', result)
    return result
