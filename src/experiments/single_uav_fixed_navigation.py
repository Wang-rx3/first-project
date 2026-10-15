"""Frozen Stage-1 local-EKF fixed navigation evaluation; never trains a policy."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from ..control.fixed_navigation import FixedNavigationConfig, FixedWaypointNavigator
from ..simulation.obstacle_model import is_physical_obstacle, segment_intersects_obstacle, segment_obstacle_clearance
from .single_uav_ppo_a_environment import create_foundation_environment
from .single_uav_ppo_baseline import load_single_uav_ppo_config
from .single_uav_fixed_navigation_scenarios import build_fixed_navigation_scenarios
from .single_uav_ppo_a_scenarios import _geometry_hash

ROOT = Path(__file__).resolve().parents[2]
UID = 'uav_00'
CONFIG = ROOT / 'data/configs/single_uav_fixed_navigation_v1_seed2026_20261011.json'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def serial(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def write_json(path, value):
    payload = json.dumps(value, ensure_ascii=False, indent=2, default=serial, allow_nan=False)
    with Path(path).open('x', encoding='utf-8') as handle:
        handle.write(payload + '\n')


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open('x', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, default=serial) if isinstance(value, (dict, list, tuple, np.ndarray)) else value
                             for key, value in row.items()})


def run_case(base, controller_config, case, phase, config_hash, coast=False):
    """Truth copies stay inside evaluator; navigator receives exactly six EKF values."""
    env = create_foundation_environment(base)
    observations, _ = env.reset(seed=case['noise_seed'], scenario_config=case['scenario'])
    physical = [obstacle for obstacle in env.obstacles if is_physical_obstacle(obstacle)]
    assert len(physical) == (1 if case['kind'] == 'single' else 0)
    env.ekfs[UID].state[:3] += np.asarray(case['belief_position_offset_m'])
    env.ekfs[UID].state[3:6] += np.asarray(case['belief_velocity_offset_m'])
    env.graph_batch = env._build_graph_batch()
    observations = env._observations()
    navigator = FixedWaypointNavigator(case['references'], controller_config)
    rows = []
    first_hold = None
    minimum_lower_bound = None
    goal = np.asarray(case['scenario']['goals'][UID], dtype=float)
    for step in range(1, base.max_steps + 1):
        # Deliberately narrow the controller API, before evaluator reads truth.
        estimate = np.asarray(observations[UID]['estimate'], dtype=float)[:6].copy()
        command = navigator.command(estimate)
        requested = np.zeros(3) if coast else np.asarray(command['command_delta_v_mps'], dtype=float)
        before_position = env.states[UID].position_m.copy()
        before_velocity = env.states[UID].velocity_mps.copy()
        observations, _, terminated, truncated, info = env.step({UID: {
            'measurement_uav_id': 'none', 'bandwidth': {}, 'delta_v': requested.copy()}})
        position = env.states[UID].position_m.copy()
        velocity = env.states[UID].velocity_mps.copy()
        assert np.array_equal(env.goals[UID], goal), 'Mission goal changed'
        actual = velocity - before_velocity
        execution = info['action_execution'][UID]
        reported = np.asarray(execution['delta_v_applied'], dtype=float).reshape(3)
        semantic_error = float(np.max(np.abs(reported - actual)))
        assert np.isfinite(reported).all() and semantic_error <= 1e-6
        reference = np.asarray(command['reference_position_m'])
        displacement = reference - estimate[:3]
        direction = displacement / max(float(np.linalg.norm(displacement)), 1e-12)
        speed_before = float(np.linalg.norm(before_velocity))
        speed = float(np.linalg.norm(velocity))
        distance = float(np.linalg.norm(position - goal))
        qualifying = distance <= 1 and speed <= .5 and not bool(info.get('collision', False))
        if qualifying and first_hold is None:
            first_hold = step
        witness = bool(np.dot(before_velocity, direction) > .05 and np.dot(requested, direction) < -1e-4
                       and -np.dot(actual, direction) > .001 and speed < speed_before)
        lower_bound = None
        intersects = False
        if physical:
            sampled = min(segment_obstacle_clearance(before_position, position, obstacle, samples=33) for obstacle in physical)
            lower_bound = max(0., sampled - float(np.linalg.norm(position - before_position)) / 64.)
            minimum_lower_bound = lower_bound if minimum_lower_bound is None else min(minimum_lower_bound, lower_bound)
            intersects = any(segment_intersects_obstacle(before_position, position, obstacle) for obstacle in physical)
        axis_limit = controller_config.max_delta_v_norm_mps / np.sqrt(3)
        legal = bool(np.isfinite(requested).all() and np.isfinite(actual).all()
                     and np.max(np.abs(requested)) <= axis_limit + 1e-9
                     and np.max(np.abs(actual)) <= axis_limit + 1e-9
                     and np.linalg.norm(requested) <= .6 + 1e-9
                     and np.linalg.norm(actual) / .2 <= 3 + 1e-8)
        rows.append({
            'phase': phase, 'scenario_id': case['scenario_id'], 'noise_seed': case['noise_seed'],
            'geometry_sha256': case['geometry_sha256'], 'config_sha256': config_hash,
            'regime': case['regime'], 'step': step, 'controller_mode': 'coast_zero_delta_v' if coast else 'fixed_navigation',
            'safety2_enabled': True, 'assist_mask': '00000', 'pass_commit': False,
            'lateral_assist': False, 'radial_assist': False, 'goal_recovery': False, 'terminal_capture': False,
            'ekf_position_pre_m': estimate[:3], 'ekf_velocity_pre_mps': estimate[3:6],
            'reference_position_m': reference, 'reference_index': command['reference_index'],
            'desired_velocity_mps': command['desired_velocity_mps'],
            'requested_acceleration_mps2': command['requested_acceleration_mps2'],
            'raw_delta_v_mps': command['raw_delta_v_mps'], 'axis_limited_delta_v_mps': command['axis_limited_delta_v_mps'],
            'requested_delta_v_mps': requested, 'env_input_command_mps': requested,
            'reported_applied_delta_v_mps': reported, 'actual_delta_v_mps': actual,
            'execution_correction_delta_v_mps': reported - requested,
            'actual_acceleration_mps2': actual / .2, 'applied_semantic_error_mps': semantic_error,
            'nominal_position_after_command_m': before_position + (2 * before_velocity + requested) * .1,
            'true_position_pre_m': before_position, 'true_velocity_pre_mps': before_velocity,
            'true_position_post_m': position, 'true_velocity_post_mps': velocity,
            'goal_distance_m_actual': distance, 'speed_mps_actual': speed,
            'estimated_goal_distance_m': float(np.linalg.norm(goal - estimate[:3])),
            'estimated_speed_mps': float(np.linalg.norm(estimate[3:6])),
            'position_estimation_error_pre_m': float(np.linalg.norm(estimate[:3] - before_position)),
            'velocity_estimation_error_pre_mps': float(np.linalg.norm(estimate[3:6] - before_velocity)),
            'strict_hold_steps': info['strict_hold_steps'], 'strict_success': info['strict_success'],
            'strict_physical_pass_complete': info['strict_physical_pass_complete'],
            'qualifying': qualifying, 'braking_witness': witness,
            'braking_before_first_hold': witness and (first_hold is None or step < first_hold),
            'requested_projection_mps': float(np.dot(requested, direction)),
            'pre_true_projection_mps': float(np.dot(before_velocity, direction)),
            'actual_projected_speed_reduction_mps': float(-np.dot(actual, direction)),
            'boundary_projected': bool(execution.get('boundary_projected', False)),
            'safety_projected': bool(execution.get('safety_projected', False)),
            'environment_action_clipped': bool(execution.get('action_clipped', False)),
            'controller_limited': bool(command['controller_limited']) if not coast else False,
            'collision': bool(info.get('collision', False)), 'physical_contact': bool(info.get('physical_contact', False)),
            'segment_intersects': intersects, 'segment_clearance_lower_bound_m': lower_bound,
            'legal_action_and_acceleration': legal, 'terminated': bool(terminated), 'truncated': bool(truncated),
            'termination_reason': info.get('termination_reason', '')})
        if terminated or truncated:
            break
    last = rows[-1]
    episode = {'phase': phase, 'scenario_id': case['scenario_id'], 'regime': case['regime'],
               'noise_seed': case['noise_seed'], 'steps': len(rows), 'success': bool(last['strict_success']),
               'final_distance_m': last['goal_distance_m_actual'], 'final_speed_mps': last['speed_mps_actual'],
               'peak_speed_mps': max(row['speed_mps_actual'] for row in rows),
               'braking_steps_before_first_hold': sum(row['braking_before_first_hold'] for row in rows),
               'first_qualifying_step': first_hold, 'collision': any(row['collision'] for row in rows),
               'physical_contact': any(row['physical_contact'] for row in rows),
               'boundary_projection_steps': sum(row['boundary_projected'] for row in rows),
               'safety_projection_steps': sum(row['safety_projected'] for row in rows),
               'illegal_steps': sum(not row['legal_action_and_acceleration'] for row in rows),
               'environment_clipped_steps': sum(row['environment_action_clipped'] for row in rows),
               'physical_pass_complete': last['strict_physical_pass_complete'],
               'minimum_clearance_lower_bound_m': minimum_lower_bound,
               'segment_intersects': any(row['segment_intersects'] for row in rows),
               'timeout': bool(last['truncated'] and not last['strict_success']),
               'max_actual_acceleration_mps2': max(np.linalg.norm(row['actual_acceleration_mps2']) for row in rows),
               'max_position_estimation_error_m': max(row['position_estimation_error_pre_m'] for row in rows)}
    return rows, episode


def clean(episodes):
    return all(not episode['collision'] and not episode['physical_contact']
               and episode['boundary_projection_steps'] == 0 and episode['safety_projection_steps'] == 0
               and episode['illegal_steps'] == 0 and episode['environment_clipped_steps'] == 0 for episode in episodes)


def isolation_audit(cases, own_stem):
    historical_geometry, historical_ids, historical_seeds = set(), set(), set()
    historical_endpoints = set()
    scanned = []
    def endpoint_hash(scene):
        payload = {key: {uid: [float(v) for v in xyz] for uid, xyz in scene[key].items()}
                   for key in ('initial_positions', 'goals')}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    def walk(value):
        if isinstance(value, dict):
            if 'initial_positions' in value and 'goals' in value:
                historical_endpoints.add(endpoint_hash(value))
                if all(all(key in obstacle for key in ('geometry_type', 'center_m', 'size_m', 'physical_collision_enabled'))
                       for obstacle in value.get('obstacle_specs', [])):
                    historical_geometry.add(_geometry_hash(value))
            for key in ('episode_id', 'scenario_id'):
                if key in value:
                    historical_ids.add(str(value[key]))
            if 'noise_seed' in value:
                historical_seeds.add(int(value['noise_seed']))
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)
    for path in (ROOT / 'data/synthetic').rglob('*.json'):
        if own_stem not in str(path) and any(token in path.name.lower() for token in ('curriculum', 'scenario', 'manifest')):
            walk(json.loads(path.read_text(encoding='utf-8-sig')))
            scanned.append(str(path.relative_to(ROOT)))
    flat = [case for phase in cases.values() for case in phase]
    ids = [case['scenario_id'] for case in flat]
    seeds = [case['noise_seed'] for case in flat]
    assert len(ids) == len(set(ids)) and len(seeds) == len(set(seeds))
    assert not set(ids) & historical_ids and not set(seeds) & historical_seeds
    assert not {case['geometry_sha256'] for case in flat} & historical_geometry
    assert not {endpoint_hash(case['scenario']) for case in flat} & historical_endpoints
    for first, second in (('development', 'normal_evaluation'), ('development', 'paired'), ('normal_evaluation', 'paired')):
        assert not {c['geometry_sha256'] for c in cases[first]} & {c['geometry_sha256'] for c in cases[second]}
    return {'scanned_manifests': scanned, 'historical_geometries': len(historical_geometry),
            'historical_endpoint_pairs': len(historical_endpoints),
            'historical_ids': len(historical_ids), 'historical_noise_seeds': len(historical_seeds),
            'new_unique_ids': len(ids), 'new_unique_noise_seeds': len(seeds), 'disjoint': True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=CONFIG)
    args = parser.parse_args()
    spec = json.loads(args.config.read_text(encoding='utf-8-sig'))
    logs, data = ROOT / spec['log_root'], ROOT / spec['data_root']
    assert not (logs / 'episodes.csv').exists(), 'Refusing to overwrite prior execution'
    controller = FixedNavigationConfig(**spec['controller'])
    base = load_single_uav_ppo_config(ROOT / spec['baseline_config'])
    stats = json.loads((ROOT / spec['frozen_environment_potential_statistics']).read_text(encoding='utf-8'))
    base = replace(base, train_potential_statistics=stats['potential_statistics'])
    assert base.dt_s == .2 and base.max_steps == 120 and base.max_speed_mps == 12 and base.max_acceleration_mps2 == 3
    assert base.safety_projection_enabled and base.safety_execution_mode == 'legacy' and base.safety_distance_m == 2
    assert not any((base.pass_commit_enabled, base.lateral_exit_enabled, base.radial_intent_enabled,
                    base.goal_recovery_correction_enabled, base.goal_recovery_terminal_capture_enabled))
    assert spec['safety2_enabled'] and spec['assist_mask'] == '00000'
    cases = build_fixed_navigation_scenarios()
    assert {phase: len(pool) for phase, pool in cases.items()} == {
        'development': 6, 'normal_evaluation': 36, 'paired': 4, 'obstacle': 4}
    for pool in cases.values():
        for case in pool:
            assert case['safety2_enabled'] is True and case['assist_mask'] == '00000'
            assert not any(case[key] for key in ('pass_commit', 'lateral_assist', 'radial_assist', 'goal_recovery', 'terminal_capture'))
    isolation = isolation_audit(cases, spec['stem'])
    source_files = ['src/control/fixed_navigation.py', 'src/experiments/single_uav_fixed_navigation.py',
                    'src/experiments/single_uav_fixed_navigation_scenarios.py', 'src/simulation/environment.py',
                    'src/simulation/motion_model.py', 'src/experiments/single_uav_ppo_a_environment.py',
                    'src/experiments/single_uav_ppo_baseline.py', 'src/simulation/ekf.py',
                    'src/simulation/sensor_model.py', 'src/simulation/graph_interface.py']
    lock = {'config_sha256': digest(args.config), 'sources': {p: digest(ROOT / p) for p in source_files},
            'base_config': asdict(base), 'controller_config': asdict(controller), 'cases': cases,
            'controller_inputs': ['local_EKF_position', 'local_EKF_velocity', 'static_reference'],
            'applied_semantic_tolerance_mps': 1e-6, 'maximum_research_env_steps': 6480,
            'safety2_enabled': True, 'assist_mask': '00000', 'no_training': True}
    lock['isolation_audit'] = isolation
    protected = json.loads((logs / 'protected_inputs_before.json').read_text(encoding='utf-8-sig'))
    assert all(digest(path) == expected for path, expected in protected.items()), 'Historical input changed'
    lock['protected_file_count_before'] = len(protected)
    write_json(data / 'execution_manifest.lock.json', lock)
    # Fail closed if accidental learning/model loading is introduced later.
    import torch
    from ..learning.mappo import MAPPOTrainer
    from ..simulation.environment import MultiUAVJointEnv
    counts = {'environment_steps': 0, 'checkpoint_loads': 0, 'trainer_constructions': 0,
              'ppo_updates': 0, 'optimizer_steps': 0}
    original_step = MultiUAVJointEnv.step
    def counted_step(self, *a, **kw):
        counts['environment_steps'] += 1
        assert counts['environment_steps'] <= 6480
        return original_step(self, *a, **kw)
    def forbidden(name):
        def fail(*a, **kw):
            counts[name] += 1
            raise RuntimeError('Forbidden Stage-1 operation: ' + name)
        return fail
    MultiUAVJointEnv.step = counted_step
    torch.load = forbidden('checkpoint_loads')
    torch.optim.Adam.step = forbidden('optimizer_steps')
    MAPPOTrainer.__init__ = forbidden('trainer_constructions')
    MAPPOTrainer.update = forbidden('ppo_updates')
    all_rows, all_episodes, gates = [], [], {}
    try:
        for phase in ('development', 'normal_evaluation', 'paired', 'obstacle'):
            phase_episodes = []
            for case in cases[phase]:
                for coast in ((False, True) if phase == 'paired' else (False,)):
                    actual_phase = ('paired_coast' if coast else 'paired_fixed') if phase == 'paired' else phase
                    rows, episode = run_case(base, controller, case, actual_phase, lock['config_sha256'], coast)
                    all_rows.extend(rows)
                    all_episodes.append(episode)
                    phase_episodes.append(episode)
                    print(f"{actual_phase} {case['scenario_id']} success={episode['success']} steps={episode['steps']} distance={episode['final_distance_m']:.4f} speed={episode['final_speed_mps']:.4f}", flush=True)
            if phase == 'paired':
                fixed = [e for e in phase_episodes if e['phase'] == 'paired_fixed']
                coast = [e for e in phase_episodes if e['phase'] == 'paired_coast']
                passed = all(e['success'] for e in fixed) and not any(e['success'] for e in coast) and clean(phase_episodes)
            else:
                passed = all(e['success'] for e in phase_episodes) and clean(phase_episodes)
                if phase == 'normal_evaluation':
                    rest = [e for e in phase_episodes if e['regime'] == 'rest']
                    moving = [e for e in phase_episodes if e['regime'] != 'rest']
                    passed = passed and all(e['peak_speed_mps'] > .5 and e['braking_steps_before_first_hold'] >= 1 for e in rest)
                    passed = passed and sum(e['braking_steps_before_first_hold'] >= 1 for e in moving) >= 14
                if phase == 'obstacle':
                    passed = passed and all(e['physical_pass_complete'] and not e['segment_intersects']
                                           and e['minimum_clearance_lower_bound_m'] >= 2.5 for e in phase_episodes)
            gates[phase] = bool(passed)
            write_csv(logs / (phase + '_steps.csv'), [r for r in all_rows if r['phase'].startswith('paired')] if phase == 'paired'
                      else [r for r in all_rows if r['phase'] == phase])
            if not passed:
                print('STAGE GATE FAILED: ' + phase, flush=True)
                break
    finally:
        write_csv(logs / 'episodes.csv', all_episodes)
        write_json(logs / 'execution_summary.json', {'gates': gates, 'counts': counts, 'episodes': all_episodes,
                                                   'history_protection': 'protected_inputs_before.json',
                                                   'no_tuning_or_reruns': True})
    assert counts['environment_steps'] == len(all_rows)
    print(json.dumps({'gates': gates, 'counts': counts}), flush=True)


if __name__ == '__main__':
    main()
