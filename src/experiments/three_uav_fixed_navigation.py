"""Stage-2 controlled 3-UAV navigation/message/fusion validation, no learning."""
from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict
from itertools import combinations
from pathlib import Path

import numpy as np

from ..control.fixed_navigation import FixedNavigationConfig
from ..control.fixed_navigation_execution import compose_fixed_navigation_actions
from ..simulation.environment import MultiUAVJointEnv, SimulationConfig
from ..simulation.motion_model import MotionConfig
from .cooperative_event_logging import accepted_counts, collect_window_events
from .three_uav_fixed_navigation_contract import UIDS, StrictParkingTracker, joint_closest_approach, make_scenario_schedule, controlled_communication_actions

ROOT = Path(__file__).resolve().parents[2]
STEM = 'three_uav_fixed_navigation_v2_seed2026_20261011'
DATA = ROOT / 'data/synthetic' / STEM
LOGS = ROOT / 'logs/preliminary' / STEM
CONFIG = ROOT / 'data/configs' / (STEM + '.json')
TABLES = ('navigation', 'joint', 'pairs', 'flows', 'messages', 'gates', 'fusion_events')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def convert(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def write_json(path, value):
    encoded = json.dumps(value, indent=2, ensure_ascii=False, default=convert, allow_nan=False)
    with Path(path).open('x', encoding='utf-8') as handle:
        handle.write(encoded + '\n')


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('x', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, default=convert) if isinstance(value, (dict, list, tuple, np.ndarray)) else value for key, value in row.items()})


def simulation_config(spec, cooperative=True):
    dynamics = json.loads((ROOT / spec['baseline_dynamics_config']).read_text(encoding='utf-8-sig'))
    motion = MotionConfig(**{key: dynamics[key] for key in (
        'dt_s', 'max_speed_mps', 'max_acceleration_mps2', 'bounds_min_m', 'bounds_max_m', 'safety_distance_m')})
    assert motion.dt_s == .2 and motion.max_speed_mps == 12 and motion.max_acceleration_mps2 == 3
    return SimulationConfig(uav_count=3, max_steps=120, seed=2026, motion=motion,
                            random_stream_mode='indexed-v1', gnss_mode='denied',
                            cooperative_ekf_enabled=cooperative, safety_projection_enabled=True,
                            safety_execution_mode='legacy')


def packet_id(source, target, send_time):
    return f'{source}->{target}@{float(send_time):.17g}'


def base_fields(case, step):
    return {'episode_id': case['episode_id'], 'case_id': case['case_id'], 'variant': case['variant'],
            'stage': case['stage'], 'seed': case['seed'], 'step': step,
            'communication_authority': 'controlled_round_robin_fixed_half', 'policy_samples_executed': False}


def run_episode(case, spec, nav_config, tables, snapshots):
    env = MultiUAVJointEnv(simulation_config(spec, case['cooperative_ekf_enabled']))
    observations, info = env.reset(seed=case['seed'], scenario_config=case['scenario'])
    assert not env.obstacles and tuple(env.uav_ids) == UIDS
    assert env.config.ekf.stale_message_limit_s == .6 and env.config.sensor.uwb_update_period_s == .8
    assert env.config.topology.max_comm_range_m == 75 and env.config.topology.uwb_range_m == 100
    tracker = StrictParkingTracker(UIDS)
    goals = {uid: np.asarray(case['scenario']['goals'][uid]) for uid in UIDS}
    first_qualifying = dict.fromkeys(UIDS)
    braking = dict.fromkeys(UIDS, 0)
    peak_speed = dict.fromkeys(UIDS, 0.)
    initial_belief_matches_truth = {uid: bool(np.array_equal(env.ekfs[uid].state[:3], env.states[uid].position_m)
                                            and np.array_equal(env.ekfs[uid].state[3:6], env.states[uid].velocity_mps)) for uid in UIDS}
    state_hash = hashlib.sha256()
    min_distance, valid, ci_calls = float('inf'), True, []
    transmissions = set()
    payload_matches = True
    original_enqueue = env._enqueue_messages
    def observed_enqueue(edges, payloads):
        nonlocal transmissions, payload_matches
        transmissions = set(edges)
        payload_matches = all(np.array_equal(payload[0], pre_beliefs[uid])
                              and np.array_equal(payload[1], pre_covariances[uid]) for uid, payload in payloads.items())
        return original_enqueue(edges, payloads)
    env._enqueue_messages = observed_enqueue
    # Observational wrapper counts the actual CI branch without changing its outputs.
    from ..simulation import environment as environment_module
    original_ci = environment_module.covariance_intersection
    def observed_ci(*args, **kwargs):
        output = original_ci(*args, **kwargs)
        ci_calls.append({'step': step, 'weight': float(output[2])})
        return output
    environment_module.covariance_intersection = observed_ci
    episode_starts = {name: len(tables[name]) for name in TABLES}
    try:
        for step in range(1, 121):
            prefix = base_fields(case, step)
            pre_positions = {uid: env.states[uid].position_m.copy() for uid in UIDS}
            pre_velocities = {uid: env.states[uid].velocity_mps.copy() for uid in UIDS}
            pre_beliefs = {uid: env.ekfs[uid].state.copy() for uid in UIDS}
            pre_covariances = {uid: env.ekfs[uid].covariance.copy() for uid in UIDS}
            pre_consumed = dict(env.neighbor_belief_message_times)
            pre_update_times = dict(env.uwb_update_times)
            pre_counts = accepted_counts(env)
            assert all(np.array_equal(np.asarray(observations[uid]['estimate']), pre_beliefs[uid]) for uid in UIDS)
            communication = controlled_communication_actions(observations, step, cut_enabled=case['variant'] == 'cut')
            actions, diagnostics = compose_fixed_navigation_actions(observations, communication, nav_config)
            for uid in UIDS:
                assert np.array_equal(actions[uid]['delta_v'], diagnostics[uid]['command_delta_v_mps'])
                assert np.asarray(observations[uid]['node_features']).shape[1] == 37
            observations, _, terminated, truncated, info = env.step(actions)
            post_positions = {uid: env.states[uid].position_m.copy() for uid in UIDS}
            post_velocities = {uid: env.states[uid].velocity_mps.copy() for uid in UIDS}
            assert all(np.array_equal(env.goals[uid], goals[uid]) for uid in UIDS)
            distances = {uid: float(np.linalg.norm(post_positions[uid] - goals[uid])) for uid in UIDS}
            speeds = {uid: float(np.linalg.norm(post_velocities[uid])) for uid in UIDS}
            parking = tracker.update(distances, speeds, collision=bool(info['collision']))
            events = collect_window_events(env, pre_counts, run_root=DATA, episode_id=case['episode_id'], split=case['stage'], step=step)
            for event in events:
                source, target = event['source_uav_id'], event['target_uav_id']
                msg = env.arrived_messages[(source, target)]
                novelty = (target, source) not in pre_consumed or not np.isclose(pre_consumed[(target, source)], msg.send_time_s, rtol=0, atol=1e-12)
                prior_update = pre_update_times.get((source, target))
                assert novelty and step * .2 - msg.send_time_s <= .6
                assert prior_update is None or step * .2 - prior_update >= .8 - 1e-9
                tables['fusion_events'].append({**prefix, **event, 'packet_id': packet_id(source, target, msg.send_time_s),
                                               'fresh': True, 'new_since_last_consumed': bool(novelty),
                                               'previous_update_time_s': prior_update})
            gate_lookup = {(event['source_uav_id'], event['target_uav_id']): event for event in env.measurement_gate_events}
            for gate in env.measurement_gate_events:
                tables['gates'].append({**prefix, **gate})
            for message in env.message_events:
                tables['messages'].append({**prefix, **message})
            nav_valid = True
            for uid in UIDS:
                execution = info['action_execution'][uid]
                requested = np.asarray(diagnostics[uid]['command_delta_v_mps'])
                applied = np.asarray(execution['delta_v_applied'])
                actual = post_velocities[uid] - pre_velocities[uid]
                actual_a = actual / .2
                request_input_error = float(np.max(np.abs(requested - np.asarray(actions[uid]['delta_v']))))
                semantic_error = float(np.max(np.abs(applied - actual)))
                direction_vector = goals[uid] - pre_beliefs[uid][:3]
                direction = direction_vector / max(float(np.linalg.norm(direction_vector)), 1e-12)
                pre_projection = float(np.dot(pre_velocities[uid], direction))
                request_projection = float(np.dot(requested, direction))
                reduction = float(-np.dot(actual, direction))
                witness = bool(pre_projection > .05 and request_projection < -1e-4 and reduction > .001
                               and speeds[uid] < np.linalg.norm(pre_velocities[uid]))
                qualifying = parking['per_uav'][uid]['qualifying']
                if qualifying and first_qualifying[uid] is None:
                    first_qualifying[uid] = step
                before_first = bool(witness and (first_qualifying[uid] is None or step < first_qualifying[uid]))
                braking[uid] += int(before_first)
                peak_speed[uid] = max(peak_speed[uid], speeds[uid])
                covariance = env.ekfs[uid].covariance.copy()
                covariance_finite = bool(np.isfinite(covariance).all())
                symmetry_error = float(np.max(np.abs(covariance - covariance.T)))
                minimum_eigenvalue = float(np.linalg.eigvalsh((covariance + covariance.T)/2).min()) if covariance_finite else -1e99
                legal = bool(np.isfinite(actual).all() and np.isfinite(requested).all()
                             and np.linalg.norm(actual_a) <= 3 + 1e-8 and speeds[uid] <= 12 + 1e-9
                             and np.max(np.abs(requested)) <= .6 / np.sqrt(3) + 1e-9
                             and np.max(np.abs(actual)) <= .6 / np.sqrt(3) + 1e-9
                             and np.linalg.norm(requested) <= .6 + 1e-9
                             and request_input_error <= 1e-12 and semantic_error <= 1e-6)
                in_bounds = bool(np.all(post_positions[uid] >= [0, 0, 2]) and np.all(post_positions[uid] <= [200, 200, 60]))
                nav_valid = nav_valid and legal and in_bounds and covariance_finite and symmetry_error <= 1e-8 and minimum_eigenvalue >= -1e-8
                nav_valid = nav_valid and not any(execution.get(key, False) for key in ('boundary_projected', 'safety_projected', 'action_clipped'))
                state_hash.update(env.ekfs[uid].state.tobytes())
                tables['navigation'].append({**prefix, 'uav_id': uid, 'goal_m': goals[uid],
                    'ekf_state_pre': pre_beliefs[uid], 'ekf_state_post': env.ekfs[uid].state.copy(),
                    'ekf_covariance_post': covariance, 'true_position_pre_m': pre_positions[uid],
                    'true_velocity_pre_mps': pre_velocities[uid], 'true_position_post_m': post_positions[uid],
                    'true_velocity_post_mps': post_velocities[uid], 'requested_delta_v_mps': requested,
                    'env_input_delta_v_mps': np.asarray(actions[uid]['delta_v']), 'reported_applied_delta_v_mps': applied,
                    'actual_delta_v_mps': actual, 'actual_acceleration_mps2': actual_a, 'execution_correction_mps': applied - requested,
                    'request_input_max_abs_error_mps': request_input_error, 'reported_actual_max_abs_error_mps': semantic_error,
                    'desired_velocity_mps': diagnostics[uid]['desired_velocity_mps'],
                    'requested_acceleration_mps2': diagnostics[uid]['requested_acceleration_mps2'],
                    'raw_delta_v_mps': diagnostics[uid]['raw_delta_v_mps'],
                    'axis_limited_delta_v_mps': diagnostics[uid]['axis_limited_delta_v_mps'],
                    'goal_distance_m_actual': distances[uid], 'speed_mps_actual': speeds[uid],
                    'position_estimation_error_m': float(np.linalg.norm(env.ekfs[uid].state[:3] - post_positions[uid])),
                    'velocity_estimation_error_mps': float(np.linalg.norm(env.ekfs[uid].state[3:6] - post_velocities[uid])),
                    'position_covariance_trace_m2': float(np.trace(covariance[:3, :3])),
                    'covariance_finite': covariance_finite, 'covariance_symmetry_max_abs_error': symmetry_error,
                    'covariance_minimum_eigenvalue': minimum_eigenvalue, 'action_legal': legal, 'in_bounds': in_bounds,
                    'qualifying': qualifying, 'hold_steps': parking['per_uav'][uid]['hold_steps'],
                    'ever_success': parking['per_uav'][uid]['ever_success'],
                    'first_qualifying_step': first_qualifying[uid], 'first_success_step': parking['per_uav'][uid]['first_success_step'],
                    'braking_witness': witness, 'braking_before_first_qualifying': before_first,
                    'pre_true_velocity_projection_mps': pre_projection, 'requested_projection_mps': request_projection,
                    'actual_projected_speed_reduction_mps': reduction,
                    'boundary_projected': bool(execution.get('boundary_projected', False)),
                    'safety_projected': bool(execution.get('safety_projected', False)),
                    'environment_action_clipped': bool(execution.get('action_clipped', False)),
                    'collision': bool(info['collision']), 'safety2_enabled': True, 'assist_mask': '00000',
                    'pass_commit': False, 'lateral_exit': False, 'radial_intent': False,
                    'goal_recovery': False, 'terminal_capture': False,
                    'valid_neighbor_count_pre': pre_counts[uid], 'valid_neighbor_count_post': env.ekfs[uid].valid_neighbor_count,
                    'requested_measurement_source': communication[uid]['measurement_uav_id'],
                    'applied_measurement_source': info['actions'][uid]['measurement_uav_id'],
                    'requested_bandwidth': communication[uid]['bandwidth'], 'applied_bandwidth': info['applied_bandwidth'][uid],
                    'requested_link_selection': [communication[uid]['bandwidth'].get(recipient, 0) > 0 for recipient in UIDS],
                    'applied_link_selection': [info['applied_bandwidth'][uid].get(recipient, 0) > 0 for recipient in UIDS]})
            pair_results = joint_closest_approach(pre_positions, post_positions)
            for pair in pair_results:
                tables['pairs'].append({**prefix, **pair})
                min_distance = min(min_distance, pair['distance_m'])
            joint_valid = nav_valid and not bool(info['collision']) and all(pair['distance_m'] >= 2 for pair in pair_results) and payload_matches
            valid = valid and joint_valid
            for source in UIDS:
                for target in UIDS:
                    if source == target:
                        continue
                    key = (source, target)
                    msg = env.arrived_messages.get(key)
                    age = step * .2 - msg.send_time_s if msg else None
                    fresh = bool(msg is not None and age <= .6)
                    effective = key in env.snapshot.effective_edges
                    actor_neighbor = observations[target].get('neighbor_messages', {}).get(source)
                    assert effective == (key in env.snapshot.potential_edges and fresh)
                    assert (actor_neighbor is not None) == effective
                    if actor_neighbor:
                        assert np.array_equal(actor_neighbor['estimate'], msg.state)
                        assert np.array_equal(actor_neighbor['covariance'], msg.covariance)
                    metrics = env.snapshot.metrics.get(key)
                    previous = pre_consumed.get((target, source))
                    new_since = bool(msg is not None and (previous is None or not np.isclose(previous, msg.send_time_s, rtol=0, atol=1e-12)))
                    newly_arrived = any(event.get('source') == source and event.get('target') == target and event.get('status') == 'arrived' for event in env.message_events)
                    node_ids = observations[target].get('node_ids', [])
                    node = np.asarray(observations[target]['node_features'])[node_ids.index(source)] if source in node_ids else None
                    gate = gate_lookup.get(key, {})
                    tables['flows'].append({**prefix, 'source': source, 'target': target,
                        'physical_distance_m': float(np.linalg.norm(post_positions[source] - post_positions[target])),
                        'potential': key in env.snapshot.potential_edges, 'candidate': key in env.snapshot.candidate_edges,
                        'uwb_available': key in env.snapshot.uwb_edges,
                        'requested_bandwidth': float(communication[source]['bandwidth'].get(target, 0)),
                        'applied_bandwidth': float(info['applied_bandwidth'][source].get(target, 0)),
                        'controlled_link_selected': communication[source]['bandwidth'].get(target, 0) > 0,
                        'transmitted': key in transmissions, 'arrived_buffer_present': msg is not None,
                        'newly_arrived_this_step': newly_arrived, 'fresh': fresh, 'effective_visible': effective,
                        'cached_visible': bool(effective and not newly_arrived), 'new_since_last_consumed': new_since,
                        'reused': bool(msg is not None and not new_since), 'previous_consumed_send_time_s': previous,
                        'message_send_time_s': msg.send_time_s if msg else None, 'message_age_s': age,
                        'message_arrival_time_s': msg.arrival_time_s if msg else None,
                        'packet_id': packet_id(source, target, msg.send_time_s) if msg else '',
                        'received_message_state': msg.state.copy() if msg else None,
                        'received_message_covariance_diagonal': np.diag(msg.covariance).copy() if msg else None,
                        'actor_neighbor_state': actor_neighbor['estimate'] if actor_neighbor else None,
                        'actor_neighbor_covariance_diagonal': np.diag(actor_neighbor['covariance']) if actor_neighbor else None,
                        'actor_node_state': node[:15] if node is not None else None,
                        'actor_node_covariance_diagonal': node[15:30] if node is not None else None,
                        'receiver_selected_source': info['actions'][target]['measurement_uav_id'],
                        'measurement_selected': info['actions'][target]['measurement_uav_id'] == source,
                        'fusion_accepted': bool(gate.get('ekf_update_accepted', False)),
                        'fusion_attempted': bool(gate.get('ekf_update_attempted', False)),
                        'fusion_gate_reason': gate.get('failure_stage', 'not_selected'),
                        'gate_new_packet': bool(gate.get('new_packet', False)),
                        'queue_next_packets': float(env.queue_state.get(key, 0)),
                        'aoi_s': float(env.aoi_state.get(key, 0)),
                        'radio_packet_loss': float(metrics.packet_loss) if metrics else None,
                        'radio_delay_s': float(metrics.delay_s) if metrics else None})
            tables['joint'].append({**prefix, 'joint_hold_steps': parking['joint_hold_steps'],
                'joint_ever_success': parking['joint_ever_success'], 'joint_first_success_step': parking['joint_first_success_step'],
                'minimum_synchronized_pair_distance_m': min(pair['distance_m'] for pair in pair_results),
                'collision': bool(info['collision']), 'joint_valid': bool(joint_valid),
                'sender_payload_matches_pre_EKF': bool(payload_matches), 'cooperative_ekf_enabled': env.config.cooperative_ekf_enabled,
                'accepted_cooperative_updates': sum(bool(event['accepted']) for event in events),
                'ci_calls_this_step': sum(event['step'] == step for event in ci_calls),
                'stale_messages_cumulative': env.stale_neighbor_messages,
                'reused_packet_skips_cumulative': env.reused_uwb_measurement_skips,
                'update_period_skips_cumulative': env.uwb_update_period_skips,
                'parent_terminated': bool(terminated), 'parent_truncated': bool(truncated)})
            if step in (1, 19, 22, 25, 51, 55, 120):
                snapshots.append({**prefix, 'provenance': 'saved_real_observation_post_step',
                                  'observations': observations, 'central_state': info['central_state']})
            if terminated or truncated:
                break
    finally:
        environment_module.covariance_intersection = original_ci
    assert step <= 120
    episode_nav = tables['navigation'][episode_starts['navigation']:]
    flows = tables['flows'][episode_starts['flows']:]
    fusion = tables['fusion_events'][episode_starts['fusion_events']:]
    moving = [uid for uid in UIDS if not (case['case_id'] == 'c2' and uid == 'uav_02')]
    all_success = all(parking['per_uav'][uid]['ever_success'] for uid in UIDS) and parking['joint_ever_success']
    nav_gate = valid and all(braking[uid] >= 1 for uid in moving)
    accepted = sum(bool(event['accepted']) for event in fusion)
    mechanism_gate = (accepted == 0 if not case['cooperative_ekf_enabled'] else accepted > 0)
    cutoff_gate = True
    if case['variant'] == 'cut':
        for source, target in spec['intervention']['blocked_edges']:
            blocked = [row for row in flows if row['source'] == source and row['target'] == target and 20 <= row['step'] <= 50]
            drained = [row for row in blocked if row['step'] >= 25]
            recovered = [row for row in flows if row['source'] == source and row['target'] == target and row['step'] >= 51]
            cutoff_gate = cutoff_gate and all(row['applied_bandwidth'] == 0 and not row['transmitted'] for row in blocked)
            cutoff_gate = cutoff_gate and all(not any(row[key] for key in ('newly_arrived_this_step', 'effective_visible', 'fusion_accepted')) for row in drained)
            cutoff_gate = cutoff_gate and all(any(row[key] for row in recovered) for key in ('transmitted', 'newly_arrived_this_step', 'fusion_accepted'))
    gate = bool(nav_gate and mechanism_gate and cutoff_gate and (all_success or case['variant'] == 'fusion_off'))
    return {'episode_id': case['episode_id'], 'case_id': case['case_id'], 'variant': case['variant'], 'stage': case['stage'],
            'seed': case['seed'], 'steps': step, 'navigation_gate': bool(nav_gate), 'mechanism_gate': bool(mechanism_gate),
            'cutoff_gate': bool(cutoff_gate), 'gate_passed': gate, 'agent_successes': sum(parking['per_uav'][uid]['ever_success'] for uid in UIDS),
            'joint_success': bool(parking['joint_ever_success']), 'joint_first_success_step': parking['joint_first_success_step'],
            'final_distance_m': distances, 'final_speed_mps': speeds,
            'final_qualifying': {uid: parking['per_uav'][uid]['qualifying'] for uid in UIDS},
            'final_hold_steps': {uid: parking['per_uav'][uid]['hold_steps'] for uid in UIDS},
            'braking_before_first_qualifying': braking, 'peak_speed_mps': peak_speed,
            'minimum_synchronized_pair_distance_m': min_distance, 'accepted_cooperative_updates': accepted,
            'ci_calls': len(ci_calls), 'ci_weights': [event['weight'] for event in ci_calls],
            'initial_belief_matches_truth': initial_belief_matches_truth,
            'belief_trajectory_sha256': state_hash.hexdigest(), 'collision_steps': sum(row['collision'] for row in episode_nav) // 3,
            'boundary_projection_steps': sum(row['boundary_projected'] for row in episode_nav),
            'safety_projection_steps': sum(row['safety_projected'] for row in episode_nav),
            'maximum_actual_acceleration_mps2': max(float(np.linalg.norm(row['actual_acceleration_mps2'])) for row in episode_nav),
            'mean_position_estimation_error_m': float(np.mean([row['position_estimation_error_m'] for row in episode_nav]))}


def main():
    spec = json.loads(CONFIG.read_text(encoding='utf-8-sig'))
    assert not (LOGS / 'execution_summary.json').exists(), 'Refusing to overwrite a run'
    nav_spec = json.loads((ROOT / spec['navigation_config']).read_text(encoding='utf-8-sig'))
    nav_config = FixedNavigationConfig(**nav_spec['controller'])
    schedule = make_scenario_schedule()
    assert len(schedule) == 8 and len({case['episode_id'] for case in schedule}) == 8
    protected = json.loads((LOGS / 'protected_inputs_before.json').read_text(encoding='utf-8-sig'))
    assert all(sha(path) == expected for path, expected in protected.items())
    source_paths = ['src/control/fixed_navigation.py', 'src/control/fixed_navigation_execution.py',
                    'src/learning/mappo.py', 'src/experiments/three_uav_fixed_navigation.py',
                    'src/experiments/three_uav_fixed_navigation_contract.py',
                    'src/analysis/three_uav_fixed_navigation_audit.py',
                    'src/analysis/three_uav_fixed_navigation_probes.py', 'src/simulation/environment.py',
                    'src/simulation/ekf.py', 'src/simulation/graph_interface.py', 'src/simulation/topology.py',
                    'src/simulation/motion_model.py', 'src/simulation/sensor_model.py', 'src/synthetic/channel_model.py']
    lock = {'spec': spec, 'simulation_config': asdict(simulation_config(spec)), 'navigation_config': asdict(nav_config),
            'schedule': schedule, 'config_sha256': sha(CONFIG), 'sources': {p: sha(ROOT / p) for p in source_paths},
            'protected_file_count': len(protected), 'controlled_authority': True, 'no_training': True,
            'original_mappo_sha256': sha(DATA / 'legacy_mappo_before.py.txt'), 'learned_motion_dim': 0,
            'neural_samples_actual_execution': 0, 'maximum_joint_steps': 960,
            'truth_initialization_inherited': True, 'no_replay_or_tuning': True}
    write_json(DATA / 'execution_manifest.lock.json', lock)
    import torch
    from ..learning.mappo import MAPPOTrainer
    counts = {'joint_environment_steps': 0, 'ppo_updates': 0, 'optimizer_steps': 0,
              'checkpoint_loads': 0, 'rollout_additions': 0, 'policy_samples_executed': 0, 'trainer_constructions_actual_run': 0}
    original_step = MultiUAVJointEnv.step
    def counted_step(self, *args, **kwargs):
        counts['joint_environment_steps'] += 1
        assert counts['joint_environment_steps'] <= 960
        return original_step(self, *args, **kwargs)
    def forbidden(name):
        def fail(*args, **kwargs):
            counts[name] += 1
            raise RuntimeError('Forbidden Stage-2 learning operation: ' + name)
        return fail
    MultiUAVJointEnv.step = counted_step
    MAPPOTrainer.__init__ = forbidden('trainer_constructions_actual_run')
    MAPPOTrainer.update = forbidden('ppo_updates')
    MAPPOTrainer.add_transition = forbidden('rollout_additions')
    MAPPOTrainer.collect_rollout = forbidden('rollout_additions')
    torch.load = forbidden('checkpoint_loads')
    for cls in (torch.optim.Adam, torch.optim.AdamW, torch.optim.SGD):
        cls.step = forbidden('optimizer_steps')
    tables = {name: [] for name in TABLES}
    snapshots, summaries = [], []
    failure = None
    try:
        for case in schedule:
            result = run_episode(case, spec, nav_config, tables, snapshots)
            summaries.append(result)
            print(json.dumps({key: result[key] for key in ('episode_id', 'steps', 'agent_successes', 'joint_success',
                  'gate_passed', 'minimum_synchronized_pair_distance_m', 'accepted_cooperative_updates', 'ci_calls')}), flush=True)
            if not result['gate_passed']:
                break
        assert counts['joint_environment_steps'] == len(tables['joint'])
        assert all(sha(ROOT / p) == expected for p, expected in lock['sources'].items())
    except Exception as exc:
        failure = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        for name, rows in tables.items():
            write_csv(LOGS / (name + '.csv'), rows)
        write_json(DATA / 'saved_observation_probes.json', snapshots)
        write_json(LOGS / 'execution_summary.json', {'episodes': summaries, 'counts': counts, 'failure': failure,
                   'all_scheduled_gates_passed': len(summaries) == 8 and all(case['gate_passed'] for case in summaries),
                   'authority': 'controlled_only_neural_probes_never_executed'})
    print(json.dumps({'counts': counts, 'scheduled_episodes_completed': len(summaries)}), flush=True)


if __name__ == '__main__':
    main()
