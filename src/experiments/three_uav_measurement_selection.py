"""Human-run Stage-3 workflows; importing this module never starts a run."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import re
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from ..control.fixed_navigation import FixedNavigationConfig
from ..control.fixed_navigation_execution import compose_fixed_navigation_actions
from ..learning.mappo import MAPPOConfig, MAPPOTrainer, MAPPOPolicy, central_state_to_vector
from ..learning.normalization import FeatureNormalizer, ObservationNormalizer
from ..simulation.environment import MultiUAVJointEnv, SimulationConfig, RewardConfig
from ..simulation.ekf import EKFConfig
from ..simulation.motion_model import MotionConfig
from ..simulation.sensor_model import SensorConfig
from ..simulation.topology import TopologyConfig
from ..synthetic.channel_model import ChannelConfig
from .three_uav_fixed_navigation_contract import StrictParkingTracker, joint_closest_approach

ROOT = Path(__file__).resolve().parents[2]
STEM = 'three_uav_measurement_selection_v3_seed2026_20261012'
CONFIG = ROOT / 'data/configs' / (STEM + '.json')
UIDS = ('uav_00', 'uav_01', 'uav_02')
SCHEMA = 'three_uav_measurement_only_v3'
SOURCE_FILES = ('src/learning/mappo.py', 'src/learning/normalization.py',
    'src/control/fixed_navigation.py', 'src/control/fixed_navigation_execution.py',
    'src/simulation/environment.py', 'src/simulation/ekf.py', 'src/simulation/graph_interface.py',
    'src/simulation/motion_model.py', 'src/simulation/sensor_model.py', 'src/simulation/topology.py',
    'src/synthetic/channel_model.py', 'src/experiments/three_uav_measurement_selection.py',
    'src/experiments/three_uav_measurement_selection_contract.py')


def contract():
    from . import three_uav_measurement_selection_contract
    return three_uav_measurement_selection_contract


def native(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def encode(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=native, allow_nan=False)


def content_hash(value):
    return hashlib.sha256(encode(value).encode('utf-8')).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path, value):
    encoded = json.dumps(value, ensure_ascii=False, indent=2, default=native, allow_nan=False)
    with Path(path).open('x', encoding='utf-8') as handle:
        handle.write(encoded + '\n')


class RunOutput:
    """Reserve all output roots before any environment/model/load operation."""
    def __init__(self, run_id, root=ROOT):
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', str(run_id)):
            raise ValueError('run-id must contain only letters, digits, underscore or hyphen')
        self.run_id = str(run_id)
        self.data = Path(root) / 'data/synthetic' / STEM / self.run_id
        self.logs = Path(root) / 'logs/preliminary' / STEM / self.run_id
        self.plots = Path(root) / 'plots/preliminary' / STEM / self.run_id
        occupied = [str(p) for p in (self.data, self.logs, self.plots) if p.exists()]
        if occupied:
            raise FileExistsError('Refusing to reuse output directories: ' + ', '.join(occupied))
        for path in (self.data, self.logs, self.plots):
            path.mkdir(parents=True, exist_ok=False)
        self.tables = {}

    def row(self, table, values):
        self.tables.setdefault(table, []).append(copy.deepcopy(values))

    def flush(self):
        for table, rows in self.tables.items():
            if not rows:
                continue
            keys = list(dict.fromkeys(key for row in rows for key in row))
            with (self.logs / (table + '.csv')).open('x', encoding='utf-8', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=keys)
                writer.writeheader()
                for row in rows:
                    writer.writerow({key: encode(value) if isinstance(value, (dict, tuple, list, np.ndarray)) else value
                                     for key, value in row.items()})


def effective_simulation(spec):
    payload = copy.deepcopy(spec['simulation'])
    constructors = {'motion': MotionConfig, 'sensor': SensorConfig, 'channel': ChannelConfig,
                    'topology': TopologyConfig, 'ekf': EKFConfig, 'reward': RewardConfig}
    for key, constructor in constructors.items():
        payload[key] = constructor(**payload[key])
    sim = SimulationConfig(**payload)
    std = float(sim.sensor.uwb_los_std_m)
    if not np.isfinite(std) or std <= 0:
        raise ValueError('Effective sensor UWB LOS standard deviation must be positive and finite')
    motion = sim.motion
    if not (sim.uav_count == 3 and sim.max_steps == 120 and motion.dt_s == .2
            and motion.max_speed_mps == 12 and motion.max_acceleration_mps2 == 3
            and np.array_equal(motion.bounds_min_m, [0, 0, 2]) and np.array_equal(motion.bounds_max_m, [200, 200, 60])
            and motion.safety_distance_m == 2 and sim.safety_projection_enabled
            and sim.measurement_overhead_enabled and sim.measurement_budget_enabled
            and sim.measurement_update_eligibility_enabled and sim.cooperative_ekf_enabled
            and sim.random_stream_mode == 'indexed-v1' and sim.gnss_mode == 'denied'):
        raise ValueError('Stage-3 simulation does not match the frozen execution profile')
    if content_hash(asdict(sim.sensor)) != content_hash(spec['effective_sensor_config']):
        raise ValueError('Sensor source-of-truth manifest differs from effective configuration')
    if float(spec['range_variance_m2']) != std ** 2 or spec['fixed_bandwidth_fraction'] != .5:
        raise ValueError('Frozen variance or bandwidth contract mismatch')
    if read_json(ROOT / spec['navigation_config'])['controller'] != asdict(FixedNavigationConfig()):
        raise ValueError('Validated fixed-navigation parameters changed')
    return sim


def policy_config(spec):
    cfg = MAPPOConfig(**spec['mappo'])
    if not (cfg.motion_action_mode == 'fixed_navigation_comm_v1' and cfg.communication_learning_scope == 'measurement_only'
            and cfg.ppo_action_logprob_mode == 'sampled' and cfg.node_feature_dim == 37 and cfg.edge_feature_dim == 11
            and cfg.critic_input_dim == 439 and cfg.episode_objective == 'finite_horizon'
            and cfg.include_remaining_time and not cfg.bootstrap_time_limits):
        raise ValueError('Stage-3 measurement-only/finite-horizon model contract is required')
    return cfg


def source_manifest():
    return {name: file_hash(ROOT / name) for name in SOURCE_FILES}


def central_accounting(info, observations):
    central = copy.deepcopy(info['central_state'])
    central['measurement_budget_remaining_ratios'] = {
        uid: float(observations[uid]['measurement_budget_remaining_ratio']) for uid in UIDS}
    return central


def critic_vector(central, cfg):
    return central_state_to_vector(central, uav_count=3, node_feature_dim=37, edge_feature_dim=11,
        max_obstacles=0, include_remaining_time=True, critic_measurement_budget_features=True)


def frozen_fit(*args, **kwargs):
    raise RuntimeError('Normalization is frozen outside TRAIN calibration')


def load_calibration(calibration_id, spec, cfg):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', str(calibration_id)):
        raise ValueError('Invalid calibration-id')
    path = ROOT / 'data/synthetic' / STEM / calibration_id / 'calibration.json'
    data = read_json(path)
    if data.get('schema') != SCHEMA or data.get('ready') is not True:
        raise ValueError('Calibration is missing, invalid, or ready=false; do not start training')
    expected = {'config_hash': content_hash(spec), 'scene_hash': content_hash(contract().make_scene_splits()),
                'sources': source_manifest(), 'effective_sensor_config': asdict(effective_simulation(spec).sensor)}
    for key, value in expected.items():
        if content_hash(data.get(key)) != content_hash(value):
            raise ValueError('Calibration mismatch: ' + key)
    if data.get('fit_split') != 'train' or data.get('fit_scene_ids') != [s['scene_id'] for s in contract().make_scene_splits()['train']]:
        raise ValueError('Calibration provenance must contain only the complete TRAIN split')
    actor = ObservationNormalizer.from_state_dict(data['actor_normalizer'])
    critic = FeatureNormalizer.from_state_dict(data['critic_normalizer'])
    if not actor.fitted or critic.feature_dim != cfg.critic_input_dim or actor.normalizers['node_features'].feature_dim != 37:
        raise ValueError('Normalizer schema mismatch')
    if actor.normalizers['candidate_incoming_edge_features'].feature_dim != 11:
        raise ValueError('Normalizer edge schema mismatch')
    for stat in [*actor.normalizers.values(), critic]:
        if not (np.isfinite(stat.mean).all() and np.isfinite(stat.scale).all() and np.all(stat.scale > 0)
                and np.all(np.asarray(stat.count) >= 0)):
            raise ValueError('Invalid normalization statistics')
    actor.fit = frozen_fit
    critic.fit = frozen_fit
    scale = float(data['credit_scale'])
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('Invalid TRAIN information-credit scale')
    return data, actor, critic, file_hash(path)


def checkpoint_contract(spec, cfg, calibration, calibration_sha):
    return {'schema': SCHEMA, 'motion_action_mode': cfg.motion_action_mode,
        'communication_learning_scope': cfg.communication_learning_scope,
        'actor_node_dim': 37, 'actor_edge_dim': 11, 'critic_dim': 439, 'uav_ids': list(UIDS),
        'model_config': asdict(cfg), 'config_hash': content_hash(spec), 'scene_hash': content_hash(contract().make_scene_splits()),
        'sources': source_manifest(), 'calibration_sha256': calibration_sha,
        'normalization_hash': content_hash([calibration['actor_normalizer'], calibration['critic_normalizer']]),
        'effective_sensor_config': asdict(effective_simulation(spec).sensor),
        'range_variance_m2': effective_simulation(spec).sensor.uwb_los_std_m ** 2,
        'fit_split': 'train', 'seeds': spec['seeds']}


def validate_checkpoint_metadata(sidecar, expected):
    if content_hash(sidecar.get('contract')) != content_hash(expected):
        raise ValueError('Checkpoint contract mismatch: old joint/Stage-2 checkpoints cannot be reused')


def save_checkpoint(output, name, trainer, expected, counters):
    path = output.data / (name + '.pt')
    if path.exists() or path.with_suffix('.json').exists():
        raise FileExistsError('Checkpoint already exists')
    torch.save({'policy_state': trainer.policy.state_dict(), 'optimizer_state': trainer.optimizer.state_dict(),
                'contract': expected, 'counters': dict(counters)}, path)
    write_json(path.with_suffix('.json'), {'contract': expected, 'counters': dict(counters), 'checkpoint_sha256': file_hash(path)})


def load_policy(checkpoint_run, spec, cfg, calibration, calibration_sha):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', str(checkpoint_run)):
        raise ValueError('Invalid checkpoint-run')
    path = ROOT / 'data/synthetic' / STEM / checkpoint_run / 'checkpoint_final.pt'
    sidecar = read_json(path.with_suffix('.json'))
    expected = checkpoint_contract(spec, cfg, calibration, calibration_sha)
    validate_checkpoint_metadata(sidecar, expected)  # Always before deserialization/model creation.
    if file_hash(path) != sidecar.get('checkpoint_sha256'):
        raise ValueError('Checkpoint file SHA mismatch')
    payload = torch.load(path, map_location=cfg.device, weights_only=True)
    validate_checkpoint_metadata({'contract': payload.get('contract')}, expected)
    policy = MAPPOPolicy(cfg, uav_ids=UIDS)
    policy.load_state_dict(payload['policy_state'], strict=True)
    policy.eval()
    return policy


def clipping_profile(actor, observations):
    result = {}
    for key, stat in actor.normalizers.items():
        arrays = [array.reshape(-1, array.shape[-1]) for local in observations.values()
                  for array in actor._values_for_key(local, key) if array.size]
        if arrays:
            values = np.concatenate(arrays)
            z = (values - stat.mean) / stat.scale
            if 'edge_features' in key:
                from ..learning.normalization import _edge_valid_mask
                valid = _edge_valid_mask(values)
            else:
                valid = np.isfinite(values)
            result[key] = {'valid_count': valid.sum(axis=0), 'clipped_count': ((np.abs(z) > float(stat.clip)) & valid).sum(axis=0)}
    return result


def parameter_groups(policy):
    critic_ids = {id(p) for p in policy.critic.parameters()}
    return {name: ('critic' if id(p) in critic_ids else 'actor') for name, p in policy.named_parameters() if p.requires_grad}


def parameter_snapshot(policy):
    return {name: p.detach().cpu().clone() for name, p in policy.named_parameters()}


def parameter_changes(policy, before):
    groups = parameter_groups(policy)
    sums = {'actor': 0., 'critic': 0., 'frozen': 0.}
    for name, parameter in policy.named_parameters():
        group = groups.get(name, 'frozen')
        sums[group] += float(((parameter.detach().cpu() - before[name]) ** 2).sum())
    return {group + '_parameter_change_norm': value ** .5 for group, value in sums.items()}


def build_parser(workflow):
    parser = argparse.ArgumentParser(description='Stage 3 ' + workflow + ': explicit human-run workflow, no import-time execution.')
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--run-id', required=True, help='New exclusive run identifier; existing directories are rejected')
    if workflow in ('train', 'evaluate', 'baselines'):
        parser.add_argument('--calibration-id', required=True)
    if workflow == 'evaluate':
        parser.add_argument('--checkpoint-run', required=True)
    return parser


def episode(output, case, sim, cfg, spec, counters, method, *, actor=None, critic=None,
            trainer=None, policy=None, credit_scale=1., seed=None, action_seed=None, bank=None):
    """One existing environment episode; only explicit human train adds PPO transitions."""
    helper = contract()
    environment = MultiUAVJointEnv(sim)
    obs, info = environment.reset(seed=int(case['seed'] if seed is None else seed), scenario_config=case['scenario'])
    obs = helper.admissible_observations(obs)
    central = central_accounting(info, obs)
    active_steps = dict.fromkeys(UIDS, 0)
    decisions, credits = [], []
    previous_graph = set(environment.snapshot.potential_edges)
    graph_changes, accepted_count, overhead_total, return_sum = 0, 0, 0., 0.
    parking_tracker = StrictParkingTracker(UIDS)
    minimum_pair_distance = float('inf')
    current_transmissions = set()
    original_enqueue = environment._enqueue_messages
    def observed_enqueue(edges, payloads):
        nonlocal current_transmissions
        current_transmissions = set(edges)
        return original_enqueue(edges, payloads)
    environment._enqueue_messages = observed_enqueue
    rng = np.random.default_rng(int(action_seed if action_seed is not None else spec['seeds']['baseline_action']))
    total_steps = int(spec['precheck']['steps_per_scene']) if method == 'precheck' else 120
    episode_id = f"{method}_{case['scene_id']}_{int(case['seed'] if seed is None else seed)}"
    if bank is not None:
        bank.append({'scene_id': case['scene_id'], 'split': case['split'], 'step': 0,
                     'observations': copy.deepcopy(obs), 'central_state': copy.deepcopy(central)})
    for step in range(1, total_steps + 1):
        prefix = {'episode_id': episode_id, 'scene_id': case['scene_id'], 'split': case['split'], 'method': method,
                  'seed': int(case['seed'] if seed is None else seed), 'step': step,
                  'credit_scale_used': credit_scale,
                  'reward_is_calibration_diagnostic_not_training': method in ('precheck', 'calibration')}
        raw = obs
        expected = {uid: {source: helper.expected_relative_credit(raw[uid], source, sim.sensor.uwb_los_std_m ** 2)
                          for source in UIDS if helper.admissible_mask(raw[uid])[UIDS.index(source)]}
                    for uid in UIDS}
        decision = None
        samples = {}
        if method == 'mappo_train':
            decision = trainer.act(raw, central, deterministic=False)
            actions, samples = decision.actions, decision.samples
            counters['neural_actions_executed'] += 1
        elif method == 'mappo':
            normalized = actor.transform(raw)
            communication = {}
            with torch.no_grad():
                for uid in UIDS:
                    communication[uid], samples[uid] = policy.act(normalized[uid], deterministic=True)
            actions, _ = compose_fixed_navigation_actions(raw, communication)
            counters['neural_actions_executed'] += 1
        else:
            if method == 'random':
                selections = {uid: helper.random_selection(raw[uid], rng) for uid in UIDS}
            elif method == 'greedy':
                selections = {uid: helper.greedy_selection(raw[uid], credit_scale, sim.sensor.uwb_los_std_m ** 2) for uid in UIDS}
            elif method == 'none' or (method == 'precheck' and step == 1):
                selections = dict.fromkeys(UIDS, 'none')
            else:
                selections = {uid: helper.round_robin_selection(raw[uid], step) for uid in UIDS}
            communication = helper.communication_actions(raw, selections)
            actions, _ = compose_fixed_navigation_actions(raw, communication)
        if actor is not None:
            output.row('normalization_clipping', {**prefix, 'actor_clipping_by_feature': clipping_profile(actor, raw)})
        pre_velocity = {uid: environment.states[uid].velocity_mps.copy() for uid in UIDS}  # diagnostics only
        pre_position = {uid: environment.states[uid].position_m.copy() for uid in UIDS}
        next_obs, _, terminated, truncated, next_info = environment.step(actions)
        counters['joint_environment_steps'] += 1
        limit = 24 if method == 'precheck' else (1440 if method == 'mappo_train' else 480 if method == 'calibration' else 1080)
        if counters['joint_environment_steps'] > limit:
            raise RuntimeError('Workflow interaction budget exceeded')
        obs = helper.admissible_observations(next_obs)
        central = central_accounting(next_info, obs)
        post_position = {uid: environment.states[uid].position_m.copy() for uid in UIDS}
        pair_distances = joint_closest_approach(pre_position, post_position)
        minimum_pair_distance = min(minimum_pair_distance, min(row['distance_m'] for row in pair_distances))
        if minimum_pair_distance < 2:
            raise RuntimeError('Minimum synchronized UAV separation violated')
        parking = parking_tracker.update(
            {uid: float(np.linalg.norm(post_position[uid] - environment.goals[uid])) for uid in UIDS},
            {uid: float(np.linalg.norm(environment.states[uid].velocity_mps)) for uid in UIDS},
            collision=bool(next_info['collision']))
        for row in pair_distances:
            output.row('pair_distances', {**prefix, **row})
        learning_rewards, terms = helper.reward_components(next_info, credit_scale,
            communication_cost=spec['reward']['communication_cost'], waste_cost=spec['reward']['waste_cost'])
        return_sum += float(np.mean(list(learning_rewards.values())))
        graph = set(environment.snapshot.potential_edges)
        graph_changed = graph != previous_graph
        graph_changes += int(graph_changed)
        previous_graph = graph
        for event in next_info['relative_range_update_events']:
            credit = helper.relative_linear_credit(event)
            if credit > 0:
                credits.append(credit)
            accepted_count += int(bool(event.get('accepted', False)))
            output.row('fusion_events', {**event, **prefix, 'linear_relative_credit_proxy': credit,
                'true_error_actor_available': False, 'true_error_reward_used': False})
        for name, events in (('measurement_gates', next_info['measurement_gate_events']),
                             ('measurement_overhead', next_info['measurement_overhead_events']),
                             ('measurement_budget', next_info['measurement_budget_events']),
                             ('invalid_measurement_actions', next_info.get('invalid_measurement_action_events', []))):
            for event in events:
                output.row(name, {**event, **prefix})
        for uid in UIDS:
            values = list(expected[uid].values())
            spread = max(values) - min(values) if len(values) >= 2 else 0.
            best = max(values) if values else 0.
            positive = bool(min(best / credit_scale, 1.) > spec['reward']['communication_cost'])
            local_decision = {'scene_id': case['scene_id'], 'split': case['split'], 'uav_id': uid,
                'legal_peer_count': len(values), 'expected_credit_spread': float(spread), 'greedy_positive': positive}
            decisions.append(local_decision)
            output.row('candidate_values', {**prefix, **local_decision, 'unscaled_expected_relative_credit': expected[uid]})
            execution = next_info['action_execution'][uid]
            velocity = environment.states[uid].velocity_mps.copy()
            actual_acceleration = (velocity - pre_velocity[uid]) / .2
            safe = (not next_info['collision'] and np.isfinite(actual_acceleration).all()
                    and np.linalg.norm(actual_acceleration) <= 3 + 1e-8
                    and not any(execution.get(key, False) for key in ('boundary_projected', 'safety_projected', 'action_clipped')))
            if not safe:
                raise RuntimeError('Physical safety/execution check failed; preserve outputs and stop')
            active_steps[uid] += int(np.linalg.norm(velocity) > .5)
            sample = samples.get(uid)
            entropy, lp_error = None, None
            if sample is not None:
                evaluated_observation = decision.observations[uid] if decision is not None else normalized[uid]
                evaluated_policy = trainer.policy if decision is not None else policy
                with torch.no_grad():
                    evaluated_lp, evaluated_entropy = evaluated_policy.evaluate_actions(evaluated_observation, sample)
                entropy = float(evaluated_entropy)
                lp_error = abs(float(evaluated_lp) - sample.old_log_prob)
                if lp_error > 2e-6 or not np.isfinite(entropy):
                    raise RuntimeError('Sampled categorical probability mismatch')
            output.row('actions', {**prefix, 'uav_id': uid, 'action_mask': raw[uid]['action_mask'],
                'raw_local_estimate': raw[uid]['estimate'], 'raw_local_covariance': raw[uid]['covariance'],
                'sampled_measurement_source': actions[uid]['measurement_uav_id'],
                'executed_measurement_source': next_info['actions'][uid]['measurement_uav_id'],
                'sampled_measurement_index': sample.measurement_index if sample is not None else None,
                'sampled_old_log_prob': sample.old_log_prob if sample is not None else None,
                'measurement_probabilities': sample.measurement_probabilities if sample is not None else None,
                'categorical_entropy': entropy, 'sampling_logprob_recompute_error': lp_error,
                'old_team_critic_value': decision.value if decision is not None else None,
                'learned_delta_dimensions': 0, 'learned_bandwidth_dimensions': 0, 'learned_link_dimensions': 0,
                'fixed_navigation_requested': actions[uid]['delta_v'], 'navigation_execution': execution['delta_v_applied'],
                'fixed_bandwidth_requested': actions[uid]['bandwidth'], 'bandwidth_applied': next_info['applied_bandwidth'][uid],
                'budget_remaining_before': raw[uid]['measurement_budget_remaining'],
                'budget_remaining_after': obs[uid]['measurement_budget_remaining'], **terms[uid]})
            estimate = environment.ekfs[uid].state
            actual_position = environment.states[uid].position_m
            covariance = environment.ekfs[uid].covariance
            if not np.isfinite(covariance).all() or np.linalg.eigvalsh((covariance + covariance.T)/2).min() < -1e-8:
                raise RuntimeError('Nonfinite/non-PSD EKF covariance')
            output.row('localization_diagnostics', {**prefix, 'uav_id': uid,
                'estimated_position_m': estimate[:3], 'true_position_m_diagnostic': actual_position,
                'position_error_m_diagnostic': float(np.linalg.norm(estimate[:3] - actual_position)),
                'velocity_error_mps_diagnostic': float(np.linalg.norm(estimate[3:6] - velocity)),
                'position_covariance_trace_m2_reported': float(np.trace(covariance[:3, :3])),
                'actual_speed_mps': float(np.linalg.norm(velocity)), 'actual_acceleration_mps2': actual_acceleration,
                'actual_goal_distance_m': float(np.linalg.norm(actual_position - environment.goals[uid])),
                'strict_parking_hold_steps': parking['per_uav'][uid]['hold_steps'],
                'strict_parking_ever_success': parking['per_uav'][uid]['ever_success'],
                'boundary_projected': bool(execution.get('boundary_projected', False)),
                'safety_projected': bool(execution.get('safety_projected', False)),
                'actor_available': False, 'reward_used': False, 'collision': bool(next_info['collision'])})
        overhead_total += sum(float(term['actual_overhead_packets']) for term in terms.values())
        output.row('joint', {**prefix, 'team_learning_reward': float(np.mean(list(learning_rewards.values()))),
            'ordinary_state_transmitted_packets': len(current_transmissions),
            'ordinary_state_payload_bytes': len(current_transmissions) * sim.channel.packet_size_bytes,
            'ordinary_state_cost_action_independent_contract': True,
            'physical_potential_edges': sorted(graph), 'physical_graph_changed': graph_changed,
            'parent_terminated': bool(terminated), 'parent_truncated': bool(truncated)})
        if method == 'mappo_train':
            trainer.reconcile_action_execution(decision, next_info)
            trainer.add_transition(decision, learning_rewards, central, done=bool(terminated or truncated),
                                   terminated=bool(terminated), truncated=bool(truncated))
            counters['rollout_additions'] += 1
        if bank is not None:
            bank.append({'scene_id': case['scene_id'], 'split': case['split'], 'step': step,
                         'observations': copy.deepcopy(obs), 'central_state': copy.deepcopy(central)})
        if terminated or truncated:
            break
    if step != total_steps:
        raise RuntimeError('Unexpected early environment termination')
    return {'scene_id': case['scene_id'], 'split': case['split'], 'method': method, 'steps': step,
        'requires_graph_change': case['requires_graph_change'], 'physical_graph_changes': graph_changes,
        'moving_fractions': {uid: active_steps[uid] / step for uid in UIDS},
        'accepted_updates': accepted_count, 'actual_measurement_overhead_packets': overhead_total,
        'minimum_synchronized_pair_distance_m': minimum_pair_distance,
        'strict_parking_agent_successes': sum(parking['per_uav'][uid]['ever_success'] for uid in UIDS),
        'strict_parking_joint_success': parking['joint_ever_success'],
        'team_return': return_sum, 'decision_rows': decisions, 'positive_credits': credits}


def precheck_workflow(output, spec, sim, cfg, counters):
    originals = []
    def forbid(target, name):
        originals.append((target, name, getattr(target, name)))
        def fail(*args, **kwargs):
            raise RuntimeError('Forbidden preparation-precheck operation: ' + name)
        setattr(target, name, fail)
    try:
        for name in ('__init__', 'update', 'add_transition', 'collect_rollout'):
            forbid(MAPPOTrainer, name)
        for name in ('__init__', 'act', 'act_with_trace'):
            forbid(MAPPOPolicy, name)
        forbid(torch, 'load')
        for optimizer in (torch.optim.Adam, torch.optim.AdamW, torch.optim.SGD):
            forbid(optimizer, 'step')
        cases = contract().make_scene_splits()['train'][:3]
        results = [episode(output, case, sim, cfg, spec, counters, 'precheck') for case in cases]
        assert counters['joint_environment_steps'] == 24
        assert all(counters[key] == 0 for key in counters if key != 'joint_environment_steps')
        passed = sum(r['accepted_updates'] for r in results) > 0 and sum(r['actual_measurement_overhead_packets'] for r in results) > 0
        report = {'passed': passed, 'training_readiness': 'not_evaluated', 'normalization_fitted': False,
            'real_PPO_updates_verified': False, 'counts': dict(counters), 'episodes': results}
        write_json(output.logs / 'precheck.json', report)
        return {'passed': passed, 'training_readiness': 'not_evaluated'}
    finally:
        for target, name, value in reversed(originals):
            setattr(target, name, value)


def calibration_workflow(output, spec, sim, cfg, counters):
    bank, metrics, decisions, credits = [], [], [], []
    for case in contract().make_scene_splits()['train']:
        result = episode(output, case, sim, cfg, spec, counters, 'calibration', bank=bank)
        metrics.append({key: value for key, value in result.items() if key not in ('decision_rows', 'positive_credits')})
        decisions.extend(result['decision_rows']); credits.extend(result['positive_credits'])
    actor = ObservationNormalizer().fit([frame['observations'] for frame in bank])
    vectors = [critic_vector(frame['central_state'], cfg) for frame in bank]
    critic = FeatureNormalizer().fit(vectors)
    counters['normalization_fits'] = 2
    scale = float(np.quantile(credits, spec['reward']['credit_scale_quantile'])) if credits else 0.
    # Greedy utility must be checked against the resulting TRAIN-only scale.
    if scale > 0:
        decisions = []
        for row in output.tables['candidate_values']:
            maximum = max(row['unscaled_expected_relative_credit'].values(), default=0.)
            positive = min(maximum/scale, 1.) > spec['reward']['communication_cost']
            row['greedy_positive_after_train_scale'] = positive
            decisions.append({**row, 'greedy_positive': positive})
    for stat in [*actor.normalizers.values(), critic]:
        if not np.isfinite(stat.mean).all() or not np.isfinite(stat.scale).all() or not np.all(stat.scale > 0):
            raise ValueError('Calibration produced nonfinite normalization statistics')
    readiness = contract().calibration_readiness(metrics, decisions, credits, scale, spec['calibration_gates'])
    artifact = {'schema': SCHEMA, 'ready': readiness['ready'], 'readiness': readiness, 'fit_split': 'train',
        'fit_scene_ids': [s['scene_id'] for s in contract().make_scene_splits()['train']],
        'config_hash': content_hash(spec), 'scene_hash': content_hash(contract().make_scene_splits()),
        'sources': source_manifest(), 'actor_normalizer': actor.state_dict(), 'critic_normalizer': critic.state_dict(),
        'credit_scale': scale, 'effective_sensor_config': asdict(sim.sensor),
        'range_variance_m2': sim.sensor.uwb_los_std_m**2, 'seeds': spec['seeds'],
        'constant_features': {key: np.flatnonzero(normalizer.scale <= normalizer.eps).tolist()
                              for key, normalizer in actor.normalizers.items()}, 'full_horizon_frames': len(bank)}
    write_json(output.data / 'observation_bank.json', bank)
    write_json(output.data / 'calibration.json', artifact)
    write_json(output.logs / 'calibration_readiness.json', readiness)
    return {'passed': readiness['ready'], 'ready': readiness['ready'], 'readiness': readiness}


def training_workflow(output, spec, sim, cfg, counters, calibration_id):
    calibration, actor, critic, calibration_sha = load_calibration(calibration_id, spec, cfg)
    expected = checkpoint_contract(spec, cfg, calibration, calibration_sha)
    trainer = MAPPOTrainer(cfg, uav_ids=UIDS, normalizer=actor, central_normalizer=critic,
                           seed=spec['seeds']['parameter_initialization'])
    assert not trainer.rollout and not trainer.optimizer.state
    save_checkpoint(output, 'checkpoint_initial', trainer, expected, counters)
    original_step = trainer.optimizer.step
    gradient_steps = []
    def counted_step(*args, **kwargs):
        if counters['optimizer_steps'] >= spec['training']['max_optimizer_steps']:
            raise RuntimeError('Optimizer-step budget exceeded')
        groups = parameter_groups(trainer.policy)
        squared = {'actor': 0., 'critic': 0.}
        for name, parameter in trainer.policy.named_parameters():
            if parameter.grad is not None:
                if name not in groups:
                    raise RuntimeError('Inactive parameter received gradient')
                if not torch.isfinite(parameter.grad).all():
                    raise FloatingPointError('Nonfinite gradient')
                squared[groups[name]] += float((parameter.grad.detach() ** 2).sum())
        gradient_steps.append({group + '_gradient_norm_postclip': value ** .5 for group, value in squared.items()})
        result = original_step(*args, **kwargs)
        counters['optimizer_steps'] += 1
        return result
    trainer.optimizer.step = counted_step
    scenes = contract().make_scene_splits()['train']
    for index in range(spec['training']['episodes']):
        torch.manual_seed(spec['seeds']['rollout_base'] + index)
        result = episode(output, scenes[index % len(scenes)], sim, cfg, spec, counters, 'mappo_train',
            trainer=trainer, actor=actor, critic=critic, credit_scale=calibration['credit_scale'],
            seed=spec['seeds']['rollout_base'] + index)
        records = trainer._training_records()
        nondegenerate = any(record['sample'].measurement_index != 3
                            and bool(record['observation']['action_mask'][record['sample'].measurement_index])
                            and np.isfinite(record['advantage']) and abs(record['advantage']) > 1e-12 for record in records)
        values = np.array([record['old_value'] for record in records])
        returns = np.array([record['return'] for record in records])
        normalized_advantages = np.array([record['advantage'] for record in records])
        old_stats = {'critic_mse_before': float(np.mean((values - returns)**2)),
            'critic_explained_variance_before': float(1 - np.var(returns-values)/np.var(returns)) if np.var(returns) > 1e-12 else None,
            'advantage_mean_normalized': float(normalized_advantages.mean()),
            'advantage_std_normalized': float(normalized_advantages.std())}
        for record in records:
            output.row('ppo_records', {'update': index + 1, 'uav_id': record['uav_id'],
                'advantage_normalized': record['advantage'], 'return': record['return'], 'old_value': record['old_value'],
                'advantage_raw': record['return'] - record['old_value'],
                'old_log_prob': record['sample'].old_log_prob, 'measurement_index': record['sample'].measurement_index,
                'measurement_action_mask': record['observation']['action_mask']})
        write_json(output.data / f'rollout_update_{index+1:03d}.json', [
            {'observation': record['observation'], 'central_vector': record['central_vector'],
             'sample': vars(record['sample']), 'return': record['return'], 'advantage_normalized': record['advantage'],
             'old_value': record['old_value'], 'uav_id': record['uav_id']} for record in records])
        before = parameter_snapshot(trainer.policy)
        steps_before, gradient_start = counters['optimizer_steps'], len(gradient_steps)
        stats = trainer.update()  # Only this explicit human-run workflow may call a true PPO update.
        counters['ppo_updates'] += 1
        if not all(np.isfinite(value) for value in stats.values()):
            raise FloatingPointError('Nonfinite PPO metrics')
        changes = parameter_changes(trainer.policy, before)
        if changes['frozen_parameter_change_norm'] != 0:
            raise RuntimeError('Frozen action head changed')
        recent = gradient_steps[gradient_start:]
        actor_grad = sum(r['actor_gradient_norm_postclip'] for r in recent)
        if nondegenerate and actor_grad > 1e-12 and changes['actor_parameter_change_norm'] == 0:
            raise RuntimeError('Parameters stalled after a completed nondegenerate update with nonzero actor gradient')
        if counters['optimizer_steps'] == steps_before or trainer.rollout:
            raise RuntimeError('Missing PPO optimizer update or uncleared rollout')
        for record in records:
            with torch.no_grad():
                updated_lp, updated_entropy = trainer.policy.evaluate_actions(record['observation'], record['sample'])
                updated_value = float(trainer.policy.value(record['central_vector']))
            ratio = float(torch.exp(updated_lp - record['sample'].old_log_prob))
            clipped = float(np.clip(ratio, 1-cfg.clip_epsilon, 1+cfg.clip_epsilon))
            output.row('ppo_post_update_records', {'update': index + 1, 'uav_id': record['uav_id'],
                'post_update_log_prob': float(updated_lp), 'old_log_prob': record['sample'].old_log_prob,
                'post_update_ratio': ratio, 'post_update_entropy': float(updated_entropy),
                'post_update_clipped_surrogate': min(ratio*record['advantage'], clipped*record['advantage']),
                'post_update_value': updated_value, 'return': record['return'],
                'post_update_squared_value_error': (updated_value-record['return'])**2,
                'note': 'final updated policy on saved online samples; minibatch PPO metrics logged separately'})
        if source_manifest() != expected['sources']:
            raise RuntimeError('Source changed during training; stop and retain artifacts')
        output.row('training_curve', {'update': index + 1, **stats, **old_stats, **changes,
            'episode_team_return': result['team_return'], 'accepted_updates': result['accepted_updates'],
            'actual_measurement_overhead_packets': result['actual_measurement_overhead_packets'],
            'optimizer_steps_this_update': counters['optimizer_steps'] - steps_before,
            'optimizer_steps_total': counters['optimizer_steps'],
            'actor_gradient_norm_postclip_mean': float(np.mean([r['actor_gradient_norm_postclip'] for r in recent])),
            'critic_gradient_norm_postclip_mean': float(np.mean([r['critic_gradient_norm_postclip'] for r in recent])),
            'nondegenerate_update': nondegenerate, 'joint_environment_steps_total': counters['joint_environment_steps']})
        save_checkpoint(output, f'checkpoint_update_{index+1:03d}', trainer, expected, counters)
        print(f"update={index+1}/{spec['training']['episodes']} steps={counters['joint_environment_steps']} Adam={counters['optimizer_steps']}", flush=True)
    assert counters['joint_environment_steps'] == 1440 and counters['ppo_updates'] == 12 and counters['rollout_additions'] == 1440
    save_checkpoint(output, 'checkpoint_final', trainer, expected, counters)
    training_plot(output)
    return {'passed': True, 'interpretation': 'Training-chain smoke only; no generalization conclusion'}


def evaluation_workflow(output, spec, sim, cfg, counters, calibration_id, checkpoint_run, workflow):
    calibration, actor, critic, calibration_sha = load_calibration(calibration_id, spec, cfg)
    policy = load_policy(checkpoint_run, spec, cfg, calibration, calibration_sha) if workflow == 'evaluate' else None
    counters['checkpoint_loads'] = int(policy is not None)
    methods = ('mappo',) if policy is not None else ('random', 'greedy', 'none')
    summaries = []
    for method in methods:
        for index, case in enumerate(contract().make_scene_splits()['validation']):
            result = episode(output, case, sim, cfg, spec, counters, method, actor=actor, critic=critic,
                policy=policy, credit_scale=calibration['credit_scale'], action_seed=spec['seeds']['baseline_action'] + index)
            summaries.append({key: value for key, value in result.items() if key not in ('decision_rows', 'positive_credits')})
    write_json(output.logs / 'evaluation_summary.json', {'episodes': summaries,
        'paired_validation_scenes_and_indexed_noise': True, 'normalization_fits': 0,
        'MAPPO_deterministic': policy is not None, 'interpretation': 'Small held-out validation, not generalization proof'})
    return {'passed': True, 'episodes': summaries}


def training_plot(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = output.tables['training_curve']
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), layout='constrained')
    for axis, field in zip(axes.flat, ('episode_team_return', 'actor_loss', 'critic_loss', 'entropy')):
        axis.plot([r['update'] for r in rows], [r[field] for r in rows], marker='o')
        axis.set_xlabel('PPO update'); axis.set_ylabel(field); axis.grid(alpha=.2)
    fig.savefig(output.plots / 'training_curve.png', dpi=160)
    plt.close(fig)


def main(workflow):
    args = build_parser(workflow).parse_args()
    spec = read_json(args.config)
    sim, cfg = effective_simulation(spec), policy_config(spec)
    # Reserving outputs must precede all environment/model/load operations.
    output = RunOutput(args.run_id)
    write_json(output.data / 'configuration.json', spec)
    write_json(output.data / 'manifest.json', {'schema': SCHEMA, 'workflow': workflow,
        'sources': source_manifest(), 'scene_splits': contract().make_scene_splits(),
        'config_hash': content_hash(spec), 'effective_simulation_config': asdict(sim),
        'effective_sensor_config': asdict(sim.sensor), 'range_variance_m2': sim.sensor.uwb_los_std_m ** 2,
        'truth_initialization_inherited': True, 'runtime_navigation_truth_input': False})
    counters = {'joint_environment_steps': 0, 'ppo_updates': 0, 'optimizer_steps': 0,
                'rollout_additions': 0, 'checkpoint_loads': 0, 'normalization_fits': 0, 'neural_actions_executed': 0}
    status, failure = 'failed', None
    try:
        if workflow == 'precheck':
            result = precheck_workflow(output, spec, sim, cfg, counters)
        elif workflow == 'calibrate':
            result = calibration_workflow(output, spec, sim, cfg, counters)
        elif workflow == 'train':
            result = training_workflow(output, spec, sim, cfg, counters, args.calibration_id)
        else:
            result = evaluation_workflow(output, spec, sim, cfg, counters, args.calibration_id,
                                         getattr(args, 'checkpoint_run', None), workflow)
        status = 'passed' if result.get('passed', False) else 'not_ready'
    except Exception as exc:
        failure = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        output.flush()
        write_json(output.logs / 'run_status.json', {'workflow': workflow, 'status': status, 'failure': failure,
                                                   'counts': counters, 'run_id': args.run_id})
    print(encode({'workflow': workflow, 'status': status, 'counts': counters, 'result': result}))
    if status != 'passed':
        raise SystemExit(2)
