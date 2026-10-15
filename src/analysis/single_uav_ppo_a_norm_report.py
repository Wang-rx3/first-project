"""Read-only quantitative report and plots for the registered A-only smoke."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

STEM = 'single_uav_ppo_a_norm_transfer_v2_seed2026_20261011'
LOG = Path('logs/preliminary') / STEM
DATA = Path('data/synthetic') / STEM
PLOT = Path('plots/preliminary') / STEM


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def rows(path):
    with Path(path).open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def write_json(path, value):
    payload = json.dumps(value, ensure_ascii=False, indent=2,
                         default=lambda item: item.item() if isinstance(item, np.generic) else str(item))
    with Path(path).open('x', encoding='utf-8') as stream:
        stream.write(payload)


def write_csv(path, values):
    with Path(path).open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(values[0]))
        writer.writeheader()
        writer.writerows(values)


def physical_goal_probes():
    """Apply +/-5 physical metres before normalization; no reset or step."""
    import torch
    from ..experiments.single_uav_ppo_a_foundation import load_run
    from ..experiments.single_uav_ppo_a_norm_calibration import load_full_statistics
    from ..experiments.single_uav_ppo_a_norm_compatibility import _offline_trainer, _node_arrays, _latent_mean
    parent, _, old = load_run('data/configs/single_uav_ppo_a_foundation_v1.json')
    new = load_full_statistics(DATA/'full_train_statistics.json')
    license_record = read_json(LOG/'output_head_license.json')
    output = []
    torch.set_num_threads(1)
    for condition, statistics in [('C', old), ('N', new)]:
        trainer, _ = _offline_trainer(statistics, source_path=parent['A_source']['checkpoint'], condition=condition)
        mean, scale, clip = _node_arrays(statistics[0])
        for probe in license_record['probe_rows']:
            for offset in (-5., 0., 5.):
                raw = np.asarray(probe['node_raw'], float).copy()
                raw[30:33] = [offset, 0., 0.]
                raw[3:6] = 0.
                normalized = np.clip((raw-mean)/scale, -clip, clip)
                latent = _latent_mean(trainer, normalized.astype(np.float32))
                action = np.tanh(latent)*trainer.policy._delta_component_limit()
                output.append({'condition': condition, 'scenario_id': probe['scenario_id'],
                    'physical_goal_x_m': offset, 'estimated_speed_mps': 0.,
                    'normalized_goal': json.dumps(normalized[30:33].tolist()),
                    'mean_delta_v': json.dumps(action.tolist()), 'mean_delta_v_norm': float(np.linalg.norm(action)),
                    'direction_cosine': float(action[0]*np.sign(offset)/max(np.linalg.norm(action), 1e-12)) if offset else None})
    write_csv(LOG/'physical_goal_counterfactual.csv', output)
    return {'env_steps': 0, 'PPO_updates': 0, 'B_loads': 0, 'probe_count': len(output),
            'earlier_compatibility_perturbation_unit': 'normalized units; not physical metres', 'rows': output}


def checkpoint_checks(arms):
    import torch
    checks = {}
    for arm in arms:
        initial = torch.load(DATA/arm/'initial.pt', map_location='cpu', weights_only=False)
        final = torch.load(DATA/arm/'final.pt', map_location='cpu', weights_only=False)
        actor_names = [name for name in final['policy_state_dict'] if not name.startswith('critic.')]
        differences = {name: float(torch.linalg.vector_norm(final['policy_state_dict'][name]-initial['policy_state_dict'][name]))
                       for name in final['policy_state_dict']}
        optimizer_steps = {name: int(value['step'].item()) for name, value in final['optimizer_state_dict']['state'].items()}
        check = {'all_weights_finite': all(torch.isfinite(value).all().item() for value in final['policy_state_dict'].values()),
                 'actor_l2_delta': float(np.linalg.norm([differences[name] for name in actor_names])),
                 'critic_l2_delta': float(np.linalg.norm([differences[name] for name in differences if name.startswith('critic.')])),
                 'changed_tensor_names': [name for name, delta in differences.items() if delta > 0],
                 'actual_optimizer_steps_by_parameter': optimizer_steps,
                 'initial_optimizer_empty': not initial['optimizer_state_dict']['state'],
                 'node_statistics_unchanged': initial['normalization_stats']==final['normalization_stats'],
                 'central_statistics_unchanged': initial['central_normalization_stats']==final['central_normalization_stats'],
                 'actual_online_steps': final['online_steps'], 'actual_updates': final['new_ppo_updates']}
        assert check['actor_l2_delta']>0 and check['critic_l2_delta']>0 and check['all_weights_finite']
        assert check['node_statistics_unchanged'] and check['central_statistics_unchanged'] and check['initial_optimizer_empty']
        # Unused single-node communication/attention tensors may have no Adam state.
        assert set(optimizer_steps.values())=={48} and check['actual_online_steps']==720 and check['actual_updates']==6
        checks[arm] = check
    write_json(LOG/'checkpoint_independent_checks.json', {'env_steps': 0, 'PPO_updates': 0, 'B_loads': 0, 'arms': checks})
    return checks


def primary_update_probes():
    """Replay training-only fixed inputs through each saved R update."""
    import torch
    from ..experiments.single_uav_ppo_a_foundation import load_run
    from ..experiments.single_uav_ppo_a_norm_calibration import load_full_statistics
    from ..experiments.single_uav_ppo_a_norm_compatibility import _offline_trainer, _node_arrays, _latent_mean
    decision = read_json(DATA/'primary_decision.lock.json')
    arm = decision['primary']
    parent, _, _ = load_run('data/configs/single_uav_ppo_a_foundation_v1.json')
    new = load_full_statistics(DATA/'full_train_statistics.json')
    trainer, _ = _offline_trainer(new, source_path=parent['A_source']['checkpoint'], condition=arm)
    mean, scale, clip = _node_arrays(new[0])
    probes = read_json(LOG/'output_head_license.json')['probe_rows']
    output = []
    for update in range(7):
        checkpoint = torch.load(DATA/arm/('initial.pt' if update==0 else f'update_{update:02d}.pt'),
                                map_location='cpu', weights_only=False)
        trainer.policy.load_state_dict(checkpoint['policy_state_dict'])
        for probe in probes:
            for goal_x, velocity_x in [(-5., 0.), (0., 0.), (5., 0.), (0., -1.2), (0., 1.2)]:
                raw = np.asarray(probe['node_raw'], float).copy()
                raw[30:33] = [goal_x, 0., 0.]
                raw[3:6] = [velocity_x, 0., 0.]
                normalized = np.clip((raw-mean)/scale, -clip, clip)
                latent = _latent_mean(trainer, normalized.astype(np.float32))
                action = np.tanh(latent)*trainer.policy._delta_component_limit()
                output.append({'arm': arm, 'update': update, 'scenario_id': probe['scenario_id'],
                    'physical_goal_x_m': goal_x, 'estimated_velocity_x_mps': velocity_x,
                    'mean_delta_v': json.dumps(action.tolist()), 'mean_delta_v_norm': float(np.linalg.norm(action)),
                    'action_x': float(action[0])})
    write_csv(LOG/'primary_fixed_training_probe_updates.csv', output)
    write_json(LOG/'primary_fixed_training_probe_audit.json', {'env_steps': 0, 'PPO_updates': 0, 'B_loads': 0,
        'primary': arm, 'input_source': 'six licensing training states; raw physical goal and estimated velocity counterfactuals',
        'rows': len(output), 'validation_used': False,
        'limitation': 'off-manifold counterfactual inputs; only goal/velocity slices changed, not coherent regenerated simulator states'})


def write_unified_configuration():
    import hashlib
    result = read_json(LOG/'smoke_result.json')
    configurations = {arm: read_json(DATA/arm/'effective_config.json') for arm in result['arms']}
    baseline = configurations[result['primary']]['baseline']
    ppo = configurations[result['primary']]['ppo']
    assert all(config['baseline']==baseline and config['ppo']==ppo for config in configurations.values())
    stat_path = DATA/'full_train_statistics.json'
    payload = {'registered_run': read_json(DATA/'run_specification.lock.json'), 'primary': result['primary'],
        'actual_shared_baseline': baseline, 'actual_shared_ppo': ppo,
        'strict_success': {'actual_goal_distance_max_m': 1., 'actual_speed_max_mps': .5,
                           'consecutive_decision_steps': 5, 'no_collision': True},
        'horizon_steps': 120, 'control_period_s': .2, 'online_steps_per_condition': 720,
        'actual_updates_per_condition': 6, 'actual_adam_steps_per_condition': 48,
        'conditions': {arm: {'definition': config['condition'], 'normalization_provenance': config['normalizer'],
                            'initialization': read_json(DATA/arm/'initialization.json')}
                       for arm, config in configurations.items()},
        'full_statistics_sha256': hashlib.sha256(stat_path.read_bytes()).hexdigest(),
        'scenario_manifest': str((DATA/'scenario_manifest.json').resolve()),
        'scenario_manifest_sha256': hashlib.sha256((DATA/'scenario_manifest.json').read_bytes()).hexdigest(),
        'actual_research_env_steps': read_json(LOG/'train_runtime_audit.json')['combined_research_env_steps'],
        'B_paused': True, 'teacher_samples': 0, 'obstacle_or_multiseed_started': False,
        'next_stage_candidate': result['next_stage_candidate']}
    write_json(DATA/'effective_unified_configuration.json', payload)


def main():
    result = read_json(LOG/'smoke_result.json')
    checkpoint_checks(result['arms'])
    primary_update_probes()
    write_unified_configuration()
    stats = read_json(DATA/'full_train_statistics.json')
    with np.load(DATA/'calibration_bank.npz', allow_pickle=False) as bank:
        raw, step = bank['node_features'], bank['step']
    distributions = []
    for condition, state in [('C', stats['old_statistics_state']['observation_normalizer']),
                             ('N', stats['observation_normalizer'])]:
        node = state['normalizers']['node_features']
        unclipped = (raw-np.asarray(node['mean']))/np.asarray(node['scale'])
        for label, low, high in [('reset', 0, 0), ('1-16', 1, 16), ('17-40', 17, 40),
                                  ('41-80', 41, 80), ('81-120', 81, 120)]:
            selection = (step >= low)&(step <= high)
            for index in list(range(3, 6))+list(range(15, 33)):
                values = raw[selection, index]
                distributions.append({'condition': condition, 'time_bin': label, 'index': index,
                    'count': len(values), 'raw_min': float(values.min()), 'raw_median': float(np.median(values)),
                    'raw_max': float(values.max()), 'clip_fraction': float(np.mean(np.abs(unclipped[selection, index])>state['clip'])),
                    'normalized_max_abs_before_clip': float(np.abs(unclipped[selection, index]).max())})
    write_csv(LOG/'normalization_time_clipping.csv', distributions)
    compact, feedback, trajectories, movement = [], {}, {}, {}
    val0 = read_json(DATA/'scenario_manifest.json')['validation'][0]['scenario_id']
    for arm, arm_result in result['arms'].items():
        for phase, evaluated in arm_result['evaluations'].items():
            compact.append({'arm': arm, 'phase': phase, **evaluated['metrics']['all']})
        step_rows = rows(LOG/arm/'steps.csv')
        trajectories[arm] = [row for row in step_rows if row['phase']=='final_validation' and row['scenario_id']==val0]
        online = [row for row in step_rows if row['phase']=='online']
        replay = {(int(row['block']), int(row['block_step'])): row for row in rows(LOG/arm/'action_drift_06.csv')}
        advantages = {(block, entry['record_index']): entry['advantage']
            for block in range(6) for entry in read_json(LOG/arm/f'update_{block+1:02d}.json')['per_record_audit']}
        groups = {}
        for name, chosen in [('boundary', [r for r in online if r['boundary_projected']=='True']),
                             ('ordinary', [r for r in online if r['boundary_projected']!='True']),
                             ('early', [r for r in online if int(r['episode_step'])<=80]),
                             ('late', [r for r in online if int(r['episode_step'])>80])]:
            if not chosen:
                groups[name] = {'steps': 0}
                continue
            groups[name] = {'steps': len(chosen),
                'reward_mean': float(np.mean([float(r['learning_reward']) for r in chosen])),
                'advantage_mean': float(np.mean([advantages[int(r['block']), int(r['block_step'])] for r in chosen])),
                'final_logprob_change_mean': float(np.mean([float(replay[int(r['block']), int(r['block_step'])]['logprob_change']) for r in chosen])),
                'actual_acceleration_max': max(float(r['acceleration_mps2_actual']) for r in chosen)}
        online_nodes = np.array([json.loads(r['node_normalized']) for r in online])
        groups['feature_clip_fractions_online'] = np.mean(np.abs(online_nodes)>=10-1e-6, axis=0).tolist()
        groups['nominal_acceleration_max'] = max(float(r['nominal_acceleration_mps2']) for r in online)
        groups['source_motion_head_parameter_delta'] = arm_result['parameter_deltas']
        feedback[arm] = groups
        movement[arm] = {}
        for phase in ('initial_validation', 'final_validation', 'initial_challenge', 'final_challenge'):
            chosen = [row for row in step_rows if row['phase']==phase]
            near = [row for row in chosen if float(row['actual_pre_goal_distance_m'])<=2.]
            movement[arm][phase] = {'steps': len(chosen),
                'distance_decreasing_fraction': float(np.mean([float(r['goal_distance_m_actual'])<float(r['actual_pre_goal_distance_m']) for r in chosen])),
                'minimum_actual_goal_distance_m': min(float(r['goal_distance_m_actual']) for r in chosen),
                'maximum_actual_speed_mps': max(float(r['speed_mps_actual']) for r in chosen),
                'maximum_parking_hold': max(int(r['parking_hold']) for r in chosen), 'near_goal_steps': len(near),
                'near_goal_deceleration_steps': sum(float(r['speed_mps_actual'])<np.linalg.norm(json.loads(r['motion_telemetry'])['pre_velocity_mps'])-1e-8 for r in near)}
    write_csv(LOG/'evaluation_comparison.csv', compact)
    write_json(LOG/'learning_feedback_comparison.json', feedback)
    write_json(LOG/'approach_and_braking_comparison.json', movement)
    counterfactual = physical_goal_probes()
    write_json(LOG/'physical_goal_counterfactual_audit.json', counterfactual)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)
    arms = list(result['arms'])
    for arm in arms:
        record = result['arms'][arm]
        metrics = record['evaluations']['final_validation']['metrics']['all']
        t = [int(r['episode_step'])*.2 for r in trajectories[arm]]
        axes[0, 0].plot(t, [float(r['goal_distance_m_actual']) for r in trajectories[arm]], label=arm)
        axes[0, 1].plot(t, [float(r['speed_mps_actual']) for r in trajectories[arm]], label=arm)
    axes[0, 0].axhline(1., color='black', linestyle='--', linewidth=.8)
    axes[0, 1].axhline(.5, color='black', linestyle='--', linewidth=.8)
    axes[0, 0].set(ylabel='Actual goal distance (m)', xlabel='Time (s)', title='First registered validation case')
    axes[0, 1].set(ylabel='Actual speed (m/s)', xlabel='Time (s)', title='Same case: actual speed')
    values = [result['arms'][arm]['evaluations']['final_validation']['metrics']['all'] for arm in arms]
    axes[1, 0].bar(arms, [m['success_stable_count'] for m in values], color=['#5173ad', '#de9957', '#60a38f', '#a269a3'][:len(arms)])
    axes[1, 0].set(ylabel='Stable successes / 8', ylim=(0, 8), title='Held-out normal cases (single smoke)')
    axes[1, 1].bar(arms, [100*m['boundary_projection_fraction'] for m in values], color=['#5173ad', '#de9957', '#60a38f', '#a269a3'][:len(arms)])
    axes[1, 1].set(ylabel='Boundary projection (% actual steps)', title='Same held-out cases')
    axes[0, 0].legend()
    axes[0, 1].legend()
    for axis in axes.flat:
        axis.grid(alpha=.2)
    PLOT.mkdir(parents=True, exist_ok=True)
    figure.savefig(PLOT/'normal_parking_comparison.png', dpi=180)
    figure.savefig(PLOT/'normal_parking_comparison.svg')
    plt.close(figure)
    write_json(LOG/'analysis_runtime_audit.json', {'env_steps': 0, 'PPO_updates': 0, 'B_loads': 0,
        'offline_physical_goal_probes': len(counterfactual['rows']), 'smoke_primary': result['primary'],
        'next_stage_candidate': result['next_stage_candidate'], 'validation_used_for_fitting_or_selection': False})
    print(json.dumps({'primary': result['primary'], 'next_stage_candidate': result['next_stage_candidate'],
                     'normalization_rows': len(distributions), 'evaluation_rows': len(compact)}, indent=2))


if __name__ == '__main__':
    main()
