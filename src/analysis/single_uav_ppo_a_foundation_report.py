"""Read-only numerical summary and plots from actual A foundation artifacts."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from ..experiments.single_uav_ppo_smoke import write_json, sha256_file

ROOT = Path('logs/preliminary/single_uav_ppo_a_foundation_v1_seed2026')
DATA = Path('data/synthetic/single_uav_ppo_a_foundation_v1_seed2026')
PLOTS = Path('plots/preliminary/single_uav_ppo_a_foundation_v1_seed2026')


def rows(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def vector(row, key):
    return np.asarray(json.loads(row[key]), dtype=float)


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    result = json.loads((ROOT / 'normal_training_result.json').read_text(encoding='utf-8'))
    diagnostic = rows(ROOT / 'diagnostics_trajectory_steps.csv')
    ranges = rows(ROOT / 'normalization_ranges.csv')
    extra = {'diagnostic_EKF_position_error': {}, 'arms': {}}
    for condition in ('SS', 'SV', 'RV'):
        selected = [row for row in diagnostic if row['condition'] == condition]
        errors = [float(np.linalg.norm(vector(row, 'estimated_position_m') - vector(row, 'actual_position_m'))) for row in selected]
        extra['diagnostic_EKF_position_error'][condition] = {'median_m': float(np.median(errors)), 'max_m': max(errors)}
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for condition, color in zip(('SS', 'SV', 'RV'), ('#238b45', '#2171b5', '#cb181d')):
        selected = [row for row in diagnostic if row['condition'] == condition and row['block_case'] == '0']
        axes[0, 0].plot([int(row['episode_step']) for row in selected], [float(row['goal_distance_actual_m']) for row in selected], label=condition, color=color)
        clip = [int(row['clip_count'])/720 for row in ranges if row['condition'] == condition]
        axes[0, 1].plot(range(42), clip, label=condition, color=color)
    axes[0, 0].axhline(1, color='gray', linestyle=':')
    axes[0, 0].set(xlabel='Decision step', ylabel='Actual goal distance (m)', title='Frozen-policy diagnosis: train case 00')
    axes[0, 1].set(xlabel='Node feature index', ylabel='Fraction clipped', title='Frozen train-statistics coverage (720 steps each)')
    for arm, color in zip(('G', 'P'), ('#e6550d', '#3182bd')):
        steps = rows(ROOT / arm / 'steps.csv')
        updates = [json.loads((ROOT / arm / f'update_{i:02d}.json').read_text(encoding='utf-8')) for i in range(1, 7)]
        online = [row for row in steps if row['phase'] == 'online']
        extra['arms'][arm] = {
            'minimum_online_goal_distance_m': min(float(row['goal_distance_m_actual']) for row in online),
            'maximum_online_parking_hold': max(int(row['parking_hold']) for row in online),
            'online_success_bonus_steps': sum(float(row['strict_success_bonus']) > 0 for row in online),
            'maximum_actual_acceleration_mps2': max(float(row['acceleration_mps2_actual']) for row in steps),
            'maximum_actual_acceleration_without_boundary_projection_mps2': max(float(row['acceleration_mps2_actual']) for row in steps if row['boundary_projected'] == 'False'),
            'max_on_policy_logprob_error': max(item['max_on_policy_logprob_error'] for item in updates),
            'all_GAE_checks_passed': all(item['gae_recurrence_matches_training_records'] and item['gae_does_not_cross_done_boundary'] for item in updates),
            'updates': [item['update_stats'] for item in updates],
        }
        axes[1, 0].plot(range(1, 7), [item['value_mse_preupdate'] for item in updates], 'o-', color=color, label=arm)
        values = [result['arms'][arm][phase]['metrics']['all']['median_final_goal_distance_m'] for phase in ('initial_validation', 'final_validation')]
        axes[1, 1].plot([0, 1], values, 'o-', label=arm, color=color)
    axes[1, 0].set(xlabel='PPO update (120 real samples each)', ylabel='Value MSE before update', title='Real PPO: 6 updates / 48 Adam steps per arm')
    axes[1, 1].set(xticks=[0, 1], xticklabels=['Before', 'After'], ylabel='Median actual final distance (m)', title='Fresh normal validation: 0/8 stable in both arms')
    for axis in axes.flat:
        axis.grid(alpha=.2)
        axis.legend(frameon=False)
    PLOTS.mkdir(parents=True, exist_ok=True)
    plot = PLOTS / 'foundation_diagnostics.png'
    if plot.exists():
        raise FileExistsError(plot)
    fig.savefig(plot, dpi=160)
    plt.close(fig)
    # Separate closed-loop trajectories and the physical feasibility witness.
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    legacy = rows(ROOT / 'legacy_feasibility_steps.csv')
    witness = [row for row in legacy if row['scenario_id'] == 'smoke_train_normal_00']
    axes[0].plot([int(row['episode_step'])+1 for row in witness], [float(row['actual_goal_distance_m']) for row in witness], label='distance (m)')
    axes[0].plot([int(row['episode_step'])+1 for row in witness], [float(row['actual_speed_mps']) for row in witness], label='speed (m/s)')
    axes[0].axhline(.5, linestyle=':', color='gray')
    axes[0].set(title='Fixed open-loop feasibility; no PPO samples', xlabel='Decision step')
    for index, arm in enumerate(('G', 'P'), 1):
        selected = [row for row in rows(ROOT / arm / 'steps.csv') if row['scenario_id'] == 'smoke_train_normal_00' and row['phase'] in ('initial_train_eval', 'final_train_eval')]
        for phase, label in (('initial_train_eval', 'Before'), ('final_train_eval', 'After')):
            episode = [row for row in selected if row['phase'] == phase]
            axes[index].plot([float(row['goal_distance_m_actual']) for row in episode], label=label)
        axes[index].set(title=f'{arm}: same normal train case 00', xlabel='Decision step', ylabel='Actual goal distance (m)')
    for axis in axes:
        axis.grid(alpha=.2)
        axis.legend(frameon=False)
    trajectory_plot = PLOTS / 'feasibility_and_policy_trajectories.png'
    if trajectory_plot.exists():
        raise FileExistsError(trajectory_plot)
    fig.savefig(trajectory_plot, dpi=160)
    plt.close(fig)
    witnesses = json.loads((DATA / 'open_loop_witnesses.json').read_text())
    extra['maximum_feasibility_action_component_mps'] = max(float(np.abs(np.asarray(w['actions'])).max()) for w in witnesses.values())
    extra['feasibility_within_policy_component_limit'] = bool(extra['maximum_feasibility_action_component_mps'] <= .6/np.sqrt(3))
    extra['plots'] = {str(path): sha256_file(path) for path in (plot, trajectory_plot)}
    write_json(ROOT / 'numerical_failure_analysis.json', extra)
    print(json.dumps({key: value for key, value in extra.items() if key != 'arms'}, indent=2))
    print(json.dumps({arm: {key: value for key, value in details.items() if key != 'updates'} for arm, details in extra['arms'].items()}, indent=2))


if __name__ == '__main__':
    main()
