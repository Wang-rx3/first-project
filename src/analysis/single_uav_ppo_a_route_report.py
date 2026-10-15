"""Consolidate independent offline route audits without collecting data."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

STEM = 'single_uav_ppo_a_route_feasibility_v3_seed2026_20261011'
OLD = 'single_uav_ppo_a_norm_transfer_v2_seed2026_20261011'
LOG = Path('logs/preliminary')/STEM
DATA = Path('data/synthetic')/STEM
PLOT = Path('plots/preliminary')/STEM


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def read_csv(path):
    with Path(path).open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def write_json(path, value):
    payload = json.dumps(value, indent=2, ensure_ascii=False)
    with Path(path).open('x', encoding='utf-8') as stream:
        stream.write(payload)


def main():
    from ..simulation.motion_model import braking_distance_target_speed
    inherited = read_json(Path('logs/preliminary')/OLD/'protected_inputs_before.json')
    protected = {**inherited, **read_json(LOG/'protected_inputs_before.json')}
    assert all(hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest for path, digest in protected.items())
    spec = read_json(DATA/'assessment_specification.json')
    assert hashlib.sha256(Path(spec['frozen_statistics_path']).read_bytes()).hexdigest()==spec['frozen_statistics_sha256']
    phases = read_csv(LOG/'reward_audit/route_reward_phase_summary.csv')
    online = {r['arm']: r for r in phases if r['phase']=='online'}
    gradient = read_json(LOG/'gradient_audit_complete_v2/gradient_audit_summary.json')
    exact = read_json(LOG/'exact_gradient_v2/exact_offline_gradient_verification.json')
    findings = {row['arm']: row for row in gradient['findings']}
    alignment = {}
    for arm in ('N', 'G', 'R'):
        with (Path('logs/preliminary')/OLD/arm/'steps.csv').open(encoding='utf-8', newline='') as stream:
            rows = [r for r in csv.DictReader(stream) if r['phase']=='online']
        distance_error = np.abs([float(r['goal_distance_m_estimated'])-float(r['goal_distance_m_actual']) for r in rows])
        velocity_error = np.abs([np.linalg.norm(json.loads(r['node_raw'])[3:6])-
                                 np.linalg.norm(json.loads(r['motion_telemetry'])['pre_velocity_mps']) for r in rows])
        alignment[arm] = {'post_goal_distance_error_median': float(np.median(distance_error)),
                          'post_goal_distance_error_max': float(np.max(distance_error)),
                          'pre_speed_error_median': float(np.median(velocity_error)),
                          'pre_speed_error_max': float(np.max(velocity_error))}
    write_json(LOG/'estimated_truth_alignment.json', {'env_steps': 0, 'PPO_updates': 0, 'arms': alignment})
    target_speed = {str(d): braking_distance_target_speed(d, goal_threshold_m=1., near_goal_radius_m=5.,
        stable_speed_mps=.5, braking_acceleration_mps2=2., braking_margin_m=2.,
        far_target_speed_mps=6., max_speed_mps=12.) for d in (1., 2., 3., 4., 5., 10., 12.)}
    write_json(LOG/'integration_corrections.json', {
        'reward_initial_output_relocated_from': str(Path('logs/preliminary')/OLD/'reward_audit'),
        'canonical_reward_output': str(LOG/'reward_audit'),
        'legacy_target_speed_verified_mps': target_speed,
        'target_speed_note': '5 m is not a hard cruise/braking switch; the legacy stopping envelope applies at all distances',
        'R_initial_validation_549_steps_reason': 'eight strict-success early terminations; not horizon cut',
        'gradient_history_note': 'gradient_audit/ was the first cumulative-snapshot attempt; unique192Adam analysis is gradient_audit_complete_v2/',
        'replay_note': 'normalized node42 and critic174 are saved; exact first-block replay was verified using the known empty single-UAV graph',
        'earlier_reward_magnitude_statement': 'weak action magnitude is not isolated as a causal explanation; persistent acceleration/no braking are directly observed',
        'C_statistics_note': 'old-stat C is a deliberate historical control, not a new refit or a violation of the current frozen-stat assessment'})
    decision = {'date': '2026-10-11', 'primary_evidence': 'R full-statistics head-reset PPO route',
        'fixed_statistics_sha256': spec['frozen_statistics_sha256'],
        'confirmed_training_coverage_problem': {'online_steps': 720, 'within_2m_steps': int(online['R']['near_goal_d_le_2_rows']),
            'within_4m_steps': int(online['R']['near_goal_d_le_4_rows']),
            'within_4m_overspeed_braking_steps': int(online['R']['near_goal_d_le_4_overspeed_braking_rows']),
            'strict_success_bonus_steps': int(online['R']['logged_bonus_triggered_rows'])},
        'clear_policy_or_reward_implementation_bug_found': False,
        'supported_local_training_repair_identified': False,
        'repair_smoke_licensed': False, 'new_smoke_started': False,
        'reason': 'coverage deficit and weak critic are real diagnostics but do not isolate a single local repair; adding reset curriculum/changing reward or clipping would test a new hypothesis',
        'technical_route_recommendation': 'fixed basic navigation baseline; local EKF + GAT + MAPPO focus on cooperative localization and communication decisions',
        'end_to_end_PPO_general_infeasibility_claim': False, 'fixed_controller_implemented': False,
        'B_paused': True, 'new_env_steps': 0, 'new_PPO_updates': 0, 'new_Adam_steps': 0, 'normalizer_refits': 0,
        'no_obstacle_or_multiseed': True}
    assert decision['confirmed_training_coverage_problem']=={'online_steps': 720, 'within_2m_steps': 0,
        'within_4m_steps': 70, 'within_4m_overspeed_braking_steps': 2, 'strict_success_bonus_steps': 0}
    assert exact['env_steps']==exact['PPO_updates']==exact['optimizer_steps']==0
    assert all(item['weights_unchanged'] and item['analytic_score_gradient_max_error']<1e-5 for item in exact['arms'].values())
    write_json(DATA/'route_decision.lock.json', decision)
    write_json(LOG/'verification.json', {'protected_files_unchanged': len(protected), 'fixed_statistics_unchanged': True,
        'exact_gradient_arms_verified': 4, 'unique_existing_Adam_steps': 192,
        'new_env_steps': 0, 'new_PPO_updates': 0, 'new_optimizer_steps': 0, 'new_teacher_samples': 0,
        'normalizer_fit_or_update': 0, 'B_checkpoint_loads': 0, 'unlicensed_smoke_started': False})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8), constrained_layout=True)
    arms = ('C', 'N', 'G', 'R')
    axes[0].bar(arms, [int(online[a]['near_goal_d_le_2_rows']) for a in arms], label='Within 2 m')
    axes[0].bar(arms, [int(online[a]['near_goal_d_le_4_rows']) for a in arms], alpha=.5, label='Within 4 m')
    axes[0].set(title='Actual online coverage', ylabel='Steps / 720 per condition', ylim=(0, 80))
    axes[0].legend(fontsize=8)
    axes[1].bar(arms, [findings[a]['critic_preupdate_explained_variance_all_records'] for a in arms])
    axes[1].axhline(0., color='black', linewidth=.6)
    axes[1].set(title='Weak value predictions', ylabel='Pooled pre-update explained variance')
    probes = read_csv(Path('logs/preliminary')/OLD/'primary_fixed_training_probe_updates.csv')
    means = [np.mean([float(r['mean_delta_v_norm']) for r in probes if int(r['update'])==i
                     and float(r['physical_goal_x_m'])==0 and float(r['estimated_velocity_x_mps'])==0]) for i in range(7)]
    axes[2].plot(range(7), means, marker='o')
    axes[2].set(title='R motion bias after PPO', xlabel='Actual PPO update', ylabel='Fixed-input mean increment norm (m/s)')
    for axis in axes:
        axis.grid(alpha=.2)
    PLOT.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOT/'route_diagnostic_evidence.png', dpi=180)
    fig.savefig(PLOT/'route_diagnostic_evidence.svg')
    plt.close(fig)
    print(json.dumps({'protected_files': len(protected), 'new_env_steps': 0, 'smoke_licensed': False,
                     'state_alignment': alignment, 'target_speed': target_speed}, indent=2))


if __name__=='__main__':
    main()
