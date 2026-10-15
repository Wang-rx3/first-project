"""Staged, bounded A-only foundation experiment; exclusive new artifacts."""
from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from dataclasses import asdict
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from .single_uav_ppo_baseline import load_single_uav_ppo_config, create_single_uav_ppo_trainer, verify_actor_observation_contract
from .single_uav_ppo_smoke import sha256_file, write_json, CSVLog, ParkingReporter, raw_action_diagnostics, assert_online_transition, _reset, canonical_hash
from .single_uav_ppo_a_scenarios import build_a_foundation_scenarios, build_open_loop_witness
from .single_uav_ppo_a_support import (load_shared_a_statistics, initialize_a_foundation, capture_policy_distribution,
    parameter_snapshot, grouped_parameter_deltas, PPOUpdateAudit, update_with_audit,
    summarize_foundation_episodes, normal_release_gate)

DEFAULT_CONFIG = Path('data/configs/single_uav_ppo_a_foundation_v1.json')


def load_run(config_path=DEFAULT_CONFIG):
    spec = json.loads(Path(config_path).read_text(encoding='utf-8'))
    assert sha256_file(spec['baseline_config']) == spec['baseline_config_sha256']
    for record in (spec['A_source'], spec['historical_A_initializer'], spec['shared_A_train_statistics']):
        assert sha256_file(record.get('path', record.get('checkpoint'))) == record['sha256']
    stats = load_shared_a_statistics(spec['shared_A_train_statistics']['path'])
    base = replace(load_single_uav_ppo_config(spec['baseline_config']), train_potential_statistics=stats[2])
    return spec, base, stats


def check_protected(spec):
    root = Path(spec['log_root'])
    before = json.loads((root / 'protected_inputs_before.json').read_text(encoding='utf-8'))
    changes = [path for path, expected in before.items() if not Path(path).is_file() or sha256_file(path) != expected]
    if changes:
        raise AssertionError(f'Protected historical inputs changed: {changes}')
    return {'protected_file_count': len(before), 'all_unchanged': True}


def phase_feasibility(spec, base):
    from .single_uav_ppo_a_diagnostics import run_legacy_feasibility
    data_root, log_root = Path(spec['data_root']), Path(spec['log_root'])
    if (log_root / 'legacy_feasibility_manifest.json').exists():
        raise FileExistsError('Feasibility already executed; preserve its artifacts')
    scenarios = build_a_foundation_scenarios(spec['seed'])
    witnesses = {str(scene['scenario_id']): build_open_loop_witness(scene, base)
                 for scene in scenarios['normal_train'] + scenarios['validation_normal']}
    write_json(data_root / 'scenario_manifest.json', scenarios)
    write_json(data_root / 'open_loop_witnesses.json', witnesses)
    result = run_legacy_feasibility(base, scenarios, witnesses, log_root)
    write_json(log_root / 'legacy_feasibility_result.json', result)
    assert result['passed_all'], 'Physics gate failed; no semantics change or PPO permitted'
    return result['manifest']


def cached_inputs(spec):
    data_root, log_root = Path(spec['data_root']), Path(spec['log_root'])
    read = lambda path: json.loads(path.read_text(encoding='utf-8'))
    scenarios = read(data_root / 'scenario_manifest.json')
    witnesses = read(data_root / 'open_loop_witnesses.json')
    legacy = read(log_root / 'legacy_feasibility_result.json')
    assert legacy['passed_all'] and legacy['manifest']['static_cause_confirmed']
    return scenarios, witnesses, legacy


def phase_semantics(spec, base):
    from .single_uav_ppo_a_environment import create_foundation_environment
    from .single_uav_ppo_a_diagnostics import run_new_semantics_feasibility
    scenarios, witnesses, legacy = cached_inputs(spec)
    result = run_new_semantics_feasibility(base, scenarios, witnesses, legacy,
                                         create_foundation_environment, Path(spec['log_root']))
    write_json(Path(spec['log_root']) / 'new_semantics_feasibility_result.json', result)
    assert result['passed_all'], 'Semantics equivalence gate failed'
    return result['manifest']


def phase_diagnostics(spec, base):
    from .single_uav_ppo_a_diagnostics import run_closed_loop_diagnostics, run_action_reward_fixtures
    scenarios, _, _ = cached_inputs(spec)
    root = Path(spec['log_root'])
    assert json.loads((root / 'new_semantics_feasibility_manifest.json').read_text())['passed_all']
    result = run_closed_loop_diagnostics(base, scenarios, spec['A_source']['checkpoint'],
        spec['historical_A_initializer']['path'], spec['shared_A_train_statistics']['path'], root)
    write_json(root / 'closed_loop_diagnostics_result.json', result)
    assert result['passed'], 'Frozen-policy diagnostics mechanics failed'
    fixtures = run_action_reward_fixtures(base, scenarios, root)
    write_json(root / 'action_reward_fixtures_result.json', fixtures)
    assert fixtures['passed']
    return {'diagnostics': result['manifest'], 'fixtures': fixtures['manifest']}


def collect_block(trainer, env, scene, *, arm, phase, stage, block, online, step_log, episode_log):
    """Collect genuine online decisions; diagnostics never enter this buffer."""
    if online:
        assert len(trainer.rollout) == 0
    episodes = []
    ended, resets = True, 0
    for block_step in range(120):
        if ended:
            episode_id = f'{arm}_{phase}_{stage}_{block}_{scene["scenario_id"]}_r{resets}'
            obs, info = _reset(env, scene, episode_id)
            resets += 1
            tracker, counts = ParkingReporter(), Counter()
        uid = env.uav_ids[0]
        pre_p = env.states[uid].position_m.copy()
        pre_v = env.states[uid].velocity_mps.copy()
        pre_est_distance = env.estimated_goal_distance()
        decision = trainer.act(obs, info.get('central_state'), deterministic=not online)
        with torch.no_grad():
            distribution = capture_policy_distribution(trainer, decision)
        sampled = np.asarray(decision.samples[uid].delta_v).copy()
        logprob = float(decision.samples[uid].old_log_prob)
        danger = raw_action_diagnostics(env, sampled)
        raw = np.asarray(obs[uid]['node_features'], dtype=float)[0]
        norm = np.asarray(decision.observations[uid]['node_features'], dtype=float)[0]
        assert np.isfinite(raw).all() and np.isfinite(norm).all()
        next_obs, _, terminated, truncated, next_info = env.step(decision.actions)
        trainer.reconcile_action_execution(decision, next_info)
        assert_online_transition(decision, sampled, logprob)
        if online:
            trainer.add_transition(decision, next_info['learning_rewards'], next_info['central_state'],
                done=terminated or truncated, terminated=terminated, truncated=truncated)
        execution = next_info['action_execution'][uid]
        executed = np.asarray(execution['delta_v_applied'], dtype=float)
        p, v = env.states[uid].position_m.copy(), env.states[uid].velocity_mps.copy()
        dist, speed = float(next_info['strict_goal_distance_m_actual']), float(next_info['strict_speed_mps_actual'])
        collision = bool(next_info.get('collision', False))
        counts['steps'] += 1
        tracker.observe(dist, speed, collision, counts['steps'])
        physical = scene['kind'] != 'normal'
        counts['physical_contact_steps'] += int(bool(next_info.get('physical_collision', False)))
        counts['raw_action_danger_steps'] += int(bool(danger['raw_action_physical_danger']))
        counts['raw_action_physical_denominator'] += int(danger['raw_action_physical_danger'] is not None)
        counts['boundary_attempt_steps'] += int(danger['raw_action_boundary_danger'])
        counts['boundary_projection_steps'] += int(bool(execution.get('boundary_projected', False)))
        counts['safety_projection_steps'] += int(bool(execution.get('safety_projected', False)))
        reward = float(next_info['learning_rewards'][uid])
        counts['learning_return'] += reward
        step_log.append({'arm': arm, 'phase': phase, 'stage': stage, 'block': block, 'block_step': block_step,
            'episode_id': episode_id, 'episode_step': counts['steps'], 'scenario_id': scene['scenario_id'],
            'split': scene['split'], 'kind': scene['kind'], 'side': scene['side'],
            'sample_source': 'online_policy_rollout' if online else 'deterministic_evaluation',
            'node_raw': raw.tolist(), 'node_normalized': norm.tolist(),
            'node_clipped': (np.abs(norm) >= float(trainer.normalizer.state_dict()['clip']) - 1e-6).tolist(),
            'estimated_position_m': env.ekfs[uid].state[:3].tolist(),
            'estimated_velocity_mps': env.ekfs[uid].state[3:6].tolist(),
            'actual_position_m': p.tolist(), 'actual_velocity_mps': v.tolist(),
            'goal_distance_m_actual': dist, 'goal_distance_m_estimated': env.estimated_goal_distance(),
            'actual_progress_m': float(np.linalg.norm(pre_p-env.goals[uid]))-dist,
            'estimated_progress_m': pre_est_distance-env.estimated_goal_distance(),
            'speed_mps_actual': speed, 'acceleration_mps2_actual': float(np.linalg.norm(v-pre_v)/.2),
            'sampled_delta_raw': distribution['sample_delta_raw'], 'sampled_delta_v': sampled.tolist(),
            'distribution_raw_mean': distribution['raw_mean'], 'distribution_raw_std': distribution['raw_std'],
            'executed_delta_v': executed.tolist(), 'total_correction_delta_v': (executed-sampled).tolist(),
            'action_execution': execution, 'old_logprob': logprob, 'logprob_action': 'sampled',
            'safety_projected': bool(execution.get('safety_projected', False)),
            'boundary_projected': bool(execution.get('boundary_projected', False)), **danger,
            'learning_reward': reward, 'reward_components': next_info['learning_reward_components'],
            'parking_hold': tracker.hold, 'strict_hold_steps': int(next_info['strict_hold_steps']),
            'strict_success': bool(next_info['strict_success']), 'strict_success_bonus': next_info['strict_success_bonus'],
            'physical_bypass': bool(next_info['strict_physical_pass_complete']) if physical else None,
            'collision': collision, 'physical_collision': bool(next_info.get('physical_collision', False)),
            'terminated': bool(terminated), 'truncated': bool(truncated)})
        ended = terminated or truncated
        cut = block_step == 119 and not ended
        if ended or cut:
            success = bool(next_info['strict_success'])
            assert not success or tracker.reached
            row = {'arm': arm, 'phase': phase, 'stage': stage, 'block': block, 'episode_id': episode_id,
                'scenario_id': scene['scenario_id'], 'split': scene['split'], 'kind': scene['kind'], 'side': scene['side'],
                'geometry_sha256': scene['geometry_sha256'], 'noise_seed': scene['noise_seed'], **dict(counts),
                'stable_arrival': success, 'task_success': success, 'parking_diagnostic': tracker.reached,
                'first_parking_step': tracker.first_reached_step, 'strict_success': success,
                'physical_bypass': bool(next_info['strict_physical_pass_complete'] and not tracker.collided) if physical else None,
                'collision': tracker.collided, 'terminated': bool(terminated), 'truncated': bool(truncated),
                'time_limit_reached_raw': bool(truncated), 'timeout_failure': bool(truncated and not success),
                'rollout_window_cut': cut, 'final_actual_goal_distance_m': dist, 'final_actual_speed_mps': speed}
            episode_log.append(row)
            episodes.append(row)
            if not online:
                break
        obs, info = next_obs, next_info
    return {'steps': block_step+1, 'episodes': episodes, 'resets': resets}


def evaluate_cases(trainer, env, cases, *, arm, phase, stage, step_log, episode_log):
    before = parameter_snapshot(trainer.policy)
    rows, steps = [], 0
    for index, scene in enumerate(cases):
        result = collect_block(trainer, env, scene, arm=arm, phase=phase, stage=stage, block=index,
                               online=False, step_log=step_log, episode_log=episode_log)
        rows.extend(result['episodes'])
        steps += result['steps']
    assert grouped_parameter_deltas(before, trainer.policy)['total_actor']['l2'] == 0
    assert len(trainer.rollout) == 0
    return {'steps': steps, 'episodes': rows, 'metrics': summarize_foundation_episodes(rows)}


def save_checkpoint(path, trainer, *, arm, online_steps, initialization, updates):
    if path.exists():
        raise FileExistsError(path)
    torch.save({'policy_state_dict': trainer.policy.state_dict(), 'optimizer_state_dict': trainer.optimizer.state_dict(),
        'trainer_config': asdict(trainer.config), 'normalization_stats': trainer.normalizer.state_dict(),
        'central_normalization_stats': trainer.central_normalizer.state_dict(), 'uav_count': 1,
        'arm': arm, 'online_steps': online_steps, 'new_ppo_updates': updates,
        'initialization': initialization, 'teacher_data_count': 0}, path)


def phase_train(spec, base, stats):
    from .single_uav_ppo_a_environment import create_foundation_environment
    root, data = Path(spec['log_root']), Path(spec['data_root'])
    scenarios, _, _ = cached_inputs(spec)
    assert json.loads((root / 'new_semantics_feasibility_manifest.json').read_text())['passed_all']
    assert json.loads((root / 'diagnostics_manifest.json').read_text())['trajectory_rows'] == 2160
    assert json.loads((root / 'action_reward_fixtures_manifest.json').read_text())['transition_count'] == 12
    report = {'primary_arm': 'P', 'B_paused': True, 'arms': {}, 'online_steps': 0, 'evaluation_steps': 0,
              'teacher_data_count': 0, 'new_calibration_steps': 0, 'assistance_invocations': 0}
    train_cases = scenarios['normal_train']
    val_cases = scenarios['validation_normal']
    for arm in ('G', 'P'):
        arm_data, arm_logs = data / arm, root / arm
        arm_data.mkdir(exist_ok=False)
        arm_logs.mkdir(exist_ok=False)
        env = create_foundation_environment(base)
        obs, _ = _reset(env, train_cases[0], f'{arm}_construction_probe')
        trainer = create_single_uav_ppo_trainer(env, base, normalizer=stats[0], central_normalizer=stats[1],
            normalizer_provenance=stats[3], action_log_path=arm_logs / 'sampled_execution.csv')
        initialization = initialize_a_foundation(trainer, spec['A_source']['checkpoint'], arm,
            old_v1_initial_path=spec['historical_A_initializer']['path'])
        if arm == 'G':
            assert initialization['old_v1_initial_verification']['target_matches_old_v1_initial']
        contract = verify_actor_observation_contract(env, obs, normalizer=trainer.normalizer)
        write_json(arm_data / 'initialization.json', initialization)
        write_json(arm_data / 'actor_contract.json', contract)
        write_json(arm_data / 'effective_training_config.json', {'simulation': base.to_dict(),
                   'ppo': asdict(trainer.config), 'statistics_provenance': stats[3]})
        initial_params = parameter_snapshot(trainer.policy)
        step_log, ep_log = CSVLog(arm_logs / 'steps.csv'), CSVLog(arm_logs / 'episodes.csv')
        pre_train = evaluate_cases(trainer, env, train_cases, arm=arm, phase='initial_train_eval', stage='normal', step_log=step_log, episode_log=ep_log)
        pre_val = evaluate_cases(trainer, env, val_cases, arm=arm, phase='initial_validation', stage='normal', step_log=step_log, episode_log=ep_log)
        # Restore the same action-sampling stream after deterministic evaluation.
        torch.manual_seed(spec['seed'])
        audit = PPOUpdateAudit(trainer)
        updates, online_episodes = [], []
        for block, scene in enumerate(train_cases):
            collected = collect_block(trainer, env, scene, arm=arm, phase='online', stage='normal', block=block,
                                      online=True, step_log=step_log, episode_log=ep_log)
            assert collected['steps'] == 120
            online_episodes.extend(collected['episodes'])
            update = update_with_audit(trainer, audit, expected_samples=120)
            assert update['on_policy_logprob_within_tolerance'] and update['normalizers_unchanged']
            assert update['gae_recurrence_matches_training_records'] and update['gae_does_not_cross_done_boundary']
            assert update['all_weights_finite'] and update['rollout_empty_after_update']
            assert update['optimizer_steps'] == (block+1)*8
            update.update(arm=arm, stage='normal', block=block, new_ppo_update=block+1, online_steps=(block+1)*120)
            write_json(arm_logs / f'update_{block+1:02d}.json', update)
            updates.append(update)
        assert audit.steps == 48 and len(updates) == 6
        after_train = evaluate_cases(trainer, env, train_cases, arm=arm, phase='final_train_eval', stage='normal', step_log=step_log, episode_log=ep_log)
        after_val = evaluate_cases(trainer, env, val_cases, arm=arm, phase='final_validation', stage='normal', step_log=step_log, episode_log=ep_log)
        deltas = grouped_parameter_deltas(initial_params, trainer.policy)
        assert deltas['total_actor']['l2'] > 0 and deltas['critic']['l2'] > 0
        assert deltas['first_gat']['l2'] > 0 or deltas['deep_gat']['l2'] > 0
        result = {'initial_train': pre_train, 'initial_validation': pre_val, 'final_train': after_train,
            'final_validation': after_val, 'online_metrics': summarize_foundation_episodes(online_episodes),
            'online_steps': 720, 'new_ppo_updates': 6, 'actual_optimizer_steps': audit.steps,
            'parameter_deltas': deltas, 'mechanics_passed': True,
            'gate': normal_release_gate(after_train['metrics'], after_val['metrics'])}
        save_checkpoint(arm_data / 'normal_final.pt', trainer, arm=arm, online_steps=720, initialization=initialization, updates=6)
        write_json(arm_logs / 'normal_result.json', result)
        report['arms'][arm] = result
        report['online_steps'] += 720
        report['evaluation_steps'] += sum(item['steps'] for item in (pre_train, pre_val, after_train, after_val))
        if arm == 'P':
            primary_trainer, primary_env, primary_initialization = trainer, env, initialization
        else:
            audit.close()
    gate = report['arms']['P']['gate']['passed']
    report['normal_release_gate_passed'] = gate
    report['single_stage_executed'] = False
    report['full_stage_executed'] = False
    report['stop_reason'] = 'primary_normal_release_gate_failed' if not gate else 'normal_gate_passed_conditional_stage_pending'
    write_json(root / 'normal_training_result.json', report)
    if gate:
        raise RuntimeError('Normal gate passed: execute preregistered conditional stages before final report')
    return {'online_steps': report['online_steps'], 'evaluation_steps': report['evaluation_steps'],
            'normal_release_gate_passed': gate, 'stop_reason': report['stop_reason']}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=str(DEFAULT_CONFIG))
    parser.add_argument('--phase', choices=['feasibility', 'semantics', 'diagnostics', 'train'], required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    spec, base, stats = load_run(args.config)
    check_protected(spec)
    from ..simulation.environment import MultiUAVJointEnv
    allowed = {Path(spec['A_source']['checkpoint']).resolve(), Path(spec['historical_A_initializer']['path']).resolve()}
    original_load, original_step = torch.load, MultiUAVJointEnv.step
    model_loads, transition_counter = [], [0]
    def guarded_load(path, *positional, **kwargs):
        resolved = Path(path).resolve()
        if resolved not in allowed:
            raise AssertionError(f'A-only phase refuses checkpoint: {resolved}')
        model_loads.append(str(resolved))
        return original_load(path, *positional, **kwargs)
    def counted_step(env, action):
        transition_counter[0] += 1
        return original_step(env, action)
    torch.load, MultiUAVJointEnv.step = guarded_load, counted_step
    source_paths = [Path(__file__), Path('src/experiments/single_uav_ppo_a_environment.py'),
                    Path('src/experiments/single_uav_ppo_a_support.py'), Path('src/experiments/single_uav_ppo_a_diagnostics.py'),
                    Path('src/experiments/single_uav_ppo_a_scenarios.py')]
    implementation_hashes = {str(path): sha256_file(path) for path in source_paths if path.exists()}
    if args.phase == 'feasibility':
        result = phase_feasibility(spec, base)
    elif args.phase == 'semantics':
        result = phase_semantics(spec, base)
    elif args.phase == 'diagnostics':
        result = phase_diagnostics(spec, base)
    elif args.phase == 'train':
        result = phase_train(spec, base, stats)
    else:
        raise RuntimeError('Later phase implementation pending physics proof')
    check_protected(spec)
    assert all(sha256_file(path) == expected for path, expected in implementation_hashes.items())
    write_json(Path(spec['log_root']) / f'{args.phase}_runtime_audit.json', {
        'phase': args.phase, 'actual_env_transitions': transition_counter[0], 'checkpoint_load_paths': model_loads,
        'B_checkpoint_loads': 0, 'source_implementation_sha256': implementation_hashes,
        'implementation_unchanged_during_phase': True, 'protected_inputs': check_protected(spec),
        'teacher_data_count': 0, 'new_calibration_steps': 0})
    torch.load, MultiUAVJointEnv.step = original_load, original_step
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
