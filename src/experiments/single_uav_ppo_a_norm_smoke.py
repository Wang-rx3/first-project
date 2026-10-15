"""Bounded normalization/transfer controls; old experiments are read-only."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .single_uav_ppo_a_foundation import load_run, collect_block, evaluate_cases, save_checkpoint
from .single_uav_ppo_a_support import (PPOUpdateAudit, update_with_audit, parameter_snapshot,
    grouped_parameter_deltas, normal_release_gate, summarize_foundation_episodes, _tensor_hash)
from .single_uav_ppo_baseline import create_single_uav_ppo_trainer, verify_actor_observation_contract
from .single_uav_ppo_smoke import CSVLog, write_json, sha256_file, canonical_hash, _reset
from .single_uav_ppo_a_norm_telemetry import create_norm_environment, boundary_probes

DEFAULT = Path('data/configs/single_uav_ppo_a_norm_transfer_v2_seed2026_20261011.json')


def protected(spec):
    previous = json.loads((Path(spec['log_root'])/'protected_inputs_before.json').read_text())
    assert all(Path(p).is_file() and sha256_file(p) == expected for p, expected in previous.items())
    return len(previous)


def make_trainer(base, stats, source, condition, manifest, logs):
    from .single_uav_ppo_a_norm_compatibility import initialize_condition
    env = create_norm_environment(base)
    obs, _ = _reset(env, manifest['train'][0], f'{condition}_construction')
    trainer = create_single_uav_ppo_trainer(env, base, normalizer=stats[0], central_normalizer=stats[1],
        normalizer_provenance=stats[3], action_log_path=logs/'sampled_execution.csv')
    initialization = initialize_condition(trainer, source, condition)
    contract = verify_actor_observation_contract(env, obs, normalizer=trainer.normalizer)
    return env, trainer, initialization, contract


class EnrichedSteps:
    def __init__(self, path, trainer, env):
        self.logger, self.env, self.trainer = CSVLog(path), env, trainer
        self.original_act = trainer.act
        def observed(*args, **kwargs):
            self.rng_before = hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()
            self.decision = self.original_act(*args, **kwargs)
            return self.decision
        trainer.act = observed

    def append(self, row):
        telemetry = self.env.norm_last_telemetry
        record = dict(row)
        ordinal = telemetry['reset_ordinal']
        record.update(motion_telemetry=telemetry, value=self.decision.value,
            critic_input_normalized=self.decision.central_vector.tolist(), torch_rng_before=self.rng_before,
            episode_ordinal=ordinal, is_filler=bool(row['phase'] == 'online' and ordinal > 0),
            completed=bool(row['terminated'] or row['truncated']),
            cut_at_block_end=bool(row['phase'] == 'online' and row['block_step'] == 119 and not (row['terminated'] or row['truncated'])))
        goal = self.env.goals['uav_00'] - np.asarray(telemetry['pre_position_m'])
        velocity = np.asarray(telemetry['pre_velocity_mps'])
        record.update(actual_pre_goal_distance_m=float(np.linalg.norm(goal)),
            actual_pre_radial_speed_mps=float(np.dot(velocity, goal)/max(np.linalg.norm(goal), 1e-12)),
            nominal_acceleration_mps2=telemetry['nominal_acceleration_mps2'],
            boundary_attempt_axes=telemetry['boundary_attempt_axes'], boundary_projected_axes=telemetry['boundary_projected_axes'])
        self.logger.append(record)


def calibrate(spec, parent, base, old):
    from .single_uav_ppo_a_norm_calibration import build_split_manifest, schema_gate, collect_calibration, fit_full_statistics
    data, log = Path(spec['data_root']), Path(spec['log_root'])
    manifest = build_split_manifest()
    write_json(data/'scenario_manifest.json', manifest)
    write_json(data/'run_specification.lock.json', spec)
    write_json(log/'schema_gate.json', schema_gate(base, manifest['calibration'][0]))
    boundary = boundary_probes(base, log)
    calibration = collect_calibration(base, manifest, parent['A_source']['checkpoint'], old, data, log)
    stats = fit_full_statistics(data/'calibration_bank.npz', old[2], data/'full_train_statistics.json', log)
    write_json(log/'calibration_phase_result.json', {'boundary': boundary, 'calibration': calibration,
                                                   'new_statistics_provenance': stats[3]})
    return {'calibration': calibration, 'statistics': stats[3]}


def freeze_decision(spec, parent, base, old, new, manifest):
    from .single_uav_ppo_a_norm_compatibility import license_output_head, compatibility_probe
    data, log = Path(spec['data_root']), Path(spec['log_root'])
    compatibility = compatibility_probe(data/'calibration_bank.npz', parent['A_source']['checkpoint'], old, new, data, log)
    write_json(log/'compatibility_result.json', compatibility)
    return license_output_head(base, manifest, parent['A_source']['checkpoint'], new, data, log)


def drift_probe(trainer, transition_rows, path, update):
    """Replay collected normalized states without sampling or environment calls."""
    from ..learning.mappo import PolicySample
    from ..analysis.single_uav_ppo_a_norm_history import single_node_observation
    logger = CSVLog(path)
    with torch.no_grad():
        for row in transition_rows:
            local = single_node_observation(json.loads(row['node_normalized']))
            sample = PolicySample(1, np.zeros(1, np.float32), np.array(json.loads(row['sampled_delta_raw']), np.float32),
                np.array(json.loads(row['sampled_delta_v']), np.float32), float(row['old_logprob']))
            distribution = trainer.policy._actor_distributions(local)[2].base_dist
            new_logprob = float(trainer.policy.evaluate_actions(local, sample)[0])
            logger.append({'update': update, 'block': int(row['block']), 'block_step': int(row['block_step']),
                'episode_id': row['episode_id'], 'mean_before': json.loads(row['distribution_raw_mean']),
                'mean_after': distribution.mean.detach().numpy().reshape(3).tolist(),
                'old_logprob': float(row['old_logprob']), 'new_logprob': new_logprob,
                'logprob_change': new_logprob-float(row['old_logprob']),
                'boundary': row['boundary_projected'] == 'True', 'episode_step': int(row['episode_step'])})


def train(spec, parent, base, old):
    import csv
    from .single_uav_ppo_a_norm_calibration import load_full_statistics
    from .single_uav_ppo_a_norm_compatibility import tensor_pair_audit
    data, log = Path(spec['data_root']), Path(spec['log_root'])
    manifest = json.loads((data/'scenario_manifest.json').read_text())
    new = load_full_statistics(data/'full_train_statistics.json')
    decision = freeze_decision(spec, parent, base, old, new, manifest)
    arms = ['C', 'N', 'G']+(['R'] if decision['licensed'] else [])
    constructed, initial_states = {}, {}
    for arm in arms:
        arm_data, arm_log = data/arm, log/arm
        arm_data.mkdir(exist_ok=False)
        arm_log.mkdir(exist_ok=False)
        stats = old if arm == 'C' else new
        env, trainer, initialization, contract = make_trainer(base, stats, parent['A_source']['checkpoint'], arm, manifest, arm_log)
        initial_states[arm] = parameter_snapshot(trainer.policy)
        constructed[arm] = (env, trainer, initialization)
        write_json(arm_data/'initialization.json', initialization)
        write_json(arm_data/'actor_contract.json', contract)
        write_json(arm_data/'effective_config.json', {'baseline': base.to_dict(), 'ppo': asdict(trainer.config),
                   'actual_A_warm_start': True, 'condition': spec['condition_definitions'][arm], 'normalizer': stats[3]})
        save_checkpoint(arm_data/'initial.pt', trainer, arm=arm, online_steps=0, initialization=initialization, updates=0)
    pairs = tensor_pair_audit(initial_states)
    write_json(log/'condition_tensor_audit.json', pairs)
    assert pairs['passed']
    result = {'primary': decision['primary'], 'arms': {}, 'B_paused': True, 'teacher_samples': 0,
              'online_steps': 0, 'evaluation_steps': 0, 'reward_or_dynamics_modified': False}
    for arm in arms:
        env, trainer, initialization = constructed[arm]
        arm_data, arm_log = data/arm, log/arm
        steps = EnrichedSteps(arm_log/'steps.csv', trainer, env)
        episode_log = CSVLog(arm_log/'episodes.csv')
        evaluated = {}
        for split in ('train', 'validation', 'challenge'):
            evaluated['initial_'+split] = evaluate_cases(trainer, env, manifest[split], arm=arm,
                phase='initial_'+split, stage='normal', step_log=steps, episode_log=episode_log)
        # Evaluation is deterministic; re-establish the registered sampling stream.
        torch.manual_seed(2026)
        audit = PPOUpdateAudit(trainer)
        online_episodes, updates = [], []
        first_block_fingerprint = None
        for block, scene in enumerate(manifest['train']):
            rng = torch.get_rng_state().numpy().tobytes()
            write_json(arm_data/f'rng_block_{block:02d}.json', {'torch_rng_hash': hashlib.sha256(rng).hexdigest(),
                'torch_rng_state_b64': base64.b64encode(rng).decode(), 'minibatch_rng': trainer.rng.bit_generator.state,
                'case': scene['scenario_id'], 'block': block})
            collected = collect_block(trainer, env, scene, arm=arm, phase='online', stage='normal', block=block,
                online=True, step_log=steps, episode_log=episode_log)
            assert collected['steps'] == 120
            online_episodes.extend(collected['episodes'])
            update = update_with_audit(trainer, audit)
            assert update['normalizers_unchanged'] and update['all_weights_finite'] and update['rollout_empty_after_update']
            assert update['on_policy_logprob_within_tolerance'] and update['gae_recurrence_matches_training_records']
            assert update['gae_does_not_cross_done_boundary'] and audit.steps == (block+1)*8
            update.update(arm=arm, block=block, online_steps=(block+1)*120)
            write_json(arm_log/f'update_{block+1:02d}.json', update)
            updates.append(update)
            save_checkpoint(arm_data/f'update_{block+1:02d}.pt', trainer, arm=arm, online_steps=(block+1)*120, initialization=initialization, updates=block+1)
            with (arm_log/'steps.csv').open(newline='', encoding='utf-8') as stream:
                replay_rows = [r for r in csv.DictReader(stream) if r['phase'] == 'online']
            drift_probe(trainer, replay_rows, arm_log/f'action_drift_{block+1:02d}.csv', block+1)
            if block == 0:
                fields = ('sampled_delta_v', 'actual_position_m', 'actual_velocity_mps', 'learning_reward', 'old_logprob')
                first_block_fingerprint = canonical_hash([{key: row[key] for key in fields} for row in replay_rows])
        for split in ('train', 'validation', 'challenge'):
            evaluated['final_'+split] = evaluate_cases(trainer, env, manifest[split], arm=arm,
                phase='final_'+split, stage='normal', step_log=steps, episode_log=episode_log)
        gate = normal_release_gate(evaluated['final_train']['metrics'], evaluated['final_validation']['metrics'])
        challenge_rows = evaluated['final_challenge']['episodes']
        challenge_types = {s['scenario_id']: s['challenge_type'] for s in manifest['challenge']}
        successful = {kind: sum(r['task_success'] for r in challenge_rows if challenge_types[r['scenario_id']] == kind)
                      for kind in ('stationary', 'fast')}
        challenge_metrics = evaluated['final_challenge']['metrics']['all']
        limit = challenge_metrics['actual_steps']//100
        challenge_ok = bool(sum(successful.values()) >= 3 and min(successful.values()) >= 1
            and challenge_metrics['collision_episode_count'] == 0 and challenge_metrics['physical_contact_steps'] == 0
            and challenge_metrics['boundary_attempt_steps'] <= limit and challenge_metrics['boundary_projection_steps'] <= limit)
        arm_result = {'evaluations': evaluated, 'normal_gate': gate, 'challenge_gate': {'passed': challenge_ok,
                      'success_by_type': successful, 'boundary_limit': limit},
            'online_metrics': summarize_foundation_episodes(online_episodes), 'online_episodes': online_episodes,
            'parameter_deltas': grouped_parameter_deltas(initial_states[arm], trainer.policy),
            'online_steps': 720, 'updates': 6, 'actual_Adam_steps': audit.steps, 'first_block_fingerprint': first_block_fingerprint}
        if arm == 'C':
            with (Path(parent['log_root'])/'P'/'steps.csv').open(encoding='utf-8', newline='') as stream:
                old_rows = [r for r in csv.DictReader(stream) if r['phase'] == 'online' and r['block'] == '0']
            previous = canonical_hash([{key: row[key] for key in fields} for row in old_rows])
            arm_result['historical_first_block_identical'] = previous == first_block_fingerprint
            arm_result['historical_first_block_fingerprint'] = previous
        save_checkpoint(arm_data/'final.pt', trainer, arm=arm, online_steps=720, initialization=initialization, updates=6)
        write_json(arm_log/'result.json', arm_result)
        result['arms'][arm] = arm_result
        result['online_steps'] += 720
        result['evaluation_steps'] += sum(item['steps'] for item in evaluated.values())
        audit.close()
        trainer.act = steps.original_act
    primary = result['arms'][result['primary']]
    result['next_stage_candidate'] = bool(primary['normal_gate']['passed'] and primary['challenge_gate']['passed'])
    result['obstacle_or_multiseed_started'] = False
    write_json(log/'smoke_result.json', result)
    return {'primary': result['primary'], 'next_stage_candidate': result['next_stage_candidate'],
            'online_steps': result['online_steps'], 'evaluation_steps': result['evaluation_steps']}


def main():
    from ..simulation.environment import MultiUAVJointEnv
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=str(DEFAULT))
    parser.add_argument('--phase', choices=('calibrate', 'train'), required=True)
    args = parser.parse_args()
    spec = json.loads(Path(args.config).read_text())
    parent, base, old = load_run(spec['parent_config'])
    torch.set_num_threads(1)
    protected(spec)
    implementation_files = [Path(__file__), Path(__file__).with_name('single_uav_ppo_a_norm_calibration.py'),
        Path(__file__).with_name('single_uav_ppo_a_norm_compatibility.py'),
        Path(__file__).with_name('single_uav_ppo_a_norm_telemetry.py')]
    implementation_hashes = {str(path.resolve()): sha256_file(path) for path in implementation_files}
    loads, counter = [], [0]
    original_load, original_step = torch.load, MultiUAVJointEnv.step
    allowed = {Path(parent['A_source']['checkpoint']).resolve(), Path(parent['historical_A_initializer']['path']).resolve()}
    def checked_load(path, *a, **kw):
        resolved = Path(path).resolve()
        assert resolved in allowed, f'Unexpected model load: {resolved}'
        loads.append(str(resolved))
        return original_load(path, *a, **kw)
    def counted_step(env, action):
        counter[0] += 1
        return original_step(env, action)
    torch.load, MultiUAVJointEnv.step = checked_load, counted_step
    try:
        result = calibrate(spec, parent, base, old) if args.phase == 'calibrate' else train(spec, parent, base, old)
        prior_steps = 0
        if args.phase == 'train':
            prior_steps = json.loads((Path(spec['log_root'])/'calibrate_runtime_audit.json').read_text())['env_steps']
        assert counter[0]+prior_steps <= spec['maximum_research_env_steps']
        assert all(sha256_file(path) == digest for path, digest in implementation_hashes.items())
        write_json(Path(spec['log_root'])/f'{args.phase}_runtime_audit.json', {'env_steps': counter[0],
            'combined_research_env_steps': counter[0]+prior_steps, 'checkpoint_loads': loads,
            'implementation_hashes': implementation_hashes, 'B_loads': 0,
            'protected_files_unchanged': protected(spec), 'teacher_samples': 0})
    finally:
        torch.load, MultiUAVJointEnv.step = original_load, original_step
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
