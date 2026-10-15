"""Offline diagnosis of prior A-only PPO; never collect or update a policy."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch

from ..learning.mappo import MAPPOConfig, MAPPOTrainer, PolicySample
from ..experiments.single_uav_ppo_smoke import CSVLog, write_json

OLD_LOG = Path('logs/preliminary/single_uav_ppo_a_foundation_v1_seed2026')
OLD_DATA = Path('data/synthetic/single_uav_ppo_a_foundation_v1_seed2026')
SOURCE = Path('data/synthetic/single_uav_external_recovery_method_study_seed2026/dagger_bc/baseline_observation_migration/checkpoints/stage2a_observation_compatible_init.pt')


def single_node_observation(node):
    """Rebuild the checked empty single-UAV graph for policy-only replay."""
    return {'node_features': np.asarray(node, dtype=np.float32).reshape(1, 42),
            'edge_index': np.empty((2, 0), dtype=np.int64), 'edge_features': np.empty((0, 11), dtype=np.float32),
            'node_ids': ['uav_00'], 'action_mask': np.array([False, True]), 'bandwidth_mask': np.array([False]),
            'effective_edge_index': np.empty((2, 0), dtype=np.int64),
            'effective_edge_features': np.empty((0, 11), dtype=np.float32),
            **{key: {} for key in ('candidate_edge_features',
                                   'candidate_incoming_edge_features', 'potential_edge_features')}}


def historical_audit(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    action_log = CSVLog(output / 'historical_actor_replay.csv')
    feature_log = CSVLog(output / 'historical_feature_distribution.csv')
    summary = {'env_steps': 0, 'PPO_updates': 0, 'B_loads': 0, 'arms': {}}
    source = torch.load(SOURCE, map_location='cpu', weights_only=False)
    for arm in ('G', 'P'):
        with (OLD_LOG / arm / 'steps.csv').open(encoding='utf-8', newline='') as stream:
            rows = [r for r in csv.DictReader(stream) if r['phase'] == 'online']
        updates = [json.loads((OLD_LOG / arm / f'update_{i:02d}.json').read_text()) for i in range(1, 7)]
        records = [record for update in updates for record in update['per_record_audit']]
        assert len(rows) == len(records) == 720
        raw = np.array([json.loads(row['node_raw']) for row in rows])
        norm = np.array([json.loads(row['node_normalized']) for row in rows])
        clipped = np.array([json.loads(row['node_clipped']) for row in rows], dtype=bool)
        for channel in range(42):
            for low, high in ((1, 16), (17, 40), (41, 80), (81, 120)):
                selected = np.array([low <= int(row['episode_step']) <= high for row in rows])
                values = raw[selected, channel]
                feature_log.append({'arm': arm, 'channel': channel, 'time_start': low, 'time_end': high,
                    'count': int(selected.sum()), 'raw_min': float(values.min()), 'raw_max': float(values.max()),
                    'raw_mean': float(values.mean()), 'raw_std': float(values.std()),
                    'q01': float(np.quantile(values, .01)), 'q50': float(np.median(values)), 'q99': float(np.quantile(values, .99)),
                    'clip_count': int(clipped[selected, channel].sum()), 'clip_fraction': float(clipped[selected, channel].mean()),
                    'raw_normalized_sign_mismatches': int(((values*norm[selected, channel]) < 0).sum())})
        config = MAPPOConfig(**json.loads((OLD_DATA / arm / 'effective_training_config.json').read_text())['ppo'])
        final = MAPPOTrainer(config, seed=2026)
        checkpoint = torch.load(OLD_DATA / arm / 'normal_final.pt', map_location='cpu', weights_only=False)
        final.policy.load_state_dict(checkpoint['policy_state_dict'])
        initial = MAPPOTrainer(config, seed=2026)
        state = initial.policy.state_dict()
        state.update({name: value for name, value in source['policy_state_dict'].items()
                      if not name.startswith('critic.') and (arm == 'P' or not name.startswith('gat.layers.0.'))})
        initial.policy.load_state_dict(state)
        groups = {name: [] for name in ('boundary', 'ordinary', 'early', 'late')}
        replay_check = []
        with torch.no_grad():
            for row, record, normalized in zip(rows, records, norm):
                local = single_node_observation(normalized)
                initial_mean = initial.policy._actor_distributions(local)[2].base_dist.mean.numpy().reshape(3)
                final_mean = final.policy._actor_distributions(local)[2].base_dist.mean.numpy().reshape(3)
                cached_mean = np.array(json.loads(row['distribution_raw_mean']))
                if int(row['block']) == 0:
                    replay_check.append(float(np.abs(cached_mean-initial_mean).max()))
                sample = PolicySample(1, np.zeros(1, np.float32), np.array(json.loads(row['sampled_delta_raw']), np.float32),
                                     np.array(json.loads(row['sampled_delta_v']), np.float32), float(row['old_logprob']))
                new_logprob = float(final.policy.evaluate_actions(local, sample)[0])
                components = json.loads(row['reward_components'])
                item = {'reward': float(row['learning_reward']), 'advantage': float(record['advantage']),
                        'braking_reward': float(components['single_uav_braking_action_reward']),
                        'progress_m': float(row['actual_progress_m']), 'logprob_change': new_logprob-float(row['old_logprob']),
                        'executed_norm': float(np.linalg.norm(json.loads(row['executed_delta_v'])))}
                boundary = row['boundary_projected'] == 'True'
                groups['boundary' if boundary else 'ordinary'].append(item)
                groups['late' if int(row['block_step']) >= 80 else 'early'].append(item)
                action_log.append({'arm': arm, 'block': int(row['block']), 'step': int(row['episode_step']),
                    'boundary': boundary, 'initial_latent_mean': initial_mean.tolist(), 'collected_latent_mean': cached_mean.tolist(),
                    'final_latent_mean': final_mean.tolist(), 'latent_mean_shift_l2': float(np.linalg.norm(final_mean-cached_mean)),
                    'old_logprob': float(row['old_logprob']), 'final_logprob': new_logprob, **item})
        assert max(replay_check) < 2e-6, 'Single-node replay does not reproduce the collected policy mean'
        summary['arms'][arm] = {'initial_replay_max_mean_error': max(replay_check), 'groups': {
            group: {'steps': len(items), **{key+'_mean': float(np.mean([r[key] for r in items])) for key in items[0]},
                    'positive_advantage_fraction': float(np.mean([r['advantage'] > 0 for r in items])),
                    'positive_reward_steps': sum(r['reward'] > 0 for r in items)} for group, items in groups.items()}}
    write_json(output / 'historical_failure_audit.json', summary)
    return summary
