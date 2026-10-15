"""Offline score/gradient validation; no rollout, optimization or normalization fit."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

PARENT = 'single_uav_ppo_a_norm_transfer_v2_seed2026_20261011'
STEM = 'single_uav_ppo_a_route_feasibility_v3_seed2026_20261011'
OLD_LOG = Path('logs/preliminary')/PARENT
OLD_DATA = Path('data/synthetic')/PARENT
LOG = Path('logs/preliminary')/STEM


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    payload = json.dumps(value, ensure_ascii=False, indent=2)
    with Path(path).open('x', encoding='utf-8') as stream:
        stream.write(payload)


def online_rows(arm):
    with (OLD_LOG/arm/'steps.csv').open(encoding='utf-8', newline='') as stream:
        return [row for row in csv.DictReader(stream) if row['phase']=='online']


def score_direction(raw, mean, std, advantage):
    """Gradient-ascent score for the latent motion mean at ratio=1."""
    return np.asarray(advantage)[:, None]*(np.asarray(raw)-np.asarray(mean))/np.asarray(std)**2


def policy_hash(policy):
    digest = hashlib.sha256()
    for name, parameter in policy.named_parameters():
        digest.update(name.encode())
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def exact_first_block_gradient(arm, rows, update):
    from ..learning.mappo import MAPPOConfig, MAPPOTrainer, PolicySample
    from .single_uav_ppo_a_norm_history import single_node_observation
    checkpoint = torch.load(OLD_DATA/arm/'initial.pt', map_location='cpu', weights_only=False)
    trainer = MAPPOTrainer(MAPPOConfig(**checkpoint['trainer_config']), uav_ids=('uav_00',), seed=2026)
    trainer.policy.load_state_dict(checkpoint['policy_state_dict'])
    before = policy_hash(trainer.policy)
    assert not trainer.optimizer.state and not trainer.rollout
    losses, entropy_values, log_errors = [], [], []
    for row, audit in zip(rows, update['per_record_audit'], strict=True):
        local = single_node_observation(json.loads(row['node_normalized']))
        sample = PolicySample(1, np.zeros(1, np.float32), np.array(json.loads(row['sampled_delta_raw']), np.float32),
                             np.array(json.loads(row['sampled_delta_v']), np.float32), float(row['old_logprob']))
        logprob, entropy = trainer.policy.evaluate_actions(local, sample)
        entropy_values.append(entropy)
        log_errors.append(abs(float(logprob.detach())-sample.old_log_prob))
        ratio = torch.exp(logprob-sample.old_log_prob)
        losses.append(-torch.minimum(ratio*audit['advantage'], ratio.clamp(.8, 1.2)*audit['advantage']))
    actor_loss = torch.stack(losses).mean()
    central = np.array([json.loads(row['critic_input_normalized']) for row in rows], np.float32)
    returns = torch.tensor([record['return'] for record in update['per_record_audit']], dtype=torch.float32)
    values = trainer.policy.value(central)
    critic_loss = .5*(values-returns).square().mean()
    (actor_loss+trainer.config.value_coef*critic_loss-trainer.config.entropy_coef*torch.stack(entropy_values).mean()).backward()
    actor_sq = critic_sq = 0.
    for name, parameter in trainer.policy.named_parameters():
        norm_sq = float(parameter.grad.detach().double().square().sum()) if parameter.grad is not None else 0.
        if name.startswith('critic.'):
            critic_sq += norm_sq
        else:
            actor_sq += norm_sq
    raw = np.array([json.loads(row['sampled_delta_raw']) for row in rows])
    mean = np.array([json.loads(row['distribution_raw_mean']) for row in rows])
    std = np.array([json.loads(row['distribution_raw_std']) for row in rows])
    advantage = [record['advantage'] for record in update['per_record_audit']]
    analytic_bias_gradient = -score_direction(raw, mean, std, advantage).mean(axis=0)
    actual_bias_gradient = trainer.policy.delta_mean_head.bias.grad.detach().numpy().astype(float)
    error = float(np.max(np.abs(actual_bias_gradient-analytic_bias_gradient)))
    assert error < 1e-5 and max(log_errors) < 1e-4
    assert policy_hash(trainer.policy)==before and not trainer.optimizer.state and not trainer.rollout
    return {'full_batch_samples': len(rows), 'max_old_logprob_error': max(log_errors),
        'actor_loss_without_entropy': float(actor_loss.detach()), 'entropy_included_in_backward': True,
        'critic_loss_unweighted': float(critic_loss.detach()),
        'preclip_actor_norm': actor_sq**.5, 'preclip_weighted_critic_norm': critic_sq**.5,
        'global_clip_scale_for_this_full_batch': min(1., .5/max((actor_sq+critic_sq)**.5, 1e-12)),
        'actual_motion_head_bias_gradient': actual_bias_gradient.tolist(),
        'analytic_score_gradient_max_error': error, 'weights_unchanged': True,
        'limitation': 'full first 120-sample batch diagnostic; actual optimizer used eight shuffled 64/56 minibatches'}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-directory', type=Path, default=LOG)
    destination = parser.parse_args().output_directory
    destination.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    from ..simulation.environment import MultiUAVJointEnv
    from ..learning.mappo import MAPPOTrainer
    def forbidden(*args, **kwargs):
        raise RuntimeError('Offline feasibility audit forbids environment/optimizer/update execution')
    MultiUAVJointEnv.step = forbidden
    MAPPOTrainer.update = forbidden
    torch.optim.Adam.step = forbidden
    protected = read_json(LOG/'protected_inputs_before.json')
    assert all(hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest for path, digest in protected.items())
    output, exact = [], {}
    for arm in ('C', 'N', 'G', 'R'):
        rows = online_rows(arm)
        for block in range(6):
            chosen = [row for row in rows if int(row['block'])==block]
            update = read_json(OLD_LOG/arm/f'update_{block+1:02d}.json')
            assert len(chosen)==len(update['per_record_audit'])==120
            raw = np.array([json.loads(row['sampled_delta_raw']) for row in chosen])
            mean = np.array([json.loads(row['distribution_raw_mean']) for row in chosen])
            std = np.array([json.loads(row['distribution_raw_std']) for row in chosen])
            advantage = np.array([record['advantage'] for record in update['per_record_audit']])
            score = score_direction(raw, mean, std, advantage)
            output.append({'arm': arm, 'block': block, 'steps': 120,
                **{f'latent_mean_ascent_score_{axis}': float(score[:, i].mean()) for i, axis in enumerate('xyz')},
                'zero_mean_advantage': float(advantage.mean()), 'recorded_preclip_global_norm_mean': update['update_stats']['gradient_norm'],
                'recorded_actor_norm_is_postclip': True, 'reported_group_clip_scale_is_not_joint_scale': True})
            if block==0:
                exact[arm] = exact_first_block_gradient(arm, chosen, update)
    with (destination/'motion_score_direction.csv').open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    write_json(destination/'exact_offline_gradient_verification.json', {'env_steps': 0, 'PPO_updates': 0, 'optimizer_steps': 0,
        'normalizer_fit_or_update': 0, 'B_loads': 0, 'protected_files_unchanged': len(protected),
        'arms': exact, 'implementation_note': 'joint group norms/scales in old update_stats computed after global clipping; 1.0 does not mean unclipped',
        'algorithm_error_found': False})
    print(json.dumps({'arms': exact, 'protected_files': len(protected)}, indent=2))


if __name__=='__main__':
    main()
