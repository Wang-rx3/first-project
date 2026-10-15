"""Offline, reward-free communication-policy/navigation isolation proofs."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import torch

from ..control.fixed_navigation import fixed_navigation_command
from ..learning.mappo import MAPPOConfig, MAPPOTrainer
from ..simulation.ekf import covariance_intersection
from .three_uav_legacy_action_fixture import observation, state_hash
from ..experiments.three_uav_fixed_navigation import ROOT, DATA, LOGS, write_json

UIDS = ('uav_00', 'uav_01', 'uav_02')


def synthetic_observations():
    result = {}
    for uid in UIDS:
        local = copy.deepcopy(observation())
        local['uav_id'] = uid
        local['node_ids'] = [uid] + [other for other in UIDS if other != uid]
        local['potential_targets'] = [other for other in UIDS if other != uid]
        local['effective_sources'] = [other for other in UIDS if other != uid]
        local['candidate_incoming_sources'] = [other for other in UIDS if other != uid]
        local['candidate_incoming_edge_features'] = {other: np.ones(11) * .01 for other in UIDS if other != uid}
        local['action_mask'] = np.array([other != uid for other in UIDS] + [True])
        local['bandwidth_mask'] = np.array([other != uid for other in UIDS])
        result[uid] = local
    return result


def main():
    import src.learning.mappo as module
    from ..simulation.environment import MultiUAVJointEnv
    counters = {'environment_steps': 0, 'optimizer_steps': 0, 'checkpoint_loads': 0, 'ppo_updates': 0,
                'rollout_additions': 0, 'motion_distribution_constructions': 0}
    def forbidden(name):
        def fail(*args, **kwargs):
            counters[name] += 1
            raise RuntimeError('Forbidden offline probe operation: ' + name)
        return fail
    MultiUAVJointEnv.step = forbidden('environment_steps')
    MAPPOTrainer.update = forbidden('ppo_updates')
    MAPPOTrainer.add_transition = forbidden('rollout_additions')
    MAPPOTrainer.collect_rollout = forbidden('rollout_additions')
    torch.load = forbidden('checkpoint_loads')
    for optimizer in (torch.optim.Adam, torch.optim.AdamW, torch.optim.SGD):
        optimizer.step = forbidden('optimizer_steps')
    config = MAPPOConfig(uav_count=3, node_feature_dim=37, edge_feature_dim=11,
                         gat_hidden_dim=8, gat_output_dim=8, gat_heads=1, gat_layers=1,
                         actor_hidden_dim=16, critic_hidden_dim=16, max_obstacles=0,
                         measurement_head_mode='action_conditioned_v3',
                         parameter_initialization_seed=2026115403,
                         motion_action_mode='fixed_navigation_comm_v1', ppo_action_logprob_mode='sampled')
    trainer = MAPPOTrainer(config, uav_ids=UIDS, seed=2026115403)
    trainer.policy.eval()
    initial_hash = state_hash(trainer.policy)
    original_normal = module.Normal
    module.Normal = forbidden('motion_distribution_constructions')
    records = []
    try:
        frames = [{'episode_id': 'synthetic', 'step': 0, 'observations': synthetic_observations(), 'central_state': None}]
        frames += json.loads((DATA / 'saved_observation_probes.json').read_text(encoding='utf-8'))
        maximum_logprob_error = 0.
        for index, frame in enumerate(frames):
            raw = frame['observations']
            torch.manual_seed(2026115500 + index)
            with torch.no_grad():
                decision = trainer.act(raw, frame.get('central_state'), deterministic=False)
            for uid, sample in decision.samples.items():
                assert np.asarray(sample.delta_v).size == np.asarray(sample.delta_raw).size == 0
                assert np.asarray(sample.link_selection).shape == (3,)
                expected_nav = fixed_navigation_command(np.asarray(raw[uid]['estimate'])[:6], raw[uid]['goal'])
                assert np.array_equal(decision.actions[uid]['delta_v'], expected_nav['command_delta_v_mps'])
                with torch.no_grad():
                    lp, entropy = trainer.policy.evaluate_actions(decision.observations[uid], sample)
                difference = abs(float(lp) - sample.old_log_prob)
                maximum_logprob_error = max(maximum_logprob_error, difference)
                assert difference <= 2e-6 and torch.isfinite(lp) and torch.isfinite(entropy)
                baseline_lp, baseline_entropy = float(lp), float(entropy)
                mutated = copy.deepcopy(sample)
                mutated.executed_delta_v = np.array([.3, -.25, .1], dtype=np.float32)
                mutated.executed_bandwidth = np.array([0, 0, 1], dtype=np.float32)
                mutated.executed_measurement_index = 3
                with torch.no_grad():
                    mutated_lp, mutated_entropy = trainer.policy.evaluate_actions(decision.observations[uid], mutated)
                assert float(mutated_lp) == baseline_lp and float(mutated_entropy) == baseline_entropy
                records.append({'episode_id': frame['episode_id'], 'step': frame['step'], 'uav_id': uid,
                    'interface_only_not_training': True, 'executed': False, 'reward_associated': False, 'rollout_added': False,
                    'node_feature_dim': 37, 'edge_feature_dim': 11, 'learned_motion_dimensions': 0,
                    'environment_navigation_dimensions': 3, 'measurement_index': sample.measurement_index,
                    'sampled_link_selection': sample.link_selection, 'sampled_bandwidth': sample.bandwidth,
                    'old_log_prob': sample.old_log_prob, 'evaluated_log_prob': baseline_lp, 'entropy': baseline_entropy,
                    'ppo_ratio': float(np.exp(baseline_lp - sample.old_log_prob)),
                    'execution_perturbation_probability_invariant': True, 'fixed_navigation_matches_raw_input': True})
        # Actual PPO surrogate formula, routing gradients only; no optimizer/update/transition.
        raw = synthetic_observations()
        decision = trainer.act(raw, None, deterministic=False)
        uid = UIDS[0]
        sample = decision.samples[uid]
        lp, entropy = trainer.policy.evaluate_actions(decision.observations[uid], sample)
        ratio = torch.exp(lp - torch.as_tensor(sample.old_log_prob))
        loss = -torch.minimum(ratio * .7, ratio.clamp(.8, 1.2) * .7) - .005 * entropy
        before_loss = float(loss.detach())
        trainer.policy.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norms = {}
        for group in ('gat', 'actor_body', 'measurement_head', 'link_selection_head', 'bandwidth_head'):
            parameters = [p for name, p in trainer.policy.named_parameters() if name.startswith(group + '.') and p.grad is not None]
            gradient_norms[group] = float(sum(float(torch.sum(p.grad.detach() ** 2)) for p in parameters) ** .5)
        assert gradient_norms['gat'] > 0 and gradient_norms['link_selection_head'] > 0 and gradient_norms['measurement_head'] > 0
        assert all(parameter.grad is None and not parameter.requires_grad for name, parameter in trainer.policy.named_parameters()
                   if name.startswith('delta_mean_head.') or name == 'delta_logstd')
        original_sample = copy.deepcopy(sample)
        trainer.reconcile_action_execution(decision, {'actions': {uid: {'measurement_uav_id': UIDS[1]}},
            'applied_bandwidth': {uid: {UIDS[1]: .1}}, 'action_execution': {uid: {'delta_v_applied': [.1, .2, .3]}}})
        assert sample.old_log_prob == original_sample.old_log_prob
        assert sample.measurement_index == original_sample.measurement_index
        assert np.array_equal(sample.link_selection, original_sample.link_selection)
        assert np.array_equal(sample.bandwidth, original_sample.bandwidth) and sample.delta_v.size == 0
        lp_after, entropy_after = trainer.policy.evaluate_actions(decision.observations[uid], sample)
        ratio_after = torch.exp(lp_after - torch.as_tensor(sample.old_log_prob))
        loss_after = -torch.minimum(ratio_after * .7, ratio_after.clamp(.8, 1.2) * .7) - .005 * entropy_after
        assert float(loss_after.detach()) == before_loss
        # A hostile normalizer changes NN inputs, but cannot supply normalized motion input.
        original_transform = trainer.normalizer.transform
        def hostile_normalizer(values):
            result = original_transform(values)
            for local in result.values():
                local['estimate'] = np.full(15, 999.)
                local['goal'] = np.full(3, -999.)
                local['node_features'] = np.full_like(local['node_features'], 999.)
            return result
        trainer.normalizer.transform = hostile_normalizer
        poisoned_decision = trainer.act(raw, {'truth_poison': [1e9]}, deterministic=True)
        assert all(np.all(np.asarray(poisoned_decision.observations[uid]['node_features']) == 999.) for uid in UIDS)
        assert all(np.array_equal(poisoned_decision.actions[uid]['delta_v'], fixed_navigation_command(raw[uid]['estimate'], raw[uid]['goal'])['command_delta_v_mps']) for uid in UIDS)
        trainer.normalizer.transform = original_transform
        mean, covariance, weight = covariance_intersection(np.array([1., 2., 3.]), np.diag([1., 2., 3.]),
                                                          np.array([1.1, 1.8, 3.2]), np.diag([2., 1., 3.]))
        ci_probe_passed = bool(np.isfinite(mean).all() and np.isfinite(covariance).all() and 0 <= weight <= 1
                               and np.max(np.abs(covariance - covariance.T)) <= 1e-8 and np.linalg.eigvalsh(covariance).min() >= -1e-8)
        assert ci_probe_passed
        assert state_hash(trainer.policy) == initial_hash and len(trainer.rollout) == 0 and not trainer.optimizer.state
        assert not trainer.normalizer.normalizers and not trainer.central_normalizer.fitted
        write_json(LOGS / 'interface_probes.json', {'passed': True, **counters, 'parameters_unchanged': True,
            'rollout_length': 0, 'normalization_fits': 0, 'normalization': 'identity_unfitted_interface_only',
            'interface_only_not_training': True, 'executed': False, 'reward_associated': False,
            'policy_samples_executed': 0, 'interface_only_policy_executed': False,
            'records': records, 'record_count': len(records), 'maximum_logprob_error': maximum_logprob_error,
            'gradient_norms': gradient_norms, 'motion_gradients_absent': True,
            'loss_execution_invariant': True, 'raw_navigation_normalization_poison_invariant': True,
            'ci_pure_probe_passed': ci_probe_passed, 'ci_weight': weight,
            'parameters_sha256_before': initial_hash, 'parameters_sha256_after': state_hash(trainer.policy)})
    finally:
        module.Normal = original_normal
    print(json.dumps({'passed': True, 'record_count': len(records), 'maximum_logprob_error': maximum_logprob_error, 'gradient_norms': gradient_norms, **counters}))


if __name__ == '__main__':
    main()
