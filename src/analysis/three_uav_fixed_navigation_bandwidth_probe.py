"""Supplemental offline K=2 bandwidth-gradient witness, without learning."""
from __future__ import annotations

import json
import torch

from .three_uav_fixed_navigation_probes import synthetic_observations, UIDS
from .three_uav_legacy_action_fixture import state_hash
from ..experiments.three_uav_fixed_navigation import LOGS, write_json
from ..learning.mappo import MAPPOConfig, MAPPOPolicy, MAPPOTrainer
from ..simulation.environment import MultiUAVJointEnv


def main():
    counters = {'environment_steps': 0, 'optimizer_steps': 0, 'ppo_updates': 0,
                'checkpoint_loads': 0, 'rollout_additions': 0}
    def forbidden(name):
        def fail(*args, **kwargs):
            counters[name] += 1
            raise RuntimeError('Forbidden offline operation: ' + name)
        return fail
    MultiUAVJointEnv.step = forbidden('environment_steps')
    MAPPOTrainer.update = forbidden('ppo_updates')
    MAPPOTrainer.add_transition = forbidden('rollout_additions')
    torch.load = forbidden('checkpoint_loads')
    for cls in (torch.optim.Adam, torch.optim.AdamW, torch.optim.SGD):
        cls.step = forbidden('optimizer_steps')
    cfg = MAPPOConfig(uav_count=3, node_feature_dim=37, edge_feature_dim=11,
                      gat_hidden_dim=8, gat_output_dim=8, gat_heads=1, gat_layers=1,
                      actor_hidden_dim=16, critic_hidden_dim=16, max_obstacles=0,
                      measurement_head_mode='action_conditioned_v3',
                      parameter_initialization_seed=2026115403,
                      motion_action_mode='fixed_navigation_comm_v1',
                      ppo_action_logprob_mode='sampled')
    policy = MAPPOPolicy(cfg, uav_ids=UIDS)
    policy.eval()
    initial_hash = state_hash(policy)
    observation = synthetic_observations()[UIDS[0]]
    for seed in range(2026115600, 2026115632):
        torch.manual_seed(seed)
        _, sample = policy.act(observation)
        if sample.link_selection.sum() == 2:
            break
    else:
        raise AssertionError('No K=2 sample in bounded offline probe')
    lp, entropy = policy.evaluate_actions(observation, sample)
    assert abs(float(lp.detach()) - sample.old_log_prob) <= 2e-6
    ratio = torch.exp(lp - torch.as_tensor(sample.old_log_prob))
    loss = -torch.minimum(ratio * .7, ratio.clamp(.8, 1.2) * .7) - .005 * entropy
    policy.zero_grad(set_to_none=True)
    loss.backward()
    gradients = {group: sum(float((p.grad.detach() ** 2).sum()) for name, p in policy.named_parameters()
                  if name.startswith(group + '.') and p.grad is not None) ** .5
                 for group in ('gat', 'measurement_head', 'link_selection_head', 'bandwidth_head')}
    assert all(value > 0 for value in gradients.values())
    assert all(p.grad is None and not p.requires_grad for name, p in policy.named_parameters()
               if name.startswith('delta_mean_head.') or name == 'delta_logstd')
    assert state_hash(policy) == initial_hash
    result = {'passed': True, **counters, 'parameters_unchanged': True,
              'interface_only_not_training': True, 'executed': False, 'reward_associated': False,
              'seed': seed, 'link_selection': sample.link_selection, 'sampled_bandwidth': sample.bandwidth,
              'old_log_prob': sample.old_log_prob, 'evaluated_log_prob': float(lp.detach()),
              'gradient_norms': gradients, 'motion_gradients_absent': True,
              'interpretation': 'K<2 has no bandwidth simplex degree of freedom; K=2 routes gradient to bandwidth head.'}
    write_json(LOGS / 'bandwidth_gradient_probe.json', result)
    print(json.dumps({'passed': True, 'seed': seed, 'gradient_norms': gradients, **counters}))


if __name__ == '__main__':
    main()
