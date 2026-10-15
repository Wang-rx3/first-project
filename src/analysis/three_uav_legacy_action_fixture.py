"""Zero-environment-step legacy MAPPO regression fixture, before/after opt-in edits."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from ..learning.mappo import MAPPOConfig, MAPPOPolicy, MAPPOTrainer, PolicyDecision

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'data/synthetic/three_uav_fixed_navigation_v2_seed2026_20261011'
IDS = ('uav_00', 'uav_01', 'uav_02')


def observation():
    return {
        'uav_id': IDS[0], 'node_ids': list(IDS),
        'node_features': np.arange(3 * 37, dtype=np.float32).reshape(3, 37) / 500,
        'effective_edge_index': np.array([[1, 2], [0, 0]], dtype=np.int64),
        'effective_edge_features': np.zeros((2, 11), dtype=np.float32),
        'candidate_incoming_sources': [IDS[1], IDS[2]],
        'candidate_incoming_edge_features': {IDS[1]: np.zeros(11), IDS[2]: np.ones(11) * .01},
        'potential_targets': [IDS[1], IDS[2]], 'effective_sources': [IDS[1], IDS[2]],
        'action_mask': np.array([False, True, True, True]), 'bandwidth_mask': np.array([False, True, True]),
        'estimate': np.array([40., 50., 20., .2, -.1, 0.]), 'goal': np.array([52., 50., 20.]),
    }


def convert(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def state_hash(policy):
    digest = hashlib.sha256()
    for key, tensor in sorted(policy.state_dict().items()):
        digest.update(key.encode()); digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def fixtures():
    result = []
    for head in ('global_mean', 'action_conditioned_v3'):
        for probability_mode in ('sampled', 'executed'):
            config = MAPPOConfig(uav_count=3, node_feature_dim=37, edge_feature_dim=11,
                                 gat_hidden_dim=8, gat_output_dim=8, gat_heads=1, gat_layers=1,
                                 actor_hidden_dim=16, critic_hidden_dim=16, max_obstacles=0,
                                 parameter_initialization_seed=2026115400,
                                 measurement_head_mode=head, ppo_action_logprob_mode=probability_mode)
            torch.manual_seed(2026115400)
            policy = MAPPOPolicy(config, uav_ids=IDS)
            policy.eval()
            records = []
            for deterministic in (True, False):
                torch.manual_seed(2026115401)
                obs = observation()
                with torch.no_grad():
                    action, sample = policy.act(obs, deterministic=deterministic)
                    lp, entropy = policy.evaluate_actions(obs, sample)
                decision = PolicyDecision(actions={IDS[0]: action}, samples={IDS[0]: copy.deepcopy(sample)},
                                          observations={IDS[0]: obs}, central_vector=np.zeros(config.critic_input_dim), value=0.)
                # Reconciliation needs only these fields: no trainer, optimizer or rollout is constructed.
                receiver = object.__new__(MAPPOTrainer)
                receiver.config, receiver.policy = config, policy
                MAPPOTrainer.reconcile_action_execution(receiver, decision, {
                    'actions': {IDS[0]: {'measurement_uav_id': IDS[1]}},
                    'applied_bandwidth': {IDS[0]: {IDS[1]: .25, IDS[2]: .75}},
                    'action_execution': {IDS[0]: {'delta_v_applied': [.1, -.2, .05]}}})
                reconciled = decision.samples[IDS[0]]
                records.append({'deterministic': deterministic, 'action': action, 'old_log_prob': sample.old_log_prob,
                                'log_prob': float(lp), 'entropy': float(entropy),
                                'reconciled_old_log_prob': reconciled.old_log_prob,
                                'reconciled_delta_v': reconciled.delta_v,
                                'reconciliation_diagnostics': decision.measurement_diagnostics,
                                'post_action_rng_sha256': hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()})
            result.append({'head': head, 'probability_mode': probability_mode,
                           'state_sha256': state_hash(policy), 'parameter_initialization_hashes': policy.parameter_initialization_hashes(),
                           'records': records})
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    result = fixtures()
    fixture_path = DATA / 'legacy_action_fixture_before_v2.json'
    encoded = json.dumps(result, indent=2, sort_keys=True, default=convert, allow_nan=False)
    if args.check:
        expected = json.loads(fixture_path.read_text(encoding='utf-8'))
        assert json.loads(encoded) == expected, 'Legacy default behavior changed'
        with (DATA / 'legacy_action_fixture_comparison.json').open('x', encoding='utf-8') as handle:
            json.dump({'passed': True, 'configurations': 4, 'actions': 8, 'exact_equality': True,
                       'environment_steps': 0, 'optimizer_steps': 0}, handle, indent=2)
    else:
        with fixture_path.open('x', encoding='utf-8') as handle:
            handle.write(encoded + '\n')
    print(json.dumps({'legacy_exact_comparison': bool(args.check), 'configurations': 4, 'actions': 8, 'environment_steps': 0}))


if __name__ == '__main__':
    main()
