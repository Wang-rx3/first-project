"""Zero-interaction legacy joint and Stage-2 communication regression."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from .three_uav_legacy_action_fixture import fixtures, observation, state_hash, convert, IDS
from ..learning.mappo import MAPPOConfig, MAPPOPolicy

ROOT = Path(__file__).resolve().parents[2]
STEM = 'three_uav_measurement_selection_v3_seed2026_20261012'
DATA = ROOT / 'data/synthetic' / STEM
LOGS = ROOT / 'logs/preliminary' / STEM

def communication_fixtures():
    result = []
    for head in ('global_mean', 'action_conditioned_v3'):
        for active in (0, 1, 2):
            config = MAPPOConfig(uav_count=3, node_feature_dim=37, edge_feature_dim=11,
                gat_hidden_dim=8, gat_output_dim=8, gat_heads=1, gat_layers=1,
                actor_hidden_dim=16, critic_hidden_dim=16, max_obstacles=0,
                parameter_initialization_seed=2026115400, measurement_head_mode=head,
                motion_action_mode='fixed_navigation_comm_v1', ppo_action_logprob_mode='sampled')
            policy = MAPPOPolicy(config, uav_ids=IDS)
            policy.eval()
            rows = []
            for deterministic in (True, False):
                raw = copy.deepcopy(observation())
                raw['potential_targets'] = list(IDS[1:1+active])
                raw['bandwidth_mask'] = np.array([False] + [i < active for i in range(2)])
                torch.manual_seed(2026115700)
                with torch.no_grad():
                    action, sample = policy.act(raw, deterministic=deterministic)
                    lp, entropy = policy.evaluate_actions(raw, sample)
                rows.append({'deterministic': deterministic, 'action': action,
                    'measurement_index': sample.measurement_index, 'link_selection': sample.link_selection,
                    'bandwidth': sample.bandwidth, 'delta_v': sample.delta_v,
                    'old_log_prob': sample.old_log_prob, 'log_prob': float(lp), 'entropy': float(entropy),
                    'post_action_rng_sha256': hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()})
            result.append({'head': head, 'active': active, 'state_sha256': state_hash(policy),
                'trainable_parameters': [name for name, p in policy.named_parameters() if p.requires_grad], 'rows': rows})
    return result

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    values = {'joint': fixtures(), 'stage2_communication': communication_fixtures()}
    values = json.loads(json.dumps(values, default=convert, allow_nan=False))
    if args.check:
        assert values == json.loads((DATA/'legacy_fixtures_before.json').read_text(encoding='utf-8'))
        path = LOGS/'legacy_regression.json'
        values = {'passed': True, 'exact_equality': True, 'joint_configurations': 4, 'joint_actions': 8,
                  'stage2_configurations': 6, 'stage2_actions': 12, 'environment_steps': 0, 'optimizer_steps': 0}
    else:
        path = DATA/'legacy_fixtures_before.json'
    with path.open('x', encoding='utf-8') as handle:
        json.dump(values, handle, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({'passed': True, 'environment_steps': 0, 'optimizer_steps': 0, 'path': str(path)}))

if __name__ == '__main__':
    main()
