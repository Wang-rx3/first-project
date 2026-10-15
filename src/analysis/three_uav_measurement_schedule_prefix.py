"""Locked 48-joint-step scheduling comparison, with no learning operations."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from src.control.fixed_navigation_execution import compose_fixed_navigation_actions
from src.experiments import three_uav_measurement_selection as v3
from src.experiments import three_uav_measurement_selection_contract as contract
from src.experiments.three_uav_fixed_navigation_contract import joint_closest_approach
from src.experiments.three_uav_measurement_selection_v4 import WindowRoundRobin, mask_observations
from src.simulation.environment import MultiUAVJointEnv

ROOT = Path(__file__).resolve().parents[2]
STEM = 'three_uav_measurement_learnability_v4_seed2026_20261012'


def run(run_id: str) -> dict:
    if not run_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in run_id):
        raise ValueError('Invalid exclusive run identifier')
    paths = [ROOT / area / STEM / run_id for area in ('data/synthetic', 'logs/preliminary')]
    if any(path.exists() for path in paths):
        raise FileExistsError('Output already exists; use a new run-id')
    for path in paths:
        path.mkdir(parents=True, exist_ok=False)
    design = v3.read_json(ROOT / 'data/synthetic' / STEM / 'diagnostic_design.lock.json')
    if design['max_joint_steps_total'] != 48 or design['prefix_steps_per_arm'] != 24:
        raise ValueError('Locked bounded design changed')
    spec = v3.read_json(v3.CONFIG)
    sim = v3.effective_simulation(spec)
    case = next(c for c in contract.make_scene_splits()['train'] if c['scene_id'] == design['prefix_scene'])
    if case['seed'] != design['prefix_seed']:
        raise ValueError('Scene/seed mismatch')
    output = v3.RunOutput.__new__(v3.RunOutput)
    output.data, output.logs, output.tables = paths[0], paths[1], {}
    forbidden = []
    def guard(owner, name):
        original = getattr(owner, name)
        forbidden.append((owner, name, original))
        def fail(*args, **kwargs):
            raise RuntimeError('Learning operation forbidden in scheduling diagnosis: ' + name)
        setattr(owner, name, fail)
    for name in ('__init__', 'update', 'add_transition', 'collect_rollout'):
        guard(v3.MAPPOTrainer, name)
    for name in ('__init__', 'act', 'act_with_trace'):
        guard(v3.MAPPOPolicy, name)
    guard(v3.torch, 'load')
    for cls in (v3.torch.optim.Adam, v3.torch.optim.AdamW, v3.torch.optim.SGD):
        guard(cls, 'step')
    counts = dict(joint_environment_steps=0, ppo_updates=0, optimizer_steps=0,
                  rollout_additions=0, normalization_fits=0, checkpoint_loads=0)
    summaries = []
    try:
        for arm in design['prefix_arms']:
            env = MultiUAVJointEnv(sim)
            native, _ = env.reset(seed=case['seed'], scenario_config=copy.deepcopy(case['scenario']))
            scheduler = WindowRoundRobin()
            multi, separated, accepted, packets = 0, 0, 0, 0
            minimum_distance = float('inf')
            for step in range(1, 25):
                physical = contract.admissible_observations(native)
                obs = (mask_observations(native, step) if arm == 'v4_window_round_robin' else physical)
                selections = {uid: (scheduler.select(obs[uid], step) if arm == 'v4_window_round_robin'
                                   else contract.round_robin_selection(obs[uid], step)) for uid in contract.UIDS}
                communication = contract.communication_actions(obs, selections)
                actions, _ = compose_fixed_navigation_actions(obs, communication)
                positions = {uid: env.states[uid].position_m.copy() for uid in contract.UIDS}  # diagnostics only
                velocities = {uid: env.states[uid].velocity_mps.copy() for uid in contract.UIDS}
                next_native, _, terminated, truncated, info = env.step(actions)
                counts['joint_environment_steps'] += 1
                if counts['joint_environment_steps'] > 48:
                    raise RuntimeError('Locked interaction cap exceeded')
                post = {uid: env.states[uid].position_m.copy() for uid in contract.UIDS}
                distance = min(row['distance_m'] for row in joint_closest_approach(positions, post))
                minimum_distance = min(minimum_distance, distance)
                if distance < 2 or info['collision'] or terminated or truncated:
                    raise RuntimeError('Unsafe or unexpectedly terminated prefix')
                for uid in contract.UIDS:
                    execution = info['action_execution'][uid]
                    acceleration = (env.states[uid].velocity_mps - velocities[uid]) / sim.motion.dt_s
                    if np.linalg.norm(acceleration) > 3 + 1e-8 or any(execution.get(k, False) for k in
                            ('boundary_projected', 'safety_projected', 'action_clipped')):
                        raise RuntimeError('Execution safety/projection check failed')
                    mask = contract.admissible_mask(obs[uid])
                    sources = [s for i, s in enumerate(contract.UIDS) if mask[i]]
                    values = [contract.expected_relative_credit(obs[uid], s, .0064) for s in sources]
                    spread = max(values) - min(values) if len(values) > 1 else 0.
                    multi += len(sources) == 2
                    separated += len(sources) == 2 and spread > 1e-6
                    output.row('prefix_actions', dict(arm=arm, step=step, uav_id=uid,
                        physical_legal_mask=physical[uid]['action_mask'], scheduled_mask=mask,
                        selected_source=selections[uid], executed_source=info['actions'][uid]['measurement_uav_id'],
                        expected_credit_spread=spread, budget_before=obs[uid]['measurement_budget_remaining'],
                        budget_after=next_native[uid]['measurement_budget_remaining'],
                        fixed_navigation_requested=actions[uid]['delta_v'], fixed_navigation_applied=execution['delta_v_applied'],
                        actual_speed_mps=float(np.linalg.norm(env.states[uid].velocity_mps)), actual_acceleration_mps2=acceleration,
                        true_position_diagnostic=post[uid], estimate_position=next_native[uid]['estimate'][:3],
                        boundary_projected=False, safety_projected=False, collision=False,
                        truth_actor_input=False, truth_navigation_input=False))
                for table, key in (('prefix_gates', 'measurement_gate_events'), ('prefix_budget', 'measurement_budget_events'),
                                   ('prefix_overhead', 'measurement_overhead_events'), ('prefix_fusion', 'relative_range_update_events')):
                    for event in info[key]:
                        output.row(table, {**event, 'arm': arm, 'step': step})
                accepted += sum(bool(e['accepted']) for e in info['relative_range_update_events'])
                packets += sum(float(e['load_packets_applied']) for e in info['measurement_overhead_events'])
                native = next_native
            summaries.append(dict(arm=arm, steps=24, two_legal_decisions=multi,
                separated_decisions_original_tolerance=separated, accepted_updates=accepted,
                measurement_overhead_packets=packets, minimum_pair_distance_m=minimum_distance,
                remaining_budgets={uid: native[uid]['measurement_budget_remaining'] for uid in contract.UIDS},
                scheduler_state=scheduler.state_dict() if arm.startswith('v4') else None))
    finally:
        for owner, name, original in reversed(forbidden):
            setattr(owner, name, original)
        output.flush()
        v3.write_json(paths[1] / 'counts.json', counts)
    result = dict(arms=summaries, counts=counts, training_ready=False,
        interpretation='One scene, two matched indexed-noise prefixes; eligibility/cost diagnosis only. No parking or generalization conclusion.')
    v3.write_json(paths[0] / 'configuration.json', spec)
    v3.write_json(paths[0] / 'manifest.json', dict(design=design, scene=case,
        sources={**v3.source_manifest(), 'src/analysis/three_uav_measurement_schedule_prefix.py': v3.file_hash(Path(__file__))},
        physical_configuration_unchanged=True))
    v3.write_json(paths[1] / 'summary.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    print(json.dumps(run(parser.parse_args().run_id), ensure_ascii=False))
