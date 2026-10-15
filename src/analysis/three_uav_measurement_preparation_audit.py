"""Independent arithmetic audit of the capped preparation precheck, no env."""
from __future__ import annotations
import csv
import json
from pathlib import Path
import numpy as np

from ..control.fixed_navigation import fixed_navigation_command

ROOT = Path(__file__).resolve().parents[2]
STEM = 'three_uav_measurement_selection_v3_seed2026_20261012'
BASE = ROOT / 'logs/preliminary' / STEM
RUN = BASE / 'prep_precheck_20261012'

def rows(name):
    with (RUN / (name + '.csv')).open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))

def yes(value):
    return str(value).lower() in ('true', '1')

def vector(value):
    return np.asarray(json.loads(value), dtype=float)

def main():
    lock = json.loads((ROOT/'data/synthetic'/STEM/'preparation_manifest.lock.json').read_text(encoding='utf-8-sig'))
    cases = {case['scene_id']: case for case in lock['scene_splits']['train']}
    actions, fusion, budget, overhead, diagnostics, joint, pairs = [rows(name) for name in (
        'actions', 'fusion_events', 'measurement_budget', 'measurement_overhead', 'localization_diagnostics', 'joint', 'pair_distances')]
    errors = []
    def check(label, condition):
        if not condition:
            errors.append(label)
    check('counts', len(actions)==72 and len(diagnostics)==72 and len(joint)==24 and len(pairs)==72)
    for row in actions:
        scene, step, uid = row['scene_id'], int(row['step']), row['uav_id']
        estimate = vector(row['raw_local_estimate'])
        requested = vector(row['fixed_navigation_requested'])
        expected = fixed_navigation_command(estimate, cases[scene]['scenario']['goals'][uid])['command_delta_v_mps']
        check(f'RAW navigation:{scene}/{step}/{uid}', np.allclose(requested, expected, rtol=0, atol=1e-12))
        mask = json.loads(row['action_mask'])
        chosen = row['sampled_measurement_source']
        index = 3 if chosen == 'none' else ('uav_00','uav_01','uav_02').index(chosen)
        check(f'mask:{scene}/{step}/{uid}', bool(mask[index]) and bool(mask[3]))
        if step == 1:
            check(f'startup:none:{scene}/{uid}', chosen == 'none')
        key_rows = lambda group: [r for r in group if r['scene_id']==scene and int(r['step'])==step]
        consumed = sum(float(r['budget_consumed']) for r in key_rows(budget) if r['uav_id']==uid)
        packets = sum(float(r['load_packets_applied']) for r in key_rows(overhead) if r['target_uav_id']==uid)
        events = [r for r in key_rows(fusion) if r['target_uav_id']==uid]
        credit = sum(max(0, min(1, (float(e['position_covariance_trace_before_m2'])-
            float(e['position_covariance_trace_after_linear_m2'])) / float(e['position_covariance_trace_before_m2'])))
            for e in events if yes(e['accepted']))
        cost, waste = .05*packets, .05*max(consumed-packets,0)
        check(f'budget:{scene}/{step}/{uid}', float(row['budget_consumed'])==consumed)
        check(f'packets:{scene}/{step}/{uid}', float(row['actual_overhead_packets'])==packets)
        check(f'proxy:{scene}/{step}/{uid}', abs(float(row['raw_relative_credit_proxy'])-credit)<1e-12)
        check(f'utility:{scene}/{step}/{uid}', abs(float(row['local_utility'])-(credit-cost-waste))<1e-12)
        for event in events:
            if yes(event['covariance_restored']):
                check('conservative trace restoration', abs(float(event['position_covariance_trace_before_m2'])-
                    float(event['position_covariance_trace_after_final_m2']))<1e-8)
        check('learned fixed dimensions', all(int(row[k])==0 for k in (
            'learned_delta_dimensions','learned_bandwidth_dimensions','learned_link_dimensions')))
    for row in joint:
        selected = [a for a in actions if a['scene_id']==row['scene_id'] and a['step']==row['step']]
        check('team reward mean', abs(float(row['team_learning_reward'])-np.mean([float(a['local_utility']) for a in selected]))<1e-12)
    # Rebuild the true velocity from saved diagnostic acceleration (not EKF).
    for scene, case in cases.items():
        data = [r for r in diagnostics if r['scene_id']==scene]
        if not data:
            continue
        velocities = {uid: np.zeros(3) for uid in ('uav_00','uav_01','uav_02')}
        previous = {uid: np.asarray(case['scenario']['initial_positions'][uid]) for uid in velocities}
        for step in range(1,9):
            current = {}
            for row in [r for r in data if int(r['step'])==step]:
                uid = row['uav_id']; acceleration = vector(row['actual_acceleration_mps2'])
                velocities[uid] += .2*acceleration
                current[uid] = vector(row['true_position_m_diagnostic'])
                check('actual speed', abs(np.linalg.norm(velocities[uid])-float(row['actual_speed_mps']))<1e-9)
                check('acceleration bound', np.linalg.norm(acceleration)<=3+1e-8)
                check('projection/collision', not any(yes(row[k]) for k in ('boundary_projected','safety_projected','collision')))
            for pair in [p for p in pairs if p['scene_id']==scene and int(p['step'])==step]:
                a,b = pair['source'],pair['target']
                relative = previous[a]-previous[b]
                change = current[a]-current[b]-relative
                norm2 = np.dot(change,change)
                fraction = np.clip(-np.dot(relative,change)/norm2,0,1) if norm2>1e-24 else 0.
                distance = np.linalg.norm(relative+fraction*change)
                check('synchronized separation', abs(distance-float(pair['distance_m']))<1e-8 and distance>=2)
            previous = current
    status = json.loads((RUN/'run_status.json').read_text(encoding='utf-8-sig'))
    check('no training costs', status['counts']['joint_environment_steps']==24 and
          all(value==0 for key,value in status['counts'].items() if key!='joint_environment_steps'))
    report = {'passed': not errors, 'errors': errors, 'independent': True, 'auditor_environment_steps': 0,
        'optimizer_steps':0, 'action_rows':len(actions), 'accepted_updates':sum(yes(e['accepted']) for e in fusion),
        'actual_overhead_packets':sum(float(r['load_packets_applied']) for r in overhead),
        'minimum_synchronized_pair_distance_m':min(float(p['distance_m']) for p in pairs),
        'training_readiness': 'not_evaluated', 'full_horizon_candidate_diversity_verified':False}
    with (BASE/'independent_precheck_audit.json').open('x',encoding='utf-8') as handle:
        json.dump(report,handle,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps(report,ensure_ascii=False))
    assert report['passed']

if __name__ == '__main__':
    main()
