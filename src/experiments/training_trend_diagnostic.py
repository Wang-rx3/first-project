"""Frozen endpoint and clipping probes. No historical optimizer steps or test reads."""
from __future__ import annotations
from dataclasses import replace
from pathlib import Path
import numpy as np
import torch

from .baseline import _read_json, _read_csv, _sha256, _reference_formal_config
from .formal import _make_environment, _scenario_for, _scale_seed
from .mainline_baseline_comparison import _checkpoint_trainer, _write_table, BASELINE_BRANCH, COMMUNICATION_BRANCH
from .mainline_figures_v2 import SourceGuard
from .cooperative_event_logging import accepted_counts, collect_window_events
from .pilot import _seed_everything, _assert_observation_finite
from ..learning.mappo import MAPPOTrainer
from ..synthetic.episode_generator import _write_json, _ekf_diagnostic_row

SOURCE=Path('data/synthetic/mainline_baseline_comparison_v2_seed42/development/rollout4/seed_42')
OLD_LOGS=Path('logs/preliminary/mainline_baseline_comparison_v2_seed42')
LOGS=Path('logs/preliminary/training_trend_diagnostic_seed42')
DATA=Path('data/synthetic/training_trend_diagnostic_seed42')
PLOTS=Path('plots/preliminary/training_trend_diagnostic_seed42')

def fresh(*paths):
    for p in paths:
        if p.exists() and any(p.iterdir()): raise FileExistsError(f'new directory required: {p}')
    for p in paths: p.mkdir(parents=True,exist_ok=True)

def trainer_at(method, endpoint):
    scale=SOURCE/method/'scale_3'
    config=_reference_formal_config(_read_json(scale/'manifest.json'),DATA,force=False,progress=False)
    stats=_read_json(scale/'normalization_stats.json')
    config=replace(config,metric_scales=stats['metric_scales'])
    env=_make_environment(config,42,metric_scales=config.metric_scales)
    env.reset(seed=42)
    final=_checkpoint_trainer(scale,config,env)
    if endpoint=='final': return config,final
    initial=MAPPOTrainer(config=final.config,uav_ids=final.policy.uav_ids,seed=42,
        normalizer=final.normalizer,central_normalizer=final.central_normalizer,
        measurement_quality_model=final.measurement_quality_model,
        measurement_proxy_model=final.measurement_proxy_model,
        measurement_persistent_proxy_model=final.measurement_persistent_proxy_model)
    expected=_read_json(scale/'diagnostics/initial_parameter_hashes.json')['modules']
    actual=initial.policy.initial_parameter_initialization_hashes()
    if any(actual.get(k)!=v for k,v in expected.items()): raise ValueError('initial module hashes differ')
    return config,initial

def policy_vector(trainer):
    return torch.cat([p.detach().flatten().cpu() for p in trainer.policy.parameters()]).clone()

def collect(config,trainer,*,seed,index,monitor,identity,run_root=DATA):
    """Fresh environment, optional on-policy collection; offline truth only for labels."""
    _seed_everything(seed)
    env=_make_environment(config,seed,metric_scales=config.metric_scales)
    obs,info=env.reset(seed=seed,scenario_config=_scenario_for(config,seed,split='train',episode_index=index))
    rows=[]; events=[]
    for step in range(config.max_steps):
        _assert_observation_finite(obs)
        with torch.no_grad():
            decision=trainer.act(obs,info['central_state'],deterministic=monitor)
        counts=accepted_counts(env)
        nxt,rewards,terminated,truncated,ninfo=env.step(decision.actions)
        events.extend(collect_window_events(env,counts,run_root=run_root,episode_id=identity,
            split='development_monitor' if monitor else 'train',step=step))
        if not monitor: trainer.add_transition(decision,rewards,ninfo['central_state'],done=bool(terminated or truncated))
        parts=ninfo['objective_components']; penalties=ninfo['reward_components'][env.uav_ids[0]]
        reward=float(np.mean(list(rewards.values())))
        components={'reward_loc':-float(parts['j_loc']),'reward_com':-float(parts['j_com']),
                    'reward_collision':-float(penalties['collision_penalty']),'reward_energy':-float(penalties['energy_penalty'])}
        if abs(reward-sum(components.values()))>1e-6: raise ValueError('reward closure failed')
        error=float(np.mean([np.sum((env.ekfs[u].position-env.states[u].position_m)**2) for u in env.uav_ids]))
        for ekf in env.ekfs.values():
            if not np.isfinite(ekf.covariance).all() or not np.isfinite(ekf.state).all(): raise ValueError('nonfinite filter')
        consistency=[_ekf_diagnostic_row({},step,u,env.states[u].position_m,env.ekfs[u].state,
            env.ekfs[u].covariance,ninfo['ekf_diagnostics'][u]) for u in env.uav_ids]
        diagnostic_fields=['position_nees','nis_vo','nis_uwb_mean','nis_uwb_max','nis_gnss']
        if not all(np.isfinite(r[k]) for r in consistency for k in diagnostic_fields):
            raise ValueError('nonfinite NIS or position NEES')
        selected=sum(ninfo['actions'][u]['measurement_uav_id']!='none' for u in env.uav_ids)
        overhead=ninfo['measurement_overhead_events']
        if config.measurement_overhead_enabled and (len(overhead)!=selected or any(
                r['payload_size_bytes']!=512 or r['load_packets_applied']!=1 for r in overhead)):
            raise ValueError('measurement payload closure failed')
        if any(env.measurement_budget_used[u]>limit for u,limit in env.measurement_budget_limits.items()):
            raise ValueError('measurement budget exceeded')
        rows.append({'step':step,'reward':reward,'squared_error':error,**components,'measurements':selected,
                     'none_rate':1-selected/3,'collision':int(ninfo['collision']),
                     'invalid_action_count':ninfo['invalid_measurement_action_count'],
                     **{k+'_max':max(r[k] for r in consistency) for k in diagnostic_fields},
                     **{k:float(ninfo['global_metrics'][k]) for k in ('packet_loss','delay','aoi','connectivity')}})
        obs,info=nxt,ninfo
        if terminated or truncated: break
    accepted=[e for e in events if e['accepted']]
    summary={k:float(np.mean([r[k] for r in rows])) for k in ('reward','reward_loc','reward_com','reward_collision','reward_energy','none_rate','packet_loss','delay','aoi','connectivity')}
    summary.update(windows=len(rows),mean_reward=summary['reward'],cumulative_return=sum(r['reward'] for r in rows),
        position_rmse_norm_m=float(np.sqrt(np.mean([r['squared_error'] for r in rows]))),
        measurements=sum(r['measurements'] for r in rows),accepted_updates=len(accepted),collision=any(r['collision'] for r in rows),
        harmful_fraction=sum(e['true_position_error_after_m']-e['true_position_error_before_m']>.001 for e in accepted)/len(accepted) if accepted else 'NA')
    summary['measurement_bytes']=512*summary['measurements']
    if not all(np.isfinite(v) for v in summary.values() if isinstance(v,(float,np.floating))):
        raise ValueError('nonfinite endpoint summary')
    return summary,rows,events

def components(trainer,records):
    cfg=trainer.config; policy=trainer.policy; zero=torch.zeros((),device=policy.device)
    logp=[]; entropy=[]; teachers=[]; costs={}
    for r in records:
        lp,en=policy.evaluate_actions(r['observation'],r['sample']); logp.append(lp); entropy.append(en)
        teachers.append(policy.measurement_proxy_teacher_loss(r['observation'],r['sample'],uav_id=r['uav_id'])[0])
        if cfg.measurement_communication_constraint_enabled:
            costs.setdefault(r['uav_id'],[]).append(policy.expected_measurement_communication_cost(r['observation']))
    tensor=lambda values:torch.as_tensor(values,dtype=torch.float32,device=policy.device)
    ratios=(torch.stack(logp)-tensor([r['sample'].old_log_prob for r in records])).exp()
    advantage=tensor([r['advantage'] for r in records])
    ppo=-torch.minimum(ratios*advantage,ratios.clamp(1-cfg.clip_epsilon,1+cfg.clip_epsilon)*advantage).mean()
    value=policy.value(np.stack([r['central_vector'] for r in records]))
    critic=cfg.value_coef*.5*(value-tensor([r['return'] for r in records])).square().mean()
    communication=[]
    for u,values in costs.items():
        violation=torch.relu(torch.stack(values).mean()-cfg.measurement_communication_cost_target_by_uav[u]-cfg.measurement_communication_cost_tolerance)
        communication.append(violation*float(trainer.measurement_communication_dual_by_uav[u]))
    return {'ppo':ppo,'entropy':-cfg.entropy_coef*torch.stack(entropy).mean(),
            'teacher':cfg.measurement_proxy_teacher_coef*torch.stack(teachers).mean(),
            'communication':torch.stack(communication).mean() if communication else zero,'critic':critic}

def gradient_probe(trainer,records):
    named=list(trainer.policy.named_parameters()); params=[p for _,p in named]
    terms=components(trainer,records); vectors={}
    for label,loss in terms.items():
        grads=torch.autograd.grad(loss,params,allow_unused=True,retain_graph=True) if loss.requires_grad else [None]*len(params)
        vectors[label]=[torch.zeros_like(p) if g is None else g.detach() for p,g in zip(params,grads)]
    for label in ('teacher','communication'):
        if any(torch.count_nonzero(g).item() for (n,_),g in zip(named,vectors[label]) if not n.startswith('measurement_head.')):
            raise ValueError(f'{label} auxiliary gradient leaked outside measurement head')
    norm=lambda values:float(torch.sqrt(sum(v.square().sum() for v in values)))
    combined=[sum(vectors[k][i] for k in vectors) for i in range(len(params))]
    actor=[g for (n,_),g in zip(named,combined) if not n.startswith('critic.')]
    critic=[g for (n,_),g in zip(named,combined) if n.startswith('critic.')]
    a,c=norm(actor),norm(critic); total=(a*a+c*c)**.5; limit=trainer.config.max_grad_norm
    joint=min(1.,limit/(total+1e-6)); separate=min(1.,limit/(a+1e-6))
    head=lambda key:torch.cat([g.flatten() for (n,_),g in zip(named,vectors[key]) if n.startswith('measurement_head.')])
    x,y=head('ppo'),head('teacher'); denominator=float(x.norm()*y.norm())
    row={f'{k}_gradient_norm':norm(v) for k,v in vectors.items()}
    row.update({f'{key}_measurement_head_gradient_norm':float(head(key).norm())
                for key in vectors})
    row.update(actor_gradient_norm=a,critic_gradient_norm=c,total_gradient_norm=total,
       critic_squared_fraction=c*c/(total*total) if total else 0.,joint_clip_factor=joint,actor_clip_factor=separate,
       critic_clip_factor=min(1.,limit/(c+1e-6)),actor_after_joint=a*joint,actor_after_group=a*separate,
       critic_after_joint=c*joint,critic_after_group=c*min(1.,limit/(c+1e-6)),
       actor_retention_ratio=separate/joint,ppo_teacher_cosine=float(torch.dot(x,y))/denominator if denominator else 'NA',
       actor_nonzero=a>0,valid=np.isfinite([a,c,total]).all(),samples=len(records))
    if not all(np.isfinite(v) for v in row.values() if isinstance(v,(float,np.floating))): raise ValueError('nonfinite probe')
    return row


def batch_metadata(records):
    returns=np.asarray([r['return'] for r in records],dtype=float)
    advantages=np.asarray([r['advantage'] for r in records],dtype=float)
    if not np.isfinite(returns).all() or not np.isfinite(advantages).all():
        raise ValueError('nonfinite returns or GAE advantages')
    available=[]
    for r in records:
        mask=np.asarray(r['observation']['action_mask']).astype(bool)
        if not mask[-1] or not mask[r['sample'].measurement_index]:
            raise ValueError('none missing or sampled measurement masked out')
        available.append(int(mask.sum()))
    return {'return_mean':float(returns.mean()),'return_std':float(returns.std()),
            'advantage_mean':float(advantages.mean()),'legal_measurement_options_mean':float(np.mean(available)),
            'masks_valid':True,'returns_finite':True}

def gate(rows):
    valid=[r for r in rows if r['valid'] and r['actor_nonzero']]
    passing=[r for r in valid if r['critic_squared_fraction']>=.8 and r['actor_retention_ratio']>=2]
    return {'valid_minibatches':len(valid),'joint_criteria_minibatches':len(passing),
            'required_passing_minibatches':int(np.ceil(.5*len(valid))),
            'eligible':len(valid)>=20 and len(passing)>=.5*len(valid),
            'interpretation':'screen for controlled clipping experiment, not causal proof; repeated probe batches are not independent samples'}

def summarize_history(logs):
    import pandas as pd
    results=[]; monitor=[]; episode_parts=[]
    for arm in ('reference','proxy-aligned','rollout4'):
        root=OLD_LOGS/'development'/arm/'figures'
        episodes=pd.read_csv(root/'training_episodes.csv'); updates=pd.read_csv(root/'training_updates.csv')
        parts=pd.read_csv(root/'reward_decomposition.csv'); points=pd.read_csv(root/'development_monitor.csv')
        for method,g in episodes.groupby('method'):
            g=g.sort_values('training_episode').copy(); g['cumulative_windows']=g.window_count.cumsum()
            u=updates[updates.method==method]
            result={'arm':arm,'method':method,'episodes':len(g),'updates':len(u),'optimizer_steps':u.optimizer_steps.sum(),
                'multiplier_updates':u.communication_multiplier_updates.sum(),'return_std':g.cumulative_return.std(),
                'mean_return':g.cumulative_return.mean(),'mean_reward':g.mean_reward.mean(),'early_count':int((g.window_count<50).sum()),
                'kl_mean':u.approximate_kl.mean(),'clip_fraction':u.clip_fraction.mean()}
            results.append(result)
            means=parts[parts.method==method].groupby('episode_id')[['reward_loc','reward_com','reward_collision','reward_energy']].mean()
            joined=g.merge(means,left_on='episode_id',right_index=True)
            episode_parts.extend({'arm':arm,**r} for r in joined[['method','training_episode','cumulative_windows','window_count','collision_occurred','mean_reward','cumulative_return','reward_loc','reward_com','reward_collision','reward_energy']].to_dict('records'))
        for (method,scene),g in points.groupby(['method','monitor_scene']):
            first=g[g.completed_training_episodes==0].iloc[0]; last=g[g.completed_training_episodes==100].iloc[0]
            monitor.append({'arm':arm,'method':method,'scene':scene,**{k+'_difference':last[k]-first[k] for k in ('mean_reward','position_rmse_norm_m','windows')}})
    _write_table(logs/'R_P_T_summary.csv',results); _write_table(logs/'R_P_T_episode_parts.csv',episode_parts); _write_table(logs/'monitor_endpoint_pairs.csv',monitor)


def evidence_summary(logs):
    """Descriptive endpoint and stratified statistics, never a causal test."""
    import pandas as pd
    endpoints=pd.read_csv(logs/'endpoint_summary.csv')
    probes=pd.read_csv(logs/'gradient_probes.csv',na_values=['NA'])
    history=pd.read_csv(logs/'R_P_T_episode_parts.csv')
    strata=[]; reward_variation=[]
    for (arm,method),g in history.groupby(['arm','method']):
        total_variance=float(g.mean_reward.var())
        for field in ['reward_loc','reward_com','reward_collision','reward_energy']:
            reward_variation.append({'arm':arm,'method':method,'component':field,
                'mean':float(g[field].mean()),'std':float(g[field].std()),
                'covariance_share_of_mean_reward_variance':float(g[field].cov(g.mean_reward))/total_variance if total_variance else 'NA'})
    _write_table(logs/'reward_variation.csv',reward_variation)
    for (arm,method,collision),g in history.groupby(['arm','method','collision_occurred']):
        strata.append({'arm':arm,'method':method,'collision':bool(collision),'episodes':len(g),
            **{field:float(g[field].mean()) for field in ['mean_reward','cumulative_return','window_count']}})
    _write_table(logs/'collision_stratified_history.csv',strata)
    endpoint_means=endpoints.groupby(['method','endpoint'])[['mean_reward','position_rmse_norm_m',
        'measurements','accepted_updates','reward_loc','reward_com','reward_collision','reward_energy']].mean().reset_index()
    _write_table(logs/'endpoint_group_means.csv',endpoint_means.to_dict('records'))
    probe_stats=[]
    for endpoint,g in probes.groupby('endpoint'):
        qualifies=(g.critic_squared_fraction>=.8)&(g.actor_retention_ratio>=2)
        probe_stats.append({'endpoint':endpoint,'minibatches':len(g),'qualifying_minibatches':int(qualifies.sum()),
             'negative_ppo_teacher_cosines':int((g.ppo_teacher_cosine<0).sum()),
             **{field+'_mean':float(g[field].mean()) for field in ['actor_gradient_norm','critic_gradient_norm',
                'critic_squared_fraction','actor_retention_ratio','ppo_teacher_cosine']}})
    _write_table(logs/'probe_endpoint_summary.csv',probe_stats)
    proposed=endpoint_means[endpoint_means.method==COMMUNICATION_BRANCH].set_index('endpoint')
    delta={field:float(proposed.loc['final',field]-proposed.loc['initial',field]) for field in ['mean_reward','position_rmse_norm_m','measurements','accepted_updates','reward_loc','reward_com','reward_collision','reward_energy']}
    return {'proposed_monitor_final_minus_initial':delta,'probe_endpoints':probe_stats,
            'interpretation':'Endpoint association only; no intermediate trajectory reconstruction, no Adam updates, no cross-seed inference.',
            'not_identified':['causal role of clipping','causal role of teacher direction','causal role of weak localization reward feedback']}

def frozen_audit():
    logs=LOGS/'frozen-audit'; fresh(logs,DATA/'frozen-audit')
    report={'audit_ok':False,'stage':'frozen-audit','parameter_updates':0,'test_accesses':0}
    _write_json(logs/'audit_report.json',report)
    endpoints=[]; windows=[]; event_rows=[]; probes=[]; prefixes=[]
    try:
        with SourceGuard([SOURCE,OLD_LOGS],[logs,DATA/'frozen-audit']) as guard:
            summarize_history(logs)
            monitor=_read_csv(OLD_LOGS/'development/rollout4/figures/development_monitor.csv')
            for method in (BASELINE_BRANCH,COMMUNICATION_BRANCH):
                traces={}
                for endpoint in ('initial','final'):
                    config,trainer=trainer_at(method,endpoint); trainer.policy.eval(); before=policy_vector(trainer)
                    for index in range(5):
                        seed=42+900000+index*1009; identity=f'{method}_{endpoint}_{index}'
                        summary,rows,events=collect(config,trainer,seed=seed,index=index,monitor=True,identity=identity)
                        old=next(r for r in monitor if r['method']==method and int(r['monitor_scene'])==index and int(r['completed_training_episodes'])==(0 if endpoint=='initial' else 100))
                        errors={k:abs(float(summary[k])-float(old[k])) for k in ('mean_reward','position_rmse_norm_m','windows','cumulative_return')}
                        if max(errors.values())>1e-6: raise ValueError(f'endpoint mismatch: {identity}: {errors}')
                        endpoints.append({'method':method,'endpoint':endpoint,'scene':index,**summary,'max_replay_error':max(errors.values())})
                        windows.extend({'method':method,'endpoint':endpoint,'scene':index,**r} for r in rows)
                        event_rows.extend({'method':method,'endpoint':endpoint,'scene':index,**e} for e in events)
                        traces[endpoint,index]=rows
                        print(f'[trend audit] endpoint {method}/{endpoint}/{index+1} matched',flush=True)
                    if not torch.equal(before,policy_vector(trainer)): raise ValueError('endpoint changed weights')
                    if method==COMMUNICATION_BRANCH:
                        trainer.rollout.clear()
                        for index in range(4):
                            collect(config,trainer,seed=_scale_seed(42,3,'train',index),index=index,monitor=False,identity=f'probe_{endpoint}_{index}')
                        records=trainer._training_records()
                        for epoch in range(trainer.config.policy_epochs):
                            order=trainer.rng.permutation(len(records))
                            for start in range(0,len(order),trainer.config.minibatch_size):
                                batch=[records[int(i)] for i in order[start:start+trainer.config.minibatch_size]]
                                probes.append({'endpoint':endpoint,'epoch':epoch,'minibatch':start//trainer.config.minibatch_size,
                                    **batch_metadata(batch),**gradient_probe(trainer,batch)})
                            print(f'[trend audit] {endpoint} probe epoch {epoch+1} complete',flush=True)
                        if not torch.equal(before,policy_vector(trainer)): raise ValueError('probe changed weights')
                for i in range(5):
                    n=min(len(traces['initial',i]),len(traces['final',i]))
                    prefixes.append({'method':method,'scene':i,'common_windows':n,**{s+'_rmse':float(np.sqrt(np.mean([r['squared_error'] for r in traces[s,i][:n]]))) for s in ('initial','final')}})
            hashes=guard.verify()
        if guard.test_attempts or not all(r['unchanged'] for r in hashes): raise ValueError('source isolation failure')
        for name,rows in [('endpoint_summary',endpoints),('endpoint_windows',windows),('endpoint_events',event_rows),('gradient_probes',probes),('common_prefix',prefixes)]: _write_table(logs/f'{name}.csv',rows)
        evidence=evidence_summary(logs)
        report.update(audit_ok=True,status='complete',gate=gate(probes),endpoint_count=len(endpoints),source_hashes=hashes,
            evidence=evidence,
            smoke_status='eligible_not_started' if gate(probes)['eligible'] else 'not_run_gate_failed',
            initial_hashes_checked=True, endpoint_parameter_vectors_unchanged=True,
            reward_closure_tolerance=1e-6, endpoint_replay_tolerance=1e-6,
            packet_and_budget_checks=True, event_counter_checks=True,
            nis_position_nees_finite=True, auxiliary_gradient_boundary_checked=True,
            findings={'length_mixture':'confirmed','fixed_monitor_no_sustained_improvement':'confirmed','critic_clipping_coupling':'supported_for_testing' if gate(probes)['eligible'] else 'insufficient_evidence',
                      'causal_effect_on_training':'insufficient_evidence','intermediate_checkpoints':'not_available'})
    except Exception as exc:
        report.update(status='failed',error=str(exc)); raise
    finally: _write_json(logs/'audit_report.json',report)
    from .training_trend_figures import figures
    figures(logs,PLOTS/'frozen-audit')
    return {'audit_ok':report['audit_ok'],'gate':report['gate'],'smoke_status':report['smoke_status'],'report':str(logs/'audit_report.json')}

def run(stage,seed=42):
    if seed!=42: raise ValueError('only seed42 is authorized')
    if stage=='frozen-audit': return frozen_audit()
    if stage=='smoke':
        report=_read_json(LOGS/'frozen-audit/audit_report.json')
        if not report['audit_ok']: raise ValueError('frozen audit must pass first')
        if not report['gate']['eligible']:
            return {'audit_ok':True,'status':'not_run_gate_failed','gate':report['gate'],
                    'message':'No training or clipping changes authorized by the prespecified screening gate.'}
        raise NotImplementedError('This version stopped at the failed frozen gate; a passing re-audit requires implementing and testing the gated clipping comparison before execution.')
    raise ValueError(stage)
