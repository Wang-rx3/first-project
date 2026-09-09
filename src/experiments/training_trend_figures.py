"""Diagnostic-only figures; single seed, no smoothing or inferential intervals."""
from pathlib import Path
import numpy as np
import pandas as pd


def figures(logs: Path, output: Path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from .training_trend_diagnostic import fresh
    fresh(output)
    endpoints=pd.read_csv(logs/'endpoint_summary.csv')
    probes=pd.read_csv(logs/'gradient_probes.csv',na_values=['NA'])
    history=pd.read_csv(logs/'R_P_T_episode_parts.csv')
    colors={'initial':'#0072B2','final':'#D55E00'}
    def save(fig,name):
        fig.savefig(output/f'{name}.png',dpi=300,bbox_inches='tight')
        fig.savefig(output/f'{name}.pdf',bbox_inches='tight')
        plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(14,8),layout='constrained')
    fields=[('mean_reward','Mean reward / window'),('position_rmse_norm_m','Position RMSE (m)'),
            ('measurements','Selected measurements / episode')]
    for row,(method,group) in enumerate(endpoints.groupby('method',sort=False)):
        for col,(field,label) in enumerate(fields):
            ax=axes[row,col]
            for endpoint,g in group.groupby('endpoint',sort=False):
                g=g.sort_values('scene')
                ax.plot(g.scene,g[field],marker='o',color=colors[endpoint],label=endpoint)
            ax.set(xlabel='Fixed monitoring scene',ylabel=label,title=('GAT baseline' if row==0 else 'Proposed method'))
            ax.set_xticks(range(5)); ax.grid(alpha=.2); ax.legend()
    fig.suptitle('Frozen endpoints | seed 42 | same five development scenes | not independent training seeds')
    save(fig,'frozen_endpoint_comparison')
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for endpoint,g in probes.groupby('endpoint',sort=False):
        indices=np.arange(len(g))
        for ax,field,title in zip(axes.flat,
                ['critic_squared_fraction','actor_retention_ratio','ppo_teacher_cosine','actor_gradient_norm'],
                ['Critic squared-gradient share','Grouped / joint retained actor norm','PPO-teacher cosine (measurement head)','Actor gradient norm before clipping']):
            ax.plot(indices,g[field],label=endpoint,color=colors[endpoint],alpha=.85)
            ax.set(xlabel='Probe minibatch index (not training update)',ylabel=title)
            ax.grid(alpha=.2); ax.legend()
    axes[0,0].axhline(.8,color='black',linestyle=':',label='screening threshold')
    axes[0,1].axhline(2.,color='black',linestyle=':')
    axes[1,0].axhline(0,color='black',linestyle=':')
    fig.suptitle('Frozen gradient probes | repeated batches are dependent | no optimizer steps')
    save(fig,'frozen_gradient_probes')
    fig,axes=plt.subplots(3,2,figsize=(14,10),layout='constrained')
    for row,arm in enumerate(['reference','proxy-aligned','rollout4']):
        g=history[(history.arm==arm)&(history.method=='communication_constrained_proxy_guided_mappo')].sort_values('training_episode')
        axes[row,0].plot(g.training_episode,g.cumulative_return,color='#0072B2',label='Episode return')
        short=g.window_count<50
        axes[row,0].scatter(g.loc[short,'training_episode'],g.loc[short,'cumulative_return'],marker='x',color='#D55E00',label='Short episode')
        for field,label in [('reward_loc','Localization'),('reward_com','Communication'),('reward_collision','Collision'),('reward_energy','Energy')]:
            axes[row,1].plot(g.training_episode,g[field],label=label,linewidth=1)
        for ax in axes[row]: ax.set(xlabel='Training episode',title=arm); ax.grid(alpha=.2); ax.legend(fontsize=8)
        axes[row,0].set_ylabel('Episode return (sum)'); axes[row,1].set_ylabel('Signed reward contribution / window')
    fig.suptitle('Historical R/P/T proposed method | raw single-seed traces | no smoothing')
    save(fig,'historical_reward_attribution')
