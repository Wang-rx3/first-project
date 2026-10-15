"""Offline report and figures for frozen fixed-navigation evaluation."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
STEM = 'single_uav_fixed_navigation_v1_seed2026_20261011'
LOGS = ROOT / 'logs/preliminary' / STEM
PLOTS = ROOT / 'plots/preliminary' / STEM
DATA = ROOT / 'data/synthetic' / STEM


def load_rows(filename):
    with (LOGS / filename).open(encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle))


def plot_box_routes():
    """Read-only regeneration after figure QA; no simulation or metric writes."""
    rows = load_rows('obstacle_steps.csv')
    lock = json.loads((DATA / 'execution_manifest.lock.json').read_text(encoding='utf-8'))
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), layout='constrained')
    for col, prefix in enumerate(('box_x', 'box_y')):
        ax = axes[col]
        geometry_cases = [case for case in lock['cases']['obstacle'] if case['scenario_id'].startswith(prefix)]
        for case in geometry_cases:
            group = [r for r in rows if r['scenario_id'] == case['scenario_id']]
            trajectory = np.array([case['scenario']['initial_positions']['uav_00']] + [json.loads(r['true_position_post_m']) for r in group])
            references = np.array([case['scenario']['initial_positions']['uav_00']] + case['references'])
            line, = ax.plot(trajectory[:, 0], trajectory[:, 1], label=case['side'])
            ax.plot(references[:, 0], references[:, 1], ':', color=line.get_color(), alpha=.4)
        spec = geometry_cases[0]['scenario']['obstacle_specs'][0]
        center, size = np.array(spec['center_m']), np.array(spec['size_m'])
        ax.add_patch(Rectangle(center[:2] - size[:2]/2, *size[:2], color='gray', alpha=.6))
        ax.set_aspect('equal'); ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
        ax.legend(loc='upper right', fontsize=9)
        ax.set_title(prefix + ': known static waypoint routes')
    fig.savefig(PLOTS / 'known_box_routes_verified.png', dpi=180)
    plt.close(fig)


def main():
    summary = json.loads((LOGS / 'execution_summary.json').read_text(encoding='utf-8'))
    episodes = summary['episodes']
    aggregate = {}
    for phase in dict.fromkeys(e['phase'] for e in episodes):
        group = [e for e in episodes if e['phase'] == phase]
        aggregate[phase] = {
            'episodes': len(group), 'successes': sum(e['success'] for e in group),
            'stable_arrival_rate': np.mean([e['success'] for e in group]),
            'final_distance_m_median': np.median([e['final_distance_m'] for e in group]),
            'final_distance_m_maximum': max(e['final_distance_m'] for e in group),
            'final_speed_mps_median': np.median([e['final_speed_mps'] for e in group]),
            'final_speed_mps_maximum': max(e['final_speed_mps'] for e in group),
            'collisions': sum(e['collision'] for e in group), 'timeouts': sum(e['timeout'] for e in group),
            'boundary_projection_steps': sum(e['boundary_projection_steps'] for e in group),
            'safety_projection_steps': sum(e['safety_projection_steps'] for e in group),
            'braking_cases_before_first_hold': sum(e['braking_steps_before_first_hold'] > 0 for e in group),
            'maximum_actual_acceleration_mps2': max(e['max_actual_acceleration_mps2'] for e in group),
            'environment_steps': sum(e['steps'] for e in group)}
    from ..experiments.single_uav_fixed_navigation import write_json
    write_json(LOGS / 'aggregate_metrics.json', aggregate)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    if (LOGS / 'paired_steps.csv').exists():
        rows = load_rows('paired_steps.csv')
        ids = list(dict.fromkeys(r['scenario_id'] for r in rows))
        fig, axes = plt.subplots(2, 4, figsize=(14, 6), layout='constrained')
        for col, sid in enumerate(ids):
            for phase, style in [('paired_fixed', '-'), ('paired_coast', '--')]:
                group = [r for r in rows if r['scenario_id'] == sid and r['phase'] == phase]
                times = np.array([int(r['step']) for r in group]) * .2
                label = 'Fixed EKF feedback' if phase == 'paired_fixed' else 'Zero delta-v'
                axes[0, col].plot(times, [float(r['goal_distance_m_actual']) for r in group], style, label=label)
                axes[1, col].plot(times, [float(r['speed_mps_actual']) for r in group], style, label=label)
            axes[0, col].axhline(1, color='gray', lw=1, alpha=.6)
            axes[1, col].axhline(.5, color='gray', lw=1, alpha=.6)
            axes[0, col].set_title(group[0]['regime'])
            axes[1, col].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Actual goal distance (m)')
        axes[1, 0].set_ylabel('Actual speed (m/s)')
        axes[0, 0].legend(fontsize=8)
        fig.suptitle('Matched initial conditions: active braking versus passive motion')
        fig.savefig(PLOTS / 'matched_navigation.png', dpi=180)
        fig.savefig(PLOTS / 'matched_navigation.svg')
        plt.close(fig)
    if (LOGS / 'normal_evaluation_steps.csv').exists():
        rows = load_rows('normal_evaluation_steps.csv')
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
        for sid in dict.fromkeys(r['scenario_id'] for r in rows):
            group = [r for r in rows if r['scenario_id'] == sid]
            times = np.array([int(r['step']) for r in group]) * .2
            axes[0].plot(times, [float(r['goal_distance_m_actual']) for r in group], alpha=.45, lw=1)
            axes[1].plot(times, [float(r['speed_mps_actual']) for r in group], alpha=.45, lw=1)
        axes[0].axhline(1, color='black', ls='--', lw=1)
        axes[1].axhline(.5, color='black', ls='--', lw=1)
        axes[0].set_ylabel('Actual goal distance (m)')
        axes[1].set_ylabel('Actual speed (m/s)')
        for ax in axes:
            ax.set_xlabel('Time (s)')
        fig.suptitle('36 independent no-obstacle cases; curves stop at strict success or horizon')
        fig.savefig(PLOTS / 'normal_navigation.png', dpi=180)
        plt.close(fig)
    if (LOGS / 'obstacle_steps.csv').exists():
        plot_box_routes()
    lines = ['# 单 UAV 固定导航第一阶段实际验证', '', '日期：2026-10-11。固定增益一次执行，未按评估结果调参或重跑。', '',
             '控制器仅用本机 EKF 位置/速度和静态任务参考点；原始物理单位，无 Actor/GAT/归一化依赖。位置增益0.6、速度反馈1.6、速度目标上限2.5 m/s；逐轴增量0.6/sqrt(3)、范数0.6 m/s。复用原 dt=0.2 s、速度12、加速度3、120步时域和真实1 m/0.5 m/s/连续5步停车。基础安全投影 ON、五辅助 OFF。', '',
             '|阶段|严格停车|距离中位数/最大 m|速度中位数/最大 m/s|碰撞/超时|边界/安全投影步|',
             '|---|---:|---:|---:|---:|---:|']
    for phase, a in aggregate.items():
        lines.append(f"|{phase}|{a['successes']}/{a['episodes']}|{a['final_distance_m_median']:.4f}/{a['final_distance_m_maximum']:.4f}|{a['final_speed_mps_median']:.4f}/{a['final_speed_mps_maximum']:.4f}|{a['collisions']}/{a['timeouts']}|{a['boundary_projection_steps']}/{a['safety_projection_steps']}|")
    lines += ['', '阶段门槛：`' + json.dumps(summary['gates']) + '`。运行计数：`' + json.dumps(summary['counts']) + '`。', '',
              '主动制动见证在首次符合停车保持条件前计算，使用动作前估计参考方向、反向请求增量及实际投影速度与速率同时下降；记录与环境执行向量一致性逐步校验。上述成功率仅适用于冻结的独立小规模场景，不是泛化结论。', '',
              '现有 EKF reset 由真实初态初始化；四个独立场景仅扰动初始 belief（位置0.4 m、速度0.2 m/s），未验证完全未知初态或持续传感器故障。三维质点运动学及现有传感器模型保持不变。障碍阶段若执行，只验证两套几何的左右四条已知地图预定义路线，不包含通用在线规划；连续 clearance 使用精确相交检查及33样本1-Lipschitz距离下界。', '',
              '下一阶段可在已有3 UAV框架中逐机接入相同本机EKF固定执行层，再单独验证估计误差、通信中断与安全约束。GAT保留图状态表达，MAPPO仅学习测量/链路/带宽等联合决策；必须从log-prob、熵和策略损失中排除固定导航的确定性速度动作。本轮没有执行该集成或训练。A/B历史不覆盖，不提交推送。', '',
              '详见 execution_manifest.lock.json、各阶段 steps.csv、episodes.csv、aggregate_metrics.json 及 independent_audit.json。']
    with (LOGS / 'fixed_navigation_summary.md').open('x', encoding='utf-8') as handle:
        handle.write('\n'.join(lines) + '\n')
    print(json.dumps(aggregate, default=lambda value: value.item(), ensure_ascii=False))


if __name__ == '__main__':
    main()
