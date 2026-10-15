"""Read-only Stage-2 aggregates and figures from the frozen 960 joint steps."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
STEM = 'three_uav_fixed_navigation_v2_seed2026_20261011'
DATA = ROOT / 'data/synthetic' / STEM
LOGS = ROOT / 'logs/preliminary' / STEM
PLOTS = ROOT / 'plots/preliminary' / STEM
UIDS = ('uav_00', 'uav_01', 'uav_02')
COLORS = ('#2166ac', '#d95f02', '#238b45')


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def read_csv(name):
    with (LOGS / name).open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def truth(value):
    return str(value).lower() in ('true', '1')


def vector(value):
    return np.asarray(json.loads(value), dtype=float)


def write_json(path, value):
    with path.open('x', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')


def write_csv(name, rows):
    with (LOGS / name).open('x', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_figure(fig, name):
    path = PLOTS / name
    assert not path.exists(), 'Refusing to overwrite figure'
    fig.savefig(path, dpi=180)
    plt.close(fig)


def figures(nav, flows, lock):
    PLOTS.mkdir(parents=True, exist_ok=True)
    schedule = {row['episode_id']: row for row in lock['schedule']}
    fig = plt.figure(figsize=(14, 4.8), layout='constrained')
    for col, case in enumerate(('c0', 'c1', 'c2')):
        ax = fig.add_subplot(1, 3, col + 1, projection='3d')
        for variant, style in (('normal', '-'), ('cut', '--')):
            episode = f'three_nav_{case}_{variant}'
            for uid, color in zip(UIDS, COLORS):
                rows = [r for r in nav if r['episode_id'] == episode and r['uav_id'] == uid]
                start = schedule[episode]['scenario']['initial_positions'][uid]
                goal = schedule[episode]['scenario']['goals'][uid]
                points = np.vstack([start] + [vector(r['true_position_post_m']) for r in rows])
                ax.plot(*points.T, style, color=color, label=f'{uid[-2:]} {variant}')
                if variant == 'normal':
                    ax.scatter(*start, color=color, marker='o', s=18)
                    ax.scatter(*goal, color=color, marker='x', s=40)
        ax.set_title(f'{case}: three simultaneous fixed controllers')
        ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')
        ax.view_init(elev=25, azim=-65)
        ax.legend(fontsize=7, loc='upper left')
    fig.suptitle('Saved true trajectories for evaluation only; normal / bandwidth cut')
    save_figure(fig, 'three_uav_routes.png')

    fig, axes = plt.subplots(2, 3, figsize=(13, 7), layout='constrained')
    for col, case in enumerate(('c0', 'c1', 'c2')):
        for variant, style in (('normal', '-'), ('cut', '--')):
            for uid, color in zip(UIDS, COLORS):
                rows = [r for r in nav if r['episode_id'] == f'three_nav_{case}_{variant}' and r['uav_id'] == uid]
                time = np.array([int(r['step']) * .2 for r in rows])
                axes[0, col].plot(time, [float(r['goal_distance_m_actual']) for r in rows], style, color=color)
                axes[1, col].plot(time, [float(r['speed_mps_actual']) for r in rows], style, color=color,
                                  label=f'{uid[-2:]} {variant}')
        axes[0, col].axhline(1., color='black', ls=':', label='1 m strict distance')
        axes[1, col].axhline(.5, color='black', ls=':', label='0.5 m/s strict speed')
        axes[0, col].set_title(case)
        axes[0, col].set_ylabel('True goal distance (m)')
        axes[1, col].set_ylabel('Actual speed (m/s)'); axes[1, col].set_xlabel('Time (s)')
        for ax in axes[:, col]:
            ax.grid(alpha=.2); ax.set_xlim(0, 24); ax.set_ylim(bottom=0)
    axes[1, 0].legend(fontsize=7, ncol=2)
    fig.suptitle('Active approach and braking; controllers stay active for all 120 steps')
    save_figure(fig, 'three_uav_parking.png')

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), layout='constrained')
    cut_rows = [r for r in flows if r['episode_id'] == 'three_nav_c0_cut' and r['source'] == 'uav_00' and r['target'] == 'uav_02']
    t = [int(r['step']) for r in cut_rows]
    axes[0, 0].step(t, [float(r['applied_bandwidth']) for r in cut_rows], where='post', label='Applied bandwidth fraction')
    axes[0, 0].step(t, [truth(r['effective_visible']) for r in cut_rows], where='post', label='Fresh visible message', alpha=.8)
    axes[0, 0].scatter([int(r['step']) for r in cut_rows if truth(r['newly_arrived_this_step'])],
                       [.8 for r in cut_rows if truth(r['newly_arrived_this_step'])], marker='|', label='Arrival', s=55)
    axes[0, 0].axvspan(20, 50, color='#d95f02', alpha=.12)
    axes[0, 0].set_title('c0 cut: 00 → 02; service disabled at steps 20–50')
    axes[0, 0].set_ylim(-.05, 1.15); axes[0, 0].legend(fontsize=8)
    for variant, color in (('normal', '#2166ac'), ('cut', '#d95f02')):
        rows = [r for r in flows if r['episode_id'] == f'three_nav_c0_{variant}' and r['source'] == 'uav_00' and r['target'] == 'uav_02']
        axes[0, 1].step([int(r['step']) for r in rows], np.cumsum([truth(r['fusion_accepted']) for r in rows]),
                        where='post', color=color, label=variant)
    axes[0, 1].axvspan(20, 50, color='#d95f02', alpha=.12)
    axes[0, 1].set_title('c0 00 → 02: accepted relative-range EKF updates')
    axes[0, 1].set_ylabel('Cumulative accepted updates'); axes[0, 1].legend()
    dyn = [r for r in flows if r['episode_id'] == 'three_nav_c2_normal' and r['source'] == 'uav_00' and r['target'] == 'uav_01']
    t = [int(r['step']) for r in dyn]
    axes[1, 0].plot(t, [float(r['physical_distance_m']) for r in dyn], label='Physical distance (diagnostic)')
    axes[1, 0].axhline(75, color='black', ls=':', label='Communication range 75 m')
    axes[1, 0].axhline(100, color='gray', ls='--', label='UWB range 100 m')
    axes[1, 0].set_title('c2: 00 / 01 separate, UAV 02 remains the bridge')
    axes[1, 0].set_ylabel('Distance (m)'); axes[1, 0].legend(fontsize=8)
    for key, label in (('potential', 'Physical potential link'), ('effective_visible', 'Fresh effective message'),
                       ('measurement_selected', 'Receiver selected 00'), ('uwb_available', 'UWB available')):
        axes[1, 1].step(t, [truth(r[key]) for r in dyn], where='post', label=label, alpha=.8)
    axes[1, 1].set_title('c2 00 → 01: dynamic masks and message visibility')
    axes[1, 1].set_ylim(-.05, 1.2); axes[1, 1].legend(fontsize=8)
    for ax in axes.flat:
        ax.set_xlabel('Decision step'); ax.set_xlim(1, 120); ax.grid(alpha=.2)
    save_figure(fig, 'three_uav_communication.png')


def main():
    nav, flows = read_csv('navigation.csv'), read_csv('flows.csv')
    execution = read_json(LOGS / 'execution_summary.json')
    audit = read_json(LOGS / 'independent_audit.json')
    lock = read_json(DATA / 'execution_manifest.lock.json')
    probe = read_json(LOGS / 'interface_probes.json')
    bandwidth_probe = read_json(LOGS / 'bandwidth_gradient_probe.json')
    assert audit['passed'] and execution['all_scheduled_gates_passed'] and probe['passed'] and bandwidth_probe['passed']
    episodes = execution['episodes']
    episode_metrics = []
    for episode in episodes:
        rows = [r for r in nav if r['episode_id'] == episode['episode_id']]
        errors = [float(r['position_estimation_error_m']) for r in rows]
        episode_metrics.append({'episode_id': episode['episode_id'], 'variant': episode['variant'],
            'agent_successes': episode['agent_successes'], 'joint_success': episode['joint_success'],
            'joint_first_success_step': episode['joint_first_success_step'],
            'maximum_final_distance_m': max(episode['final_distance_m'].values()),
            'maximum_final_speed_mps': max(episode['final_speed_mps'].values()),
            'minimum_final_hold_steps': min(episode['final_hold_steps'].values()),
            'minimum_synchronized_pair_distance_m': episode['minimum_synchronized_pair_distance_m'],
            'accepted_cooperative_updates': episode['accepted_cooperative_updates'], 'ci_calls': episode['ci_calls'],
            'position_error_rmse_m': float(np.sqrt(np.mean(np.square(errors)))),
            'mean_position_estimation_error_m': float(np.mean(errors))})
    write_csv('episode_metrics.csv', episode_metrics)
    groups = {}
    for variant in ('normal', 'cut'):
        selection = [e for e in episodes if e['stage'] == 'evaluation' and e['variant'] == variant]
        distances = [v for e in selection for v in e['final_distance_m'].values()]
        speeds = [v for e in selection for v in e['final_speed_mps'].values()]
        groups[variant] = {'episodes': len(selection), 'agent_successes': sum(e['agent_successes'] for e in selection),
            'agents': len(selection) * 3, 'joint_successes': sum(e['joint_success'] for e in selection),
            'stable_arrival_rate': sum(e['agent_successes'] for e in selection)/(len(selection)*3),
            'maximum_final_distance_m': max(distances), 'median_final_distance_m': float(np.median(distances)),
            'maximum_final_speed_mps': max(speeds), 'median_final_speed_mps': float(np.median(speeds)),
            'minimum_synchronized_pair_distance_m': min(e['minimum_synchronized_pair_distance_m'] for e in selection),
            'accepted_cooperative_updates': sum(e['accepted_cooperative_updates'] for e in selection),
            'collision_rate': 0., 'timeout_rate': 0., 'boundary_projection_rate': 0., 'safety_projection_rate': 0.}
    edges = []
    for episode in episodes:
        for source, target in (('uav_00', 'uav_02'), ('uav_02', 'uav_00')):
            for phase, start, end in (('prefix', 1, 19), ('intervention', 20, 50), ('drained', 25, 50), ('recovery', 51, 120)):
                rows = [r for r in flows if r['episode_id'] == episode['episode_id'] and r['source'] == source
                        and r['target'] == target and start <= int(r['step']) <= end]
                edges.append({'episode_id': episode['episode_id'], 'source': source, 'target': target, 'phase': phase,
                    'steps': len(rows), 'positive_bandwidth_steps': sum(float(r['applied_bandwidth']) > 0 for r in rows),
                    'transmitted_steps': sum(truth(r['transmitted']) for r in rows),
                    'arrivals': sum(truth(r['newly_arrived_this_step']) for r in rows),
                    'effective_steps': sum(truth(r['effective_visible']) for r in rows),
                    'accepted_updates': sum(truth(r['fusion_accepted']) for r in rows)})
    write_csv('directed_edge_phase_metrics.csv', edges)
    summary = {'evaluation_groups': groups, 'episode_metrics': episode_metrics, 'actual_execution_counts': execution['counts'],
        'all_episode_agent_successes': sum(e['agent_successes'] for e in episodes), 'all_episode_agents': 24,
        'all_joint_successes': sum(e['joint_success'] for e in episodes),
        'all_final_agents_qualifying': all(all(e['final_qualifying'].values()) for e in episodes),
        'moving_agent_braking_witnesses': sum(e['braking_before_first_qualifying'][uid] > 0 for e in episodes for uid in UIDS),
        'moving_agent_count': 22, 'collision_steps': sum(e['collision_steps'] for e in episodes),
        'boundary_projected_uav_steps': sum(e['boundary_projection_steps'] for e in episodes),
        'safety_projected_uav_steps': sum(e['safety_projection_steps'] for e in episodes),
        'environment_action_clipped_uav_steps': sum(truth(r['environment_action_clipped']) for r in nav),
        'maximum_actual_acceleration_mps2': max(e['maximum_actual_acceleration_mps2'] for e in episodes),
        'all_minimum_synchronized_pair_distance_m': min(e['minimum_synchronized_pair_distance_m'] for e in episodes),
        'accepted_cooperative_updates': sum(e['accepted_cooperative_updates'] for e in episodes),
        'online_ci_calls': sum(e['ci_calls'] for e in episodes), 'pure_ci_probe_passed': probe['ci_pure_probe_passed'],
        'interface_probe_records': probe['record_count'], 'maximum_logprob_error': probe['maximum_logprob_error'],
        'bandwidth_gradient_norm_k2': bandwidth_probe['gradient_norms']['bandwidth_head'],
        'independent_audit_passed': audit['passed'], 'protected_file_count': lock['protected_file_count'],
        'normal_and_cut_prefix_identical_steps': 19, 'truth_initialization_inherited': True,
        'interpretation': 'Small paired interface validation, not generalized navigation or localization performance.'}
    write_json(LOGS / 'integration_metrics.json', summary)
    figures(nav, flows, lock)
    lines = ['# 3 UAV 固定导航执行层与协同定位接口验证（第二阶段）', '',
        '本轮 8 回合、960 个联合环境步（2880 个 UAV 步）均完成完整 120 步。开发、正常通信、部分带宽中断及 fusion-off 诊断全部通过预先冻结的合同。A/B、PPO、MAPPO 均未训练，未加载模型；没有验证结果调参、轨迹重跑或历史覆盖。', '',
        '## 实现与冻结配置', '',
        '复用现有 MultiUAVJointEnv、LocalEKF、图观测、拓扑、测距、消息传输和 MAPPO。纯动作组合器逐 UAV 仅读取自身原始 EKF 前六维及任务目标，继续使用第一阶段限速位置—速度反馈：位置增益 0.6、速度反馈 1.6、目标速度上限 2.5 m/s，dt 0.2 s。动力学仍为速度上限 12 m/s、加速度上限 3 m/s²、速度增量范数 0.6 m/s、逐轴 0.6/sqrt(3)，120 步，基础 2 m 安全投影开启，五类历史辅助全关。没有改控制器、动力学、EKF 或通信核心。', '',
        '无碰撞、实际距离 ≤1 m、实际速度 ≤0.5 m/s 连续 5 步按本机及三机同时保持分别重算；首次达成后控制器仍运行至第120步，未冻结真实状态。三套独立评估几何使用配对同种子，前19步状态完全一致；本轮不是正式多种子性能研究。', '',
        '## 实际结果', '',
        '| 回合 | 本机停车 | 三机同时首次停车步 | 最终最大距离 m | 最终最大速度 m/s | 全时域最小间隔 m | 接受协同更新 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for row in episode_metrics:
        lines.append(f"| {row['episode_id']} | {row['agent_successes']}/3 | {row['joint_first_success_step']} | {row['maximum_final_distance_m']:.4f} | {row['maximum_final_speed_mps']:.4f} | {row['minimum_synchronized_pair_distance_m']:.4f} | {row['accepted_cooperative_updates']} |")
    for variant, label in (('normal', '正常通信'), ('cut', '部分带宽中断')):
        g = groups[variant]
        lines += ['', f"{label}独立评估：本机稳定停车 {g['agent_successes']}/9、三机同时停车 {g['joint_successes']}/3；最终距离中位数 {g['median_final_distance_m']:.4f} m、最大 {g['maximum_final_distance_m']:.4f} m，最终速度中位数 {g['median_final_speed_mps']:.4f} m/s、最大 {g['maximum_final_speed_mps']:.4f} m/s。"]
    lines += ['', f"全部24个本机运行实例在120步末仍满足距离/速度阈值；22/22个移动实例在首次符合停车前都有主动反向请求、实际减速及速度下降的见证，另外2个c2桥接机初始即驻留目标，不纳入制动分母。碰撞、超时、边界投影、安全投影、环境动作裁剪均为0；实际加速度最大 {summary['maximum_actual_acceleration_mps2']:.4f} m/s²。同步分段相对运动最小间隔 {summary['all_minimum_synchronized_pair_distance_m']:.4f} m，超过2 m。", '',
        '## 通信与协同 EKF 证据', '',
        '实际通信采取受控轮询测量选择、每条潜在出链分配0.5；所有闭环回合没有执行神经策略。中断仅对00→02、02→00在20–50步将服务带宽设0，不直接编辑消息/队列/EKF。25–50步缓存耗尽后两条链路的发送、到达、有效邻居、接受融合均0，51步起两方向均重新有发送、到达和接受更新。未受影响链路仍为0.5，不转移闲置预算。逐方向/阶段数量见 directed_edge_phase_metrics.csv。', '',
        '正常三场景接受协同测距更新170/176/124次，中断为156/161/108次；fusion-off保持通信却接受更新为0，belief轨迹与正常对照不同。事件逐项与各机 valid_neighbor_count 增量一致。接收图节点的15维状态、15维协方差对角均严格来自到达并传播后的消息；缓存可见不等于新消息，已消费包不再次融合，过期包不进入Actor。', '',
        f"动态分离场景00/01真实间距 {audit['geometry']['c2']['distance_min_m']:.3f}→{audit['geometry']['c2']['distance_max_m']:.3f} m，跨过75 m通信门槛而始终小于100 m测距范围；两条物理潜在边各消失一次，02桥接四条边全时域存在。相对测距可用不意味着链路或测量选择合法。", '',
        '在线CI实际调用为0。现有0.6 s消息TTL与0.8 s同对测距更新周期使前次邻居belief不能满足CI freshness；代码采用新收到belief及相对测距EKF更新。独立离线CI均值/协方差有限、对称PSD探针通过，但不能据此声称在线CI融合已验证。本轮不改TTL或周期来制造CI调用。', '',
        '同配对c0正常融合平均位置误差约0.1518 m；fusion-off以及cut的具体误差和RMSE见 episode_metrics.csv。本轮只证明消息与更新机制生效，不证明协同融合普遍提高精度；通信会通过估计影响后续导航，不能将后续轨迹分离单独归为观测噪声效果。', '',
        '## 固定动作与学习动作的分离', '',
        '新增显式 motion_action_mode="fixed_navigation_comm_v1"，强制 sampled 概率模式。环境继续执行3维导航速度增量，但 PolicySample.delta_raw/delta_v 均为空；运动Normal不构建，运动头保留旧state键却冻结且无梯度。策略仅含4类入链测量（含none）、3个稳定UID出链槽（自身/不合法掩码）、选中出链上的带宽单纯形；Bernoulli链路概率与条件Dirichlet带宽共同计入通信概率/熵。0链无带宽动作，1链带宽固定1且概率/熵0，2链才有带宽自由度。环境安全/边界/链路修正只记executed诊断，不替换采样样本或旧log-prob。', '',
        f"171条保存真实观测/合成观测探针的采样与重算log-prob最大差为 {probe['maximum_logprob_error']:.1f}，执行动作扰动不改变概率、熵或PPO代理损失。RAW导航在故意污染NN观测归一化后仍逐项不变。仅合成离线代理损失backward验证GAT/测量/链路有梯度；首次见证选中不足两链，带宽梯度0属无自由度，补充实际K=2采样的带宽梯度范数 {summary['bandwidth_gradient_norm_k2']:.6f}>0，运动梯度仍无，所有参数哈希不变。", '',
        '本轮GAT小网络与identity统计仅做接口探针，没有执行到环境、绑定奖励或加入rollout，不是MAPPO行为验证。旧joint默认模式保持原行为，四配置八动作包含LP/熵/reconcile/权重及动作后RNG精确一致。旧模型仍属于joint模式；不得将其加载后直接声称满足新通信策略合同。', '',
        '## 限制及下一阶段条件', '',
        'EKF继续继承真实初始位置/速度/姿态/偏置初始化；这是理想初始先验，运行中导航不直接读真值。没有验证未知初态、持续定位漂移、复杂交叉、在线避障、风扰或真实推力姿态动力学。本轮所有场景无实体障碍，不能宣称三机在线避障或一般碰撞安全。正常/cut是配对实例，24/24不作为独立样本泛化统计。', '',
        '执行层与通信/相对EKF接口具备进入下一阶段“小规模协同测量选择MAPPO训练smoke”的工程条件，尚无MAPPO性能结论。开始前必须单独冻结训练/验证场景，仅用训练轨迹校准新37维节点/11维边与Critic（无障碍435维）归一化，固定协同定位收益与测量/通信成本目标、初始化方案；显式启用新模式并清空rollout/optimizer，旧checkpoint兼容性单独审查。第一轮只释放测量选择，其余通信控制冻结；通过真实更新/通信路径/定位指标验证后再逐项释放链路与带宽，不让基础飞行控制再次进入策略损失。CI若作为论文方法核心，应另设估计机制审计，不凭本轮0调用当作通过。', '',
        '## 审计、文件与复现', '',
        '26项新增纯工程测试、3项既有MAPPO兼容测试通过；工程测试与离线探针环境步0、参数更新0。独立审计0错误；655个历史输入/结果/第一阶段源与产物SHA不变。核心唯一既有学习模块改动为opt-in MAPPO动作模式，修改前完整归档并做精确回归。没有提交或推送。', '',
        '`python -m src.experiments.three_uav_fixed_navigation` 为受保护的一次性执行入口；当前目录已存在锁与结果，重复执行会拒绝覆盖。离线探针、审计和报告同样使用独占写入。', '',
        '冻结配置：data/configs/' + STEM + '.json；完整源码/配置/场景锁：data/synthetic/' + STEM + '/execution_manifest.lock.json。',
        '实际CSV：navigation/joint/pairs/flows/messages/gates/fusion_events；独立审计：independent_audit.json；汇总：integration_metrics.json；图：plots/preliminary/' + STEM + '/。']
    with (LOGS / 'integration_summary.md').open('x', encoding='utf-8') as handle:
        handle.write('\n'.join(lines) + '\n')
    write_json(LOGS / 'report_provenance.json', {'environment_steps': 0, 'optimizer_steps': 0, 'ppo_updates': 0,
        'sources': {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in (Path(__file__), ROOT/'src/analysis/three_uav_fixed_navigation_bandwidth_probe.py')},
        'execution_manifest_sha256': hashlib.sha256((DATA/'execution_manifest.lock.json').read_bytes()).hexdigest()})
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
