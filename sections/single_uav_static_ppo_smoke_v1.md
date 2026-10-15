# 单 UAV 静态绕障 PPO smoke v1

日期：2026-10-10。目的为验证真实在线 PPO 链路和记录完整性。此次单种子、小预算运行不构成泛化、安全性或初始化优劣结论，不自动启动多种子训练，不提交或推送。

## 冻结输入与初始化

环境原样采用上一阶段的 `StrictSingleUAVPPOEnv`，基础配置为 `data/configs/single_uav_ppo_baseline_v1.json`（SHA256 `bd34f163de64fc814cd21f705029343c4c4d45efa0ff7ace8a3ef6acbf07c609`）。控制周期 0.2 s，最大速度 12 m/s，最大加速度 3 m/s²，单步速度增量范数上限 0.6 m/s；边界 `[0,0,2]` 至 `[200,200,60]`，基础 legacy 安全投影开启，安全距离 2 m，时域 120 步/24 s。PASS_COMMIT、lateral-exit、radial-intent、Goal Recovery、Terminal Capture 全部关闭。

A 从已有 PPO 的 42 维迁移检查点初始化（迁移本身未训练）；B 从 `dagger_bc/bc_final/scale_1/checkpoints/best_validation.pt` 的 BC/DAgger Actor 初始化。精确路径、SHA 和网络兼容记录写入 smoke 配置及各组 `compatibility.json`。旧 `update_count=100` 是共享 PPO 祖先的计数，不计为本次更新，也不计为 B 的 BC/PPO 微调次数。

42 维 Actor 观测、79 维 Actor body 输入、174 维 Critic 输入结构一致。旧统计的障碍方向槽计数为零，且新场景分布不同，因此两组共享训练池校准的新统计。完整首层 GAT 与 Critic 保持 fresh constructor 参数；保留匹配的深层 GAT、Actor body、动作 heads 和动作标准差。两组解冻全部参数，使用全新 Adam，清空旧 rollout。基础模板的 `from_scratch=true` 仅描述 fresh trainer 构造步骤；本 smoke 的最终 Actor 初始化明确为上述 warm start，不代表从零训练。

## 场景与成本

训练池 18 场：6 无障碍、6 单障碍（3 左右镜像对）、6 完整绕障归航（3 对）。独立验证池 10 场：2 无障碍、4 单障碍、4 完整任务。几何哈希、场景 ID、场景 seed、传感噪声 seed 在两个 split 之间无交集。左右定义为目标 +X 坐标下 +Y/-Y；任务直线确实穿过实体 box。无障碍场景显式设置 `obstacles=[]`，避免空 specs 触发历史默认障碍。历史 20 回合不参与训练、归一化、模型选择或本次验证。

每组在同一 18 个训练场景各执行 16 个相同有界随机动作，共 288 个校准交互步；两组校准流逐项相同，只从一份流拟合共享观测、中央观测与 reward potential 统计。速度尺度下限 6 m/s，potential 的 MAD 尺度下限 0.001。允许校准专用单位尺度占位值使环境计算奖励；校准动作与奖励无关，占位值在 PPO 前替换，校准样本从不进入 rollout。统计在训练和验证期间冻结。

每组额外执行 2160 个真实 PPO 在线决策步，共 18 次 update，每次 120 个新样本、4 epochs、minibatch 64，即 8 次实际 optimizer step/update、总计 144 次。学习率 0.0001，gamma 0.99，GAE lambda 0.95，clip 0.2，梯度裁剪 0.5，entropy 0.005；其余 PPO 设置沿用统一 baseline。校准加在线训练为 2448 交互步/组。验证、reset-only 观测探针、单元测试成本分别记录，不算入 PPO 预算。

| 阶段 | 每组 6 个 PPO 块 | 左右平衡及回放 |
|---|---|---|
| 1 | 6 无障碍归航停车 | normal 00–05 |
| 2 | 2 无障碍 + 4 单障碍 | 保留 normal 00–01，single 00–03 左右各 2 |
| 3 | 2 无障碍 + 2 单障碍 + 2 完整绕障归航 | 保留 normal 04–05 和 single 00–01，完整任务左右各 1 |

实际在线 PPO 接触 12 个独立训练场景；其余 2 个 single、4 个 full 仅用于训练池校准。可用池规模、实际曝光场景及校准专用场景分别写入 curriculum provenance。

每个块严格消耗 120 个决策步，场景和噪声 seed 对两组一致。早终止后重置同一指定场景继续填满预算。块尾未物理结束的 episode 记录为 `rollout_window_cut`，最后 transition 保持 nonterminal、使用下一状态 value bootstrap；update 后清空 buffer，不跨块传递 GAE。物理 time limit 沿用原配置，不混同为窗口切分。

## 评估与指标

初始化时评估全体 10 个验证场景；阶段 1 评估 2 个 normal；阶段 2 评估累计 6 个 normal/single；阶段 3 评估完整 10 场（同时为最终评估），均采用 deterministic action 与相同 120 步时域。评估不写入 PPO buffer，实际评估步数、reset 次数单独计费。

障碍场景严格成功：全程无碰撞、已通过任务相关实体障碍、实际位置距目标不超过 1 m、实际速度不超过 0.5 m/s，连续 5 步。沿用已冻结的通过门槛：越过下游面加 0.75 m、当前运动段 clearance 至少 2.5 m。无障碍阶段只读 reporter 记录同样的实际停车条件；实体通过与 strict success 在无障碍场景记为不适用。reporter 不改变环境奖励、bonus、终止或控制，无障碍环境因此不会因停车 reporter 提前终止。

`stable_arrival/task_success` 对 normal 使用实际停车，对障碍场景使用完整 strict success。分别报告原始物理时限 `time_limit_reached_raw`、到时且尚未成功的 `timeout_failure` 和非终止窗口切分。绕障率的分母仅为实体障碍 episode；normal 不计入。原始动作危险比例在范数/速度限幅后、安全与边界投影前检查候选运动段：接触或实体 clearance 小于 2 m 为危险；其分母仅为实体障碍决策步。边界危险另列，基础安全投影介入比例按全部决策步统计。

明确 `ppo_action_logprob_mode="sampled"`；importance ratio 使用采样动作和其原始 log-prob，执行动作、安全修正、总修正及修正原因分别记入 CSV。每条在线样本来自当前策略真实环境交互，拒绝教师动作、BC target、expert/DAgger 样本；已有教师数据仅为 B 初始化来源。

## 输出与完成门槛

smoke 规格：`data/configs/single_uav_static_ppo_smoke_v1_seed2026.json`。代码入口：`python -m src.experiments.single_uav_ppo_smoke --config data/configs/single_uav_static_ppo_smoke_v1_seed2026.json`。输出独立存放于 `data/synthetic/`、`logs/preliminary/`、`plots/preliminary/` 下的 `single_uav_static_ppo_smoke_v1_seed2026`，已存在目录直接拒绝执行，不覆盖。保留初始、三个阶段和最终 checkpoint；阶段验证、step/episode、sample/execution、安全修正、update/梯度/参数变化及预训练成本独立记录。

实际完整 trainer 参数、保留的初始策略标准差和预算另存 `effective_smoke_training_config.json`；模板的 train/warmup/validation/test episode 数不驱动 smoke。完成结果以独立 `smoke_summary.md`、`smoke_report.json` 和 `independent_artifact_audit_v2.json` 为准，原始记录保留。

链路通过要求：预算与实际更新一致、损失有限、Actor/GAT/Critic 均有梯度和实际参数变化、统计冻结、数据 split 隔离、教师在线样本为零、动作 log-prob 配对正确、日志可重算、冻结输入 SHA 未变。成功率为零不会伪报链路失败，但必须报告策略能力不足。进入多种子研究前需登记独立 seed、扩大后的等量在线预算、训练/校准/验证及 B 预训练的成本账本，并按任务类型核查失败与原始动作风险；单次 smoke 不提供性能或泛化依据。

本文件未引入论文数学符号。
