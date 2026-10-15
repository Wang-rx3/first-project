# 单 UAV PPO 新训练前工程协议 v1（2026-10-10）

本协议对应独立配置 `data/configs/single_uav_ppo_baseline_v1.json`。本阶段只完成离线记录修复、配置构造和工程预检，正式训练尚未启动。后续检查结果登记在 `logs/diagnostics/single_uav_ppo_pretraining_20261010_v1/`。

## 历史 PASS_COMMIT 汇总修复边界

原始修复版轨迹来自 `logs/preliminary/single_uav_pass_commit_diagnostic_seed2026/run_action_decomposition_fix_v1/`；预注册及冻结记录来自其同级 `preparation_action_decomposition_fix_v1/`。原 `run/` 的 ON 中止尝试、旧准备目录和 ZIP 归档保留。恢复入口只读取已有 CSV/JSON，不构造环境，不调用环境步骤或学习更新。

最终汇总原先两次引用未绑定的 `ALL_EPISODE_IDS`。修复改为复用独立纯汇总函数，并以预注册 `population.ordered_episode_ids` 显式提供总体。离线恢复检查 OFF/ON 有序身份、种子、步数、关键字段、已有聚合和源文件哈希；失败时不生成正式汇总。恢复输出使用独立目录并拒绝覆盖。

新的恢复记录区分原 rollout 与本次离线操作，登记输入清单、调用参数和代码哈希。旧 runtime source lock 保留其历史含义；有意修复后的当前代码与旧冻结代码的差异另行记录，不改写旧锁。历史异常退出的具体原因仍未取得终端堆栈，代码缺陷的修复不等于补证该退出原因。

已从原 CSV 完成离线恢复，输出为 `logs/preliminary/single_uav_pass_commit_diagnostic_seed2026/offline_recovery_action_decomposition_fix_v1_20261010/`，含 `evaluation_summary.json`、`recovery_provenance.json`、`recovery_checks.csv` 和 `paired_summary.csv`。两臂均为 20 回合、2,185 行逐步记录、11 次通过、10 次稳定到达、0 次碰撞；PASS_COMMIT 生效步数为 OFF 0、ON 73。OFF fingerprint、预注册哈希和三点冻结组件身份检查通过，恢复前后输入哈希一致。这些是历史契约下的结果，保留原分类。

## 统一物理与控制配置

| 项目 | 固定值 |
| --- | --- |
| UAV 数量 | 1 |
| 学习初始化 | 从头初始化；不加载历史 BC/DAgger checkpoint |
| 决策周期 | 0.2 s |
| 最大实际速度 | 12 m/s |
| 最大加速度 | 3 m/s² |
| 每步速度增量范数上限 | 0.6 m/s |
| 位置边界 | 最小 `(0, 0, 2)` m；最大 `(200, 200, 60)` m |
| 基础安全距离 | 2 m |
| 安全执行模式 | `legacy` 基础实体障碍安全投影和边界处理 |
| 评估时域 | 120 个决策步，即 24 s |
| PPO 动作概率模式 | `ppo_action_logprob_mode="sampled"` |

PASS_COMMIT、lateral-exit assistance、radial-intent assistance、Goal Recovery、Terminal Capture 全部关闭。实际环境与训练器由独立构造入口创建；运行时组件检查确认这些辅助控制没有注册。目标附近的额外 braking safety projection 也关闭。基础物理安全投影使用原实现。

## 成功、奖励和终止

成功必须同时满足全程无碰撞、已按真实状态通过实体障碍、真实任务目标距离不超过 1 m、实际状态速度不超过 0.5 m/s，并连续保持 5 个决策步。5 步对应 1 s。

成功门在动作后的实际状态上更新，先于成功奖励和终止判定。碰撞历史永久锁存；未完成实体越障时不累计保持；距离或实际速度超限立即清零。重置回合同时清空越障、碰撞和保持状态。EKF 目标距离与估计速度可用于 Actor 观测、导航塑形及诊断，不用于决定严格成功。

实体越障使用独立几何判定，不依赖已关闭辅助控制的状态：选取初始真实位置到任务目标连线上的活动实体障碍；真实位置需越过每个障碍沿任务方向的下游面再加 0.75 m，且当前通过段的净空达到 2.5 m，随后锁存通过。轨迹曾经低于 2.5 m 的净空只作诊断，不永久阻止后续通过。通信遮挡体不能充当实体障碍；无实体障碍场景不能记为此协议的严格绕障成功。

## Actor 观测与归一化

启用目标相关观测、估计速度、braking-distance 附加量和相对实体障碍方向，节点特征维数为 42。下表索引均为 Python 零基切片。

| 信息 | Actor 节点切片 |
| --- | --- |
| EKF 估计速度三分量 | `3:6` |
| 目标相对 EKF 位置的三分量 | `30:33` |
| 最近实体障碍在估计机体系内的相对方向 | `39:42` |

预检从上游 EKF 状态、目标和实体障碍位置分别施加受控扰动，重新生成观测，检查上述切片在原始及归一化输入中变化，并检查 Actor 实际接收到的维数和数值。预检 normalizer 只用于临时 reset fixture；其统计不保存为正式训练统计。

正式训练归一化统计及奖励 potential 校准当前为待生成状态。后续只用训练划分拟合，并保存来源、特征顺序、维数和 artifact 哈希。验证/测试入口必须加载一致的冻结统计；未拟合、来源或哈希不符时拒绝构造。评估入口不允许调用 fit/update。

正式统计 artifact 沿用观测 normalizer 的 JSON state，另含 `central_normalizer` state；也接受 `observation_normalizer` 包装。provenance 必须含 `fit_split="train"`、`artifact_path`、`source_sha256` 和 `feature_index_map`。factory 比对文件内容与传入的观测/中心统计，并复制为冻结 view，外部原统计的后续修改不会影响验证输入。

## 动作与安全修正记录

PPO buffer 中的动作和 old log-probability 始终对应采样动作。安全执行后的动作另行保存，不替换 PPO 的采样动作。日志记录采样动作、实际执行动作、总修正向量与范数、安全投影修正及原因，同时标记 `logprob_action="sampled"`。动作限幅、速度限幅、安全投影和边界修正须能区分，不能把全部执行差异都归为障碍安全干预。

纯 reconciliation 检查构造采样动作与执行动作不同的情形，验证采样动作及 old log-probability 不变、未更新策略上的采样动作 PPO ratio 为 1，且没有梯度回传或 optimizer step。

## 工程验收

验收范围包括纯离线汇总、身份/哈希失败关闭、成功门边界、配置实例与组件图、采样动作概率一致性、Actor 特征传递和冻结归一化。预检只允许 reset 和纯函数/奖励夹具，不运行轨迹、正式评估或训练。检查命令、实际结果和历史保护清单以诊断目录的记录为准；最终进展同步追加到中文根目录 README。

本文件沿用现有参数及代码字段名，没有新增论文数学符号。

验收已完成：75 项 PASS_COMMIT 测试及 3 个子测试、27 项 PPO 测试通过。正式预检 `checks_passed=true`，实际 Actor encoder 输入 `[1,42]`、原始及归一化三组扰动均通过、采样动作 old log-probability 不变、PPO ratio=1，环境推进/backward/optimizer step 为 0。最终输出为 `logs/diagnostics/single_uav_ppo_pretraining_20261010_v1/preflight_final/`，配置、代码来源和物理/成功契约冻结于其 `config.lock`；采样执行动作单行证据位于 `action_contract_probe.csv`，标明为未训练策略的合成执行修正探针。
