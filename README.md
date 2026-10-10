# 多无人机通信—定位联合优化研究工程

本项目用于研究纯 Python 三维多无人机环境中的协同定位、动态通信拓扑、GAT 图表示和 MAPPO 联合优化。所有实验数据均由项目内的 NumPy/PyTorch 仿真环境生成；`knowledge/` 中的论文只作为理论和模型设计参考，不作为训练、验证、测试或参数校准数据源。

## 目录职责

| 路径 | 职责 |
| --- | --- |
| `knowledge/` | 参考文献、文献笔记和理论资料 |
| `src/` | 环境、传感器、信道、EKF、GAT、MAPPO 和实验代码 |
| `data/` | Python 仿真数据、校准扫描和模型检查点 |
| `plots/` | 实验图和诊断图 |
| `logs/` | 与实验图对应的 CSV/JSONL 日志 |
| `sections/` | 论文章节材料；`sections/2_system_model.md` 是唯一符号登记源 |
| `tests/` | 环境、EKF、schema、配置入口和绘图测试 |
| `main.py` | 轻量化入口，转发到 `src/cli.py` |

## 当前状态

### 研究进度审计索引（2026-10-10）

最新全局审计见 [硕士课题进度审计](knowledge/research_progress_audit_20261010.md)。下文包含历史阶段记录，不能把“准备完成/未执行”直接当作当前磁盘状态。当前多 UAV 主线已完成框架实现与单训练 seed 开发比较，尚未完成当前最终版本的多 seed、跨规模与独立泛化确认；最近工作集中在冻结单 UAV 策略的绕障、通过前衔接和终端稳定捕获。

PASS_COMMIT `run_action_decomposition_fix_v1/` 已保存 OFF/ON 各 20 回合、各 2,185 行逐步轨迹。ON 在六个主要失败回合中实际介入 73 步，但两臂均为 11/20 通过障碍、10/20 稳定到达。最终 `evaluation_summary.json` 缺失；当前 evaluator 的 PASS_COMMIT 汇总引用了未定义的 `ALL_EPISODE_IDS`，因此只能记为“配对轨迹已有，最终汇总/验收未闭合”，不能记为完整实验验收通过。该缺陷与缺失产物一致，但没有该次终端堆栈，历史退出原因仍须保留这一证据边界。

本次复核为 `1147 passed, 5 skipped, 3 failed`：两项绘图接口兼容失败，一项测试仍假定修复版结果目录不存在。符号审计通过，含 300 个规范符号、10 个历史符号。本次未修改研究代码、未训练或重新运行研究评估，未提交或推送；详细证据与审计范围见上述报告。

### 单 UAV 静态障碍物主动避障阶段 1 Smoke（2026-09-13）

已新增独立入口 `src/experiments/single_uav_static_obstacle_avoidance.py`，从
`single_uav_braking_progress025_seed2026` 的 baseline checkpoint 初始化，保持原有动力学、动作空间、PPO/MAPPO 参数、稳定到达判据和 safety execution 不变。训练 Smoke 使用 30 个 episode，实际调度为 21 个 normal full-distance 与 9 个 static pressure（70%/30%）；静态奖励是可选且默认关闭的独立接口，本次训练仅在静态场景中按原始候选动作的 swept clearance 计算，未使用 safety projection 后的位置。

静态压力 sanity 使用相同场景和动作轨迹进行了 safety-on/off 配对：safety-off 出现 1 个 pre-candidate unsafe step 和 1 个碰撞 step，safety-on 出现 1 个安全投影 step且碰撞为 0，压力门控通过。Smoke 的 5 个 normal validation 与 5 个 normal test 均保持 curriculum-free；normal test 为 5/5 进入 8 m、1/5 进入 1 m，静态 holdout 暂未产生碰撞或安全投影。`validate_preliminary` 为 `ok=true`，项目 `tests/` 为 `418 passed, 5 skipped`。这些是工程链路和压力场景验证结果，不足以证明策略已经学会泛化的主动绕障。

静态阶段 1 正式 `100 train / 20 validation / 20 test` 实验已在 Smoke 通过版本上完成；训练实际为 `70% normal + 30% static pressure`，validation/test 分别包含 20 个 normal full-distance 与 20 个 static-obstacle holdout。正式 normal test 为 `20/20` 进入 8 m、`0/20` 进入 1 m、`0/20` stable；static test 为 `20/20` pre-risk、`1316` 个 safety projection step、`0/20` collision，且未进入 8 m。这保护了远距离导航，但尚未证明策略学会主动绕过静态障碍物；static holdout 的 `collision=0` 主要来自安全执行层介入。

阶段 1 输出目录：

```text
data/synthetic/single_uav_static_obstacle_avoidance_stage1_seed2026/
logs/preliminary/single_uav_static_obstacle_avoidance_stage1_seed2026/
plots/preliminary/single_uav_static_obstacle_avoidance_stage1_seed2026/
```

### 实验产物清理（2026-09-09）

已清理 2026-09-02 00:00 至 2026-09-07 12:00 期间生成的 `data/`、`logs/`、`plots/` 实验文件及对应 Python 缓存，共 235,186 个文件、约 54.11 GB。`src/`、`tests/`、`sections/`、入口文件、配置、知识库和答辩文档未删除。历史实验输出不再作为当前运行的隐式依赖；若版本化 manifest 不存在，实验模板使用显式的 3-UAV 默认配置构造，保证清理后代码仍可执行。

随后按用户明确列出的目录名，又从 `data/`、`logs/`、`plots/` 中删除了 37 个版本化实验结果目录（包括 `mainline_core_framework_v5` 至 `v11` 的截图所列结果目录及 `advisor_report_seed42_v10`），共 684 个文件、约 143 MB。此次删除仅作用于这些实验结果目录及其内容；源代码、测试、论文章节、配置和目录骨架保留。README 中对应的历史说明仅作过程记录，已删除目录不再作为可复现实验产物。

### 当前论文主线命名（2026-09-04）

当前论文主方法统一写作 **Local-EKF-GAT-MAPPO (Proposed Method)**，其代码和结果目录中的兼容内部标识仍为 `baseline_mappo`。这里的
`baseline` 只表示“未叠加历史 proxy/通信约束实验块的核心框架参照”，不是论文中的弱基线；核心方法本身包含本地 EKF、动态图 GAT 和 MAPPO。

`communication_constrained_proxy_guided_mappo` 已从当前论文主线和新图表中停用，不再作为主方法、基线或新实验分支。它的历史源代码、checkpoint、日志和结果目录暂时保留并标记为 deprecated，仅用于追溯已经完成的历史实验；删除这些文件会破坏旧结果复现，因此不做物理删除。

当前主线图表使用 `paper_scope.active_mainline_methods` 和 `Local-EKF-GAT-MAPPO (Proposed Method)` 显示标签；CSV、manifest 和 checkpoint 仍保留内部 ID，便于精确复现。

当前主线的新实验协议为 `core-framework-v3`。它只训练和绘制 `Local-EKF-GAT-MAPPO (Proposed Method)`，并在 validation 后两张主图中与 `local_only_fixed`、`random_cooperative` 和 `budgeted_greedy_ig` 三个规则参照比较；不训练、不读取或展示 MLP。相较历史协议，v3 保留本地 EKF、GAT、通信、奖励和物理目标定义，采用逐候选动作头、四回合 rollout 稳定配置，并将 PPO 旧对数概率固定对应策略采样动作；安全投影后的执行动作单独记录。历史协议仍可用来复现实验档案，但不应作为论文主线入口。

`core-framework-v3` 的 seed=42 Smoke 已完成，输出在 `mainline_core_framework_v3_seed42/smoke/reference/`，
结构审计为 `audit_ok=true`，test 访问为 0，且只包含 Proposed 训练与三个规则 validation 结果。
Smoke 中 Proposed 的 validation RMSE 为约 `0.4197 m`，低于 Local、Random 和 Greedy 的约
`0.4382/0.4554/0.4401 m`；该结果仅用于接口和数值链路检查。`20/100/20/20` development
已完成，结构审计为 `audit_ok=true`，但仍只是 seed=42 描述性结果：Proposed 的 validation
RMSE/P95 为约 `0.41084/0.70071 m`；RMSE 低于 Local 和 Random（分别为
`0.43604 m`、`0.41431 m`），P95 低于 Local 但略高于 Random（分别为
`0.73382 m`、`0.69932 m`），而不如
`budgeted_greedy_ig` 的 `0.40070/0.67771 m`。Proposed 实际测量负载约 `53.66 kB`、
`J_com=0.07649`，高于三个规则参照的负载或通信目标，不能表述为全面优于基线。
训练 approximate KL 最大约 `0.01318`、95% 分位约 `0.00967`，未再出现历史数量级尖峰；
但训练报告回报仍受 15/100 个碰撞回合影响，固定监测场景最终 RMSE 低于初始值而平均报告
奖励略下降。因此本轮仅确认 PPO 采样概率语义和数值链路改善，不继续自动运行 actor 学习率
或反馈对齐变体，也不宣称训练曲线已收敛或跨 seed 稳定。

v3 输出目录：

```text
data/synthetic/mainline_core_framework_v3_seed42/development/reference/
logs/preliminary/mainline_core_framework_v3_seed42/development/reference/
plots/preliminary/mainline_core_framework_v3_seed42/development/reference/
```

五张主图只保留 Proposed 的前三张训练图，后两张保留 Proposed、Local、Random、Greedy
的 validation 对照；每张图均同时导出 PNG 和 PDF。全量测试为 `284 passed`，
`main.py audit-symbols` 通过（285 个规范符号，无重复）。

### `core-framework-v4` 训练反馈改善协议（seed=42 开发）

v4 直接基于 v3 的 Local-EKF-GAT-MAPPO 代码，报告奖励、EKF、GAT、通信物理模型和动作空间
保持不变，只新增训练期定位进度与实际测量负载反馈。三个反馈臂按 `loc-progress`、
`actual-load`、`loc-progress-actual-load` 顺序检验；反馈目标只从 v3 baseline 的 `train`
数据得到，validation 用于描述性比较，test 仅生成并封存。前三张图只显示 Proposed 训练曲线，
后两张与 `local_only_fixed`、`random_cooperative`、`budgeted_greedy_ig` 比较。

初次 v4 运行发现 `actual_measurement_cost_target` 曾按“每窗口总测量数”拟合，而环境的 `b_t`
定义是“每 UAV 每窗口测量率”，导致 C/LC 的负载反馈恒为零。该量纲问题已修正，并以同一实现
版本在独立目录 `mainline_core_framework_v4_corrected_seed42` 重跑 L/C/LC；旧目录保留为可追溯
历史记录，不纳入本轮有效结论。修正后的三个 `2/5/5/5` Smoke 和三个 `20/100/20/20`
development 均已完成，报告 `audit_ok=true`。修正目标为 `0.583141942028134`，来源为 v3
baseline 的 100 个 train 回合（定义为每 UAV 每窗口测量率）。

修正后的 seed=42 描述性 validation 结果显示，L/C/LC 三臂 Proposed 的 RMSE 分别约为
`0.40820/0.40453/0.42293 m`，实际测量负载分别约为 `61.57/31.74/13.13 kB`；对应的
Local/Random/Greedy 参照约为 `0.43604 m/0 bytes`、`0.41355 m/33.18 kB`、
`0.39686 m/33.25 kB`（L、C 数值分别按各自 validation 运行记录）。因此 C 臂降低了负载并
保持相对 Local/Random 的定位优势，但 LC 臂过度抑制测量，RMSE 反弹；不能把组合反馈称为
同时优化成功，也不能由单个 seed 宣称稳定性。Smoke/development 结果只支持 seed=42 的
机制诊断，最终论文结论仍需新 seed 的确认实验。

修正后的 v4 输出根目录：

```text
data/synthetic/mainline_core_framework_v4_corrected_seed42/
logs/preliminary/mainline_core_framework_v4_corrected_seed42/
plots/preliminary/mainline_core_framework_v4_corrected_seed42/
```

### V4 P0--P3 有序曲线归因诊断（2026-09-06）

新增只读入口 `run-ordered-v4-diagnostic`，只读取上述三个 development arm 的 train 和
validation，不打开 test，不执行训练更新，也不修改 checkpoint。诊断结果写入：

```text
logs/preliminary/mainline_core_framework_v4_corrected_seed42/ordered_diagnostic_v2/
plots/preliminary/mainline_core_framework_v4_corrected_seed42/ordered_diagnostic_v2/
```

P0 数据闭合通过：100 个 train episode、25 个 PPO update、每次 update 绑定 4 个 episode；
报告/学习奖励闭合误差不超过 `1.1e-9`，测量事件闭合和 512-byte 负载闭合均为 0 个不匹配，
test 访问次数为 0。P1 未发现 V4 的 PPO 数值发散（各 arm 最大 approximate KL 约
`0.0120--0.0129`，低于目标 `0.05`），但非碰撞回报仍有明显方差，且总 loss 主要跟随
critic loss。P2 确认定位进度差分错误地复用了绝对 logdet 的 warm-up scale（`u_scale≈0.0403`），
组合反馈同时明显压低测量数量；P3 确认协同 UWB 能即时改善状态均值误差，但最终协方差被恢复，
因此 `J_loc` 几乎不变，后续有效更新约 97% 为视觉里程计，beneficial 更新到 h5 的保持率约
`0.41--0.45`。因此 P4 五种子确认实验暂缓，下一步先单独修正差分反馈尺度，再做 seed=42
Smoke；若持久性仍不足，再单独检验执行可见的 VO 质量/协方差校准。

### `core-framework-v6` 原始主方法训练稳定化开发结果（2026-09-06）

v6 直接基于 `Local-EKF-GAT-MAPPO (Proposed Method)` 的核心框架，未加入历史
proxy/quality teacher/通信约束分支，也未训练 MLP。此次只调整训练链路中已确认的三个量：
不确定度使用 warm-up 的 Q05--Q95 归一化、碰撞惩罚配置从 10.0 调整为 2.0、梯度裁剪阈值从
0.5 调整为 1.0；报告奖励与学习奖励均采用 `report_direct`，EKF、GAT、信道和动作空间保持不变。
结果写入独立的 `mainline_reward_rebalanced_seed42` 目录，旧结果未覆盖。

seed=42 的 `2/5/5/5` Smoke 和 `20/100/20/20` development 均完成，结构审计为
`audit_ok=true`，共 25 次 PPO update，test 仅生成并封存。训练中 approximate KL 最大约
`0.01154`、逐 update 的 95% 分位约 `0.01100`，未见历史数量级的 KL 发散；critic loss
由前四分之一约 `2.34` 降至后四分之一约 `1.53`，但原始回报仍有明显随机波动，且 16/100
个训练回合发生碰撞或提前终止。固定五个监测场景的平均 RMSE 从约 `0.3884 m` 变为
`0.3902 m`，所以不能称为已收敛或已证明训练后定位变好。

development validation 的 Proposed 结果为 RMSE/P95 `0.41473/0.70275 m`、实际负载
`11.392 kB`、`J_com=0.02339`、有效更新约 `0.1718/window`；Local、Random、Greedy
的 RMSE 分别约为 `0.43604/0.41538/0.39824 m`。因此 Proposed 明显优于 Local，
与 Random 接近但略差于 Random，仍不如 Greedy；它的负载显著低于 Random/Greedy，但有效
协同更新也更少。该结果支持“核心框架具备定位—通信权衡趋势”，不支持“单种子下全面最优”
或“跨随机初始化稳定”。

v6 输出目录：

```text
data/synthetic/mainline_reward_rebalanced_seed42/development/reference/
logs/preliminary/mainline_reward_rebalanced_seed42/development/reference/
plots/preliminary/mainline_reward_rebalanced_seed42/development/reference/
```

对应五张主图只保留 Proposed 的前三张训练图，后两张保留 Proposed、Local、Random、Greedy
的 validation 对照；每张图均导出 PNG 和 PDF。下一步在不改变主方法定义的前提下，应优先解释
“较少测量/更新”与“定位改善”的关系，再决定是否需要独立的多种子确认实验。

### `core-framework-v7` 有限时域与训练方差稳定化开发结果（2026-09-06）

在保留 `Local-EKF-GAT-MAPPO (Proposed Method)`、本地 EKF、GAT、MAPPO、信道和
通信模型的前提下，v7 针对训练链路的七个已知干扰因素建立了独立开发协议：将 50 步任务
作为有限时域目标，加入归一化剩余时间，关闭时间上限 bootstrap；使用 8 个完整 episode
汇总后再执行 PPO 更新；actor 学习率减半；保留 Huber critic、value clipping、
actor/critic 分组梯度裁剪和 sampled-action log-prob；报告奖励保持物理定义不变，PPO 学习
通道仅对稀疏碰撞惩罚限幅。新增 `return_per_window` 仅用于消除回合长度对训练趋势图的
混淆，不能替代报告奖励。

结果写入独立目录，v6 及更早结果未覆盖：

```text
data/synthetic/mainline_core_framework_v7_seed42/development/reference/
logs/preliminary/mainline_core_framework_v7_seed42/development/reference/
plots/preliminary/mainline_core_framework_v7_seed42/development/reference/
```

v7 的 Smoke 和 development 均通过结构审计；全量测试为 `305 passed, 5 skipped`，
`main.py audit-symbols` 通过。development 共 100 个训练 episode、13 次 PPO update，
报告回报均值约 `-10.17`，前/后四分之一约为 `-10.61/-10.34`；平均窗口奖励约为
`-0.222/-0.230`，所以原始奖励仍有场景和提前终止引起的波动，但没有出现数值发散。
14/100 个回合提前终止，平均有效窗口约 46.39。Loss 从首个 update 的约 `0.97` 降至
末个 update 的约 `0.20`，approximate KL 最大约 `0.010`，说明 PPO 更新链路稳定，不能
再把主要问题归咎于灾难性 KL 发散。

validation 的 Proposed 结果为 RMSE/P95 `0.40878/0.69379 m`，Local、Random、Greedy
分别为 `0.43604/0.73382`、`0.41787/0.69637`、`0.40055/0.67992 m`。因此 v7 在该
单种子开发场景下优于 Local 和 Random，但仍不如 Greedy，不能称为全面最优。Proposed 的
实际测量负载约 `52.147 kB/回合`，高于 Random/Greedy 的约 `36.4 kB/回合`，且 harmful
更新比例约 `0.356`；这说明稳定化改动降低了训练目标的数值风险，却使策略偏向更频繁测量，
通信—定位闭环仍未达到受控状态。固定监测 RMSE 从约 `0.380` 降至 `0.354 m`，属于有利
的开发信号，但不能抵消 validation 的通信非劣效要求。

因此 v7 的正确结论是“训练链路稳定性得到改善、定位有改善趋势，但通信代价尚未受控，
尚未证明主方法全面优于基线”。下一步应只处理测量预算/实际通信代价与测量动作之间的
衔接，并继续保留 v7 的有限时域和 PPO 稳定设置；不得通过隐藏高负载或删除碰撞回合改善图形。

```text
C:\Users\lenovo\anaconda3\python.exe main.py run-mainline-baseline-comparison --protocol core-framework-v1 --stage smoke
C:\Users\lenovo\anaconda3\python.exe main.py run-mainline-baseline-comparison --protocol core-framework-v1 --stage development
```

`core-framework-v2` 的 seed=42 入口为：

```text
C:\Users\lenovo\anaconda3\python.exe main.py run-mainline-baseline-comparison --protocol core-framework-v2 --stage smoke
C:\Users\lenovo\anaconda3\python.exe main.py run-mainline-baseline-comparison --protocol core-framework-v2 --stage development
```

v2 仍是单个开发 seed 的实现与描述性比较；在 Smoke 和 development 完成前，不宣称跨随机初始化稳定，也不把训练专用 `R^{learn}_t` 当作最终性能指标。输出保存在 `mainline_core_framework_v2_seed42/<stage>/reference/`，旧 v1、历史实验和 checkpoint 保持不变。

### `stabilized-v4` 持久信息收益—通信代价一致化版本（历史归档，2026-09-04）

在不覆盖历史结果、不给基线人为降级的前提下，新增了版本化协议
`stabilized-v4`。该版本仅对 `communication_constrained_proxy_guided_mappo` 接入三步
持久信息收益代理，并令 teacher 与通信约束共用包含固定 512 bytes 测量负载的候选代价；
teacher 使用 baseline-train 目标的相对代价。`baseline_mappo`、`mappo_mlp`、三个规则
基线和旧协议的默认行为保持不变。代理拟合只读取 baseline train 的 1332 条候选行，
validation/test 均未用于拟合；新的实验根目录为
`mainline_baseline_comparison_v4_seed42`。该次运行保留用于追溯；由于当时规则贪心基线没有显式接收
baseline-train 的代理模型，不能把其中的 `budgeted_greedy_ig` 结果作为有效强基线比较。

seed=42、3 UAV 的 `2/5/5/5` Smoke 已完成，报告为 `audit_ok=true`，所有六种方法的
`episode-v1.3` 数据、checkpoint 和五张紧凑诊断图均已生成；全量测试为 `277 passed`，
`main.py audit-symbols` 通过。Smoke 只证明实现与数据链路正确，`cross_seed_stability`
仍为 `not_evaluated`，不能据此宣称 MAPPO 已稳定优于基线。

Smoke 报告：

```text
logs/preliminary/mainline_baseline_comparison_v4_seed42/smoke/reference/audit_report.json
```

Smoke 图表：

```text
plots/preliminary/mainline_baseline_comparison_v4_seed42/smoke/reference/
```

### `stabilized-v4-corrected` 有效强基线重跑（2026-09-04）

为修复上述比较缺口，新增独立协议 `stabilized-v4-corrected`，不覆盖历史目录。
`budgeted_greedy_ig` 在运行前接收同一份只由 `baseline_mappo/train` 拟合的执行期代理模型，
无代理模型时接口现在直接失败，不再静默退化为全 `none` 动作。该模型的来源、行数以及
validation/test 未参与拟合的标记会写入 `rule_measurement_proxy_model.json` 和
`audit_report.json`；持久代理、目标率、场景、EKF、GAT、信道、队列、奖励和动作空间均保持
原协议定义。

完整 development 已完成于：

```text
data/synthetic/mainline_baseline_comparison_v4_corrected_seed42/development/reference/
logs/preliminary/mainline_baseline_comparison_v4_corrected_seed42/development/reference/
plots/preliminary/mainline_baseline_comparison_v4_corrected_seed42/development/reference/
```

六种方法均完成 `20/100/20/20`，结构审计为 `audit_ok=true`。validation 的单种子描述性结果为：
通信约束主方法 RMSE `0.40050 m`、实际负载 `43008 bytes`；GAT-MAPPO 基线为 `0.40797 m`、
`64640 bytes`；预算贪心基线为 `0.40015 m`、`36224 bytes`。因此主方法相对 GAT 基线呈现
约 `0.00747 m` 的 RMSE 改善和约 `33.5%` 的负载下降，但与贪心基线的 RMSE 几乎持平且略高，
不能宣称全面优于所有基线，也不能把单个 seed 当作稳定性证据。
本次代码验收为 `278 passed`，`main.py audit-symbols` 通过；这些结构验收不等同于算法
性能显著性检验。

### seed=42 训练趋势冻结诊断 v2（2026-09-04）

已完成 `python main.py run-training-trend-diagnostic --audit-version v2 --stage frozen-audit --seed 42`。该版本只读重放 corrected-v2/rollout4 的 GAT-MAPPO 基线和通信约束主方法，在五个固定开发监测场景上比较初始/最终 checkpoint，并对主方法执行 80 个冻结 autograd 梯度探针；不调用 `trainer.update()`、`optimizer.step()` 或 dual 更新，不读取 test，不修改源数据和 checkpoint。

最终审计报告位于 `logs/preliminary/training_trend_diagnostic_seed42/frozen-audit-v2/audit_report.json`，结果为 `audit_ok=true`：20 个端点全部与历史监测值匹配，556 个协同更新事件、2661 条动作/通信证据和 80 个梯度探针已生成；参数更新、优化器步数和乘子更新均为 0，test 访问为 0，checkpoint 哈希前后一致。五张诊断图位于 `plots/preliminary/training_trend_diagnostic_seed42/frozen-audit-v2/`，包括奖励—回合长度归因、冻结端点对比、更新质量、期望/实际通信代价和梯度路径。

审计确认训练奖励低谷与回合长度/碰撞混合、定位奖励反馈弱以及期望通信代价与实际选中代价存在口径差异；固定监测端点没有显示持续改善。critic 梯度主导和 PPO—教师方向仅作为进一步检验或非因果诊断，不能据此宣称算法收敛或跨随机初始化稳定。全量测试 `273 passed`，`main.py audit-symbols` 通过。

### seed=42 训练趋势冻结诊断（2026-09-04）

新增 `python main.py run-training-trend-diagnostic --seed 42 --stage frozen-audit`，只读 corrected-v2 的 R/P/T 训练汇总及 T 组冻结参数。重建初始策略并核对模块哈希，在原五个开发监测场景重放 GAT 基线与主方法的初始/最终策略；另在前四个 train 场景收集初始/最终策略梯度探针，不执行优化器更新、不读取 test。

诊断结果保存到 `logs/preliminary/training_trend_diagnostic_seed42/frozen-audit/`，诊断图在对应 `plots/` 目录。新目录非空即停止，保留旧实验和旧五张主图。探针记录加权损失梯度、测量头 PPO/教师方向余弦、联合及假设分组裁剪保留量；这些不是历史训练梯度，也不是 Adam 参数更新。

筛选门槛固定为至少 20 个有效 actor 非零 minibatch，至少一半同时满足 critic 平方范数占比不低于 80%、分组裁剪保留 actor 范数至少为联合裁剪两倍。未通过时 `--stage smoke` 只报告 `not_run_gate_failed`，不启动训练、不新增裁剪算法。本轮不授权 100 回合对照或多种子实验，不以图形平滑程度判定收敛。

本次 v1 结果已由上方 v2 复核；旧目录和旧报告保留用于追溯，不与 v2 输出混用。

### 五种子主线绘图 mainline-v2（2026-09-03）

`run-final-mainline-audit --figure-profile mainline-v2` 新增只读绘图模式，默认旧模式仍保留。使用最新 VO 校准组的 3 UAV、五训练种子、三个分支；训练图只读 train，性能图只读 validation。独立 split 日志优先于混合日志；v2 文件访问保护会拒绝 test、混合诊断日志读取及源文件写入。

```text
C:\Users\lenovo\anaconda3\python.exe main.py run-final-mainline-audit --reference data/synthetic/communication_constrained_proxy_mappo_vo_calibrated_seedset_27281_31327 --output logs/preliminary/communication_constrained_proxy_mappo_vo_calibrated_seedset_27281_31327/mainline_figures_v2 --plots-output plots/preliminary/communication_constrained_proxy_mappo_vo_calibrated_seedset_27281_31327/mainline_figures_v2 --figure-profile mainline-v2 --seeds 27281 28290 29309 30318 31327 --expected-episodes 20 --expected-training-updates 100 --bootstrap-samples 10000 --bootstrap-seed 2026
```

生成八张 300 dpi PNG 与矢量 PDF：训练奖励/总损失、损失分量、四项奖励分解、定位性能、通信代价、EKF 更新质量、通信—定位配对权衡、逐 seed 配对差异。奖励总图在左侧，定位/通信/碰撞/能耗四项贡献在右侧；碰撞和能耗插图采用全部分支及其区间的共同中央范围并保留零线，主图保留完整尖峰。四项从冻结 reward_config 与归一化日志还原，按 episode_id 绑定 Update，闭合容差为 1e-6。

训练置信带重采样完整 seed 曲线，并对分支复用同一索引；它是逐点区间，不是同时置信带。validation 先 episode 后 seed 等权汇总；更新质量比例的缺失值记 NA，每窗口更新数在无更新时记零。收益保持率属于真实轨迹上的描述性误差诊断，不是无更新反事实。通信贡献可为正，保留原符号；实际代理代价不等于字节数或 J_com。

对应 CSV、中文图注、源文件/检查点 SHA-256 与 audit_report.json 写入日志目录；旧风格复用审计图放在新图目录的 diagnostics_legacy 子目录。v2 即使指定 --force 也拒绝覆盖非空目录，重复生成应选择新的输出后缀。既有算法判定保留，不能通过绘图更新有效性结论。安全场景与无人机规模不在本次绘图范围内。

### seed=42、3 架 UAV 弱强基线比较（2026-09-03）

已完成 `run-mainline-baseline-comparison` 的 Smoke 和 development 流程。该实验固定 3 架 UAV、Moderate、GNSS denied、50 步，使用 seed=42；Smoke 为 `2/5/5/5`，development 为 `20/100/20/20`。A 组完整系统比较包含 `local_only_fixed`、`random_cooperative`、`budgeted_greedy_ig`、`mappo_mlp`、`baseline_mappo` 和 `communication_constrained_proxy_guided_mappo` 六种方法；B 组是冻结 checkpoint 后统一固定飞行/带宽的测量选择诊断，另保留 `proxy_guided_adaptive_budgeted_quality_aware_mappo` 作为机制消融。

规则方法不调用 `trainer.update()`；`local_only_fixed` 保留背景通信，不能解释为零通信。贪心基线使用 baseline-train 的硬预算和执行期可见信息，已修正为只记录当前时间步新增的协同更新事件。B 组现在保存 `validation_cooperative_updates.jsonl`，因此 accepted、beneficial 和 harmful 更新指标不再以缺失日志替代为零。

结果和图表位于：

```text
data/synthetic/mainline_baseline_comparison_dev_seed42
logs/preliminary/mainline_baseline_comparison_dev_seed42
plots/preliminary/mainline_baseline_comparison_dev_seed42/A
plots/preliminary/mainline_baseline_comparison_dev_seed42/B
```

结构审计 `audit_ok=true`，20 个 validation episode 在 A 组六种方法和 B 组四个冻结学习策略中均完整，奖励分解闭合最大误差约 `1.3e-9`，test 未用于拟合、选择或绘图。A 组的单种子描述性结果中，通信约束方法的向量范数 RMSE 约为 `0.3987 m`、GAT-MAPPO baseline 约为 `0.4081 m`，但两者不能据此形成跨随机初始化稳定性或普遍优越性结论；训练曲线、validation 图和 B 组诊断图必须结合 `figure_index.md` 及 CSV 一起解释。

本轮仅新增比较接口、`mlp_flat` 架构对照、规则基线重放和绘图审计；未修改 EKF、奖励、GAT 算法、信道、物理模型或旧实验目录。完整测试和 `main.py audit-symbols` 均通过。

本次 VO 校准组已完成八张主图及 PDF 导出：1500 个训练 Update、71966 个训练窗口和 300 个 validation 回合。奖励闭合最大误差约 1.25e-9；5177 个源文件（含 15 个检查点）哈希未变，test 访问次数为 0。最终全测试集 247 项通过，符号审计通过。绘图结构审计 `audit_ok=true`；原有算法判定仍为 `decision_supported=false`。日志目录中的 `figure_index.md` 可浏览八张图，`verification_report.json` 记录测试和图像检查。

代码联通性已完成：环境、EKF、GAT、MAPPO、episode-v1.3 日志、schema 验证和主图生成链路均可运行。当前已经完成 EKF 相对 VO 的时间一致性修正、UWB EKF 更新周期、配置文件统一入口和 3 架无人机小规模验证。

当前唯一作为后续检查起点的结果是：

```text
data/synthetic/python_sim_calibrated_v4_small
```

它是 3 架无人机、20/10/10 回合、每回合 50 步、单随机种子的 preliminary 结果，schema 为 `episode-v1.3`，验证和主图均已通过。它只能证明当前工程链路能够运行，不能直接作为论文最终结论。

### 通信—协同定位稳定化第一阶段（2026-09-01）

已完成代码层面的公平性与评价口径修正：开启新实验时，GAT、actor body、带宽头、速度头、对数标准差头和 centralized critic 使用固定的模块级随机流；每个 `scale_3/diagnostics/initial_parameter_hashes.json` 和 checkpoint 保存初始参数哈希。质量测量头仍可独立初始化，旧实验默认保持原有行为。

新增的主要诊断字段包括 `position_rmse_norm_m`、`position_p95_norm_m`、`update_coverage_any`、`accepted_update_rate`、`beneficial_per_window`、`harmful_per_window`、`no_update_episode_count` 和 `j_loc_saturation_fraction`。通信代价同时报告期望代理代价、实际选中测量代价、512 字节负载和队列相关指标，避免用代理代价替代真实开销。

seed=2026 的三分支 Smoke 已完成并通过结构审计和 `episode-v1.3`；共享模块初始哈希一致，旧结果目录未覆盖。Smoke 的 `decision_supported=false` 仅表示单种子不能形成算法有效性结论。新的五随机种子正式实验仍需使用独立目录运行完成后再判断 RMSE、更新质量和实际通信代价是否稳定改善。

当前实际使用的候选 EKF 配置为：

```text
data/synthetic/python_sim_calibration_seed2026_v4/selected_ekf_config.json
```

该配置来自纯 Python 训练场景扫描，不是公开数据集或真实传感器校准。扫描选中参数为 `VO=0.05 m`、过程位置标准差 `0.08 m`、过程速度标准差 `0.25 m/s`、NLOS 协方差倍率 `1.0`。但选中候选的 NEES 95% 覆盖率约为 `0.0093`，没有达到预设目标，因此只能称为 preliminary 候选，不能称为最终校准参数。当前结果表现为协方差偏保守，而不是先前的过度自信。

另外，UWB 测量仍按决策窗口记录，但本地 EKF 的成对 UWB 更新按 `sensor_config.uwb_update_period_s = 0.8 s` 执行。这一规则已写入 scale 的 `manifest.json`，用于避免同一时刻重复使用相同协同测距信息。

### 协同 UWB 后续视觉交互的最小修正（2026-09-01）

新增了只读入口 `run-ekf-sensor-interaction-audit`，用于在当前通信约束正式实验的 validation 数据上追踪协同 UWB 更新之后的视觉里程计、预测和后续滤波变化。该审计确认后续窗口主要由视觉里程计更新构成，协同 UWB 的短期误差收益随窗口延长而衰减；这说明当前首要问题是 VO 观测与协同 UWB 的后续交互，不是“通信动作完全不起作用”。审计不读取 test、不训练、不修改原始 episode 或 checkpoint。

按照单一主因原则，代码增加了默认关闭的 `visual_odometry_quality_calibration_enabled`。启用时，仅使用本次运行的 train warm-up 中的 tracking quality 和执行期 VO 创新统计拟合一个固定的、质量单调的 VO 协方差映射，之后原样复用于 train、validation 和 test。低质量或缺失质量不会被当作高质量观测；历史实验默认仍使用固定 VO 协方差，因此旧结果不受影响。该机制只改变 VO 更新的观测协方差，不改奖励、协同 UWB、通信队列、GAT 或 MAPPO 动作。

首轮验证应使用新的独立输出目录运行 3 架无人机、seed=2026 的 `2/5/5/5` Smoke，并检查 `calibration/visual_odometry_quality_model.json`、`integration_summary.json` 和每个 episode 的 `visual_odometry.csv`/`ekf_diagnostics.csv`。Smoke 通过后再在预注册的新 seeds 上做正式验证；在此之前不能把 VO 校准描述为 MAPPO 已经解决通信—协同定位问题。

以下目录均为历史或旧实验结果，保留用于追溯，不作为当前有效结果：

```text
python_sim_preliminary
python_sim_preliminary_100
python_sim_preliminary_fixed
python_sim_preliminary_fixed_v2
python_sim_preliminary_fixed_v3
python_sim_calibrated_v2_small
python_sim_calibrated_v3_small
```

### 近期代理教师—有效 EKF 更新质量审计（2026-08-30）

已对 `proxy_guided_adaptive_quality_mappo` 的 5 个训练随机种子、3 个分支和 validation 数据完成只读归因审计。审计报告位于：

```text
logs/preliminary/proxy_guided_adaptive_quality_mappo_seedset_7071_11127/proxy_update_quality_audit/audit_report.json
```

结构验收已通过：动作和队列开销重放一致，教师分布可复算，代理尺度来自 train，质量模型来自 warm-up，未读取 test，未调用训练更新，checkpoint 哈希未改变。代理效用与线性协方差收益总体一致（4/5 个 seed 对齐），但代理分支平均测量次数和通信代价更高，主要判定为 `communication_cost_dominant`；该审计不是算法有效性证明，`decision_supported=false`。后续应优先设计预注册的通信预算/代价约束实验，不同时修改 EKF、奖励或 GAT。

### 有效协同 EKF 更新持续性审计（2026-08-31）

为解释通信约束 MAPPO 中单次协同更新的收益是否能够保持，新增只读入口 `run-ekf-update-persistence-audit`。审计读取 `communication_constrained_proxy_mappo_seedset_12136_16172` 的 5 个训练 seed、3 个分支和 validation 数据，使用更新后的第 1、2、3、5 个决策窗口追踪同一目标 UAV 的误差、协方差和 `J_loc`；不重放环境、不训练、不读取 test，事件只作机制诊断，Bootstrap 先按 episode 汇总再按 seed 推断。

输出位于：

```text
logs/preliminary/communication_constrained_proxy_mappo_seedset_12136_16172/ekf_update_persistence_audit/
plots/preliminary/communication_constrained_proxy_mappo_seedset_12136_16172/ekf_update_persistence_audit/
```

结构验收通过：300 个 validation episode 完整，更新误差变化与线性协方差减少量公式闭合，审计未调用训练更新且未读取 test。主分支的 seed-level 诊断初步判定为 `benefit_not_persistent`：线性协方差每次均有下降，但立即有益的误差改善在后续窗口中仅约 42% 保持；单次更新的 `J_loc` 变化为 0，说明当前保守协方差恢复使 `J_loc` 对定位均值改善不敏感。该结果不是算法有效性证明；下一步应先审查 EKF 更新后的误差传播/持续性与协方差恢复机制，不应直接调奖励或宣称 MAPPO 已解决通信—协同定位问题。

## 当前唯一执行顺序

以下命令在 PyCharm 的“运行/调试配置 → 脚本参数”中填写，工作目录设为 `D:\research project`，解释器使用 `C:\Users\lenovo\anaconda3\python.exe`。

### 1. 已完成的 EKF 扫描

```text
calibrate-ekf --output "D:\research project\data\synthetic\python_sim_calibration_seed2026_v4" --seed 2026 --uav-count 3 --episodes 10 --steps 50 --force
```

校准结果包括 `ekf_parameter_scan.csv` 和 `selected_ekf_config.json`。参数选择只使用训练场景的 NEES/NIS 和定位误差诊断，不使用 validation/test。由于 NEES 覆盖率目标未满足，不能据此直接进入论文级实验；应先明确初始估计误差/先验协方差的定义，再决定下一轮校准规则。

### 2. 已完成的 3 架无人机小规模验证

```text
run-preliminary --output "D:\research project\data\synthetic\python_sim_calibrated_v4_small" --uav-counts 3 --seed 2026 --steps 50 --warmup-episodes 20 --train-episodes 20 --validation-episodes 10 --test-episodes 10 --ekf-config "D:\research project\data\synthetic\python_sim_calibration_seed2026_v4\selected_ekf_config.json" --device cpu --force
```

使用 `--ekf-config` 后，实际参数、配置文件路径和完整命令会写入每个 scale 的 `manifest.json`。如果同时显式传入冲突的四个 EKF 参数，程序会直接报错，避免配置混用。

### 3. 验证和绘图

```text
validate-preliminary "D:\research project\data\synthetic\python_sim_calibrated_v4_small"
```

```text
plot-main "D:\research project\data\synthetic\python_sim_calibrated_v4_small"
```

数据、图和日志分别位于：

```text
D:\research project\data\synthetic\python_sim_calibrated_v4_small
D:\research project\plots\preliminary\python_sim_calibrated_v4_small
D:\research project\logs\preliminary\python_sim_calibrated_v4_small
```

小规模验证只回答三个问题：EKF 是否一致、MAPPO 是否数值稳定、训练—验证—测试—日志—绘图链路是否闭合；不用于论文最终性能结论。

### 4. 当前暂缓：中等规模实验

小规模工程链路已经通过，但由于当前 EKF 扫描没有达到 NEES 覆盖率目标，中等规模实验暂不作为论文结果运行。解决初始估计先验和 NEES 评价口径后，才运行 3、5、7 架无人机：

```text
run-preliminary --output "D:\research project\data\synthetic\python_sim_medium_seed2026" --uav-counts 3 5 7 --seed 2026 --steps 100 --warmup-episodes 20 --train-episodes 100 --validation-episodes 20 --test-episodes 20 --ekf-config "D:\research project\data\synthetic\python_sim_calibration_seed2026_v4\selected_ekf_config.json" --device cpu --force
```

确认三种规模均通过 schema、loss/梯度/熵有限、动作不过度饱和、碰撞终止记录正确、validation/test 场景确实不同，并且 NEES 评价口径已经固定后，才进入论文级实验。

### 5. 论文级实验

论文级配置为 300/60/60 回合、每回合 300 步、3/5/7 架无人机，并至少使用多个随机种子，例如 `2026`、`3035` 和 `4044`。最终报告 validation 和 test 的均值、标准差或置信区间，同时报告核心目标、定位 RMSE、NEES/NIS、丢包率、时延、AoI、最大连通分量比例、碰撞率、安全距离、能耗代理和三维轨迹。

## 结果解释边界

- `J_t = J_loc,t + J_com,t` 是通信—定位核心目标；碰撞和能耗只进入强化学习团队奖励惩罚，不并入核心目标。
- GNSS 当前为 `denied`；图中不得伪造局部 GNSS 失效区间。
- NIS 必须区分 accepted 和 rejected；NEES 只在 episode 写出阶段使用真值计算，不进入 actor 或 critic。
- 零碰撞不能单独证明策略学会避碰，因为环境安全投影会阻止危险动作；还必须报告最小安全距离、安全投影次数和动作饱和率。
- 论文结论只适用于当前纯 Python 仿真环境，不能表述为已经完成真实飞行验证。

## 代码检查

在项目根目录运行：

```text
C:\Users\lenovo\anaconda3\python.exe -m pytest -q
C:\Users\lenovo\anaconda3\python.exe main.py audit-symbols
```

旧结果不删除、不覆盖；每次修正使用新的输出目录，并在 README、manifest 和日志中记录实验状态。

## 3 架无人机配对基线与动作—KPI 链路审计

为落实“先证明问题存在，再判断完整方法解决了什么”的实验顺序，当前已在冻结的 Moderate 场景上完成 20 组同随机种子、同场景配对测试：

```text
MAPPO：
data/synthetic/python_sim_moderate_seed2026_formal/scale_3

独立固定最弱基线：
data/synthetic/python_sim_independent_baseline_moderate_seed2026_formal/scale_3
```

独立固定基线不使用 GAT、MAPPO、中心状态、邻居状态或协同 UWB 更新。现有配对结果已经证明该环境会自然暴露连通性下降、碰撞、安全间距减小和能耗增加等问题；但尚未证明 MAPPO 能显著降低核心目标 (J_t)。因此当前不调整奖励权重，先执行窗口对齐的链路审计：

```text
audit-causal-chain --reference "D:\research project\data\synthetic\python_sim_moderate_seed2026_formal\scale_3" --baseline "D:\research project\data\synthetic\python_sim_independent_baseline_moderate_seed2026_formal\scale_3" --split test --bootstrap-samples 10000 --bootstrap-seed 2026
```

该命令按动作实际生效窗口对齐 `actions.csv`、`links.csv` 和 `rewards.csv`，并检查：

```text
动作 → 通信/定位 KPI → J_loc/J_com → J → 团队奖励
```

输出位于：

```text
logs/preliminary/python_sim_moderate_seed2026_formal/causal_audit/
plots/preliminary/python_sim_moderate_seed2026_formal/causal_audit/
```

统计不把同一 episode 的时间步当作独立重复。`action_kpi_relation_summary.csv` 中的窗口内相关只用于定位链路弱点，不能解释为动作的因果效应；若要确认因果效应，下一阶段必须在冻结场景上预先登记并执行受控动作扰动实验。test 结果只用于审计和最终报告，不用于反向调节权重。

### Validation 配对复核（2026-08-16）

为避免根据 test 结果修改目标函数，已使用同一冻结 Moderate 场景另外生成 20 组 validation 配对 episode：

```text
MAPPO：
data/synthetic/python_sim_moderate_seed2026_formal/scale_3/validation/

独立固定最弱基线：
data/synthetic/python_sim_independent_baseline_moderate_validation_seed2026/scale_3/validation/
```

对应审计输出为：

```text
logs/preliminary/python_sim_moderate_seed2026_formal/causal_audit_validation/
plots/preliminary/python_sim_moderate_seed2026_formal/causal_audit_validation/
```

验证集复核确认了与 test 审计相同的结构性现象：独立固定基线的原始时延、AoI 尾部和失连程度更严重，MAPPO 的连通性和能耗明显更好；但当前采用训练集 95% 分位数缩放并裁剪到 `[0,1]` 的平均型通信代价，会压平少量灾难性窗口的严重程度，使基线在 `J_com` 和核心目标 `J` 上反而更低。目标恒等式和奖励恒等式的最大数值误差均约为 `1e-9`，因此这不是日志公式实现错误，而是当前目标对尾部风险不敏感。下一阶段如需修改目标聚合，只能使用 train/validation 预先比较平均代价、失连事件和尾部风险项；test 继续封存用于最终报告。

## 独立固定策略配对基线

为先确认当前场景是否真实暴露通信—定位联合优化问题，项目新增 `run-baseline`。该入口不会训练或调用 GAT/MAPPO，而是在现有 3 架无人机 MAPPO 测试场景上运行独立固定策略：每架无人机只使用本机 EKF 估计和目标，关闭协同 UWB 更新，并对潜在出边平均分配带宽。

当前参考结果保持不变：

```text
data/synthetic/python_sim_current_seed2026/scale_3
```

在 PyCharm“脚本参数”中运行：

```text
run-baseline --reference "D:\research project\data\synthetic\python_sim_current_seed2026\scale_3" --output "D:\research project\data\synthetic\python_sim_independent_baseline_seed2026" --controller independent-fixed
```

`run-baseline` 支持 `--split validation` 或 `--split test`。目标函数、归一化方式和超参数只能根据 train/validation 确定；test 基线只在方案冻结后用于最终独立评价。

如需重新生成该基线目录，才添加 `--force`。输出包括：

```text
data/synthetic/python_sim_independent_baseline_seed2026/scale_3/test/
data/synthetic/python_sim_independent_baseline_seed2026/scale_3/diagnostics/paired_metrics.csv
data/synthetic/python_sim_independent_baseline_seed2026/scale_3/diagnostics/paired_summary.csv
data/synthetic/python_sim_independent_baseline_seed2026/scale_3/diagnostics/baseline_comparison.png
```

## 通信—定位环境筛选

当前三无人机基线比较显示网络连通性长期饱和。为避免根据 MAPPO 是否获胜反向调整环境，项目新增独立的 `screen-scenarios` 阶段。该入口不训练、不加载也不评价 GAT/MAPPO，只使用本机估计驱动的固定目标跟踪、潜在出边等比例带宽和首个有效 UWB 邻居选择，扫描以下四个 Python 仿真参数：

```text
最大通信距离             25--65 m
总带宽                   20 kHz--2 MHz（对数扫描）
NLOS 附加损耗            16--40 dB
每决策窗口消息到达量      1--30 packet
```

筛选场景还会增加一个位于两架无人机连线中部、沿垂直方向移动的三维遮挡体，使链路通过真实几何相交产生 NLOS→LOS 转换。程序不会按时间直接篡改丢包率；所有丢包、速率、队列和 AoI 仍由统一信道模型计算。

设计采用带种子的随机化拉丁超立方。每个参数点用独立场景种子重复，episode 是独立重复单位，时间步不是独立样本。场景只有同时通过预先登记的连通性、候选边转化率、丢包、消息到达、EKF 邻居更新、碰撞和有限性门槛，才允许被标记为后续训练候选。若没有点通过全部门槛，`selected_profiles/*.json` 会明确写入 `allowed_for_training=false`，不得强行选择“最接近”的点。

PyCharm 中新建一个 Python 运行配置，脚本仍为 `D:\research project\main.py`，工作目录为 `D:\research project`，脚本参数填写：

```text
screen-scenarios --output data/synthetic/scenario_screening_seed2026 --seed 2026 --design-points 36 --episodes-per-point 10 --steps 50 --ekf-config "D:\research project\data\synthetic\python_sim_calibration_seed2026_v4\selected_ekf_config.json"
```

首次确认接口时可运行较小的 smoke 配置：

```text
screen-scenarios --output data/synthetic/scenario_screening_smoke --seed 2026 --design-points 4 --episodes-per-point 2 --steps 10
```

正式筛选结果默认分开保存：

```text
data/synthetic/scenario_screening_seed2026/
├── manifest.json
└── selected_profiles/
    ├── easy.json
    ├── moderate.json
    └── hard.json

logs/screening/scenario_screening_seed2026/
├── design_matrix.csv
├── run_schedule.csv
├── episode_metrics.csv
├── step_metrics.csv
├── parameter_summary.csv
└── eligibility_report.csv

plots/screening/scenario_screening_seed2026/
├── saturation_diagnostics.png
├── communication_coupling.png
└── feasible_region.png
```

场景筛选结果只证明环境是否暴露联合优化问题，不证明 MAPPO 有效。只有冻结 `moderate.json` 后，才能重新训练三无人机 MAPPO，并在相同场景与种子上进行配对基线比较。

统计以 episode 为独立重复单位，并按相同场景和随机种子进行配对；同一 episode 内的时间步不作为独立样本。该结果只用于判断当前场景是否存在值得优化的问题，不直接作为最终论文性能结论。

当前已将冻结的 Moderate profile 接入正式训练入口。该入口会同时复用筛选阶段的通信参数和移动三维 LOS 遮挡几何；不提供 profile 时，`run-preliminary` 仍使用原有默认环境。第一次闭环只运行 3 架无人机的短实验：

```text
run-preliminary --output data/synthetic/python_sim_moderate_seed2026 --uav-counts 3 --seed 2026 --steps 50 --warmup-episodes 5 --train-episodes 10 --validation-episodes 2 --test-episodes 2 --scenario-profile data/synthetic/scenario_screening_seed2026/selected_profiles/moderate.json --ekf-config "D:\research project\data\synthetic\python_sim_calibration_seed2026_v4\selected_ekf_config.json" --device cpu --force
```

该输出标记为 `moderate_profile_closed_loop_pilot`，只用于确认筛选配置、环境、EKF、GAT、MAPPO 和日志已经闭环联通，不用于最终论文结论。完成该闭环后，才进入相同测试种子的独立固定策略基线配对比较。

## 带宽动作受控扰动（2026-08-24）

项目新增 `run-bandwidth-intervention`，用于只验证下面这一条机制链：

```text
带宽动作 → 信道/队列/数据包状态 → 丢包、时延、AoI、连通性 → J_com
```

该入口不重新训练 MAPPO，不修改奖励权重、目标公式、定位模型或碰撞场景，也禁止使用 test 数据。它读取冻结的 3 架无人机 Moderate validation 集，将每个 episode 按相同场景、seed、速度动作和测量选择重放三次：原 MAPPO 带宽、潜在出边均分带宽、原带宽比例置换。episode 是独立统计单位，窗口数据只用于机制诊断。

PyCharm 中脚本仍选择 `D:\research project\main.py`，脚本参数填写：

```text
run-bandwidth-intervention --reference data/synthetic/python_sim_moderate_seed2026_formal/scale_3 --output data/synthetic/python_sim_bandwidth_intervention_moderate_validation_seed2026 --split validation --bootstrap-samples 10000 --bootstrap-seed 2026
```

如需替换同名输出，再添加 `--force`。本次完成的正式诊断输出为：

```text
data/synthetic/python_sim_bandwidth_intervention_moderate_validation_seed2026/
logs/preliminary/python_sim_moderate_seed2026_formal/bandwidth_intervention/
plots/preliminary/python_sim_moderate_seed2026_formal/bandwidth_intervention/
```

20 个 validation episode 的配对审计结果为：

- 原 MAPPO 重放与参考 validation 的轨迹、链路 KPI 和目标值完全复现；
- 三种处理的真实轨迹、障碍物、速度动作、测量动作、接收功率和 RSSI 扰动完全一致；
- 98.7% 的全局决策窗口产生了有效带宽扰动，高于预设 20% 门槛；
- 带宽改变后速率、丢包尾部、时延尾部和 AoI 尾部发生可检测变化，`J_com` 也随之响应；
- 原 MAPPO 的平均 `J_com` 为 0.09079，均分为 0.09137，置换为 0.09217。原策略相对两种简单分配的配对差异 95% Bootstrap 区间均低于零，但绝对效应较小；
- 原策略降低了丢包和 AoI，但时延 p95 略高，连通性在三种处理中没有变化。因此不能把本结果解释为所有通信 KPI 均优，也不能据此形成最终算法性能结论。

当前决策门结论是：带宽动作确实进入了信道、队列和通信目标，代码链路不存在“动作完全不起作用”的问题；下一阶段可单独验证通信变化是否通过消息到达、邻居更新和协方差传播影响定位。所有图属于工程机制诊断图，权威数值以 `episode_metrics.csv`、`paired_contrasts.csv` 和 `audit_report.json` 为准。

## 协同定位耦合受控实验（2026-08-24）

项目新增 `run-localization-coupling-intervention`，用于在带宽动作链已经通过审计后，单独验证：

```text
已接收邻居消息 → Distributed EKF 协同融合 → 定位性能与 J_loc
```

该实验只使用冻结的 3 架无人机 Moderate validation 20 个 episode，不重新训练 MAPPO，也不使用 test。每个 episode 在完全相同的场景、seed、动作、通信、传感器和真实轨迹下重放两次：正常执行协同 EKF；保留全部通信与消息过程但禁止 EKF 消费邻居状态/UWB 对。episode 是独立统计单位，窗口数据仅用于诊断。

PyCharm 脚本参数：

```text
run-localization-coupling-intervention --reference data/synthetic/python_sim_moderate_seed2026_formal/scale_3 --output data/synthetic/python_sim_localization_coupling_moderate_validation_seed2026 --split validation --bootstrap-samples 10000 --bootstrap-seed 2026
```

默认输出为：

```text
data/synthetic/python_sim_localization_coupling_moderate_validation_seed2026/
logs/preliminary/python_sim_moderate_seed2026_formal/localization_coupling/
plots/preliminary/python_sim_moderate_seed2026_formal/localization_coupling/
```

`SimulationConfig.cooperative_ekf_enabled` 默认为 `True`，因此不改变原训练、验证和测试入口。禁用时只阻断满足消息到达、UWB 有效、测量动作允许、新消息和更新周期等全部条件后的 EKF 融合；链路、消息缓冲、GAT 可见图、动作和物理状态仍按原模型运行。最终判断必须读取 `audit_report.json`，不能只凭图形判断算法有效。

本次20个配对 validation episode 的实际审计结果为：

- 正常协同组与原 validation 的定位估计和奖励逐值一致；两组的真实轨迹、障碍物、动作、链路和数据包事件逐值一致；
- 正常协同组共有50次可执行邻居 UWB 更新并全部接受；禁用组接受0次、阻止50次，说明干预位置准确；
- 正常协同组位置 RMSE 为0.25410 m，禁用组为0.25620 m，禁用减启用为0.00209 m，95% Bootstrap 区间为[0.00010, 0.00472] m；
- 该差异方向符合“协同融合改善定位”，但远低于预设的0.05 m实际效应门槛，因此当前只能证明邻居信息确实进入并轻微改变 EKF 均值，不能证明协同定位具有足够工程收益；
- `J_loc` 和位置协方差迹在两组间不变。当前相对测距更新为避免未知交叉相关导致虚假收缩，保留了保守协方差，因此本次干预主要影响状态均值；
- 每个 episode 的协同更新机会中位数仅为1.5次，且 NEES 覆盖率很低，显示当前协同信息密度和滤波一致性表达仍需单独审计。

当前决策门为 `localization_response_too_small_audit_ekf_update_strength`。下一步不调整奖励，也不直接比较 MAPPO 测量选择；应先检查相对测距增益、观测噪声、创新门限、邻居协方差传播以及协同更新机会密度，在保持配对场景合理的前提下确认协同定位能够产生具有实际意义的定位改善。

## EKF更新强度与定位目标灵敏度审计（2026-08-24）

项目新增只读入口 `run-ekf-update-strength-audit`。该入口重放同一组20个 Moderate validation episode，只记录相对 UWB 更新内部已经计算的创新、NIS、方差分解、Kalman增益、状态修正、协方差恢复和 `J_loc` 响应；不改变滤波公式、奖励、动作或随机数序列，也不使用 test。

PyCharm 脚本参数：

```text
run-ekf-update-strength-audit --reference data/synthetic/python_sim_moderate_seed2026_formal/scale_3 --output data/synthetic/python_sim_ekf_update_strength_audit_seed2026 --split validation
```

输出位置：

```text
data/synthetic/python_sim_ekf_update_strength_audit_seed2026/
logs/preliminary/python_sim_moderate_seed2026_formal/ekf_update_strength/
plots/preliminary/python_sim_moderate_seed2026_formal/ekf_update_strength/
```

本次审计逐值复现了参考 validation，50次更新的方差与状态修正公式残差均为0。实际发现：

- 只有13/20个 episode 包含协同更新，更新窗口占全部1000个窗口的4.7%，每个 episode 更新次数中位数为1.5，协同机会明显稀疏；
- 位置修正量中位数为0.04725 m，并非“增益几乎为零”；
- 未知交叉相关上界占有效测量方差的中位比例为66.59%，是方差预算中的主要部分；
- 100%的相对测距更新在均值修正后恢复了更新前位置协方差，因此50次更新的 `J_loc` 变化全部为0。

当前决策门为 `cooperative_update_opportunities_sparse_design_paired_density_intervention`。这说明当前首要问题是协同更新机会不足；同时已确认 `J_loc` 对现有保守相对测距更新存在结构性不敏感。下一步应先用配对受控实验合理提高更新机会，确认均值定位收益是否随更新密度增强；在此之前不调整奖励、不比较MAPPO测量选择，也不删除未知交叉相关保护。

## 协同测量选择密度配对干预（2026-08-24）

项目新增 `run-measurement-density-intervention`，用于单独验证：

```text
actor可见候选邻居 → 测量对象选择 → 消息/UWB门控 → EKF更新 → 定位性能与J_loc
```

实验只使用冻结的3架无人机 Moderate validation 20个 episode，不重新训练MAPPO、不使用test，也不改变带宽、速度、信道、UWB参数、EKF公式或奖励。每个episode配对重放原MAPPO测量动作和确定性轮换基线。轮换基线只从动作产生时actor已经可见的候选邻居中按无人机ID排序循环选择，不读取未来数据包成功、消息到达或仿真真值。

PyCharm脚本参数：

```text
run-measurement-density-intervention --reference data/synthetic/python_sim_moderate_seed2026_formal/scale_3 --output data/synthetic/python_sim_measurement_density_moderate_validation_seed2026 --split validation --bootstrap-samples 10000 --bootstrap-seed 2026
```

默认输出：

```text
data/synthetic/python_sim_measurement_density_moderate_validation_seed2026/
logs/preliminary/python_sim_moderate_seed2026_formal/measurement_density/
plots/preliminary/python_sim_moderate_seed2026_formal/measurement_density/
```

`window_metrics.csv`和`gate_funnel.csv`记录从候选存在、选择非空、UWB有效、通信边有效、消息到达、更新周期、新数据包到EKF接受的完整门控漏斗。窗口仅用于机制诊断；配对差异和Bootstrap区间以episode为独立统计单位。该实验只判断测量选择过稀是否限制协同定位，不直接证明MAPPO优于基线。

本次20个配对validation episode的实际结果为：

- 原MAPPO重放与参考validation逐值一致；两组真实轨迹、障碍物、链路、数据包、带宽和速度动作的最大差异均为0；
- 原MAPPO在存在候选邻居时只以4.53%的比例选择非空测量，轮换基线为100%；
- 接受的协同EKF更新从平均2.50次/episode提高到64.35次/episode，配对增加61.85次，95% Bootstrap区间为[58.20, 65.35]，说明测量选择过稀确实是更新机会不足的直接原因；
- 位置RMSE由0.25410 m降至0.24321 m，平均变化为-0.01090 m，但95% Bootstrap区间为[-0.02668, 0.00354] m，包含0且低于预设0.05 m实际效应门槛，因此不能证明高密度轮换测量稳定改善定位；
- `J_loc`、位置协方差迹及全部通信KPI保持不变，符合“只改变测量选择”的配对设计，也再次确认当前保守协方差恢复使`J_loc`无法响应相对测距更新；
- 门控漏斗显示轮换组在2963条候选可用动作中全部选择邻居，最终1287次更新全部被EKF接受；主要损失发生在UWB/有效边筛选以及0.8 s更新周期，而不是消息到达或创新拒绝。

当前决策门为`updates_increased_but_localization_response_small_audit_measurement_quality`。这说明“MAPPO大量选择none”已经被确认，但盲目提高测量频率并没有形成稳定且具有实际意义的定位收益。下一阶段不应继续提高频率或修改奖励，而应在这些新增更新中比较LOS/NLOS、NIS、创新幅度、Kalman增益和更新前后真实误差，建立质量感知测量选择依据。

## 协同EKF单次更新质量审计（2026-08-24）

项目新增只读入口`run-cooperative-update-quality-audit`，对原MAPPO选择与actor可见轮换选择进行相同种子、场景和非测量动作的配对重放。真实位置只用于更新完成后的离线效果标签，不进入EKF、actor、critic或奖励。事件按固定1 mm门槛标记为有益、有害或近似无效，但统计推断仍以20个episode为独立单位，不能把更新事件作为独立样本。

PyCharm脚本参数：

```text
run-cooperative-update-quality-audit --reference data/synthetic/python_sim_moderate_seed2026_formal/scale_3 --output data/synthetic/python_sim_cooperative_update_quality_audit_seed2026 --split validation --bootstrap-samples 10000 --bootstrap-seed 2026
```

默认输出：

```text
data/synthetic/python_sim_cooperative_update_quality_audit_seed2026/
logs/preliminary/python_sim_moderate_seed2026_formal/cooperative_update_quality/
plots/preliminary/python_sim_moderate_seed2026_formal/cooperative_update_quality/
```

实际审计精确复现了原组50次和轮换组1287次接受更新，所有配对不变量、公式闭合与episode-v1.3验证通过。轮换组中52.84%的更新有益、38.46%有害、8.70%近似无效，单次更新平均使真实位置误差降低0.00310 m；这解释了高密度测量整体收益很小：有益更新占优，但存在大量有害更新抵消收益。

可执行期可见因素中发现：

- NLOS更新覆盖20个episode、181次事件，episode平均误差变化为+0.03080 m，95% Bootstrap区间为[+0.01784, +0.04437] m；
- 消息年龄大于0.5 s的更新覆盖10个episode、19次事件，episode平均误差变化为+0.06058 m，95% Bootstrap区间为[+0.02694, +0.10312] m；
- 最低Kalman增益四分位和最低未知交叉上界比例四分位也与误差增加相关，但两者高度关联，当前只能作为诊断线索，不能单独解释为因果关系；
- 未知交叉相关上界仍占有效测量方差约66.59%，且每次线性协方差缩减后都会恢复至更新前水平，因此`J_loc`不变是当前保守协方差机制的预期结果。

当前决策门为`observable_quality_strata_predict_harm_design_quality_aware_mask`。下一步应只使用RSSI、SNR、预测丢包率、时延、AoI、队列、链路状态和已接收邻居不确定度等执行期可见量建立质量感知测量掩码，并与原MAPPO和轮换基线做新的配对干预；物理LOS/NLOS和真实误差只能用于离线评价，不能作为策略输入。

## 执行期可见的质量感知测量选择（2026-08-24）

项目新增`run-quality-aware-measurement-intervention`，用于比较原MAPPO测量选择、actor可见候选轮换选择和质量感知选择。该实验不修改EKF、奖励、信道或MAPPO参数，只替换`measurement_uav_id`，并保持场景、随机种子、带宽和速度动作一致。

质量感知处理只使用动作产生时已经存在于局部观测中的RSSI、SNR、预测丢包率、时延、AoI、队列、当前链路状态和已接收邻居协方差。物理LOS/NLOS、真实位置误差、下一窗口数据包成功结果和未来消息到达时间仅允许作为离线评价标签，不能参与候选筛选或排序。

所有筛选门限和归一化区间只从冻结参考运行的`train`目录拟合：RSSI和SNR采用训练集25%分位下限，丢包、时延、AoI和队列采用训练集75%分位上限；存在邻居协方差时同样使用训练集75%分位上限。validation只用于20个episode的配对评价，test继续封存。

PyCharm脚本参数：

```text
run-quality-aware-measurement-intervention --reference data/synthetic/python_sim_moderate_seed2026_formal/scale_3 --output data/synthetic/python_sim_quality_aware_measurement_seed2026 --split validation --bootstrap-samples 10000 --bootstrap-seed 2026
```

默认输出：

```text
data/synthetic/python_sim_quality_aware_measurement_seed2026/
logs/preliminary/python_sim_moderate_seed2026_formal/quality_aware_measurement/
plots/preliminary/python_sim_moderate_seed2026_formal/quality_aware_measurement/
```

统计推断仍以episode为独立单位。`update_events.csv`中的有益、有害和近似无效标签只用于解释机制，不能把单次更新当作独立重复。最终后续方向以`audit_report.json`中的`decision_gate`为准；在结果通过之前，不把质量掩码接入MAPPO训练，也不宣称质量感知策略优于原算法。

本次20个validation episode的实际配对结果为：原MAPPO、轮换和质量感知处理平均每个episode分别接受2.50、64.35和31.40次协同更新。质量感知处理相对轮换处理排除了绝大多数NLOS与陈旧更新，但接受更新数仅为轮换处理的48.80%，触发“训练门限偏严”决策门。

质量感知处理相对轮换处理的有害更新比例平均下降0.02294，但95% Bootstrap区间为[-0.06454, 0.01776]；位置RMSE平均下降0.00761 m，但95%区间为[-0.02364, 0.00752] m，均不足以证明稳定改善。相对原MAPPO，质量感知处理的RMSE平均下降0.01851 m，95%区间为[-0.03370, -0.00597] m，但由于原MAPPO测量极少，该比较不能替代与高密度轮换基线的质量验证。

因此，本阶段只能证明：执行期可见指标能够过滤NLOS和陈旧更新，并保持严格配对重放；尚不能证明当前硬门限质量选择优于轮换选择。下一步只能使用train数据重新设计较温和的质量筛选或排序规则，并在validation上重新评价；不得根据validation结果直接调门限，也不得提前接入MAPPO训练。

## 软质量评分测量选择（2026-08-24）

项目新增`run-soft-quality-measurement-intervention`，在不修改EKF、奖励、信道和MAPPO的条件下，对原MAPPO、候选轮换、硬门限质量选择和软质量评分选择进行20个validation episode的配对重放。软评分使用RSSI、SNR、预测丢包率、时延、AoI、队列和已接收邻居位置协方差迹，各项先用train四分位区间归一化，再采用预先固定的等权平均；仅保留`link_state=connected`基本条件，不再要求每项指标同时通过硬门限。

`none`阈值冻结为train连接候选风险分数的90%分位数0.628251。若当前最佳候选风险超过该值才选择`none`。该规则没有读取validation或test结果，也没有使用物理LOS/NLOS、真实误差、未来通信成功或未来消息到达信息。

PyCharm脚本参数：

```text
run-soft-quality-measurement-intervention --reference data/synthetic/python_sim_moderate_seed2026_formal/scale_3 --output data/synthetic/python_sim_soft_quality_measurement_seed2026 --split validation --bootstrap-samples 10000 --bootstrap-seed 2026
```

默认输出：

```text
data/synthetic/python_sim_soft_quality_measurement_seed2026/
logs/preliminary/python_sim_moderate_seed2026_formal/soft_quality_measurement/
plots/preliminary/python_sim_moderate_seed2026_formal/soft_quality_measurement/
```

实际结果表明，软评分组平均接受47.70次协同更新，是硬门限组31.40次的1.52倍、轮换组64.35次的74.13%。相对轮换组，有害更新比例从0.38476下降至0.33186，配对差异为-0.05289，95% Bootstrap区间为[-0.09524, -0.01275]；位置RMSE从0.24321 m下降至0.22573 m，配对差异为-0.01748 m，95%区间为[-0.03320, -0.00322]；位置误差P95下降0.05090 m，95%区间为[-0.09952, -0.01028]。NLOS更新比例下降0.14308，陈旧更新比例降至0。

四组真实轨迹、障碍物、通信链路、数据包事件、带宽和速度动作的最大差异均为0，原MAPPO处理也精确复现参考validation结果。当前决策门为`soft_quality_effective_freeze_then_add_random_seeds`：软评分机制已通过单随机种子机制验证，但仍属于工程诊断证据。下一步应冻结评分规则，增加多个独立训练/评价随机种子确认稳定性；在此之前不接入MAPPO重新训练，也不使用封存test数据。

## 冻结软质量规则的多随机种子验证（2026-08-24）

项目新增`run-multiseed-soft-quality-validation`，固定3架无人机Moderate场景，对`2026、3035、4044、5053、6062`五个独立基础随机种子分别从头运行20个warm-up、100个MAPPO训练回合、20个validation回合和20个test回合。test数据只生成并封存，不参与软评分拟合、比较或决策。

每个种子的软评分公式、等权规则、执行期可见信息边界和`none`判定规则保持不变；归一化尺度和train-q90阈值只在该种子的train数据中拟合。validation阶段仍比较原MAPPO、候选轮换、硬质量门限和软质量评分四种处理，episode是配对统计单位，更新事件和时间窗口不是独立重复。

PyCharm脚本参数：

```text
run-multiseed-soft-quality-validation --output data/synthetic/python_sim_multiseed_soft_quality_seed2026 --seeds 2026 3035 4044 5053 6062 --steps 50 --warmup-episodes 20 --train-episodes 100 --validation-episodes 20 --test-episodes 20 --scenario-profile data/synthetic/scenario_screening_seed2026/selected_profiles/moderate.json --ekf-config data/synthetic/python_sim_calibration_seed2026_v4/selected_ekf_config.json --bootstrap-samples 10000 --bootstrap-seed 2026 --device cpu
```

输出包括每个种子的独立参考训练结果、四组validation干预数据以及：

```text
logs/preliminary/python_sim_multiseed_soft_quality_seed2026/
├── seed_runs.csv
├── treatment_means.csv
├── seed_contrasts.csv
├── cross_seed_summary.csv
└── audit_report.json
```

决策门要求至少4/5个种子中软质量选择相对轮换选择同时改善位置RMSE和有害更新比例，并且两项跨种子Bootstrap区间均支持改善。通过后才进入“将质量评分接入MAPPO并重新训练”；未通过则先检查跨种子场景、归一化和阈值稳定性，不修改EKF、奖励、碰撞、GAT或软评分权重。

本次五种子验证已经完成。相对`round_robin_candidate`，软质量选择在5/5个种子中同时降低位置RMSE和有害更新比例：位置RMSE配对差异均值为`-0.01419 m`，跨种子95% Bootstrap区间为`[-0.01884, -0.00858] m`；有害更新比例配对差异均值为`-0.04813`，区间为`[-0.05860, -0.03719]`。软质量选择平均少接受约`15.66`次协同更新，但同时将NLOS更新比例平均降低`0.14004`；这说明改善来自更谨慎的更新筛选，而不是单纯增加更新次数。25个episode-v1.3目录（5个参考目录和20个干预目录）均通过验证，全套93项测试通过，符号审计通过。

该结果支持“软质量规则在当前3架无人机Moderate场景下具有初步跨种子稳定性”，但仍只是工程机制证据，不等于完整MAPPO算法优越性，也没有解决`J_{\mathrm{loc}}`结构、碰撞风险和安全/能耗权衡。下一步可以在保持规则冻结的前提下，将质量信息作为执行期测量动作掩码或候选约束接入MAPPO并重新训练；test数据仍保持封存。

## 软质量机制正式接入 MAPPO（2026-08-25）

项目新增 `run-quality-aware-mappo`，将已经冻结并通过五随机种子机制验证的软质量规则接入 MAPPO 的测量动作头。实验保留两个分支：

```text
baseline_mappo
    → 原有 MAPPO actor；不接收测量质量特征

quality_aware_mappo
    → 仅测量动作头接收 measurement_quality_risk 和 measurement_quality_valid
```

质量向量按无人机 ID 对齐，只由当前局部观测中的 RSSI、SNR、预测丢包率、时延、AoI、队列、链路状态和已接收邻居协方差计算。每个随机种子先用 warm-up 观测拟合一次并冻结；validation/test 不参与拟合。带宽动作、速度动作、GAT、centralized critic、EKF、信道、奖励和碰撞模型均不改变。

单种子联通 smoke：

```text
run-quality-aware-mappo --output data/synthetic/python_sim_quality_aware_mappo_smoke --seeds 2026 --steps 3 --warmup-episodes 2 --train-episodes 2 --validation-episodes 2 --test-episodes 2 --bootstrap-samples 100 --device cpu --force
```

正式五随机种子实验：

```text
run-quality-aware-mappo --output data/synthetic/python_sim_quality_aware_mappo_seed2026 --seeds 2026 3035 4044 5053 6062 --steps 50 --warmup-episodes 20 --train-episodes 100 --validation-episodes 20 --test-episodes 20 --scenario-profile data/synthetic/scenario_screening_seed2026/selected_profiles/moderate.json --ekf-config data/synthetic/python_sim_calibration_seed2026_v4/selected_ekf_config.json --bootstrap-samples 10000 --bootstrap-seed 2026 --device cpu
```

输出目录为：

```text
data/synthetic/python_sim_quality_aware_mappo_seed2026/
    seed_2026/baseline_mappo/scale_3/
    seed_2026/quality_aware_mappo/scale_3/
    ...

logs/preliminary/python_sim_quality_aware_mappo_seed2026/
    episode_metrics.csv
    treatment_means.csv
    paired_contrasts.csv
    cross_seed_summary.csv
    seed_runs.csv
    audit_report.json

plots/preliminary/python_sim_quality_aware_mappo_seed2026/
    quality_aware_mappo_comparison.png
```

`audit_report.json` 中的 `audit_ok` 只表示输出、schema、分支和有限性检查通过；`decision_supported` 才表示五随机种子下位置 RMSE 与有害更新比例同时满足预设改善门槛。test 数据会生成但封存，不能用于规则拟合、模型选择或训练调参。该实验不自动证明最终算法优越性，也不扩展到5/7架无人机、碰撞风险或能耗权衡。

## 目标函数—定位性能—通信开销一致性审计（2026-08-25）

项目新增只读入口`run-objective-sensitivity-audit`，用于检查质量感知MAPPO五随机种子结果中的定位RMSE、协方差型定位代价`J_loc`、通信代价`J_com`、核心目标`J_t`和团队奖励是否方向一致。审计只读取`validation`目录，不读取或使用`test`目录，不重新训练、不重放环境，也不修改EKF、奖励、GAT、信道和碰撞模型。真实位置只用于离线计算RMSE和位置误差P95，不能用于构造目标、奖励或策略输入。

PyCharm脚本参数：

```text
run-objective-sensitivity-audit --reference data/synthetic/python_sim_quality_aware_mappo_seed2026 --split validation --bootstrap-samples 10000 --bootstrap-seed 2026 --force
```

输出为：

```text
logs/preliminary/python_sim_quality_aware_mappo_seed2026/objective_sensitivity/
├── episode_sensitivity.csv
├── seed_sensitivity.csv
├── metric_correlations.csv
└── audit_report.json

plots/preliminary/python_sim_quality_aware_mappo_seed2026/objective_sensitivity/
└── objective_sensitivity.png
```

五个训练随机种子、每个分支20个validation episode均被读取，统计推断单位仍为训练随机种子，不把窗口或单次EKF更新当作独立样本。审计确认：

- `J_t=J_loc+J_com`最大绝对误差约`8.0×10^-10`，奖励闭合最大绝对误差约`1.0×10^-9`，公式实现闭合；
- quality-aware相对baseline的位置RMSE平均下降`0.01167 m`，4/5个种子改善，但95% Bootstrap区间为`[-0.03363, 0.01472] m`，尚不支持稳定改善；
- `J_loc`平均增加`0.0003305`，相对于baseline约`0.34475`的1%实用容差`0.003447`属于基本不变；协方差迹平均增加`0.459`，而RMSE下降，说明当前协方差型`J_loc`不能反映这次定位均值改善；
- `J_com`平均增加`0.03387`，95%区间为`[0.02720, 0.04025]`；同时丢包率平均增加`0.02968`，连通性下降`0.01587`，团队奖励平均下降`0.03436`且95%区间不跨0，质量感知MAPPO的定位收益伴随明确通信代价；
- `audit_ok=true`，但`decision_supported=false`；当前决策门为`objective_break_jloc_insensitive_to_rmse`，不是算法有效性通过。

因此，当前不扩展5/7架无人机，也不根据本次validation结果调奖励权重或修改EKF。下一步应先设计不使用真实误差的执行期不确定度代理，并单独控制测量带来的通信开销，再进行新的配对验证；test数据继续封存。

## 执行期预期信息增益代理审计（2026-08-25）

项目新增只读入口`run-execution-uncertainty-proxy-audit`，用于建立不依赖真实误差的定位不确定度代理。该阶段只读取每个随机种子的`baseline_mappo/train`拟合尺度，并读取两个分支的`validation`候选邻居；不打开`test`目录，不重新训练MAPPO，也不修改EKF、奖励、GAT、信道或碰撞模型。

代理使用本机估计位置/协方差、已接收邻居估计与协方差（不可见时使用训练集统计回退）、UWB测量标准差、AoI/消息年龄、RSSI、SNR、预测丢包率、预测时延、队列和当前链路状态。真实位置、真实距离、LOS/NLOS标签、未来消息结果和`central_state_json`均被排除。validation中的位置误差仅作为离线解释标签。

PyCharm脚本参数：

```text
run-execution-uncertainty-proxy-audit --reference data/synthetic/python_sim_quality_aware_mappo_seed2026 --force
```

输出为：

```text
logs/preliminary/python_sim_quality_aware_mappo_seed2026/execution_uncertainty_proxy/
├── proxy_parameters.json
├── candidate_proxy.csv
├── episode_proxy_summary.csv
├── seed_proxy_summary.csv
├── proxy_strata.csv
└── audit_report.json

plots/preliminary/python_sim_quality_aware_mappo_seed2026/execution_uncertainty_proxy/
└── execution_uncertainty_proxy.png
```

本次五种子validation审计结果为：代理字段有效率`100%`，多候选窗口中具有非零代理差异的比例约`97.53%`，排序确定性检查通过，训练/validation隔离和test封存检查通过；但高风险分层的离线定位误差没有呈现预设的非下降趋势。因此`audit_ok=false`，当前不能进入预算受控干预，也不能将该代理接入MAPPO。这个结果不是代码联通失败，而是说明当前代理与定位误差之间的解释力尚未被验证；下一步应先修正代理构造或分层定义，再重新审计。

## 执行期信息增益代理目标对齐审计（2026-08-25）

项目新增只读入口`run-execution-uncertainty-proxy-alignment-audit`，用于把执行期预期信息增益代理与它真正预测的单次测量收益对齐。该审计继续只用每个随机种子的`baseline_mappo/train`拟合尺度，并读取两个分支的`validation`候选邻居和协同更新诊断；不读取`test`，不重放环境，不重新训练MAPPO，也不修改EKF、奖励、GAT、信道或碰撞模型。

本次对齐标签采用线性协方差减少量：

```text
Delta P_linear = trace(P_before) - trace(P_after_linear)
```

其中真实位置误差变化和LOS/NLOS仅作为validation离线诊断标签，不能参与代理拟合、候选排序、动作选择或模型更新。审计同时记录最终保守协方差恢复量、状态/位置修正量、NIS、消息年龄、未知交叉相关上界和更新接受状态。单次更新只用于机制解释，统计独立单位仍是seed和episode。

PyCharm脚本参数：

```text
run-execution-uncertainty-proxy-alignment-audit --reference data/synthetic/python_sim_quality_aware_mappo_seed2026 --bootstrap-samples 10000 --bootstrap-seed 2026 --force
```

输出为：

```text
logs/preliminary/python_sim_quality_aware_mappo_seed2026/execution_uncertainty_proxy_alignment/
├── proxy_alignment_events.csv
├── proxy_alignment_episode_summary.csv
├── proxy_alignment_seed_summary.csv
├── proxy_alignment_strata.csv
├── proxy_alignment_correlations.csv
└── audit_report.json

plots/preliminary/python_sim_quality_aware_mappo_seed2026/execution_uncertainty_proxy_alignment/
└── proxy_alignment.png
```

本次实际审计读取五个训练随机种子、两个分支的validation候选记录，共`53,822`条候选记录，匹配`3,503`条已接受协同更新。代理字段和更新字段均有限，线性协方差减少量公式闭合，训练/validation隔离、test封存、原始数据不变和“更新事件不作为独立重复”检查均通过。低风险分层的平均线性协方差减少量高于高风险分层，且低风险有害更新比例不高于高风险分层，因此`audit_ok=true`、`decision_supported=true`，当前代理可以进入预算受控配对干预。

该结论的边界是：代理获得了与线性协方差收益一致的初步分层支持，但散点相关仍较弱，不能解释为已经证明真实定位RMSE或论文最终算法有效；下一步只做固定预算的`recorded_mappo`、无预算信息增益和预算受控信息增益三组validation配对实验，继续封存test，不修改核心模型。

## 预算受控的信息增益配对干预（validation-only，2026-08-25）

项目新增`run-information-gain-budget-intervention`，以五个随机种子的`baseline_mappo`分支为共同参考，配对重放validation数据。三种处理只替换`measurement_uav_id`：

```text
recorded_mappo
    → 保留baseline原始测量选择
unconstrained_information_gain
    → 从当前可见候选中选择执行期信息增益代理最大的邻居，不设测量上限
budgeted_information_gain
    → 使用同一排序，但每个seed、每架无人机、每个validation episode使用train集baseline平均测量次数取整后的固定预算
```

信息增益代理只使用估计状态/协方差、已接收邻居信息、UWB测量噪声、当前候选链路质量、预测丢包/时延、AoI、队列和消息年龄；不使用真实位置、真实误差、LOS/NLOS真值、实际成功结果或未来消息。训练集只用于每个seed的代理尺度和测量预算拟合，validation只用于配对评价，test不读取。

PyCharm脚本参数：

```text
run-information-gain-budget-intervention --reference data/synthetic/python_sim_quality_aware_mappo_seed2026 --reference-branch baseline_mappo --output data/synthetic/python_sim_information_gain_budget_intervention_seed2026 --split validation --bootstrap-samples 10000 --bootstrap-seed 2026 --force
```

输出包括：

```text
data/synthetic/python_sim_information_gain_budget_intervention_seed2026/
    seed_2026/{recorded_mappo,unconstrained_information_gain,budgeted_information_gain}/scale_3/
    ...

logs/preliminary/python_sim_quality_aware_mappo_seed2026/information_gain_budget_intervention/
    proxy_parameters.json
    measurement_budgets.csv
    window_metrics.csv
    update_events.csv
    episode_metrics.csv
    seed_metrics.csv
    paired_contrasts.csv
    budget_usage.csv
    audit_report.json

plots/preliminary/python_sim_quality_aware_mappo_seed2026/information_gain_budget_intervention/
    information_gain_budget_intervention.png
```

统计独立单位是五个base seed；episode用于配对，窗口和单次EKF更新仅作机制诊断。该实验不重新训练MAPPO，不修改EKF、奖励、GAT、信道、碰撞和能耗模型。其目的首先是判断“定位收益—通信开销”权衡是否真实存在；只有预算受控处理在不增加通信开销的情况下改善定位指标，才考虑把相应信息接入MAPPO。碰撞风险和能耗仍在通信—定位主线完成后单独验证。

本次五种子validation-only运行已完成，`audit_ok=true`且`decision_supported=true`。`recorded_mappo`与baseline validation逐项重放一致；真实轨迹、障碍物、链路、数据包、带宽和速度动作在三种处理之间保持一致，预算上限和episode-v1.3校验通过，test未读取。相对于`recorded_mappo`，无预算信息增益组的平均位置RMSE差异为`-0.02690 m`（Bootstrap 95% CI：`[-0.03176, -0.02206]`），预算受控组为`-0.02121 m`（Bootstrap 95% CI：`[-0.02657, -0.01586]`）。

但两组的`J_loc`、`J_com`、团队奖励及通信KPI在本实验中均未发生差异；这是因为方案按设计只替换测量无人机，不改变带宽、信道随机过程和通信过程。因此本次结果支持“预算受控信息增益选择在固定通信条件下具有初步定位收益”，不支持宣称已经证明完整的通信—定位开销权衡，也不等同于重新训练后的MAPPO策略优越性。当前`J_loc`对RMSE改善不敏感的问题仍保留；若要研究真实通信开销权衡，后续必须单独建立测量消息对带宽、队列或通信资源的显式影响，而不能从本次固定通信重放中推断。

## 测量消息通信开销验证（validation-only，2026-08-25）

为验证“测量选择是否真正占用通信资源”，项目新增默认关闭的`measurement_overhead_enabled`机制。每次选择一个有效潜在邻居时，向对应发送边加入一个等效的512字节数据包负载；该负载进入队列，并影响速率、丢包、时延、AoI和下一时刻队列。该机制不是完整MAC或请求—响应协议，只用于第一版通信开销机制验证；EKF消息到达语义、运动、带宽动作、速度动作、奖励和核心模型均未改变。旧实验默认关闭该开关，因此结果保持兼容。

项目新增`run-measurement-overhead-intervention`，使用五个随机种子的`baseline_mappo` validation分支配对比较`recorded_mappo`、`unconstrained_information_gain`和`budgeted_information_gain`，并额外运行关闭开关的`recorded_mappo_no_overhead`控制重放。训练集只用于信息增益尺度和预算拟合，validation用于机制评价，test未读取；统计独立单位为base seed，时间窗口和EKF事件不作为独立样本。

PyCharm脚本参数：

```text
run-measurement-overhead-intervention --reference data/synthetic/python_sim_quality_aware_mappo_seed2026 --reference-branch baseline_mappo --output data/synthetic/python_sim_measurement_overhead_intervention_seed2026 --split validation --bootstrap-samples 10000 --bootstrap-seed 2026 --force
```

结果位于：

```text
data/synthetic/python_sim_measurement_overhead_intervention_seed2026/
logs/preliminary/python_sim_quality_aware_mappo_seed2026/measurement_overhead_intervention/
plots/preliminary/python_sim_quality_aware_mappo_seed2026/measurement_overhead_intervention/
```

本次审计报告为`audit_ok=true`、`decision_supported=true`，控制重放与原validation逐项一致，三种处理的真实轨迹、障碍物、带宽和速度动作一致，预算上限、开销负载对应关系、episode-v1.3验证和test封存检查均通过。相对于开启开销的`recorded_mappo`，无预算信息增益组位置RMSE平均降低`0.01414 m`（95% Bootstrap CI：`[-0.01946,-0.01022]`），但`J_com`平均增加`0.01139`（CI：`[0.00213,0.02066]`），同时团队奖励下降；这明确暴露了定位收益—通信开销权衡。预算受控组位置RMSE平均降低`0.00757 m`（CI：`[-0.01394,-0.00273]`），`J_com`平均变化为`-0.00138`且区间跨零，因此只能称为固定预算下的初步定位收益，不能宣称已经完成最终MAPPO算法证明。

本实验仍是validation-only机制证据，不是论文最终结果，也不代表完整无线MAC实现。`J_loc`在本实验中保持不变，说明现有协方差型定位代价对定位均值改善仍不敏感；碰撞风险、能耗权衡、5/7架无人机扩展和重新训练后的策略优越性仍需后续单独验证。

## 通信开销条件下的 MAPPO 联合优化训练（2026-08-25）

为验证“测量动作—通信队列—通信KPI—MAPPO策略”这一完整链路，质量感知MAPPO训练入口新增了`measurement_overhead_enabled`开关。开启时，两个训练分支的每次有效非空测量选择都会向对应通信边加入一个等效的512字节队列负载；该负载进入队列、丢包、时延和AoI计算。EKF、GAT、信道公式、奖励公式、带宽动作、速度动作和软质量规则保持不变，旧配置默认关闭开关，因此历史实验结果不被覆盖。

PyCharm中的正式命令为：

```text
run-quality-aware-mappo --output data/synthetic/python_sim_quality_aware_mappo_overhead_seed2026 --seeds 2026 3035 4044 5053 6062 --steps 50 --warmup-episodes 20 --train-episodes 100 --validation-episodes 20 --test-episodes 20 --scenario-profile data/synthetic/scenario_screening_seed2026/selected_profiles/moderate.json --ekf-config data/synthetic/python_sim_calibration_seed2026_v4/selected_ekf_config.json --measurement-overhead-enabled --device cpu --force
```

正式训练前已完成3架无人机、seed=2026的smoke检查：warm-up 2回合、train/validation/test各5回合；baseline和quality-aware两个分支均生成`episode-v1.3`数据，`measurement_overhead.csv`已写入每个episode，开销开关在配置和环境中均为`true`，结构审计为`audit_ok=true`。由于smoke只有一个随机种子，`decision_supported=false`是预期结果，不能用于论文算法结论。

正式五种子结果将写入：

```text
data/synthetic/python_sim_quality_aware_mappo_overhead_seed2026/
logs/preliminary/python_sim_quality_aware_mappo_overhead_seed2026/
plots/preliminary/python_sim_quality_aware_mappo_overhead_seed2026/
```

正式验收只使用五个随机种子作为独立统计单位，并分别检查位置RMSE、`J_loc`、`J_com`、团队奖励、测量次数、512字节等效负载、丢包率、时延、AoI、连通性、有害协同更新比例、碰撞率和能耗。只有在两个分支的结构审计、episode-v1.3验证和训练/验证/测试隔离均通过后，才根据配对置信区间判断质量感知MAPPO是否学会通信—定位权衡；在此之前不扩展5/7架无人机，也不宣称最终算法有效。

本次正式五种子训练已完成。结果目录为：

```text
data/synthetic/python_sim_quality_aware_mappo_overhead_seed2026/
logs/preliminary/python_sim_quality_aware_mappo_overhead_seed2026/
plots/preliminary/python_sim_quality_aware_mappo_overhead_seed2026/
```

结构验收结果：`audit_ok=true`、全部10个训练分支均开启`measurement_overhead_enabled=true`，每个分支均包含140个episode（20个warm-up、100个train、20个validation、20个test），每个episode均写出`measurement_overhead.csv`，validation schema检查通过，test已生成但未用于选择。全量测试为`110 passed`，符号审计通过。

算法判定结果：`decision_supported=false`。validation中，quality-aware相对于baseline的位置RMSE在5个种子中仅有3个改善，有害更新比例仅有2个下降；RMSE配对差异均值为`-0.01431 m`，Bootstrap 95% CI为`[-0.03035, 0.00374]`；有害更新比例差异均值为`+0.00198`，Bootstrap 95% CI为`[-0.10622, 0.11320]`。两个区间均跨越0，且未达到预设的4/5种子改善门槛。因此当前结论是：通信开销机制已成功接入并完成可重复的3架无人机Moderate场景正式验证，但尚不能证明质量感知MAPPO稳定优于baseline，也不能宣称已经学会通信—定位联合权衡。

validation中还观察到`J_com`平均增加约`0.02118`，团队奖励平均下降约`0.02067`；`J_loc`平均变化约`-0.00007`，仍显示对定位均值改善不敏感。test结果仅作为封存后的最终报告材料，不参与上述判定。下一步应先检查质量特征是否真正改变测量动作概率、`none`选择率与队列开销响应，再决定是否调整测量动作分支；暂不扩展5/7架无人机、不修改EKF、奖励、GAT、信道或碰撞模型。

## 质量感知 MAPPO 动作—通信开销机制审计（2026-08-26）

本阶段新增只读入口`run-policy-mechanism-audit`，不重新训练、不改变EKF、奖励、GAT、信道或碰撞模型。它从已有五种子、3架无人机Moderate开销实验的冻结checkpoint重放validation，记录每个候选邻居的质量风险、有效性掩码、完整测量动作概率向量、`none`概率、实际选择以及512字节等效队列开销。test目录不读取，episode schema不修改；统计独立单位仍为训练随机种子。

PyCharm脚本参数：

```text
run-policy-mechanism-audit --reference data/synthetic/python_sim_quality_aware_mappo_overhead_seed2026 --output logs/preliminary/python_sim_quality_aware_mappo_overhead_seed2026/policy_mechanism_audit --plots-output plots/preliminary/python_sim_quality_aware_mappo_overhead_seed2026/policy_mechanism_audit --seeds 2026 3035 4044 5053 6062 --split validation --bootstrap-samples 10000 --bootstrap-seed 2026 --device cpu --force
```

输出为：

```text
logs/preliminary/python_sim_quality_aware_mappo_overhead_seed2026/policy_mechanism_audit/
plots/preliminary/python_sim_quality_aware_mappo_overhead_seed2026/policy_mechanism_audit/
```

五种子审计结果为`audit_ok=true`：所有重放动作和队列事件逐项一致，质量模型均来自warm-up，29,763条诊断轨迹有限且未包含真值/未来字段。质量感知分支的`none`选择率相对baseline在3/5个种子明显下降，同时测量次数与等效负载上升；对应决策为`policy_overmeasurement`，而非算法有效性通过。`actor_response_status`为`not_responsive`，说明当前候选级风险—概率关系尚未形成稳定负相关，不能直接声称策略已经学会按风险选择邻居。下一步应在不改核心模型的前提下，检查质量特征的归一化、方向和测量动作头响应；若确认策略确实过度测量，再设计固定预算或显式测量通信成本约束并重新训练。 

## 质量特征语义与 MAPPO 动作响应审计（validation-only，2026-08-26）

本阶段新增只读入口`run-quality-feature-sensitivity-audit`，针对上一阶段发现的风险—动作概率关系不稳定问题，分离检查三件事：风险定义是否单调、缺失邻居协方差是否按最差风险处理、quality-aware测量动作头是否对风险反事实输入产生预期响应。反事实扫描只把同一执行期观测中的一个候选风险依次替换为`0、0.25、0.50、0.75、1.00`，保持图状态、动作掩码、带宽和速度输入不变；不执行反事实动作，不读取真实位置、真实误差、LOS/NLOS、未来结果或test数据。

PyCharm脚本参数：

```text
run-quality-feature-sensitivity-audit --reference data/synthetic/python_sim_quality_aware_mappo_overhead_seed2026 --output logs/preliminary/python_sim_quality_aware_mappo_overhead_seed2026/quality_feature_sensitivity --plots-output plots/preliminary/python_sim_quality_aware_mappo_overhead_seed2026/quality_feature_sensitivity --seeds 2026 3035 4044 5053 6062 --split validation --device cpu --force
```

输出为：

```text
logs/preliminary/python_sim_quality_aware_mappo_overhead_seed2026/quality_feature_sensitivity/
├── feature_semantics.csv
├── counterfactual_policy_response.csv
├── missing_value_audit.csv
├── seed_summary.csv
└── audit_report.json

plots/preliminary/python_sim_quality_aware_mappo_overhead_seed2026/quality_feature_sensitivity/
└── quality_feature_sensitivity.png
```

统计独立单位仍是base seed；validation episode只用于配对机制诊断，test完全封存。该审计不重新训练、不修改原始episode、checkpoint、EKF、奖励、GAT、信道或碰撞模型。只有在风险语义、缺失值处理和反事实动作响应均通过后，才考虑调整测量动作头接口或开展小规模重新训练；在审计结论形成前，不扩展5/7架无人机。

全量五种子审计已完成。结构检查全部通过：`feature_semantics_ok=true`、`missing_value_handling_ok=true`、`baseline_quality_invariant=true`、`quality_features_present=true`、`replay_actions_exact=true`、`all_rows_finite=true`、`all_validation_only=true`；质量模型均来自warm-up，反事实动作未执行，test数据未读取。共生成218,780条反事实策略响应记录和58,637条缺失值诊断记录。

机制结论为`decision=actor_direction_unstable`、`decision_supported=false`。质量感知分支按种子呈现预期“风险升高、候选概率下降”的比例依次为`1.000、0.399、0.379、0.568、0.264`，只有2026号种子稳定满足该方向；其余种子方向不稳定。baseline对质量输入的最大概率变化为0，说明分支隔离有效；风险定义和缺失协方差最差回退也均通过。因此当前问题定位为“风险语义本身正确，但quality-aware actor尚未稳定利用该特征”，不能据此宣称质量感知策略有效。

本次回归测试为`120 passed`，`main.py audit-symbols`通过（234个规范符号、10个遗留符号、无重复和缺失登记）。下一步只应调整或隔离测量动作头的质量特征接口并做单种子smoke；在此之前不重新进行五种子正式实验，不修改EKF、奖励、GAT、信道和碰撞模型，也不扩展到5/7架无人机。

## 质量感知测量动作头单调性修正与单种子Smoke（2026-08-26）

针对上一阶段发现的“质量风险已经进入actor，但候选风险—动作概率关系不稳定”问题，本阶段只修改quality-aware测量动作head，不改变baseline、EKF、奖励、GAT、信道、碰撞模型或episode-v1.3。新增固定的候选logit风险惩罚：

```text
z'_k = z_k - lambda_q * q_k * v_k
```

其中`lambda_q = 1.0`仅用于`quality_aware_mappo`，`q_k`为执行期可见候选风险，`v_k`为质量有效性；`none`动作logit和原有动作掩码不变。baseline分支强制使用`lambda_q = 0`，因此即使旧调用方传入非零配置，也不会改变baseline行为。旧checkpoint缺少该字段时默认兼容为0；新的quality-aware实验必须从头训练。

同时在`measurement_quality.jsonl`中记录原始logit、风险惩罚、调整后logit、候选概率、`none`概率、惩罚系数和分支名，便于区分“固定单调机制”与“网络学习响应”。

单种子Smoke命令为：

```text
run-quality-aware-mappo --output data/synthetic/python_sim_quality_aware_mappo_quality_head_fix_smoke --seeds 2026 --steps 50 --warmup-episodes 2 --train-episodes 5 --validation-episodes 5 --test-episodes 5 --measurement-overhead-enabled --quality-logit-penalty 1.0 --bootstrap-samples 100 --device cpu --force
```

结果保存于：

```text
data/synthetic/python_sim_quality_aware_mappo_quality_head_fix_smoke/
logs/preliminary/python_sim_quality_aware_mappo_quality_head_fix_smoke/
plots/preliminary/python_sim_quality_aware_mappo_quality_head_fix_smoke/
```

Smoke结构检查为`audit_ok=true`；baseline和quality-aware两个分支均完成2个warm-up、5个train、5个validation和5个test回合，开启512字节测量开销，生成episode-v1.3和诊断日志。由于只有一个随机种子，`decision_supported=false`是预期结果，不能作为论文级算法结论。

随后对同一Smoke结果执行validation-only质量特征敏感性审计。结果为：

- `feature_semantics_ok=true`、`missing_value_handling_ok=true`、`baseline_quality_invariant=true`、`replay_actions_exact=true`、`all_rows_finite=true`；
- test未读取，反事实风险扫描不执行环境动作；
- baseline在风险从0变化到1时候选概率变化为0；
- quality-aware分支候选概率的预期单调方向比例为1.0，候选概率端点平均变化约`-0.20724`，`none`概率端点平均变化约`+0.14420`；
- `audit_ok=true`，但这只是单种子机制验证，不代表五随机种子性能改善。

本阶段全量测试为`123 passed`，`main.py audit-symbols`通过（234个规范符号、10个遗留符号、无重复或缺少登记信息）。下一步只有在该单种子反事实审计通过后，才重新执行冻结系数下的五随机种子正式验证；在此之前不扩展5/7架无人机，也不修改EKF、奖励、GAT、信道、碰撞模型或定位目标。

## 固定风险惩罚后的五随机种子正式验证（2026-08-26）

针对质量感知测量动作头，使用固定`lambda_q=1.0`完成了3架无人机Moderate场景的五随机种子正式验证。两个分支均开启测量通信开销，唯一预设差异为：`baseline_mappo`使用惩罚系数`0`，`quality_aware_mappo`使用惩罚系数`1.0`；两组均使用20个warm-up、100个train、20个validation和20个test回合。结果保存于：

```text
data/synthetic/python_sim_quality_aware_mappo_quality_head_fix_seed2026/
logs/preliminary/python_sim_quality_aware_mappo_quality_head_fix_seed2026/
plots/preliminary/python_sim_quality_aware_mappo_quality_head_fix_seed2026/
```

结构验收通过：10个`scale_3`分支全部为`episode-v1.3`，每个分支包含140个episode，所有episode验证通过；两个分支的配置分别记录了`measurement_overhead_enabled=true`以及惩罚系数`0/1.0`；test数据已生成但未参与方案选择。正式训练的结构审计为`audit_ok=true`。

算法有效性门槛未通过，`decision_supported=false`。validation中，quality-aware的定位RMSE在5个种子中有4个改善，但配对差异均值为`-0.00761 m`，Bootstrap 95% CI为`[-0.01651, 0.00479]`，区间跨越0；有害更新比例只有2/5个种子下降，配对差异均值为`+0.03816`，CI为`[-0.07897, 0.14613]`，同样跨越0。位置误差P95这一辅助指标的区间支持改善，但不足以替代预先定义的RMSE和有害更新双重门槛。与此同时，`J_com`平均增加约`0.00533`，连通性平均下降约`0.01227`，因此不能宣称质量感知MAPPO已经稳定优于baseline或已经学会完整的通信—定位联合权衡。

随后完成了同一批五种子的validation-only质量特征敏感性审计，输出为：

```text
logs/preliminary/python_sim_quality_aware_mappo_quality_head_fix_seed2026/quality_feature_sensitivity/
plots/preliminary/python_sim_quality_aware_mappo_quality_head_fix_seed2026/quality_feature_sensitivity/
```

该审计确认：`feature_semantics_ok=true`、`missing_value_handling_ok=true`、`baseline_quality_invariant=true`、`quality_features_present=true`、`replay_actions_exact=true`、`all_rows_finite=true`、`all_validation_only=true`；质量模型均来自warm-up，反事实动作未执行，test未读取。五个种子下，固定风险惩罚的反事实候选概率均呈单调不升，`none`概率呈单调不降，说明动作头的局部单调机制已按设计生效。但审计决策仍为`scene_confounding_or_observed_correlation_instability`、`decision_supported=false`：反事实单调性是机制正确性证据，不等于重新训练后的论文级性能改善证据。

本轮不扩展到5/7架无人机，不修改EKF、奖励、GAT、信道、碰撞模型或定位目标。后续应先基于这些结果检查质量感知分支的实际测量概率、`none`选择率、测量负载和通信代价；在确认策略行为与机制目标一致前，不应再次调大惩罚系数或宣称算法有效。

## 最新五种子动作—通信开销机制闭环审计（2026-08-26）

针对固定风险惩罚后的最新正式结果，完成了validation-only的动作—通信开销机制审计。审计使用已有checkpoint和episode，不重新训练、不读取test、不修改原始数据。输出为：

```text
logs/preliminary/python_sim_quality_aware_mappo_quality_head_fix_seed2026/policy_mechanism_audit/
plots/preliminary/python_sim_quality_aware_mappo_quality_head_fix_seed2026/policy_mechanism_audit/
```

结构验收全部通过：`audit_ok=true`、`feature_injection_ok=true`、`quality_models_warmup_only=true`、`trace_rows_finite=true`、`replay_actions_exact=true`、`replay_queue_events_exact=true`、`all_validation_only=true`，共记录`29,682`条候选级诊断轨迹，未发现禁止字段。说明质量特征确实进入quality-aware actor，512字节测量负载也确实进入队列重放链路。

机制结果为：`actor_response_status=responsive`，5个随机种子的候选风险—测量概率相关性均达到预设负相关阈值；baseline与quality-aware的动作和队列事件均能精确重放。实际测量率具有明显种子差异：quality-aware相对baseline的测量次数分别为`163/165`、`1187/440`、`264/2104`、`30/2692`和`1683/2128`。只有seed=3035被判定为明显过度测量，因此总体决策为`statistical_evidence_insufficient`、`decision_supported=false`，不是“特征未使用”或“通信队列断裂”。

当前结论是：固定风险惩罚的局部动作机制和通信开销链路均已验证，但策略行为尚未跨场景/种子稳定，不能立即加入新的测量预算或继续调大惩罚系数。下一步应冻结当前代码和`lambda_q=1.0`，基于预注册的新场景或随机种子重复机制审计；在获得稳定的测量率、通信代价和定位收益关系前，不扩展5/7架无人机，也不修改EKF、奖励、GAT或信道模型。

## 新增五随机种子复现实验（7071、8080、9099、10118、11127，2026-08-26）

为检验固定`lambda_q=1.0`的质量感知测量动作头是否具有跨随机种子的可重复性，新增一组独立随机种子复现实验。实验使用3架无人机Moderate场景，`baseline_mappo`与`quality_aware_mappo`均开启512字节测量通信开销，配置为20个warm-up、100个train、20个validation和20个test回合；结果没有覆盖旧的五种子目录：

```text
data/synthetic/python_sim_quality_aware_mappo_quality_head_fix_replication_seedset_7071_11127/
logs/preliminary/python_sim_quality_aware_mappo_quality_head_fix_replication_seedset_7071_11127/
plots/preliminary/python_sim_quality_aware_mappo_quality_head_fix_replication_seedset_7071_11127/
```

10个`scale_3`分支均完成运行并通过`episode-v1.3`验证，每个分支包含140个episode。随后完成validation-only的动作—通信开销机制审计：`audit_ok=true`，动作重放和队列事件重放均无差异，质量模型均来自warm-up，test未用于审计或方案选择，生成了29,154条候选级诊断轨迹。

复现实验的算法判定为`decision=statistical_evidence_insufficient`、`decision_supported=false`。quality-aware actor在5个新种子中均表现出预期的风险—候选测量概率负相关，但实际测量行为不稳定：部分种子测量次数减少，部分种子增加，且仅1/5个种子的定位RMSE改善、仅1/5个种子的有害更新比例下降；跨种子配对区间支持不足。因此当前只能确认“质量特征—动作概率—队列开销”链路存在且可审计，不能宣称质量感知MAPPO已经稳定改善定位或完成联合优化。

本次复现实验不修改`lambda_q`、EKF、奖励、GAT、信道、碰撞模型或无人机规模。另：批量报告中的`branch_configs_equal_except_variant=false`来自审计比较器未排除预期的`measurement_quality_logit_penalty`字段；逐项核对表明两分支除预期的分支名、惩罚系数和输出路径外，其余物理模型及训练配置一致。该报告字段属于审计器待改进项，不应解释为实验物理配置不一致。

## 开启通信开销的预算受控信息增益干预（7071、8080、9099、10118、11127，2026-08-26）

为验证“信息增益测量选择—512字节队列负载—通信KPI—定位收益”之间的权衡，本轮在新增五个随机种子的`baseline_mappo` validation数据上进行了配对干预。测量通信开销开关为`true`：每次有效非空测量动作对应一个512字节等效队列负载；不重新训练MAPPO，不读取test，不修改EKF、GAT、信道、奖励或动作中的带宽和速度分量。

比较处理为：`recorded_mappo`、`unconstrained_information_gain`和`budgeted_information_gain`。代理尺度与测量预算仅由各随机种子的train数据确定，validation用于配对评价；五个base seed是统计独立单位，时间窗口和EKF事件不作为独立样本。

结果目录为：

```text
data/synthetic/python_sim_information_gain_budget_replication_overhead_seedset_7071_11127/
logs/preliminary/python_sim_information_gain_budget_replication_overhead_seedset_7071_11127/
plots/preliminary/python_sim_information_gain_budget_replication_overhead_seedset_7071_11127/
```

结构验收通过：`audit_ok=true`、`overall_ok=true`；15个处理目录全部为`episode-v1.3`，每个包含20个validation episode且逐目录验证通过；`recorded_mappo`精确复现参考validation结果；真实轨迹、障碍物轨迹、速度增量和带宽动作在处理间保持不变；test数据未读取；代理选择不使用真值、LOS/NLOS真值或未来通信结果。测量开销日志共记录`28,954`条事件，非空测量与512字节负载逐项对应，队列和通信字段均有限。

主要validation结果（处理减`recorded_mappo`，以5个base seed为单位）为：

| 处理 | 位置RMSE变化 | (J_{\mathrm{loc}})变化 | (J_{\mathrm{com}})变化 | 95% Bootstrap区间 |
|---|---:|---:|---:|---|
| `unconstrained_information_gain` | `-0.01960 m` | 近似为0 | `+0.01285` | RMSE `[-0.02551,-0.01333]`；(J_{\mathrm{com}}) `[0.00351,0.02180]` |
| `budgeted_information_gain` | `-0.01056 m` | 近似为0 | `+0.00356` | RMSE `[-0.01518,-0.00514]`；(J_{\mathrm{com}}) `[-0.00544,0.01198]` |

因此本轮的结构性结论为：无预算信息增益处理在validation中降低了定位RMSE，同时显著增加了通信代价，`decision_supported=true`且决策为`localization_gain_communication_cost_tradeoff_exposed`，说明通信—定位权衡已被真实暴露。预算处理也降低了RMSE，但其(J_{\mathrm{com}})区间跨越0，尚不能宣称预算方案已经稳定优于参考策略；审计中的`budgeted_effective=false`应予保留。

本轮只建立机制证据，不证明MAPPO最终策略优越性，也不覆盖碰撞、能耗和5/7架无人机扩展问题。后续应冻结当前信息增益代理和实验结果，先决定是否将预算受控测量机制作为独立方法分支进行正式训练；在此之前不调节惩罚系数、不修改EKF或奖励公式。

## 预算约束下的 MAPPO 联合优化实现（2026-08-26）

为把已经暴露的“定位收益—通信开销”权衡交给策略学习，代码新增了兼容旧分支的`budgeted_quality_aware_mappo`。该分支的固定测量预算只由同一随机种子的`baseline_mappo/train`动作日志计算，每架无人机的预算为训练episode非`none`测量次数均值取整；validation和test只复用该预算，不重新拟合。每次`reset`清零已用预算，预算耗尽时动作掩码只保留`none`，合法非空测量每次消耗一个单位，并写入`measurement_budget.csv`。剩余预算比例只进入预算分支的测量动作头，不进入带宽、速度或centralized critic。

本次实现涉及：

```text
src/simulation/environment.py
src/learning/mappo.py
src/experiments/formal.py
src/experiments/quality_aware_mappo.py
src/experiments/pilot.py
tests/test_measurement_budget.py
```

系统模型和方法文档已同步补充固定预算、剩余预算、预算动作掩码、训练集拟合和日志规则；符号表登记了`$K^{\mathrm{meas}}_{\nu,i}$`、`$B^{\mathrm{rem}}_{\nu,i,t}$`和`$M^{\mathrm{baseline}}_{\nu,i,e}$`。旧的`baseline_mappo`和`quality_aware_mappo`默认不启用预算约束，历史结果不覆盖。

先运行单种子Smoke，不直接启动五种子长实验：

```text
run-quality-aware-mappo --output data/synthetic/python_sim_budgeted_quality_aware_mappo_smoke --logs-output logs/preliminary/python_sim_budgeted_quality_aware_mappo_smoke --plots-output plots/preliminary/python_sim_budgeted_quality_aware_mappo_smoke --seeds 2026 --steps 50 --warmup-episodes 2 --train-episodes 5 --validation-episodes 5 --test-episodes 5 --scenario-profile data/synthetic/scenario_screening_seed2026/selected_profiles/moderate.json --ekf-config data/synthetic/python_sim_calibration_seed2026_v4/selected_ekf_config.json --measurement-overhead-enabled --quality-logit-penalty 1.0 --bootstrap-samples 100 --device cpu --force
```

Smoke只验收三分支（baseline、原quality-aware、预算quality-aware）的接口、预算上限、动作掩码、队列日志、loss、协方差和episode-v1.3；通过后才使用`7071、8080、9099、10118、11127`进行正式五种子训练。正式判定仍以seed为独立样本：预算分支必须在不超预算的前提下改善RMSE，且通信代价不显著恶化；否则只能报告定位—通信权衡，不能宣称全面优于baseline。

## 无效测量动作修正后的五种子验收（2026-08-27）

本轮针对“非潜在边被当作有效测量动作”的接口问题进行了修正。测量动作掩码现在只开放当前决策时刻的潜在出边，`none`始终保留；环境对异常非空动作进行防御性归一化，不产生测量通信负载，也不进入EKF。合法非空测量统一记录为一个512字节等效数据包。动作日志新增`invalid_measurement_action`和`measurement_transition_invalidated`两个区分字段：前者表示策略提交的动作在决策时无效，后者表示动作提交后由于运动导致链路离开潜在边，二者不混为一谈。

修正后的五种子、三分支实验使用3架无人机、Moderate场景、20个warm-up、100个train、20个validation和20个test回合，结果保存于：

```text
data/synthetic/python_sim_budgeted_quality_aware_mappo_action_mask_fix_20260826/
logs/preliminary/python_sim_budgeted_quality_aware_mappo_action_mask_fix_20260826/
plots/preliminary/python_sim_budgeted_quality_aware_mappo_action_mask_fix_20260826/
```

结构验收全部通过：15个`scale_3`分支均生成完成，均为`episode-v1.3`，每个分支包含100个train、20个validation和20个test episode，逐目录验证通过；Anaconda Python全量测试为`130 passed`，符号审计通过。独立字段审计发现预动作无效测量动作和掩码违规均为`0`，143645条测量开销记录全部满足潜在边、512字节、请求1包和实际加载1包，99684条预算记录无超预算或负余额。另记录645条运动后拓扑过渡失效事件；这些事件已转换为实际`none`，不属于策略提交的无效动作。

正式算法判定仍为`decision_supported=false`。以budgeted分支相对baseline的validation结果为准，位置RMSE五种子平均差异为`+0.00359 m`（95% Bootstrap区间`[-0.00939, 0.01775]`），有害更新比例平均差异为`+0.07476`（区间`[-0.10917, 0.27236]`）；两项均只有2/5个种子改善且区间跨越0。`J_com`平均增加`+0.01289`，团队奖励平均下降`-0.02459`，因此当前结论是：无效动作接口问题已修正，环境—动作—队列链路结构正确，但质量感知MAPPO尚未证明稳定优于baseline。旧实验目录未覆盖，暂不扩展到5/7架无人机或新的碰撞场景。

## 最新五种子三分支动作—通信开销机制审计

本阶段新增`--branches`参数，默认仍审计历史的`baseline_mappo`和`quality_aware_mappo`两分支；需要时可显式加入`budgeted_quality_aware_mappo`。审计为只读、validation-only，不重新训练、不修改EKF、奖励、GAT、信道、碰撞模型或既有episode与checkpoint。它分别记录决策前潜在边、运动后潜在边、提交时无效动作、运动导致的拓扑过渡失效、最终有效测量、512字节队列负载、通信KPI、`J_com`和协同EKF更新，避免把两类无效性混为一谈。

本轮针对动作掩码修正后的最新五种子结果使用：

```text
run-policy-mechanism-audit --reference data/synthetic/python_sim_budgeted_quality_aware_mappo_action_mask_fix_20260826 --output logs/preliminary/python_sim_budgeted_quality_aware_mappo_action_mask_fix_20260826/policy_mechanism_audit_v2 --plots-output plots/preliminary/python_sim_budgeted_quality_aware_mappo_action_mask_fix_20260826/policy_mechanism_audit_v2 --branches baseline_mappo quality_aware_mappo budgeted_quality_aware_mappo --seeds 7071 8080 9099 10118 11127 --split validation --bootstrap-samples 10000 --bootstrap-seed 2026 --device cpu --force
```

输出为：

```text
logs/preliminary/python_sim_budgeted_quality_aware_mappo_action_mask_fix_20260826/policy_mechanism_audit_v2/
plots/preliminary/python_sim_budgeted_quality_aware_mappo_action_mask_fix_20260826/policy_mechanism_audit_v2/
```

其中`measurement_policy_trace.jsonl`只保留执行期可见质量、动作、边状态和队列字段；训练集仅用于确认warm-up质量模型和预算来源，test完全不读取。验收重点是：提交的非空动作均属于决策时潜在边；运动后失效单独统计；最终有效测量与一个512字节等效数据包一一对应；budgeted分支不超预算；三分支动作、队列重放和所有字段均可复现。审计完成前不扩展5/7架无人机、不调整`lambda_q`，不修改EKF、奖励、GAT或信道模型。

## 最新五种子三分支机制审计结果（2026-08-27）

上述审计已在修正后的代码上完成重放，使用`baseline_mappo`、`quality_aware_mappo`和`budgeted_quality_aware_mappo`三个分支，以及`7071、8080、9099、10118、11127`五个随机种子；只读取validation，未读取test，未覆盖原始episode、checkpoint或既有日志。结果位于：

```text
logs/preliminary/python_sim_budgeted_quality_aware_mappo_action_mask_fix_20260826/policy_mechanism_audit_v2/
plots/preliminary/python_sim_budgeted_quality_aware_mappo_action_mask_fix_20260826/policy_mechanism_audit_v2/
```

结构验收为`audit_ok=true`：提交时无效非空动作数量为`0`，动作和队列事件重放无差异，512字节负载规则及事件数量对应关系通过，预算分支无超预算记录，质量模型来源均为warm-up。15个`scale_3`分支目录均通过`episode-v1.3`验证；全量测试为`133 passed`；符号审计为`ok=true`。

机制报告给出的决策为`post_motion_transition_invalidation`、`decision_supported=false`。这不是提交动作掩码错误：它表示少量合法动作在执行运动后离开了潜在通信边，需要与提交时无效动作分开解释。全局最高分支内比例为`0.2`，出现在seed=`9099`的budgeted分支，分母仅为5次提交动作（1次运动后失效），不能据此直接认定普遍存在严重时序问题。quality-aware和budgeted分支的质量特征—测量概率响应均为`responsive`，但本审计不证明定位性能或算法优越性。

当前后续方向应先针对运动后拓扑过渡失效检查动作生效时序与边界条件；在该机制问题解释清楚前，不扩展到5/7架无人机，不调整`lambda_q`，也不修改EKF、奖励、GAT或信道模型。

## 运动后拓扑失效审计口径校准（2026-08-27）

针对 v2 审计把单个小分母单元的比例误认为全局时序问题，本轮只校准审计口径，不改变环境物理执行逻辑。`measurement_transition_invalidated` 的固定含义为：动作提交时属于潜在出边，但无人机完成本决策窗口运动后，该边离开通信范围；它与提交时的`invalid_measurement_action`严格区分。

v3 只读取已有三分支、五随机种子的 validation 数据，输出到：

```text
logs/preliminary/python_sim_budgeted_quality_aware_mappo_action_mask_fix_20260826/policy_mechanism_audit_v3/
plots/preliminary/python_sim_budgeted_quality_aware_mappo_action_mask_fix_20260826/policy_mechanism_audit_v3/
```

新的判定规则为：运动后失效统计分母是提交的非空测量动作；分母小于`30`的单元只报告原始计数，不触发时序问题判定；只有汇总失效率严格超过`5%`且分母不少于`30`时，才标记为可行动时序问题。报告同时保存 seed×分支、分支、seed 和全局四级汇总。

v3 结果为：全局运动后失效`69/12835=0.5376%`，全局状态为`low_rate_expected_link_breakage`；提交时无效动作数仍为`0`。分支级比例分别为 baseline=`48/6132=0.7828%`、quality-aware=`12/3981=0.3014%`、budgeted=`9/2722=0.3306%`。`seed=9099`的 budgeted 单元为`1/5=20%`，已标记为`small_denominator=1`，不会触发全局时序告警。

v3 的`audit_ok=true`，动作和队列重放、512字节负载对应关系、预算约束、warm-up来源和 validation-only 隔离均通过；未修改`src/simulation/environment.py`、论文文档、episode schema、原始 episode 或 checkpoint。该结果支持冻结当前“动作选择→运动更新→信道计算”的物理语义，下一步回到通信开销约束下的 MAPPO 性能验收。

## 通信开销约束下的 MAPPO 最终验收（2026-08-27）

本轮使用已有的3架无人机、Moderate场景、`7071、8080、9099、10118、11127`五个随机种子结果完成最终验收，不重新训练，不读取test进行方案选择，也不修改EKF、GAT、信道、奖励、`lambda_q`或动作时序。主要比较为`budgeted_quality_aware_mappo`与`baseline_mappo`，`quality_aware_mappo`仅作辅助参考。

结构检查全部通过：15个分支均通过`episode-v1.3`验证；`audit_ok=true`；预算分支无超预算记录；提交时无效测量动作为0；合法测量与512字节等效负载的对应关系通过；质量模型仅来自warm-up；动作和队列事件可以精确重放；test数据未用于选择或调参。运动后拓扑失效为`69/12835=0.5376%`，低于5%的可行动阈值，状态为`low_rate_expected_link_breakage`，不判定为动作时序错误。

以随机种子为独立统计单位，预算质量感知分支相对baseline的validation结果为：位置RMSE平均变化`+0.00359 m`，仅2/5个种子改善，95%区间为`[-0.00939, 0.01775]`；有害更新比例平均变化`+0.07476`，仅2/5个种子下降，95%区间为`[-0.10917, 0.27236]`；`J_com`平均增加`+0.01289`，团队奖励平均下降`-0.02459`，相关区间均跨越0。因此`decision_supported=false`。

最终研究结论为：环境已经真实暴露了“协同定位收益—测量通信开销”的联合优化问题，环境—动作—队列—通信指标链路也已验证；但当前质量感知MAPPO在五个种子下尚未稳定解决该问题，不能宣称算法优于baseline。当前结果应冻结为3架无人机机制和算法验收证据，暂不扩展到5/7架无人机，不新增碰撞场景，不继续调大风险惩罚系数。后续若继续改进，只能以本结果为基线，针对跨种子测量决策稳定性设计一个单独、可预注册的最小策略改动。

## 安全风险场景筛选（2026-08-27）

为验证安全与能耗问题是否在环境中真实存在，本项目新增独立的安全风险场景筛选模块。该模块固定3架无人机、GNSS denied、Moderate通信配置和已选EKF配置，不重新训练MAPPO，也不修改EKF、GAT、信道、奖励或核心定位目标。场景采用安全初始位置、同一飞行高度的动态球形障碍物穿越无人机走廊，以及不使用邻居信息的独立目标跟踪控制器；场景种子是统计独立单位，时间窗口只作机制诊断。

运行入口为：

```text
run-safety-risk-screening --output data/synthetic/safety_risk_screening_seed2026 --logs-output logs/preliminary/safety_risk_screening_seed2026 --plots-output plots/preliminary/safety_risk_screening_seed2026 --scenario-seeds 2026 3035 4044 5053 6062 7071 8080 9089 10098 11107 --steps 50 --force
```

筛选结果保存于：

```text
data/synthetic/safety_risk_screening_seed2026/
logs/preliminary/safety_risk_screening_seed2026/
plots/preliminary/safety_risk_screening_seed2026/
```

本轮结构审计和预设场景门槛均通过：10个场景种子中初始碰撞为0，投影开启下实际碰撞回合比例为20%，最小安全距离分布包含低于2 m安全阈值的风险场景，状态、协方差和能耗均有限。日志同时区分投影前风险与投影后实际碰撞，并保留投影关闭的诊断重放；这组结果只证明场景适合进行安全—能耗算法比较，不代表MAPPO已经改善安全性能。

随后在完全相同且已冻结的安全风险场景种子上，配对运行`baseline_mappo`和完整MAPPO，比较碰撞率、最小安全距离、安全投影、能耗和轨迹长度；场景生成规则未根据算法结果反向调整。

## 安全风险场景上的 MAPPO 配对安全—能耗评估（2026-08-27）

在冻结的安全风险场景上完成了不重新训练的确定性配对评估。评估固定3架无人机、GNSS denied、projection on，使用5个策略种子`7071、8080、9099、10118、11127`；每个策略种子在10个场景种子上推理，统计时先在同一场景内平均5个策略种子，再以10个场景种子作为独立统计单位。未调用`trainer.update()`，未读取test数据。

评估入口为：

```text
run-safety-mappo-evaluation --output data/synthetic/safety_mappo_evaluation_seed2026 --model-root data/synthetic/python_sim_budgeted_quality_aware_mappo_action_mask_fix_20260826 --scenario-root data/synthetic/safety_risk_screening_seed2026/scenarios --logs-output logs/preliminary/safety_mappo_evaluation_seed2026 --plots-output plots/preliminary/safety_mappo_evaluation_seed2026 --policy-seeds 7071 8080 9099 10118 11127 --scenario-seeds 2026 3035 4044 5053 6062 7071 8080 9089 10098 11107 --branches baseline_mappo budgeted_quality_aware_mappo --steps 50 --bootstrap-samples 10000 --bootstrap-seed 2026 --device cpu --force
```

结果位于：

```text
data/synthetic/safety_mappo_evaluation_seed2026/
logs/preliminary/safety_mappo_evaluation_seed2026/
plots/preliminary/safety_mappo_evaluation_seed2026/safety_energy_comparison.png
```

结构验收全部通过：100个episode行、4895个时间步行，`audit_ok=true`；两个分支均正确加载冻结检查点，所有源分支通过`episode-v1.3`验证，预算和状态有限性检查通过，碰撞后回合正确终止且不标记为任务完成。运动后拓扑失效为`69/12835=0.5376%`，属于低比例正常链路断开，不构成动作时序错误。

安全结果显示：baseline实际碰撞为`0/50`，`budgeted_quality_aware_mappo`实际碰撞为`4/50=8%`，对应3个场景种子出现碰撞；碰撞率差异（完整MAPPO减baseline）的场景级Bootstrap 95%区间为`[0, 0.16]`，方向是完整MAPPO更差而不是改善。最小安全距离差异为`-0.1099 m`，其区间`[-0.4928, 0.2427]`跨越0，不能证明安全距离改善。

完整MAPPO的能耗代理平均低`0.0315`、轨迹长度平均短`10.57 m`，但其4个碰撞回合提前终止，因此这两个下降值存在回合长度混杂，不能作为能耗优化或安全改善证据。完整MAPPO位置RMSE略低同样受到提前终止影响。最终报告的判定为`full_mappo_collision_rate_worse`，`decision_supported=false`。

本轮结论是：安全风险场景确实暴露了安全问题，但当前冻结的完整MAPPO检查点没有优于baseline的安全性能，不能宣称算法已经解决安全—能耗问题。下一步应先分析碰撞回合的动作轨迹、障碍物局部观测、投影前风险和投影触发位置，再决定是否设计安全约束训练或增加已预注册的安全场景重复验证；暂不修改奖励权重，也不把提前终止造成的低能耗解释为优化成功。

## 安全碰撞回合的观测—动作—投影归因审计（2026-08-27）

在冻结的安全场景和检查点上完成了只读机制审计。审计使用`baseline_mappo`和`budgeted_quality_aware_mappo`两个分支、5个策略种子和10个场景种子；每个场景内先平均策略种子，场景种子作为独立统计单位。推理使用CPU确定性模式，不调用`trainer.update()`，不读取test，也不修改环境、EKF、GAT、信道、奖励或碰撞逻辑。

结果保存于：

```text
logs/preliminary/safety_mappo_evaluation_seed2026/safety_mechanism_audit/
plots/preliminary/safety_mappo_evaluation_seed2026/safety_mechanism_audit/
```

结构验收为`audit_ok=true`：100个策略—场景—分支episode组合和14,685条时间步记录完整，新增的逐组合覆盖检查通过；提交时无效动作数为0，actor观测不含真值，名义动作—实际动作及投影差值闭合，碰撞回合均立即终止且不标记任务完成，所有状态、协方差和诊断量有限。冻结源目录未被覆盖。

审计发现4个碰撞回合，均为动态障碍物移动后才在实际位置上形成碰撞：运动后、障碍物更新前的实际碰撞标记为0，随后动态障碍物更新并触发环境碰撞检查；碰撞事件均包含该时序对照字段。只有1个碰撞回合在运动前的名义危险判定中被标记，只有1个碰撞回合发生安全投影；因此不能把其余碰撞简单归因于“actor忽略了障碍物”或“投影普遍失效”。

该审计支持的结论是：actor确实获得了局部障碍物特征，但当前完整MAPPO在动态障碍物更新时序下仍存在安全归因风险；碰撞回合提前终止造成的较低能耗、轨迹长度和位置RMSE不能解释为优化成功。下一步应根据该审计结果单独处理安全执行时序或补充预注册安全场景验证，暂不调整奖励权重、重新训练MAPPO或扩展到5/7架无人机。

## 安全碰撞时序反事实审计（2026-08-27）

为区分“策略危险动作”和“动态障碍物更新后形成碰撞”，本项目新增只读入口`run-safety-timing-counterfactual-audit`。审计使用冻结的3架无人机、5个策略种子和10个安全场景种子，在CPU确定性推理下重放`baseline_mappo`和`budgeted_quality_aware_mappo`；不调用`trainer.update()`，不读取test，也不修改环境实际执行顺序、EKF、GAT、信道、奖励或碰撞模型。

实际语义保持为：

```text
运动与当前障碍物投影 → 动态障碍物更新 → 碰撞检查
```

同时对每个决策窗口执行不回写的一步反事实：先按照当前障碍物的速度、边界反弹和时间步预测下一位置，再使用相同名义动作和相同安全投影规则进行离线几何检查。反事实结果不进入后续状态，不消耗随机数；障碍物几何仅用于该离线诊断。

结果保存于：

```text
logs/preliminary/safety_mappo_evaluation_seed2026/safety_timing_counterfactual/
plots/preliminary/safety_mappo_evaluation_seed2026/safety_timing_counterfactual/
```

结构验收为`audit_ok=true`：100个策略—场景—分支回合、14,685条时间步记录和4个碰撞事件完整；场景哈希、动作—运动闭合、障碍物预测闭合、碰撞终止、任务完成标记、有限性、状态不变性和源目录保护检查均通过；当前重放与既有安全机制审计的100个组合签名一致。

4个当前碰撞回合均满足“运动与障碍物更新前未碰撞、动态障碍物移动后形成碰撞”的时序特征，但使用预测障碍物提前投影的一步反事实仍有4个碰撞，未避免任何当前碰撞。因此当前结论为`dynamic_timing_present_but_shadow_not_sufficient`：动态障碍物更新时序确实是碰撞归因的一部分，但不是仅靠单步提前投影即可消除的唯一原因，还需要进一步检查观测范围、执行时序和安全约束设计。该审计不是新的性能比较，不能把提前终止造成的能耗、轨迹长度或RMSE差异解释为算法优势。

主要报告和图：

```text
logs/preliminary/safety_mappo_evaluation_seed2026/safety_timing_counterfactual/timing_audit_report.json
plots/preliminary/safety_mappo_evaluation_seed2026/safety_timing_counterfactual/safety_timing_counterfactual.png
```

## 动态障碍物可观测性与安全包络审计（2026-08-27）

在上述冻结时序重放基础上，新增了只读的`run-safety-observability-audit`。该审计使用5个策略种子、10个安全场景种子和`baseline_mappo`、`budgeted_quality_aware_mappo`两个分支，直接复用validation timing replay；不读取test、不调用训练更新、不改变环境实际状态，也不覆盖已有安全评估和时序审计结果。

审计结果为`audit_ok=true`，处理4895个决策窗口并复现全部4个动态车辆碰撞事件。当前actor的障碍物输入是有限的四维聚合特征：当前最小障碍物距离、box数量、sphere数量和时间步；该特征不包含动态障碍物速度、相对接近趋势、TTC或具体障碍物ID。因此碰撞障碍物虽然在聚合距离特征中可间接反映，但无法被actor按对象识别，4个碰撞事件的主诊断分类均为`observable_but_no_motion_feature`。

安全包络诊断显示，当前投影不是动态障碍物扫掠包络检查；对障碍物进行1、2、3步离线预测后，4个碰撞事件仍均存在动态碰撞风险，说明“只提前一步预测”不能单独解释或消除碰撞。该结果支持先补充执行期可见的动态障碍物运动/接近信息，再做3架无人机、seed=2026的单种子Smoke；在Smoke通过前不重新进行五种子训练，不调整奖励、EKF、GAT、信道或碰撞物理逻辑。

审计输出：

```text
logs/preliminary/safety_mappo_evaluation_seed2026/safety_observability_audit/
plots/preliminary/safety_mappo_evaluation_seed2026/safety_observability_audit/safety_observability_audit.png
```

## 动态障碍物运动趋势观测与单种子 Smoke（2026-08-28）

针对安全可观测性审计发现的“actor 只有四维静态障碍物聚合、缺少动态运动趋势”问题，代码已增加一个默认关闭的动态障碍物运动观测开关。开启后，局部障碍物特征在原有四维之后追加当前动态最小间隙、最大非负接近速率、下一决策窗口预测间隙和运动观测有效性标记；节点输入由历史 37 维扩展为 41 维。新增量只以局部聚合标量进入 actor/GAT，不传递障碍物绝对坐标、对象 ID、真实误差、碰撞标签或未来真值；一步预测也不改变现有“运动—当前投影—动态障碍物更新—碰撞检查”的物理执行时序。

历史实验和 37 维检查点保持兼容，只有显式安全 Smoke 使用 41 维并从头训练。运行入口为：

```text
run-safety-motion-feature-smoke --scenario data/synthetic/safety_risk_screening_seed2026/scenarios/scenario_2026.json --output data/synthetic/safety_motion_feature_smoke_seed2026 --logs-output logs/preliminary/safety_motion_feature_smoke_seed2026 --plots-output plots/preliminary/safety_motion_feature_smoke_seed2026 --uav-count 3 --seed 2026 --steps 50 --warmup-episodes 2 --train-episodes 5 --validation-episodes 5 --test-episodes 5 --measurement-overhead-enabled --quality-logit-penalty 1.0 --device cpu --force
```

该 Smoke 仅验证新观测接口、41 维图输入、测量开销/预算闭环、有限性和 `episode-v1.3` 契约，不构成安全性能改善结论。输出写入独立的 `data/synthetic/safety_motion_feature_smoke_seed2026/`、`logs/preliminary/safety_motion_feature_smoke_seed2026/` 和 `plots/preliminary/safety_motion_feature_smoke_seed2026/` 目录；Smoke 通过后，才可在冻结的新检查点上重新进行十个安全场景的配对评估。

## 41维动态障碍物观测的十场景冻结评估（2026-08-28）

Smoke 完成后新增只读入口 run-safety-motion-feature-evaluation，使用 seed=2026 的两个 41 维冻结检查点，在相同的十个安全场景中进行确定性配对重放。该入口支持 Smoke 的直接目录布局 branch/scale_3，不调用 trainer.update()，不读取 test，也不改变实际环境时序。

结果保存于：

    data/synthetic/safety_motion_feature_evaluation_seed2026/
    logs/preliminary/safety_motion_feature_evaluation_seed2026/
    plots/preliminary/safety_motion_feature_evaluation_seed2026/

结构审计为 audit_ok=true，两个分支均使用 41 维节点输入，动态运动特征、状态、协方差和碰撞终止检查通过。单种子诊断中 baseline 在 10 个场景发生 3 次碰撞，budgeted_quality_aware_mappo 发生 1 次碰撞；但完整分支最小安全距离平均下降约 4.33 m，能耗代理平均增加约 0.0497，因此只能记录为“碰撞次数下降但安全裕度和能耗未同步改善”的诊断信号，不能宣称动态特征已经解决安全问题。该结果仍需五个独立策略种子的 41 维正式重训和安全评估确认。

## 41维动态障碍物观测的正式五种子安全验证（已完成，2026-08-28）

为把单种子 Smoke 扩展为可解释的正式消融，本项目新增独立安全场景池和三分支训练入口 `run-safety-motion-feature-formal-training`。场景池固定分为 train（12026–12045）、validation（13026–13030）和 test（14026–14030），与既有十个安全 hold-out 场景完全不重叠；场景文件、哈希、初始安全状态和分组写入 `manifest.json`。`--force` 只管理本次训练输出，不覆盖已存在的场景池文件。

三分支固定为：`baseline_37d`（37维历史观测）、`baseline_41d`（仅增加四个动态障碍物运动聚合特征）和 `budgeted_quality_aware_41d`（41维观测、执行期质量特征和训练集预算）。三个分支保持 EKF、GAT、信道、队列、奖励、测量开销和动态障碍物物理时序一致；预算只由每个策略种子的 `baseline_37d/train` 测量次数计算，质量模型只在完整分支的 warm-up 中拟合。测试回合只生成并封存，不参与拟合、模型选择或调参。

先用 seed=2026、2/5/5/5 回合完成三分支 Smoke，检查 37/41 维检查点、场景池分配、预算接口、episode-v1.3 和有限性。Smoke 已通过，输出位于：

```text
data/synthetic/safety_motion_feature_formal_mappo_smoke_seed2026/
logs/preliminary/safety_motion_feature_formal_mappo_smoke_seed2026/
plots/preliminary/safety_motion_feature_formal_mappo_smoke_seed2026/
```

正式训练已完成。15 个分支均包含 100 个 train、20 个 validation 和 20 个 test 回合，并全部通过 `episode-v1.3`；全量测试为 163 passed，`main.py audit-symbols` 通过。正式评估使用既有十个冻结安全 hold-out 场景，评估阶段不调用 `trainer.update()`，不读取正式训练 test。统计先在每个场景内平均五个策略种子，再以场景种子作为独立单位计算 Bootstrap 区间。

正式评估输出位于：

```text
data/synthetic/safety_motion_feature_formal_mappo_seedset_7071_11127/
logs/preliminary/safety_motion_feature_formal_evaluation_seed2026/
plots/preliminary/safety_motion_feature_formal_evaluation_seed2026/
```

结构审计为 `audit_ok=true`，但 `decision_supported=false`。碰撞回合数为：`baseline_37d=9/50`、`baseline_41d=7/50`、`budgeted_quality_aware_41d=4/50`。完整分支相对 `baseline_41d` 的场景级碰撞率差异为 -0.06，95% Bootstrap 区间为 [-0.12, 0.00]，未满足严格的“区间上界小于 0”；最小安全距离差异为 -0.082 m，区间为 [-0.514, 0.285]；能耗代理差异为 +0.0399，区间为 [0.0248, 0.0537]。因此完整分支虽然描述性地减少了碰撞回合，但安全裕度变差且能耗显著增加，不能宣称已经解决安全问题。

消融结论为：`baseline_41d-baseline_37d` 的动态运动特征使最小安全距离提高约 1.243 m，但能耗代理也增加约 0.0466；在相同 41 维观测下，质量感知与预算分支尚未证明优于 `baseline_41d`。该结果确认安全风险真实存在，并定位到“动态观测有效性与质量/预算策略仍需改进”的研究问题；碰撞提前终止造成的能耗、轨迹长度和 RMSE 只作为删失描述，不能直接解释为算法优势。当前不扩展无人机规模，也不修改 EKF、奖励、GAT、信道或碰撞物理模型。

## 41维安全机制归因审计、动态扫掠投影最小修正与重新验收（已完成，2026-08-28）

针对上述三分支正式安全结果，新增只读入口 `run-safety-motion-feature-formal-mechanism-audit`，使用5个策略种子、10个冻结安全场景和 `baseline_37d`、`baseline_41d`、`budgeted_quality_aware_41d` 三个分支。审计阶段不调用 `trainer.update()`，不读取正式训练 `test`，以场景种子为独立统计单位，并记录41维动态观测、名义/投影/实际动作、投影原因、碰撞时刻、测量预算和512字节通信负载等诊断字段。

审计结构验收通过：`audit_ok=true`；150个回合、30个场景—分支汇总完整；提交时无效测量动作为0；37/41维输入和动态特征到达actor；动作—运动闭合、预算约束、有限性、场景哈希、warm-up质量模型来源和碰撞终止语义均通过。归因主因按预注册顺序判定为 `projection_coverage_insufficient`：动态特征已经进入actor，碰撞前没有稳定的名义危险动作证据，但动态障碍物更新后仍会使实际位置进入风险区域，原有投影未覆盖该运动过程。

随后只实施一个默认关闭的最小修正 `dynamic_obstacle_swept_projection_enabled`：在安全投影中加入动态障碍物一步扫掠包络检查；不改变EKF、奖励、GAT、信道、队列、碰撞模型和历史37维检查点。修正后的三分支Smoke通过，五种子正式重训和十场景冻结评估也通过结构验收。

修正后的正式评估仍不支持算法安全改善：碰撞回合为 `baseline_37d=13/50`、`baseline_41d=12/50`、`budgeted_quality_aware_41d=10/50`；完整分支相对 `baseline_41d` 的碰撞率差异为 `-0.04`，场景级Bootstrap 95%区间 `[-0.10, 0.00]`，严格判定仍未通过；最小安全距离差异为 `-0.259 m`，区间 `[-0.812, 0.271]`；能耗差异为 `+0.0583`，区间 `[0.0464, 0.0708]`。重新执行的机制审计仍给出 `projection_coverage_insufficient`，但动态特征、动作掩码、预算和数据隔离检查全部通过。因此当前严谨结论是：安全问题已被真实暴露，最小扫掠投影修正已完成并验证，但仍未证明完整MAPPO解决安全—能耗问题；不把碰撞减少或提前终止造成的指标差异写成算法成功。

本阶段报告和图位于：

```text
logs/preliminary/safety_motion_feature_formal_swept_projection_mechanism_audit_seed2026/audit_report.json
plots/preliminary/safety_motion_feature_formal_swept_projection_mechanism_audit_seed2026/safety_motion_feature_formal_mechanism_audit.png
logs/preliminary/safety_motion_feature_formal_swept_projection_evaluation_seed2026/audit_report.json
```

当前应冻结这组结果，暂不扩展到5/7架无人机，也不继续修改核心模型。若继续推进，应把“动态障碍物更新后仍可能穿入当前安全位置”的剩余问题单独预注册为下一轮安全执行/包络实验，不能把它与质量策略、奖励或EKF同时修改。

## 当前最终环境下的最弱独立基线复核（已完成，2026-08-28）

为再次验证导师要求的“问题必须真实存在”，在启用 `dynamic_obstacle_swept_projection_enabled` 的当前最终环境中，使用不读取邻居信息、不进行MAPPO更新的 `independent_fixed` 控制器，重放10个冻结安全场景。`projection_on` 与 `projection_off` 均保留在诊断输出中，正式筛选只采用 `projection_on`；统计独立单位为场景种子，结果写入新目录，未覆盖既有安全筛选和MAPPO评估。

复核通过：10个场景初始碰撞数为0，全部20个（开启/关闭投影各一组）场景状态有限；开启扫掠投影的独立基线发生 `1/10=10%` 碰撞，预投影风险为 `2/10=20%`，`screening_passed=true`。因此当前最终环境仍产生非零且不饱和的安全风险，问题不是由初始非法状态或MAPPO训练过程人为制造的。该基线只证明安全优化对象存在，不证明任何MAPPO分支有效。

复核结果位于：

```text
data/synthetic/safety_risk_screening_swept_projection_seed2026/
logs/preliminary/safety_risk_screening_swept_projection_seed2026/
plots/preliminary/safety_risk_screening_swept_projection_seed2026/safety_risk_screening.png
```

结合当前正式结果，三层证据链为：`independent_fixed` 证明安全问题存在，`baseline_41d` 提供动态观测控制，`budgeted_quality_aware_41d` 检验完整方法。当前完整分支仍未通过安全改善判定，不能将“问题已暴露”表述为“问题已解决”。

## 测量策略跨种子稳定性审计（已完成，2026-08-28）

针对最新 `baseline_mappo`、`quality_aware_mappo` 和 `budgeted_quality_aware_mappo` 五种子结果，新增只读入口 `run-measurement-policy-stability-audit`。审计将训练期 `training_metrics.jsonl` 与既有 validation-only `policy_mechanism_audit_v3` 汇总对齐，不重新训练、不读取 test、不修改 episode、checkpoint 或核心模型。

本次审计输出位于：

```text
logs/preliminary/python_sim_budgeted_quality_aware_mappo_action_mask_fix_20260826/measurement_policy_stability_audit/
plots/preliminary/python_sim_budgeted_quality_aware_mappo_action_mask_fix_20260826/measurement_policy_stability_audit/
```

结构审计为 `audit_ok=true`，但机制结论不是算法有效性通过。三个分支的验证期测量选择率均表现出明显跨种子差异：baseline 范围为 `0.0000--0.9457`，quality-aware 范围为 `0.0014--0.5413`，budgeted 范围为 `0.0017--0.4347`。例如 seed `8080` 的 budgeted 分支有效测量约 `1284` 次，而 seed `9099` 仅约 `4` 次；相应的 `J_com`、团队奖励和定位结果也随测量行为改变。训练日志中的 loss、熵、梯度和裁剪指标均为有限值；动作饱和被记录为诊断信号，但不直接解释为训练崩溃。

当前唯一结论为 `stabilize_measurement_policy_before_core_model_changes`：质量特征已进入 actor、队列开销链路已闭合，但测量策略仍存在显著的训练/跨种子行为不稳定，暂不能宣称质量感知 MAPPO 已稳定解决通信—定位联合优化问题。下一步应冻结核心物理与目标模型，先针对测量动作率和训练稳定性做一个变量的诊断或受控修正，再决定是否重新训练；暂不扩展无人机规模或继续加入动态障碍物变量。

## 训练期测量率约束分支 Smoke（已完成，2026-08-28）

为针对上述“测量策略跨种子波动过大”问题进行单变量稳定性验证，新增独立入口 `run-rate-controlled-quality-mappo` 和分支 `rate_controlled_quality_aware_mappo`。该分支保留质量特征、固定测量预算、测量通信开销、EKF、GAT、信道、奖励和动作头惩罚，仅在 PPO 训练更新中加入基于 baseline `train` 的期望非空测量率约束；baseline 与旧 budgeted 分支默认关闭该约束。

目标测量率严格由每个 seed 的 `baseline_mappo/train/actions.csv` 按实际决策窗口计算，validation 和 test 不参与目标拟合。约束系数固定为 `0.5`、容差固定为 `0.05`，并写入 checkpoint 配置和训练日志。该实现同时保存 `measurement_rate_targets.csv`、`training_rate_constraint.csv`、`validation_policy_metrics.csv`、`seed_comparison.csv`、`paired_contrasts.csv` 和 `audit_report.json`。

已完成 seed=2026 的 3 UAV Smoke（2/5/5/5 回合、50 steps、Moderate、GNSS denied、测量通信开销开启），输出位于：

```text
data/synthetic/measurement_rate_controlled_mappo_smoke_seed2026/
logs/preliminary/measurement_rate_controlled_mappo_smoke_seed2026/
plots/preliminary/measurement_rate_controlled_mappo_smoke_seed2026/
```

三个分支的 15 个 episode 均通过 `episode-v1.3`；约束损失、状态、协方差、NIS、NEES 和预算记录有限且无超预算。Smoke 只有一个随机种子、训练回合数很少，`stability_supported=false` 和 `decision_supported=false` 是预期结果，不能据此宣称新分支有效。下一步才是使用冻结的系数和同一配置完成五个正式随机种子验证。

## 训练期测量率约束五随机种子正式验证（已完成，2026-08-28）

在保持 `measurement_rate_constraint_coef=0.5`、`measurement_rate_tolerance=0.05`、测量通信开销和质量动作头惩罚不变的条件下，完成了 `7071、8080、9099、10118、11127` 五个随机种子、三个分支（`baseline_mappo`、`budgeted_quality_aware_mappo`、`rate_controlled_quality_aware_mappo`）的正式运行。每个分支均包含 `warm-up=20、train=100、validation=20、test=20`；15 个分支的内部 `episode-v1.3` 验证均通过，状态、协方差、NIS、NEES、loss 和测量率记录有限，预算超限为 0。

测量率目标严格来自各随机种子的 `baseline_mappo/train`，质量模型来自 warm-up；报告确认 test 已生成并封存，未用于目标拟合、模型选择或方案判断。正式输出位于：

```text
data/synthetic/measurement_rate_controlled_mappo_seedset_7071_11127/
logs/preliminary/measurement_rate_controlled_mappo_seedset_7071_11127/
plots/preliminary/measurement_rate_controlled_mappo_seedset_7071_11127/
```

结构审计为 `audit_ok=true`，但稳定性和算法判定均未通过：rate-controlled 分支的验证测量率跨种子范围为 `0.39879`，标准差为 `0.18015`，虽略优于旧 budgeted 分支的范围 `0.43095` 和标准差 `0.19651`，但只有 `2/5` 个种子落在对应 baseline 训练目标的 `±0.10` 范围内（要求至少 `4/5`）。相对 baseline，rate-controlled 分支的位置 RMSE 仅 `2/5` 个种子改善，平均变化约为 `-0.00183 m`；有害更新比例仅 `1/5` 个种子下降，平均变化约为 `+0.10804`；`J_com` 平均增加约 `+0.03025`，团队奖励平均下降约 `-0.04125`，`J_loc` 平均变化约为 `+0.000005`。因此 `stability_supported=false`、`decision_supported=false`。

该结果说明训练期测量率约束已经正确接入并能稳定运行，但固定约束尚未稳定地控制五个种子的实际策略行为，也没有证明定位或联合优化性能改善。当前应冻结这组结果，下一步只检查目标测量率统计、约束梯度与预算动作掩码之间的关系；不修改 EKF、奖励、GAT、信道模型，不扩展到 5/7 架无人机，也不把 test 结果用于调参。

## 测量率约束机制审计（已完成，2026-08-29）

新增只读入口 `run-measurement-rate-constraint-audit`，对上述五种子结果重新计算 baseline `train` 的每个 seed/UAV 目标率，并使用 `measurement_quality.jsonl` 的 actor-visible 概率诊断 validation 行为；不重新训练、不读取 test、不修改原始 episode 或 checkpoint。输出位于：

```text
logs/preliminary/measurement_rate_controlled_mappo_seedset_7071_11127/rate_constraint_audit/
plots/preliminary/measurement_rate_controlled_mappo_seedset_7071_11127/rate_constraint_audit/
```

审计显示：5 个 seed、3 个 UAV 的 15 个目标率均可从 baseline/train actions 完整复算；rate 分支训练约束日志有 500 条记录，其中 488 条为非零约束损失，训练期期望率也可由 actor 概率日志重建；分支 manifest 中 baseline/budgeted/rate 的约束开关隔离正确，validation 45 个 seed-branch-UAV 单元有限且完整，test 未读取。当前首要门控为 `target_provenance_error`，原因不是目标率公式被证伪，而是既有 `measurement_rate_targets.csv` 缺少 `base_seed`，15 行目标无法由该汇总文件逐 seed 绑定；同时既有 rate 分支的质量日志仍带有旧版分支标记，属于诊断日志追溯缺口。目标重算和约束行为的机制证据保留，但不能把现有结果解释为已证明约束有效。

本轮已在代码中补充 seed 绑定、逐 UAV 目标/期望率/上下界残差/原始与加权损失，以及 rate 分支质量日志的正确标记；这些改动不改变算法公式。既有正式结果未被覆盖。下一步应先用 seed=2026 做最小 Smoke 验证新增追溯字段，再决定是否需要重新完成五种子正式验证；在此之前不盲目增大约束系数、不修改 EKF/奖励/信道，也不扩展动态障碍物或无人机规模。

## 自适应测量率约束五随机种子验证（已完成，2026-08-29）

针对固定测量率约束在五个随机种子上控制能力不足的问题，新增了 `adaptive_budgeted_quality_aware_mappo` 分支。该分支保留质量特征、固定 baseline-train 测量预算、512 字节测量通信开销、EKF、GAT、信道、奖励和动作空间，仅在训练更新后为每架无人机维护独立的自适应测量率乘子：

```text
dual_initial=0.0
dual_step_size=0.25
dual_max=5.0
measurement_rate_tolerance=0.05
```

乘子只由 train 更新，validation/test 不更新；test 仅生成并封存。旧的 `baseline_mappo` 和 `budgeted_quality_aware_mappo` 行为保持兼容。新分支的 Smoke 和五随机种子正式运行均完成，输出位于：

```text
data/synthetic/adaptive_budgeted_quality_aware_mappo_smoke_seed2026/
logs/preliminary/adaptive_budgeted_quality_aware_mappo_smoke_seed2026/
plots/preliminary/adaptive_budgeted_quality_aware_mappo_smoke_seed2026/

data/synthetic/adaptive_budgeted_quality_aware_mappo_seedset_7071_11127/
logs/preliminary/adaptive_budgeted_quality_aware_mappo_seedset_7071_11127/
plots/preliminary/adaptive_budgeted_quality_aware_mappo_seedset_7071_11127/
```

结构验收通过：正式实验的 15 个分支均有完整 train/validation/test 数据，每个正式分支的 140 个持久化 episode 均通过 `episode-v1.3`；预算超限为0，乘子状态有限且位于 `[0,5]`，训练目标只来自 baseline `train`，test 未参与选择或调参。正式审计为 `audit_ok=true`，且自适应分支满足目标测量率的 seed 数为 `5/5`，超过预设的 `4/5` 稳定性门槛。

但性能判定仍为 `decision_supported=false`，不能宣称算法已经有效。validation 的五种子均值为：自适应分支位置 RMSE `0.2430 m`、baseline `0.2451 m`；自适应分支有害协同更新比例 `0.3660`、baseline `0.2614`；自适应分支 `J_com=0.0722`、baseline `0.0478`；平均团队奖励分别为 `-0.4480` 和 `-0.4213`。RMSE 配对区间为 `[-0.01615, 0.01416]`，跨越0；有害更新比例差异区间为 `[-0.00782, 0.24709]`，也不支持改善。该结果说明自适应乘子改善了测量率稳定性，但没有同时改善协同更新质量和通信—定位综合性能，且通信代价与有害更新比例上升。

因此当前严谨结论是：测量率约束的目标来源、per-UAV 聚合和梯度路径已经验证正确；自适应机制能够控制测量率，但尚未稳定解决通信—协同定位问题。下一步不应继续调测量率乘子或扩展到5/7架无人机，而应冻结这组结果，转向“质量选择—有效 EKF 更新质量—通信代价”的机制分析；碰撞、安全和动态障碍物支线也不与该问题同时修改。正式审计报告为：

```text
logs/preliminary/adaptive_budgeted_quality_aware_mappo_seedset_7071_11127/audit_report.json
```

## 期望测量率—实际动作审计（已完成，2026-08-29）

新增只读入口 `run-measurement-rate-sampling-audit`，用于区分 actor 输出的期望非空概率与环境最终记录的测量动作。审计只重新读取各 seed 的 `baseline_mappo/train/actions.csv` 计算目标，并读取三个分支的 validation 动作和 `measurement_quality.jsonl`；不重新训练、不打开 test episode、不修改原始数据。

输出位于：

```text
logs/preliminary/measurement_rate_controlled_mappo_constraint_trace_seedset_7071_11127/rate_sampling_audit/
plots/preliminary/measurement_rate_controlled_mappo_constraint_trace_seedset_7071_11127/rate_sampling_audit/
```

本次结构验收为 `audit_ok=true`：5 个 seed 的 applied/raw 目标均可复算，raw action 与 actor 概率日志错配为 `0`，动作掩码违规为 `0`，预算违规为 `0`，验证动作模式明确为 `deterministic_argmax`。75 个 raw 动作在环境侧发生了 transition invalidation，被单独计数，未被误判为概率或采样日志错误。validation 不是随机 Bernoulli 采样，因此不使用有限采样误差解释 raw 动作与期望概率的差异。

rate 分支的期望概率接近 raw actor 目标的 seed 比例为 `3/5=0.6`，因此唯一主要判定为 `expected_rate_not_at_target`，而不是 `sampled_expected_rate_gap`。这说明当前约束目标来源和日志链路可追溯，但期望率本身尚未在足够多的 seed 上收敛到目标。下一步只检查 per-UAV 约束聚合、train/validation 分布差异和梯度路径；暂不增大系数、不重新训练、不修改 EKF、奖励或信道模型。

## 测量率约束聚合—梯度路径审计（已完成，2026-08-29）

新增只读入口 `run-measurement-rate-constraint-mechanism-audit`，读取上述五个随机种子的 `baseline_mappo`、`budgeted_quality_aware_mappo` 和 `rate_controlled_quality_aware_mappo` 训练/验证产物；不重新训练、不读取 test、不执行 optimizer step，也不修改原始 episode、checkpoint 或既有日志。输出位于：

```text
logs/preliminary/measurement_rate_controlled_mappo_constraint_trace_seedset_7071_11127/rate_constraint_mechanism_audit/
plots/preliminary/measurement_rate_controlled_mappo_constraint_trace_seedset_7071_11127/rate_constraint_mechanism_audit/
```

## 信息增益—通信代价代理教师分支（五种子正式验证已完成，结构通过但效果未支持）

为解决“测量数量已稳定，但测量质量仍不稳定且通信代价上升”的问题，新增了
`proxy_guided_adaptive_budgeted_quality_aware_mappo` 分支。它在已有自适应测量率和固定 baseline-train 测量预算基础上，使用执行期可见的估计协方差、UWB 测量噪声、消息年龄和链路质量构造信息增益代理与预测通信代价，按

```text
utility = expected_information_gain_norm
          - communication_cost_weight * predicted_communication_cost_norm
```

生成 `none` 始终保留的掩码 teacher 分布，并以 `teacher_coef=0.25`、`temperature=0.25` 的 KL 辅助损失只更新 measurement head；actor body、GAT、带宽/速度头、centralized critic、EKF、奖励和信道模型不变。代理尺度和测量预算仅由同一 seed 的 baseline `train` 数据拟合，validation 用于判断，test 只生成并封存。

代理教师只读审计和 seed=2026 Smoke 已完成，随后使用 `7071、8080、9099、10118、11127` 完成了三分支五种子正式训练与 validation 对比。该分支仍不构成最终算法有效性结论，也不扩展5/7架无人机或安全支线。

正式结果写入：

```text
data/synthetic/proxy_guided_adaptive_quality_mappo_seedset_7071_11127/
logs/preliminary/proxy_guided_adaptive_quality_mappo_seedset_7071_11127/
plots/preliminary/proxy_guided_adaptive_quality_mappo_seedset_7071_11127/
```

结构审计为 `audit_ok=true`、预算超限为 `0`，五个 seed 均满足测量率目标，且训练/validation/test 隔离通过。validation 中代理分支相对 baseline 的位置 RMSE 在 `4/5` 个 seed 改善，均值差异为 `-0.00904 m`，Bootstrap 95% 区间为 `[-0.01692,-0.00288]`；但有害协同更新比例均值增加 `0.09612`，团队奖励均值下降 `0.01930`，综合性能门控为 `decision_supported=false`。因此当前结论是：代理教师能够改善部分定位误差，但尚未稳定解决通信—协同定位问题，不能据此宣称算法整体有效。

审计结果为 `audit_ok=true`，且目标率、manifest 配置、分支隔离和训练日志均可按 `seed × UAV` 追溯。训练约束日志的 500 条更新记录全部通过 per-UAV 字段有限性、非负性、目标绑定和加权损失闭合检查；代码路径也确认是“同一 UAV 内先平均期望非空概率，再对有效 UAV 的惩罚取平均”。由于不同 minibatch 对各 UAV 的样本覆盖数可能不同，不能把三个 per-UAV 汇总均值再次简单平均来重构 minibatch 级 aggregate loss，这种差异被记录为诊断信息而不是误判为聚合错误。

对五个 rate-controlled checkpoint 的 autograd dry-run 全部完成；正约束残差存在时，measurement head 和质量输入路径的梯度均非零，checkpoint 哈希保持不变。train/validation 分布审计显示，约束实现和梯度路径存在，但固定系数 `0.5`、容差 `0.05` 尚未使期望测量率在五个 seed 上稳定贴近目标，当前主要原因判定为 `constraint_active_but_insufficient`，不是目标率绑定错误、per-UAV 聚合错误或梯度断路。该结论只说明测量率约束机制的诊断原因，不等于质量感知 MAPPO 已经取得性能优势；下一步应冻结当前代码和系数，按预注册方案重新设计约束能力验证，不盲目调大系数，也不修改 EKF、奖励、信道或无人机规模。

## 执行期通信代价约束分支（实现与只读审计完成，2026-08-30）

代理教师—EKF 更新质量审计将当前唯一主因判定为 `communication_cost_dominant`：代理排序和教师梯度路径基本正确，定位 RMSE 有改善趋势，但测量次数与通信代价增加并使团队奖励下降。为此新增 `communication_constrained_proxy_guided_mappo`，不修改环境奖励、EKF、GAT、信道或原有代理教师，而是对 measurement head 增加仅在 train 更新的单侧预期通信代价约束。

候选代价定义为 `0.5 + 0.5 * (1 - q)`，其中固定 `0.5` 表示每次有效测量必然产生的一个 512 字节等效负载，剩余部分来自执行期可见链路风险；`none` 代价为 0。每个 seed/UAV 的目标只由 `baseline_mappo/train` 拟合，validation/test 不参与。新分支关闭原双侧测量率约束，保留硬预算、质量惩罚和代理教师；通信乘子范围固定为 `[0,5]`，步长 `0.25`、容差 `0.05`，辅助梯度只进入 measurement head。

只读入口 `run-measurement-communication-cost-audit` 已在旧开发五种子上复算 133324 条 baseline-train 候选记录，确认代价全部有限且位于 `[0.5,1]`、`none=0`、五个 seed 的三架 UAV 目标均可追溯，未读取 validation/test 拟合目标。下一步按顺序执行 seed=2026 的 `2/5/5/5` Smoke；Smoke 和全量测试通过后，才使用预注册新种子 `12136、13145、14154、15163、16172` 完成确认性训练。

## 定位指标口径与 EKF 更新持久性审计（2026-08-31）

为避免把正式实验中逐坐标汇总的 RMSE 与 EKF 审计中的位置向量误差混用，新增了共享指标函数和 validation-only 的 `run-metric-consistency-audit`。新的主指标定义为：先对每个 UAV—时间戳计算
`||p_hat - p_true||_2`，再对这些标量误差计算 RMSE 和 P95；历史的逐坐标 RMSE 保留为 `position_rmse_component_m`，仅用于追溯，不覆盖旧结果。审计使用 5 个训练 seed、3 个分支、每个分支 20 个 validation episode，确认两种口径存在系统性尺度差异：旧口径约为向量范数口径的 `0.57735`，与三维各向同性误差下的 `1/sqrt(3)` 一致。因此后续新的配对比较统一使用向量范数口径。

指标审计输出：

```text
logs/preliminary/communication_constrained_proxy_mappo_seedset_12136_16172/metric_consistency_audit/
plots/preliminary/communication_constrained_proxy_mappo_seedset_12136_16172/metric_consistency_audit/
```

同时扩展了 `run-ekf-update-persistence-audit` 的只读诊断：对每次接受的相对 UWB 更新追踪第 1、2、3、5 个后续诊断窗口，并区分 prediction-only、视觉里程计、机载 UWB 和后续协同 UWB 更新；事件级记录只作机制诊断，不作为独立样本。没有协同更新的 episode，其更新效果比例和条件性持久性指标写为 NA，并通过 `eventful_episode` 与 `cooperative_update_coverage` 单独报告覆盖率。

持久性分解输出：

```text
logs/preliminary/communication_constrained_proxy_mappo_seedset_12136_16172/ekf_update_persistence_decomposition_audit/
plots/preliminary/communication_constrained_proxy_mappo_seedset_12136_16172/ekf_update_persistence_decomposition_audit/
```

该审计的结构检查为 `audit_ok=true`：validation 的 300 个 episode 完整，测试集未读取，没有训练更新或 optimizer step，立即误差差值和线性协方差减少量均公式闭合。当前 focal branch 的机制诊断为 `benefit_not_persistent`：立即有益更新约占 `0.564`，有害更新约占 `0.351`，到第 5 个后续窗口仍保留的有益比例约为 `0.429`；在有协同更新的 seed 汇总中，线性协方差减少事件比例为 `1.0`，但由于保守协方差恢复，单次更新的 `J_loc` 立即变化仍为 0。该结果说明需要继续区分预测传播、更新调度、陈旧/NLOS 更新和协方差恢复的作用，不能据此直接修改奖励、增大权重或宣称 MAPPO 已解决通信—协同定位问题。

本轮新增代码和审计均为只读分析，不改变 EKF、奖励、GAT、信道、检查点或既有实验目录。下一步应在冻结当前结果的前提下，基于持久性分解结果只选择一个可检验的最小修正方向；修正后先做 seed=2026 Smoke，再进行五种子验证。

## 持久信息收益代理 MAPPO（代码已实现，实验待执行，2026-08-31）

针对协同 EKF 单次更新收益在后续预测中衰减的问题，新增分支
`persistent_proxy_guided_quality_mappo`。该分支不删除未知交叉相关条件下的保守协方差恢复，也不修改 EKF、奖励、GAT、信道或动作空间；它只在 measurement head 中加入固定三步预测的持久信息收益代理和代理教师损失。

持久代理比较当前不更新与线性相对测距更新后的预测位置协方差迹差异，通信代价仍单独使用执行期可见链路质量和固定 512 bytes 负载建模。代理、预算和通信代价目标只使用同一随机种子的 baseline `train` 数据拟合，validation 用于诊断，test 不参与拟合、调参或模型选择。代理教师的 `teacher_coef=0.25`、`temperature=0.25` 和通信代价权重 `1.0` 固定，训练与验证结果不会自动替代旧实验。

新增入口：

```text
run-persistent-proxy-audit
run-persistent-proxy-guided-quality-mappo
```

当前状态：核心代码、分支隔离和测试已完成；修复配置透传与扩展尺度兼容性后，全量 `pytest` 已通过 `217 passed`，`main.py audit-symbols` 通过。持久代理只读审计已通过，输出位于：

```text
logs/preliminary/persistent_proxy_guided_quality_mappo_seedset_17181_21227/persistent_proxy_audit/
plots/preliminary/persistent_proxy_guided_quality_mappo_seedset_17181_21227/persistent_proxy_audit/
```

`seed=2026` 的 `2/5/5/5` Smoke 已通过结构验收，输出位于：

```text
data/synthetic/persistent_proxy_guided_quality_mappo_smoke_seed2026_retry4/
logs/preliminary/persistent_proxy_guided_quality_mappo_smoke_seed2026_retry4/
plots/preliminary/persistent_proxy_guided_quality_mappo_smoke_seed2026_retry4/
```

Smoke 只证明接口、训练闭环、预算、通信负载、代理和数据隔离正常，不构成算法有效性结论。下一步才是使用固定配置和预注册种子 `17181、18190、19209、20218、21227` 进行 `20/100/20/20` 正式验证，并以 seed-level 配对区间作出结论。

## 持久信息收益代理 MAPPO 五随机种子正式验证（已完成，结构通过但算法门未通过，2026-08-31）

已使用预注册种子 `17181、18190、19209、20218、21227`，完成 `baseline_mappo` 与 `persistent_proxy_guided_quality_mappo` 两个分支的正式运行；每个分支均包含 `warm-up=20、train=100、validation=20、test=20`，共 10 个 `scale_3` 目录。逐目录 `episode-v1.3` 验证均为 `ok=true`，所有约束、预算、代理、状态和通信代价字段通过结构检查，test 已生成并封存，未用于拟合、模型选择或方案判断。

正式结果位于：

```text
data/synthetic/persistent_proxy_guided_quality_mappo_seedset_17181_21227/
logs/preliminary/persistent_proxy_guided_quality_mappo_seedset_17181_21227_formal/
plots/preliminary/persistent_proxy_guided_quality_mappo_seedset_17181_21227_formal/
```

正式审计为 `audit_ok=true`，但 `decision_supported=false`，因此不能宣称该分支已经稳定解决通信—协同定位问题。相对 baseline，validation 位置 RMSE 仅 `3/5` 个 seed 改善，均值差异为 `-0.00146 m`，Bootstrap 95% 区间为 `[-0.01725, 0.01292]`，跨越 0；有害更新比例在 `5/5` 个 seed 下降，均值差异为 `-0.25142`，区间为 `[-0.39732, -0.09878]`，支持有害更新减少；`J_com` 均值差异为 `-0.02215`，但区间 `[-0.07891, 0.02424]` 跨越 0；团队奖励均值差异为 `+0.01846`，区间 `[-0.02019, 0.05710]` 也跨越 0。

机制上，持久代理减少了测量和有害更新，但在 `17181`、`20218` 两个 seed 的 validation 中没有接受协同更新，在 `19209` 中仅接受 `0.1` 次（按 episode 平均）。这提示当前教师/通信代价路径可能过度抑制测量，或有效协同更新覆盖不足；由于正式结果尚未完成专门机制归因审计，暂不把它定性为唯一代码原因。当前应冻结本轮代码和结果，下一步只做 validation-only 的“代理效用—测量动作—有效 EKF 更新覆盖”机制审计，不调整教师系数、通信约束、EKF、奖励、GAT、信道或无人机规模，也不使用 test 选择方案。

## 持久代理目标相对通信代价修正及新种子复现实验（已完成，2026-09-01）

针对持久信息收益教师将每次测量的固定通信代价与链路风险重复相减、导致部分种子 `none` 选择过多的问题，新增了目标相对通信代价模式：只惩罚超过 baseline-train 通信目标的部分，保留固定测量预算、通信乘子、512 字节测量负载以及原有 EKF、奖励、GAT、信道和队列模型。baseline 分支保持关闭该模式，旧结果未被覆盖。

首先对旧正式结果进行 validation-only 机制审计，结果为 `audit_ok=true`，主要原因一度显示为持久代理覆盖不足；随后完成目标相对代价修正的 seed=2026 Smoke，`2/5/5/5` 回合结构、预算、通信负载、代理、EKF 和 `episode-v1.3` 检查通过。

之后使用新预注册种子 `22236、23245、24254、25263、26272` 完成 baseline 与 `persistent_proxy_guided_quality_mappo` 两个分支的 `20/100/20/20` 正式运行。结果目录为：

```text
data/synthetic/persistent_proxy_guided_quality_mappo_teacher_calibrated_seedset_22236_26272/
logs/preliminary/persistent_proxy_guided_quality_mappo_teacher_calibrated_seedset_22236_26272_formal/
plots/preliminary/persistent_proxy_guided_quality_mappo_teacher_calibrated_seedset_22236_26272_formal/
```

正式结构报告为 `audit_ok=true`、`decision_supported=false`。RMSE 在 `4/5` 个 seed 改善，但均值差异为 `-0.01591 m`，Bootstrap 95% 区间为 `[-0.03098, 0.00273]`，跨越 0；有害更新比例在 `3/5` 个 seed 未恶化，均值差异为 `-0.02082`，区间为 `[-0.06310, 0.02205]`，也跨越 0。`J_com` 均值差异为 `-0.00714`，区间为 `[-0.04194, 0.02822]`；团队奖励均值差异为 `-0.00891`，区间为 `[-0.05084, 0.03302]`。因此不能宣称持久代理 MAPPO 已稳定解决通信—协同定位问题。

新种子机制审计为 `audit_ok=true`，但唯一主要原因是 `statistical_evidence_insufficient`：五个 seed 均未达到原“测量塌缩”触发阈值，说明目标相对代价修正消除了结构性 `none` 塌缩判据；但持久分支相对 baseline 的测量次数比例仍为约 `0.70、10.58、0.73、0.74、1.44`，跨种子异质性仍然很大。审计未读取 test、未调用训练更新或优化器，checkpoint 哈希保持不变；10 个正式分支的 `episode-v1.3` 验证均通过。

当前应冻结这组代码和结果，不继续盲目修改教师、通信约束或 EKF。研究结论应写成：环境中的通信—协同定位权衡已经被暴露，目标相对通信代价修正改善了测量塌缩的结构风险，但 MAPPO 的稳定算法优势尚未得到 seed-level 统计支持。下一步若继续，应预注册更多训练 seed 或场景重复，或专门分析 train/validation 状态分布与测量动作概率的泛化；不得使用 test 反向调参，也不应扩展到 5/7 架无人机或同时改动核心物理模型。

## 通信—协同定位 RMSE 稳定化（2026-09-01，当前执行中）

当前新一轮修正仅包含两项可追溯性改进：正式 episode 指标新增按位置误差向量范数计算的 `position_rmse_norm_m`、`position_p95_norm_m` 和样本数，旧的逐坐标 `position_rmse_m` 保留；无接受协同更新的 episode 在诊断中标记 `update_quality_status=NA`，并单独报告 episode-level `update_coverage_rate`。此外，通信约束分支的代理 teacher 只惩罚超过 baseline-train 目标的通信代价，真实 512 字节队列负载、$J_{\mathrm{com}}$、预算、EKF、奖励、GAT 和信道实现不变。

旧实验目录不覆盖。下一步先运行 seed=2026 的 `2/5/5/5` Smoke，通过后再用预注册新种子 `27281、28290、29309、30318、31327` 做五种子验证；结论只使用 validation 的 seed-level 配对统计，test 只生成并封存。

## 通信—协同定位主线最终审计与可视化（2026-09-02）

针对当前冻结的 `communication_constrained_proxy_mappo_stabilized_init_seedset_27281_31327`，新增 validation-only 的 `run-final-mainline-audit`。审计覆盖五个训练 seed、三个分支（`baseline_mappo`、`proxy_guided_adaptive_budgeted_quality_aware_mappo` 和 `communication_constrained_proxy_guided_mappo`），重新计算位置误差向量范数 RMSE/P95，并同时报告实际测量次数、512 字节负载、通信 KPI、$J_{\mathrm{com}}$、EKF 更新质量和 $J_{\mathrm{loc}}$。训练曲线按 `Update` 对齐，并以 seed 均值及 Bootstrap 95% 区间呈现；validation 指标与训练诊断不混用，test 未读取。

最终输出位于：

```text
logs/preliminary/communication_constrained_proxy_mappo_stabilized_init_seedset_27281_31327/final_mainline_audit/
plots/preliminary/communication_constrained_proxy_mappo_stabilized_init_seedset_27281_31327/final_mainline_figures/
```

结构报告为 `audit_ok=true`：五个 seed、三个分支和每个分支 20 个 validation episode 完整，episode-v1.3、指标一致性、EKF 持久性、512 字节负载不变量、有限性和数据隔离检查均通过。相对 baseline，通信约束主分支的向量范数 RMSE 均值差为 `-0.02026 m`，Bootstrap 95% 区间为 `[-0.04183,-0.00225]`，4/5 个 seed 改善；有害更新比例差为 `-0.04533`，区间为 `[-0.07138,-0.02104]`，4/5 个 seed 改善；但实际通信代价区间为 `[-0.20940,0.12438]`，$J_{\mathrm{com}}$ 区间为 `[-0.01940,0.00696]`，均未满足预注册的非劣效门限。因此 `decision_supported=false`，当前论文结论应限定为：主分支呈现定位和有害更新改善趋势，但尚未证明在通信代价不恶化条件下稳定优于 baseline；不能称为全面解决通信—协同定位问题。

## 3-UAV 主线曲线归因审计（2026-09-02）

新增只读命令 `run-mainline-curve-cause-audit`，仅使用冻结训练诊断和 validation 汇总解释训练奖励、MAPPO loss、通信代价、EKF 更新质量及 $J_{\mathrm{loc}}$ 曲线。该审计不读取 test、不重放环境、不调用训练更新，并在运行前后核对正式 checkpoint 与源表哈希。

当前五种子结果确认：奖励低于 `-0.7` 的 135 个 update 全部对应碰撞；total loss 与 critic loss 的平均相关系数约为 `0.99993`，碰撞 update 的 critic loss 约为非碰撞 update 的 `3.21` 倍；4/5 个 seed 存在超过 `0.10` 的 baseline-train 通信目标到 baseline-validation 实际代价漂移；完整分支 5/5 个 seed 的确定性 argmax 实际代价比概率期望代价高 `0.05` 以上。当前种子上的 EKF 交互复核仍判定为 `visual_odometry_interaction`：约 `90.8%` 的下一有效更新是视觉里程计，至第 5 个窗口仅约 `32.5%` 的即时收益保持。$J_{\mathrm{loc}}$ 同时存在约 `24.8%` 的裁剪饱和和保守协方差恢复断点。因此下一步仍限定在 3 架 UAV，并按“VO 交互—确定性通信代价—单 episode PPO 方差”的顺序一次只验证一个最小修正。

## 视觉里程计质量校准后的 3-UAV 正式验证（2026-09-02）
## 视觉里程计质量校准后的 3-UAV 正式验证（2026-09-02）

按照上述归因结果，已在不修改 EKF、奖励、GAT、信道、队列和动作空间的前提下，开启 `--vo-quality-calibration-enabled`，先完成 `seed=2026` 的 `2/5/5/5` Smoke，再完成五个种子 `27281、28290、29309、30318、31327` 的 `20/100/20/20` 正式训练与 validation 对比。正式实验覆盖 `baseline_mappo`、`proxy_guided_adaptive_budgeted_quality_aware_mappo` 和 `communication_constrained_proxy_guided_mappo` 三个分支，均为 3 UAV、Moderate、GNSS denied、CPU；视觉里程计质量模型只由 warm-up 数据拟合，validation/test 不参与拟合、调参或模型选择。

正式结果位于：

```text
data/synthetic/communication_constrained_proxy_mappo_vo_calibrated_seedset_27281_31327/
logs/preliminary/communication_constrained_proxy_mappo_vo_calibrated_seedset_27281_31327/
plots/preliminary/communication_constrained_proxy_mappo_vo_calibrated_seedset_27281_31327/
```

15 个分支均通过 `episode-v1.3`，每个分支包含 100 个 train、20 个 validation 和 20 个 test episode；结构报告为 `audit_ok=true`，test 未用于判断。通信约束日志包含 500 条 train 更新，乘子只在 train 更新且范围为 `[0, 0.0645]`，512 字节测量负载和预算检查通过。

但算法有效性门仍未通过：通信约束主分支相对 baseline 的位置向量范数 RMSE 均值差为约 `-0.0000075 m`，Bootstrap 95% 区间为 `[-0.02480, 0.02500]`，仅 3/5 个 seed 改善；$J_{\mathrm{com}}$ 均值差约 `+0.00071`，区间为 `[-0.01548, 0.02532]`；有害更新比例均值差约 `-0.02509`，区间为 `[-0.05580, 0.02627]`。因此视觉里程计质量校准改善了执行期可追溯性，但尚未证明 RMSE 在通信代价不恶化条件下稳定改善，不能将 MAPPO 表述为已经稳定解决通信—协同定位问题。

正式报告和图表：

```text
logs/preliminary/communication_constrained_proxy_mappo_vo_calibrated_seedset_27281_31327/audit_report.json
plots/preliminary/communication_constrained_proxy_mappo_vo_calibrated_seedset_27281_31327/communication_constrained_proxy_mappo.png
```

后续若继续推进，只能在冻结本轮结果后单独验证 `rollout_episodes_per_update=4` 的训练方差修正；不与通信约束、奖励或 EKF 修改同时进行，也不扩展到 5/7 架 UAV。

## seed=42 弱强基线 corrected-v2（2026-09-03）

已修复逐窗口协同 EKF 事件漏记与规则日志重复追加，采用每 UAV 累计接受计数增量核对。历史六方法共 120 个 validation 回合冻结重放通过：动作、真值轨迹、估计和奖励一致，原文件哈希未变。该修复修正机制统计，不意味着历史 RMSE 无效。

新协议保持 3 UAV、50 步、CPU、原奖励和保守 EKF；仅运行 seed=42。R 为固定标称 VO 协方差及历史教师参照，P 只统一教师通信代价，T 只进一步改为四回合 PPO 更新并同步学习基线。各方案先 Smoke 后开发实验，最多三个开发配置，不自动搜索参数或扩展种子。当前实现和验收过程不等于 R/P/T 已完成，实际完成状态以各目录的 `audit_report.json` 为准。

```powershell
& 'C:\Users\lenovo\anaconda3\python.exe' main.py run-mainline-baseline-comparison --protocol corrected-v2 --stage replay-audit --experiment-arm reference
& 'C:\Users\lenovo\anaconda3\python.exe' main.py run-mainline-baseline-comparison --protocol corrected-v2 --stage smoke --experiment-arm reference --figure-profile compact-mainline
# 按 R Smoke→R 开发→P Smoke→P 开发→T Smoke→T 开发顺序运行；仅复用同版本已验收阶段
& 'C:\Users\lenovo\anaconda3\python.exe' -m src.experiments.mainline_corrected_suite
```

输出位于 `data/synthetic/`、`logs/preliminary/`、`plots/preliminary/` 下的 `mainline_baseline_comparison_v2_seed42/<阶段>/<方案>/`；已有非空目录拒绝覆盖。旧输出保留。主图精简为累计回报、平均奖励、训练损失、定位误差、通信性能五张（PNG/PDF）；规则方法不伪造 loss。test 只由独立 schema 子进程检查，不用于拟合、候选选择或绘图。最终候选由预先固定的定位/通信/质量门选择；无候选合格时保留 R 参照并明确失败，不宣称跨初始化稳定。

实现对应表：

| 事项 | 实现 | 解释限制 |
|---|---|---|
| 完整窗口事件与累计计数核对 | `src/experiments/cooperative_event_logging.py` | 修复日志，不改变 EKF 更新 |
| 历史冻结重放 | `src/experiments/mainline_corrected.py` | 只重放 validation，原随机机制 |
| R/P/T 与配对共同时间前缀 | `src/experiments/mainline_corrected_stages.py` | 单训练种子的开发选型 |
| 外生随机流 | `src/simulation/random_streams.py` | 新协议使用，旧模式保留 |
| 通信候选代价 | `src/learning/measurement_cost.py` | 代理代价不等于字节或 J_com |
| 槽位 MLP、多回合及辅助梯度 | `src/learning/mappo.py`、`src/experiments/formal.py` | T 的更新次数也变化 |
| 固定监测与五张主图 | `src/experiments/training_monitor.py`、`src/experiments/mainline_compact_figures.py` | 不以曲线好看作为算法成功 |

方法说明见 `sections/3_methods.md` 的 corrected-v2 小节：VO 使用相对位移及时间交叉协方差；邻居 CI 不解决全部交叉相关；J_loc 是裁剪协方差指标，不是 RMSE。当前 UWB 滤波使用仿真 LOS/NLOS 的噪声/偏差处理须作为理想化假设报告。

## 训练稳定化协议 stabilized-v3（2026-09-04，seed=42 开发实验完成）

针对 seed=42、3-UAV 开发实验中累计回报尖峰和 critic loss 尖峰，本协议在不改变物理环境、EKF、GAT、通信模型、动作空间和原始报告奖励的前提下，增加了可追溯的训练稳定化配置：PPO 使用独立的有界学习奖励通道，时间上限截断进行价值 bootstrap 但不跨 episode 递推 GAE，critic 可使用 Huber 损失和 value clipping，并设置目标 KL 早停；rollout 按 4 个完整 episode 聚合后更新。环境仍保留原始 `reward_total`、碰撞事件和四项奖励分解，`learning_reward_total` 只用于优化，不用于替换论文中的报告奖励。

新协议使用分支一致初始化、固定标称视觉里程计协方差和 indexed 外生随机流；原 corrected-v2 及更早结果保持不变。新结果写入：

```text
data/synthetic/mainline_baseline_comparison_v3_seed42/
logs/preliminary/mainline_baseline_comparison_v3_seed42/
plots/preliminary/mainline_baseline_comparison_v3_seed42/
```

执行顺序 `replay-audit → stabilized-v3 smoke → stabilized-v3 development` 已完成。Smoke 和开发实验均通过接口、有限性、奖励闭合、终止掩码、预算和 512 字节负载检查；开发阶段为 `20/100/20/20`，六种方法的 validation 结果和规则基线均已生成。结果位于：

```text
data/synthetic/mainline_baseline_comparison_v3_seed42/development/reference/
logs/preliminary/mainline_baseline_comparison_v3_seed42/development/reference/
plots/preliminary/mainline_baseline_comparison_v3_seed42/development/reference/
```

结构报告为 `audit_ok=true`，但 `development_candidate_eligible=false`；这是单种子描述性比较，不是跨初始化稳定性证明。validation 中完整通信约束方法的向量范数 RMSE 为约 `0.4065 m`，GAT-MAPPO baseline 为约 `0.4080 m`；实际测量负载约 `42.7 kB` 对 `64.6 kB`，但 $J_{\mathrm{com}}$ 约 `0.0942` 对 `0.0376`，因此不能把本次结果表述为全面优于 baseline。精简主图保留累计报告回报、平均报告奖励、优化学习回报辅助面板、MAPPO total/actor/critic loss、定位误差和通信性能；原始曲线不被平滑或删除。

## 相邻窗口定位反馈尺度修正 `core-framework-v4-delta-scale`（2026-09-06）

针对 v4 中使用绝对协方差尺度归一化相邻窗口定位变化的问题，新增独立的
`loc_progress_v2` 反馈模式。它只在同一 episode 内计算
`d_t = u_(t-1) - u_t`，并仅用 warm-up 的相邻窗口差分拟合
`s_delta_u = max(1.4826*MAD(d_t), 1e-3)`；训练反馈再将该比值裁剪到 `[-3, 3]`。
episode 边界不求差分，validation/test 不重新拟合。报告奖励、RMSE、J_loc、J_com、
EKF、GAT、通信模型和旧 v4 结果均保持不变。

新协议入口为：

```powershell
& 'C:\Users\lenovo\anaconda3\python.exe' main.py run-mainline-baseline-comparison --protocol core-framework-v4-delta-scale --experiment-arm loc-progress-v2 --stage smoke
& 'C:\Users\lenovo\anaconda3\python.exe' main.py run-mainline-baseline-comparison --protocol core-framework-v4-delta-scale --experiment-arm loc-progress-v2 --stage development
```

输出分别位于 `data/synthetic/mainline_core_framework_v4_delta_scale_seed42/`、
`logs/preliminary/mainline_core_framework_v4_delta_scale_seed42/` 和
`plots/preliminary/mainline_core_framework_v4_delta_scale_seed42/`。当前实现已完成
单元测试，seed=42 的 Smoke 与 `20/100/20/20` development 均已完成，且对应
`audit_report.json` 为 `audit_ok=true`；v2 统计文件保存 `delta_u_scale`、MAD、
中位数、样本数、来源和下限保护标志，缺少尺度时拒绝运行。当前 development 的
`delta_u_scale=0.001`，其 warm-up MAD 触发了下限保护，说明该反馈信号仍需谨慎解释。
源码的 episode writer 现同时记录原始/裁剪后的定位进度、使用的尺度及尺度来源；本次
已完成的 development 数据保留原始 `position_logdet_previous/current` 与裁剪反馈，
不改写既有实验文件，原始差分可由冻结尺度重构。

本轮还完成了 validation-only 的 VO 交互诊断，未读取 test、未更新参数、未修改
checkpoint。结果显示主方法 accepted UWB 更新后的下一次有效更新中约 `97.3%`
为视觉里程计；beneficial 更新的保持率由 h1 的约 `55.4%` 降至 h5 的约 `42.2%`，
且协方差恢复使事件前后 `J_loc` 变化为 0。因此 v2 修正了相邻窗口反馈的归一化口径，
但没有单独解决协同收益被后续 VO 交互削弱的问题。诊断输出位于：

```text
logs/preliminary/mainline_core_framework_v4_delta_scale_seed42/development/loc-progress-v2/vo_interaction_audit/
plots/preliminary/mainline_core_framework_v4_delta_scale_seed42/development/loc-progress-v2/vo_interaction_audit/
```

本轮结论仍是 seed=42 的开发性证据，不能宣称跨随机初始化稳定或全面优于基线。
以上VO交互结果仅说明相关性，不能单独证明VO造成收益衰减。

## 2026-09-06：固定50步任务与直接报告奖励对照

按更新后的研究决策，当前先执行有限时域对齐。新增 `report_direct` 真正绕过所有学习奖励裁剪与反馈；剩余时间输入actor body与critic，50步末端和碰撞终止均不bootstrap。原有模式保留。两组只训练seed=42的Proposed，规则基线共享直接奖励组train产生的预算与测量机会。

入口：`python -m src.experiments.finite_horizon --stage smoke --arm report_direct --run-name mainline_finite_horizon_seed42/verified`。依次运行direct Smoke、existing Smoke、direct development、existing development；每组development需同版本Smoke通过。输出在data、logs、plots各自的 `mainline_finite_horizon_seed42/verified/` 下；首轮Smoke保留在上一层作为实现调试记录。当前不启动多种子训练。

两组Smoke和100回合development均已完成，结构验收通过。最终全量测试：299 passed、5 skipped；符号审计293项通过。两组初始参数和共同归一化尺度一致，分析不读取test。

直接奖励组相对共同参照：validation RMSE由0.428358降至0.425369 m，P95由0.723173降至0.712582 m；平均测量负载由0增至15308.8 bytes/episode，J_com由0.015546升至0.023216。两组validation碰撞均为3/20。直接奖励组固定监测RMSE从0.380399升至0.381571 m，因此本轮未达到综合改善标准。参照组无接受协同更新，有害更新比例为NA，不代表零有害率或证明直接奖励组更有害。

配对表、共同时间前缀敏感性及有效配置见 `logs/preliminary/mainline_finite_horizon_seed42/verified/paired_comparison/`。两组各五张PNG/PDF及图注已生成；只作为seed=42开发证据，不追加训练或多种子实验。

## 2026-09-06：测量动作密度干预诊断

基于冻结的 `finite_horizon_report_direct` development Proposed checkpoint，完成20个validation回合的只读、配对测量选择干预。保持场景、障碍物、速度动作、带宽和外生过程不变，仅将测量源替换为当前actor可见候选的轮询选择；未读取test、未更新参数。

结果：记录MAPPO在有候选窗口的非none测量比例为0.2167，接受协同EKF更新平均12.3次/回合；轮询选择比例为1.0，接受更新平均60.65次/回合。轮询相对记录策略的RMSE差值为-0.00945 m，Bootstrap 95%区间为[-0.02510, 0.00699]，P95差值为-0.03682 m；`J_loc`完全不变。轮询的`J_com`增加约0.02547，说明增加更新数量伴随通信代价上升。

该诊断支持“当前策略协同测量不足、有效更新覆盖低”是优先问题，但尚不能证明轮询选择的邻居质量优于MAPPO，也不能证明定位改善具有统计确定性。输出见：

```text
logs/preliminary/mainline_finite_horizon_seed42/measurement_action_intervention/
plots/preliminary/mainline_finite_horizon_seed42/measurement_action_intervention/measurement_density_intervention.png
```

干预审计将链路KPI和包事件差异保留为下游结果，不再错误地作为不变性失败；场景、运动动作、带宽和记录回放一致性仍严格检查。下一步应进行等测量预算的邻居选择干预，以区分“测得更多”与“选得更好”。

## 2026-09-06：等测量预算邻居选择干预

在同一20个validation场景上完成等预算配对回放。每个UAV/窗口严格保留Proposed记录的`none`/非`none`模式、带宽和速度动作，只替换测量邻居；质量感知候选的风险阈值只由train拟合，未使用validation/test结果。比较了Recorded MAPPO、等预算轮询和等预算质量感知三组。

两种干预均保持每回合平均29.9次测量、15308.8 bytes负载。相对Recorded MAPPO，轮询的RMSE差值为-0.00538 m，95%区间[-0.02488, 0.01253]，效果不确定；等预算质量感知选择的RMSE差值为-0.00882 m，95%区间[-0.01831, -0.00157]，P95差值为-0.02137 m，95%区间[-0.04662, -0.00298]。质量感知选择使接受协同更新增加3.5次/回合，95%区间[1.75, 5.45]，J_com差值为-0.000225，区间[-0.000647, 0.000035]。

该结果支持：在通信负载和测量密度不变时，邻居选择质量本身具有可观测影响；此前的性能不足不能只归因于测量次数过少。但质量感知选择是基于train拟合的actor可见启发式，不是重新训练的MAPPO，也不构成主方法优越性证明。当前下一步应将这种“质量感知、等预算”的选择反馈正式接入测量动作学习，再做独立Smoke；本轮不自动改写奖励或启动多种子确认。

输出：

```text
logs/preliminary/mainline_finite_horizon_seed42/equal_budget_measurement_intervention/
plots/preliminary/mainline_finite_horizon_seed42/equal_budget_measurement_intervention/equal_budget_measurement_intervention.png
```

## 2026-09-06：`core-framework-v5` 单种子稳定化开发

按“先统一定义，再逐项修正训练反馈，再验证性能”的顺序，完成了 seed=42、3-UAV、Moderate、GNSS denied、CPU、50 步的四阶段开发：`report-clean`、`bounded-collision`、`loc-progress-v2`、`actual-load`，以及定位进度与实际负载的组合臂。每个阶段先完成 `2/5/5/5` Smoke，再完成 `20/100/20/20` development；所有阶段 `audit_ok=true`，test 未用于拟合、选型或绘图。

结果根目录为：

```text
data/synthetic/mainline_core_framework_v5_report_direct_seed42/
logs/preliminary/mainline_core_framework_v5_report_direct_seed42/
plots/preliminary/mainline_core_framework_v5_report_direct_seed42/
```

其中 `actual-load` 与组合 development 由于源码版本在原 Smoke 后发生变化，按签名门禁重新生成到 `actual-load-rerun` 和 `loc-progress-actual-load-rerun`；旧目录保留，不覆盖。组合 development 的五张主图位于：

```text
plots/preliminary/mainline_core_framework_v5_report_direct_seed42/development/loc-progress-actual-load-rerun/
```

`report-clean` 的 Proposed validation 向量范数 RMSE/P95 为约 `0.4157/0.7092 m`，实际负载约 `9.16 kB`；单独加入定位进度反馈后为 `0.3872/0.6480 m`，但负载增至约 `59.65 kB`，说明定位反馈会诱导过度测量。组合反馈将其调整为约 `0.4036/0.6836 m`、`18.97 kB`，相对 `local_only_fixed`（`0.4360 m`）和 `random_cooperative`（`0.4138 m`）具有定位改善，同时低于 Random/Greedy 的约 `31.5 kB` 负载；`budgeted_greedy_ig` 的 RMSE 约 `0.3973 m`，仍是当前即时信息增益较强的规则基线，不能隐藏或宣称 Proposed 全面支配。

训练诊断显示组合臂 25 次 PPO 更新的 loss 有局部波动，但 approximate KL 最大约 `0.011`、95% 分位约 `0.009`，没有灾难性策略发散；累计奖励仍会受碰撞和回合长度影响，不能要求逐回合单调收敛。定位进度尺度的 warm-up MAD 触发了 `0.001` 下限保护，因此该训练反馈的数值敏感性仍需在后续多 seed 验证中谨慎解释。

本轮新增了同 arm 后缀 Smoke 的严格签名查找和阶段依赖查找，避免重试目录被误判为缺失，同时保持旧实验和 checkpoint 不变。当前结论仅是 seed=42 开发性通信—定位权衡，不是跨随机初始化稳定性证明；尚未启动五种子确认实验。

### `core-framework-v8` 报告目标—有限时域—定位进度有序验证（2026-09-06）

针对 v5/v7 中“学习通道与报告目标错位”和“50 步时间上限处理不一致”的问题，新增独立的 v8
协议，不覆盖既有结果。`reference` 臂严格使用
`R_learning = R_report`，保留真实碰撞惩罚和原始奖励，不使用势函数、附加负载反馈或碰撞分量截断；
同时将 50 步作为有限时域终点，加入剩余时间特征并关闭时间上限 bootstrap，使用四个完整 episode
后再更新一次 PPO。只有当直接报告奖励参照完成后，`loc-progress-v2` 臂才在相同有限时域设置下，
叠加同一 episode 内、由 warm-up 相邻窗口 MAD 拟合尺度的定位进度反馈。

v8 输出写入：

```text
data/synthetic/mainline_core_framework_v8_seed42/
logs/preliminary/mainline_core_framework_v8_seed42/
plots/preliminary/mainline_core_framework_v8_seed42/
```

该协议的 `report_plus_feedback` 仅用于定位进度臂：其基础项仍为物理报告奖励，附加项单独记录，
不再沿用历史“碰撞惩罚上限 + 总奖励裁剪”的学习目标。两臂均使用 sampled-action log-prob、
Huber critic、value clipping、actor/critic 分组梯度裁剪和 action-conditioned measurement head。
Smoke 通过只说明定义、数据闭合和训练链路可运行；最终是否改善仍以 validation 的 RMSE、P95、
实际通信负载和更新质量判断，seed=42 不代表跨随机初始化稳定性。

### `core-framework-v8` 时序资格掩码修正开发结果（2026-09-07）

在三链路诊断确认决策间隔为 `0.2 s`、UWB 同一链路更新周期为 `0.8 s` 后，新增可选的
`measurement_update_eligibility_enabled`。开启时，观测动作掩码和环境执行校验共同禁止在
链路更新周期尚未到达时重复选择同一测量源；关闭时仍保持历史行为兼容。该修正不改变 EKF、
GAT、奖励、信道、队列、512-byte 负载或动作空间，只消除已知的无效重复测量。

seed=42、3-UAV 的 Smoke 与 `20/100/20/20` development 均完成，独立输出为：

```text
data/synthetic/mainline_core_framework_v9_timing_seed42/
logs/preliminary/mainline_core_framework_v9_timing_seed42/
plots/preliminary/mainline_core_framework_v9_timing_seed42/
```

Smoke 和 development 的 `audit_ok` 均为 `true`；训练、validation 和 test schema 链路通过，
全量测试为 `311 passed, 5 skipped`，符号审计通过。development 中 Proposed 的训练阶段选中
非 `none` 动作由 `8375` 次降至 `4863` 次，而有效协同更新由 `4051` 次升至 `4269` 次，
选中动作到有效更新的比例约由 `0.484` 提升至 `0.878`，说明主要修复确实作用于时序无效测量。

validation 的 Proposed 结果为 RMSE/P95 `0.42463/0.71183 m`、实际负载约 `4.15 kB/episode`、
`J_com=0.02663`；相较历史 v8 Proposed 的 `0.42537/0.71258 m`、`15.31 kB/episode`，
定位精度基本持平而实际负载明显下降。与同一轮规则方法相比，Proposed 优于 `local_only_fixed`
（RMSE `0.43604 m`）并显著低于 Random/Greedy 的通信负载，但 RMSE 仍略差于
`random_cooperative`（`0.42242 m`）和 `budgeted_greedy_ig`（`0.40343 m`）。因此该轮只能
确认“时序约束减少了无效通信并保留了接近原水平的定位性能”，不能宣称 Proposed 已全面优于
强规则基线或已获得跨随机种子稳定性。`core-framework-v9` 的根目录名称仅用于区分输出；其
有效配置仍记录为 `core-framework-v8` 加时序资格掩码，避免把一次实现修正误称为完整新协议。

### `core-framework-v10-safety` 联合安全执行与奖励曲线终止归因（2026-09-07）

本轮只修正安全执行时序和奖励图标记，不改变奖励权重、EKF、GAT、MAPPO、信道、队列或
物理噪声。v10 使用 `safety_execution_mode=joint_swept_v1`：所有 UAV 的候选运动与动态
障碍物一步预测在同一副本上联合检查，覆盖 UAV--UAV 端点距离、线段交叉、动态障碍物扫掠
路径和边界越界；冲突动作对称投影后再提交，最终仍执行物理碰撞检查。旧 v3/v4/v9 及其他
历史结果和 checkpoint 保持不变。

seed=42、3-UAV、Moderate、GNSS denied、50 步、四回合 rollout 的 Smoke 与
`20/100/20/20` development 均完成，`audit_ok=true`。针对新增几何与终止分类的测试为
`36 passed`；全量测试为 `318 passed, 5 skipped`，符号审计通过。v10 输出位于：

```text
data/synthetic/mainline_core_framework_v10_safety_seed42/
logs/preliminary/mainline_core_framework_v10_safety_seed42/
plots/preliminary/mainline_core_framework_v10_safety_seed42/
```

训练图中的 `X` 只表示 `collision` 终止，空心 `△` 只表示非碰撞 `early_termination`；
达到 50 步的 `time_limit` 回合不标记，缺少可靠原因的 `unknown_short` 只在审计中报告。development
的 Proposed 训练归因表包含 100 个回合，其中 87 个为 `time_limit`、13 个为 `collision`，
没有被错误归为普通提前终止。五张主图保留原始累计报告回报、PPO 学习回报、回合长度、训练
loss 以及 validation 的定位/通信基线比较；该结果用于安全执行和数据语义验证，seed=42 不代表
跨随机初始化稳定性，也不等于已经证明算法性能全面优于基线。

### `core-framework-v11-comm-occluder` 纯通信遮挡区域语义（2026-09-07）

v11 将 `screening_moving_nlos_screen` 明确定义为纯通信遮挡区域：它继续参与
`line_of_sight`、LOS/NLOS 判定、丢包与链路质量计算，但不参与物理碰撞、保护距离违规、
安全投影或动态实体障碍物风险特征。真实建筑物和动态车辆仍然是实体障碍物。v11 同时
开启动态实体障碍物运动特征，使策略能够观察物理动态障碍物的运动/接近趋势；通信遮挡区域
仍只作为通信几何存在。v10 及更早结果保留历史语义，不被重写。

v11 的独立输出目录为：

```text
data/synthetic/mainline_core_framework_v11_comm_occluder_seed42/
logs/preliminary/mainline_core_framework_v11_comm_occluder_seed42/
plots/preliminary/mainline_core_framework_v11_comm_occluder_seed42/
```

该版本用于验证模型语义和碰撞归因；碰撞减少不能自动等同于策略性能全面提升，仍需结合
validation 的 RMSE、P95、实际通信负载和安全指标判断。

### `core-framework-v8` 动作—EKF—延迟反馈链路诊断（2026-09-07）

在不覆盖 v8/v9/v10 历史结果的前提下，新增 `action-linked-feedback` 独立 Smoke 和
`20/100/20/20` development。该分支只把“当前选中的邻居是否产生了可接受的 EKF 更新”转换为
延迟一个决策窗口的训练诊断反馈；反馈使用接受更新前的线性协方差变化和 NIS 一致性，明确标记为
训练信用代理，保守恢复后的本机边际协方差仍是物理报告状态。动作 CSV 同时增加候选数、时序资格数、
时序过滤数以及 `none` 选择原因等可选扩展列，旧 `episode-v1.3` 数据仍可读取。

独立输出位于：

```text
data/synthetic/mainline_core_framework_v8_action_feedback_v2_seed42/
logs/preliminary/mainline_core_framework_v8_action_feedback_v2_seed42/
plots/preliminary/mainline_core_framework_v8_action_feedback_v2_seed42/
```

新增 `action_feedback_diagnostic/` 四张表和一张四面板诊断图，分别检查逐邻居 EKF 结果、
时序掩码下的 `none` 来源、逐回合有效更新—RMSE 关系以及延迟反馈是否真正产生。Smoke 与
development 的 `audit_ok=true`；全量测试为 `319 passed, 5 skipped`，符号审计通过，诊断报告
明确 `test_access_attempts=0`。

development 的 Proposed validation 向量范数 RMSE/P95 约为 `0.4004/0.6869 m`，低于同轮
`local_only_fixed` 的 `0.4360/0.7338 m`、`random_cooperative` 的 `0.4240/0.6982 m` 和
`budgeted_greedy_ig` 的 `0.4055/0.6769 m`，但实际负载约 `29.54 kB/episode`，显著高于
Random/Greedy 的约 `16.67/16.54 kB/episode`；有效更新约 `1.134/window`，也明显高于规则方法。
因此这次链路修正支持“协同更新数量增加可以带来定位收益”，但尚不支持“Proposed 通过更优邻居
选择获得通信效率优势”。六个有向邻居对的线性协方差收益均约 `0.08`，区分度很弱；当前最
根本瓶颈是 EKF 对候选邻居的可区分反馈不足，下一轮应优先使质量/链路差异在不泄露真值的前提下
进入实际测量不确定度或候选动作特征，再单独验证，不能继续靠提高更新频率或反馈系数制造优势。
### `core-framework-v8` 动作—EKF反馈方向修正后的独立结果（2026-09-07）

在上一轮链路诊断中发现总位置协方差迹对各邻居几乎没有区分度后，新增了不使用真值的方向信息
占比：
`local_direction_variance / (local_direction_variance + effective_measurement_variance)`，
并与 NIS 一致性共同构成延迟反馈。该项只改变 Proposed 的训练信用代理，不改变 EKF 更新、保守
协方差恢复、报告奖励或通信模型；v2 结果保留为历史诊断。

当前代码一致的 Smoke/development 独立输出为：

```text
data/synthetic/mainline_core_framework_v8_action_feedback_v3_seed42/
logs/preliminary/mainline_core_framework_v8_action_feedback_v3_seed42/
plots/preliminary/mainline_core_framework_v8_action_feedback_v3_seed42/
```

两阶段 `audit_ok=true`，全量测试与符号审计通过。development 的 Proposed validation
RMSE/P95 为 `0.4318/0.7288 m`，实际负载约 `1.97 kB/episode`；相较之下 Random/Greedy
分别约为 `0.4162/0.6900 m`、`0.4034/0.6792 m`，负载约 `15.46/15.31 kB/episode`。
该修正显著抑制了 Proposed 的测量数量，但也使有效更新约降到 `0.088/window`，定位收益随之
消失。因而本轮执行结果否定了“只加入方向信息反馈即可同时获得定位和通信优势”的假设；当前主要
问题已从“反馈没有延迟”收敛为“可用邻居的可区分质量差异不足，且方向反馈使 Proposed 过度
选择 `none`”。按修正后的动作行口径，validation 中有候选可用的 2709 个动作行里仅 77 个
选择非 `none`，约 `97.2%` 仍选择 `none`；其中仅 33 个动作行是时序掩码强制产生的 `none`。
不能把低负载误写成算法更优，下一步应在固定测量预算的动作干预中验证候选质量是否真正改变
UWB 更新质量，再决定是否保留该反馈。seed=42 仍只是开发证据。

### `core-framework-v9-action-selection` 候选动作选择修正（2026-09-07，Smoke 待执行）

针对 v8 action-feedback-v3 中 validation 合法候选下约 `97.2%` 选择 `none`、而训练阶段非
`none` 比例明显更高的训练—验证动作分布错位，新增独立 v9 协议。v9 不修改 Local EKF、
GAT 消息传递、MAPPO 物理报告奖励、通信队列或 UWB 更新公式，也不删除 `none`。控制臂使用
`report_direct + action_conditioned_v2`；动作头修正臂使用 `report_direct + action_conditioned_v3`，
把当前 actor 可见的候选协方差/测量方差代理、链路质量、AoI、队列和可行性特征逐候选输入，
并给 `none` logit 增加候选集合摘要；反馈臂才使用 warm-up 归一化的、仅对实际选择且被接受
的 EKF 更新释放的相对 `none` 延迟反馈。三臂均保留 sampled-action log-prob、四 episode
rollout、有限时域和原有稳定训练配置。

v9 预定输出根目录为：

```text
data/synthetic/mainline_core_framework_v9_action_selection_seed42/
logs/preliminary/mainline_core_framework_v9_action_selection_seed42/
plots/preliminary/mainline_core_framework_v9_action_selection_seed42/
```

除五张主图外，新增 `action_selection_diagnostic.png`，仅用于比较 train/validation 的
候选可用时非 `none` 率、`none` 概率、最佳候选与 `none` 的 logit 差和候选质量—选择关系；
它不是论文性能证据。计划先分别完成三臂 `2/5/5/5` Smoke，只有动作分布诊断支持动作头
修正后才最多选择一臂进行 `20/100/20/20` development。旧 v8/v3 结果、checkpoint 和
图表不覆盖；test 不参与拟合、选型或动作诊断。当前 v9 代码和测试接口已落地，正式 Smoke
尚未执行，故不能提前宣称 `none` 率或定位—通信性能已经改善。

### Typora 阶段性汇报文档（2026-09-07）

根据最新 seed=42、3-UAV development 结果整理了图文汇报：

```text
sections/advisor_report_seed42_v10.md
plots/preliminary/advisor_report_seed42_v10/
```

报告重点说明 Proposed 相对 Local/Random 的定位改善、相对 Greedy 的差距、实际通信负载偏高，
以及训练奖励尖峰、碰撞终止和单种子证据边界。

### 单无人机最小闭环 sanity 训练（2026-09-09）

为降低起步复杂度，正式实验入口现在支持 `uav_count=1`，并新增 `--scenario-seed` 用于固定
生成场景的几何布局。单无人机运行中图网络只保留虚拟 self-loop，测量动作只有合法的 `none`，
不存在无人机间通信、协同 UWB 或邻居更新，因此该结果只用于验证单智能体 PPO/MAPPO 代码、
EKF、日志、checkpoint 和绘图链路，不能作为协同定位算法有效性结论。

已完成一轮短训练：

```text
data/synthetic/single_uav_sanity_seed2026/
logs/preliminary/single_uav_sanity_seed2026/
plots/preliminary/single_uav_sanity_seed2026/
```

配置为固定场景布局、1 架 UAV、5 步/回合、2 个 warm-up、5 个训练回合、2 个 validation 回合
和 2 个 test 回合；训练更新 5 次，`parameter_delta.delta_l2=0.1399812698`，schema 验证通过。
相关代码测试为 `38 passed`。下一阶段应先在 2 架 UAV、单条可用链路的简单 LOS 场景中恢复协同
通信，再回到 3 架 Moderate 场景；单机 sanity 的 validation/test 不用于反向调参。

#### 单 UAV 通信/定位图的趋势横轴（2026-09-09）

当汇总目录只包含一个 UAV scale 时，`communication_localization_metrics.png` 不再把
`UAV count` 作为横轴，因为单个 scale 只能产生一个点。现在该图读取
`diagnostics/episode_metrics.csv` 的 `train` split，横轴为 `Training episode`，每个点对应一个
训练回合，不做跨回合平均；因此它是训练过程诊断图，不是独立 test 性能证据。包含多个 UAV
scale 时仍保留原来的 `UAV count` 横轴和 test 回合均值比较。横轴模式和数据来源同步记录在
`summary.json` 的 `figure_contract.communication_localization_metrics` 中。

本次横轴修正后的验证输出位于：

```text
logs/preliminary/single_uav_sanity_seed2026_trend/
plots/preliminary/single_uav_sanity_seed2026_trend/
```

#### 单 UAV episode 诊断图（2026-09-09）

新增 `src/experiments/single_uav_diagnostics.py` 和 `plot-single-uav-diagnostics` 入口，直接读取
一个已记录 episode 的 `trajectory.csv`、`ground_truth.csv`、`estimates.csv`、`rewards.csv` 和
`obstacles.csv`，不重新运行仿真或修改策略。当前已对训练后 test episode `episode_500001` 生成：

```text
plots/preliminary/single_uav_sanity_seed2026/single_uav_diagnostics_test_episode_500001/
```

其中包括定位误差时间序列、XY/XYZ 飞行轨迹和 reward 时间序列；横轴均为仿真时间（s）。复现命令为：

```text
python main.py plot-single-uav-diagnostics data/synthetic/single_uav_sanity_seed2026/scale_1/test/episode_500001 --output plots/preliminary/single_uav_sanity_seed2026/single_uav_diagnostics_test_episode_500001
```

该图组是单回合、单种子的诊断结果，不支持收敛性、泛化性或协同通信性能结论；单 UAV 没有
UAV-to-UAV 链路，packet loss、delay 和 AoI 不能解释为学习到的无人机间通信行为。

#### 单 UAV 长回合训练与完整诊断（2026-09-09）

在不改变 UAV 数量和场景几何的前提下，完成了固定 `scenario_seed=2026` 的长回合训练：1 架 UAV、
100 步/回合（每回合最长 20 s）、10 个 warm-up、100 个训练回合、20 个 validation 回合和 20 个
test 回合，MAPPO 更新 100 次。运行命令为：

```text
run-preliminary --output data/synthetic/single_uav_long_seed2026 --uav-counts 1 --seed 2026 --scenario-seed 2026 --steps 100 --warmup-episodes 10 --train-episodes 100 --validation-episodes 20 --test-episodes 20 --device cpu --force
```

结果和图表分别保存于：

```text
data/synthetic/single_uav_long_seed2026/
logs/preliminary/single_uav_long_seed2026/
plots/preliminary/single_uav_long_seed2026/
```

其中 `training_cumulative_reward.png` 使用每个训练回合的原始累计回报，`mappo_training_loss.png`
使用每次 PPO update 的 total/actor/critic loss；test episode `episode_500001` 的
`single_uav_diagnostics_test_episode_500001/` 包含完整 XY/XYZ 轨迹、定位误差时间序列和 reward
时间序列。schema 验证通过，所有回合均记录 100 个时间步。

本轮只是单 UAV 工程诊断：训练损失下降，但累计回报没有改善，test 轨迹也没有到达目标，因此不能
据此宣称策略已经收敛或定位性能已经提升；单 UAV 的通信指标保持为无邻居链路下的占位诊断值。

#### 单 UAV 定位优先基线与多 UAV 通信回归（2026-09-09）

新增显式的 `single_uav_learning_mode`。默认值 `off` 保持旧实验兼容；本轮使用
`loc_goal_v1`，将单 UAV 的 PPO 学习奖励与历史 `reporting_reward` 分离。学习奖励只使用 EKF
位置协方差 log-det 的改善、EKF 估计目标距离的改善、能耗和碰撞项，其中定位进度权重为 `0.8`、
目标进度权重为 `0.2`，并限制在 `[-1, 1]`。进度尺度仅由 warm-up 的相邻时间步拟合，并冻结到
train/validation/test；packet loss、delay、AoI 等 UAV-to-UAV 指标继续记录，但在单 UAV 中标记为
`communication_applicable=false`，不进入学习奖励。

首轮实验使用固定 `scenario_seed=2026`、1 架 UAV、100 步/回合、10 个 warm-up、100 个训练回合、
20 个 validation 回合和 20 个 test 回合，输出为：

```text
data/synthetic/single_uav_loc_goal_v1_seed2026/
logs/preliminary/single_uav_loc_goal_v1_seed2026/
plots/preliminary/single_uav_loc_goal_v1_seed2026/
```

本轮 schema 校验通过；训练日志包含 100 个 PPO update，actor gradient norm 100/100 为有限正值，
actor loss 约为 `-0.1038` 至 `0.0313`，KL、entropy、clip fraction 均有记录。测试 episode
`episode_500001` 生成 100 个时间步的 XY/XYZ 轨迹、定位误差、EKF 协方差、目标距离和奖励分解图。
当前累计学习奖励和目标距离仍显示策略尚未收敛，因此该单种子结果仍是奖励设计与训练流程验证，
不是最终性能结论。

另完成 3 UAV 小规模回归：`communication_applicable=true`、18 条候选/有效链路记录、测量与带宽
动作掩码恢复为多机形式，且通信奖励重新进入学习奖励。回归输出为：

```text
data/synthetic/three_uav_comm_regression_seed2026/
```

#### 单 UAV 导航奖励 v2（2026-09-10）

按照单无人机奖励设计，新增并启用 `loc_goal_v2`。该模式只作用于 1 UAV 的 PPO 学习通道，历史
`reporting_reward`、多 UAV 通信奖励路径和 `loc_goal_v1` 均保持不变。训练奖励为：

```text
r_learning = goal_progress + goal_arrival_bonus
              - localization_cost - collision_cost - energy_cost
goal_progress = alpha * (estimated_goal_distance_previous - estimated_goal_distance_current)
```

其中 `alpha=1.0`、定位不确定度代价权重为 `0.2`、首次进入估计目标半径 `1 m` 时给予一次性
`+1.0` 奖励；进度奖励使用米制距离差，避免 warm-up 的小尺度和 `[-1, 1]` 裁剪把大多数动作
都压成同一个负奖励。归一化的 goal-progress score 仍保留在日志中，仅用于诊断。

正式运行保持固定场景和 1 UAV 不变，配置为 100 步/回合、10 个 warm-up、100 个训练回合、20 个
validation 回合和 20 个 test 回合：

```text
data/synthetic/single_uav_nav_goal_v2_raw_seed2026/
logs/preliminary/single_uav_nav_goal_v2_raw_seed2026/
plots/preliminary/single_uav_nav_goal_v2_raw_seed2026/
```

schema 验证通过，100 次 PPO update 和全部 140 个回合均有完整日志。训练学习回报由第 1 回合约
`-68.07` 改善到第 100 回合约 `-51.96`，说明目标进度项已经进入学习信号；但 test 的固定策略仍
表现为先靠近目标、随后越过目标继续飞行，20 个 test 回合尚未触发 1 m 到达奖励。因此本轮证明
的是奖励通道和诊断链路已正确接入，不是策略已经收敛。该结果仍属于单种子工程诊断，不能替代多
种子性能实验或多 UAV 协同通信结论。

#### 单 UAV 稳定到达奖励与近目标减速（2026-09-10）

在 `loc_goal_v2` 上继续执行“先接近、再减速、稳定停靠”的奖励设计。新的学习奖励为：

```text
r_learning = goal_progress + stable_goal_bonus
              - localization_cost - near_goal_speed_cost
              - collision_cost - energy_cost
goal_progress = alpha * (estimated_goal_distance_previous - estimated_goal_distance_current)
near_goal_speed_cost = gamma * ||v_t||, when estimated_goal_distance_current <= 5 m
```

本轮稳定到达判据为：估计目标距离不超过 `1 m`，且无人机实际速度不超过 `0.5 m/s`。只有首次满足
这两个条件时才发放一次性 `+5.0` 稳定到达奖励；`loc_goal_v2` 同时将该状态作为成功终止条件。
旧版 `loc_goal_v1`、默认奖励路径和多 UAV 通信路径不变。诊断图中的目标距离曲线、实际速度曲线、
`1 m` 稳定目标线、`5 m` 近目标线和 `0.5 m/s` 稳定速度线均来自对应 episode 日志。

正式运行仍固定为 1 UAV、`scenario_seed=2026`、100 步/回合、10 个 warm-up、100 个训练回合、
20 个 validation 回合和 20 个 test 回合，输出为：

```text
data/synthetic/single_uav_nav_goal_v2_stable_seed2026/
logs/preliminary/single_uav_nav_goal_v2_stable_seed2026/
plots/preliminary/single_uav_nav_goal_v2_stable_seed2026/
```

本轮 schema 验证通过，140 个 episode 和 100 次 PPO update 均有完整记录。训练学习回报由第 1 回合
约 `-68.07` 改善到第 100 回合约 `-49.06`，训练期间最优回报约为 `-19.24`；但固定策略在 20 个
test 回合中没有进入 `1 m` 目标半径，也没有触发稳定到达奖励，全部因时间上限结束。全部 140 个
回合的最小估计目标距离为约 `1.69 m`，说明近目标速度惩罚已经被记录，但当前 100 回合、单种子配置
仍未学会“减速并稳定停靠”。因此本轮完成了奖励和成功判据的代码修改及可复现实验，但不能把该结果
表述为导航已收敛；下一步应在保持判据不变的前提下做多种子或更长训练，并单独报告稳定到达率。

#### 单 UAV 导航失败分析与制动敏感性实验（2026-09-10）

为区分“不会接近目标”和“接近后不会刹车”，新增了跨 episode 的导航分析：每个回合记录最小
估计目标距离、进入近目标区域的指标、进入目标半径的指标、稳定到达指标，以及进入近目标区域时
的速度；同时从原始时间步日志绘制速度—目标距离关系 `v(d)`。分析结果不做平滑、重采样或跨回合
平均，速率的分母是各 split 的 episode 数。

基线 `single_uav_nav_goal_v2_stable_seed2026` 的结果为：训练集进入 `5 m` 的比例为 `5/100`，
进入后的平均速度约 `4.22 m/s`；validation 和 test 进入 `5 m` 的比例均为 `0/20`。进入 `5 m`
的训练回合，其首次进入速度约为 `3.77 m/s`，明显高于 `0.5 m/s` 稳定速度阈值；进入 `1 m`
和稳定到达的比例均为 0。这确认主要问题是制动信号不足。

在不改变 UAV 数量、场景种子、步数和训练回合数的情况下，完成了三组制动敏感性对照：

```text
single_uav_nav_goal_v2_braking_seed2026
  near_goal_radius=8 m, speed_penalty_weight=0.4, learning_reward_clip=1.0

single_uav_nav_goal_v2_braking5_seed2026
  near_goal_radius=5 m, speed_penalty_weight=0.4, learning_reward_clip=1.0

single_uav_nav_goal_v2_braking_shaped_seed2026
  near_goal_radius=8 m, speed_penalty_weight=0.4, learning_reward_clip=3.0
```

为避免阈值混淆，`braking5` 与基线都以 `5 m` 作为近目标半径：test 进入 `5 m` 的比例由 `0/20`
变为 `2/20`，最小目标距离由约 `7.60 m` 降到约 `4.63 m`；但进入时平均速度仍约 `4.71 m/s`，
稳定到达仍为 `0/20`。`braking` 提前在 `8 m` 开始惩罚，test 进入 `8 m` 为 `2/20`，进入时平均速度
约 `3.70 m/s`，同样没有稳定到达。`braking_shaped` 放宽学习奖励裁剪后未进入 `8 m`，test 最小
目标距离约 `13.23 m`。因此当前结论是：增大近目标速度惩罚能让少量回合更接近目标，但还没有形成
有效的低速制动策略；直接放大学习奖励幅值反而破坏导航行为，三组参数均不覆盖默认配置。

新增分析代码和命令为：

```text
src/experiments/single_uav_navigation_analysis.py
main.py analyze-single-uav-navigation <scale_1>
```

上述各组结果的 CSV 和 `v(d)` 图均保存在对应的 `logs/preliminary/` 与 `plots/preliminary/` 目录中，
均属于单种子固定场景诊断，不能替代多种子成功率或多 UAV 协同通信实验。

#### 单 UAV 距离—期望速度制动策略（2026-09-10）

针对“接近目标后不会提前刹车”，在 `loc_goal_v2` 中新增了可选的
`distance_target_speed` 模式。该模式在 `8 m` 外允许 `6 m/s` 巡航，在 `1--8 m` 之间将目标速度
从 `6 m/s` 线性降至稳定速度阈值 `0.5 m/s`，只惩罚实际速度超过该距离相关目标速度的部分：

```text
run-preliminary --output data/synthetic/single_uav_nav_goal_v2_distance_target_speed_seed2026 --uav-counts 1 --seed 2026 --scenario-seed 2026 --steps 100 --warmup-episodes 10 --train-episodes 100 --validation-episodes 20 --test-episodes 20 --single-uav-learning-mode loc_goal_v2 --single-uav-speed-control-mode distance_target_speed --single-uav-near-goal-radius-m 8.0 --single-uav-near-goal-speed-penalty-weight 0.4 --single-uav-target-speed-far-mps 6.0 --learning-reward-clip 1.0 --device cpu --force --quiet
```

```text
v_target(d) = 6.0                         , d >= 8 m
              0.5 + (d - 1) * 5.5 / 7     , 1 m < d < 8 m
              0.5                         , d <= 1 m
r_speed = -0.4 * max(0, actual_speed - v_target(d))
```

实验只改变速度控制模式，保持 `uav_count=1`、`seed=2026`、`scenario_seed=2026`、100 步/回合、
100/20/20 个训练/validation/test 回合和 `learning_reward_clip=1.0` 不变。与相同 `8 m` 半径、
固定速度惩罚的对照相比，新模式训练集最小目标距离约为 `1.09 m`，进入 `8 m` 为 `14/100`，
近目标样本的平均超速量约 `0.32 m/s`；但 validation/test 均为 `0/20` 进入 `8 m`，test 最小
目标距离约 `8.06 m`，稳定到达仍为 `0/20`。因此该奖励改善了训练集接近行为，但尚未形成可泛化
的刹车策略，不能据此认为单 UAV 导航已经收敛。默认模式仍为 `fixed_near_goal`，后续应优先检查
动作空间的减速可控性并进行多随机种子验证。

本轮新增日志字段包括 `single_uav_target_speed_mps`、`single_uav_speed_excess_mps` 和
`single_uav_speed_control_mode`；`v(d)` 图会叠加实际目标速度曲线。输出位于：

```text
data/synthetic/single_uav_nav_goal_v2_distance_target_speed_seed2026/
logs/preliminary/single_uav_nav_goal_v2_distance_target_speed_seed2026/
plots/preliminary/single_uav_nav_goal_v2_distance_target_speed_seed2026/
```

#### 单 UAV 动作空间与制动能力审计（2026-09-10）

针对距离—期望速度实验中“能接近目标但未稳定到达”的现象，进一步检查了
`Actor 输出 → 动作映射 → 运动模型 → 速度 → 位置` 的完整链路，并直接覆盖重新运行了上面的
`single_uav_nav_goal_v2_distance_target_speed_seed2026` 结果。审计确认当前动作空间已经支持有效制动，
不需要额外添加离散 `brake` 动作：动作是三维连续速度增量 `Δv`，每个 `0.2 s` 决策窗口的最大范数为
`0.6 m/s`（等效最大加速度 `3.0 m/s²`），允许负向速度增量；运动模型按
`v_next = clip(v_pre + Δv_applied, v_max)` 更新速度。

新增 `actions.csv` 字段记录动作前后速度、命令/实际速度增量、纵向制动分量、实际减速量、制动是否被速度上限阻塞，
并生成 `braking_action_response.png` 与 `braking_action_summary.png`。制动定义为实际 `Δv` 在动作前速度方向上的负投影，
有效制动定义为动作后速度严格低于动作前速度；制动率的分母是动作样本数，近目标制动率的分母是带有精确时间戳匹配的近目标动作样本数。

最终 1 UAV、固定场景、`seed=2026` 结果显示：训练集 `1427/10000=14.27%` 的动作产生制动，制动动作中
`1297/1427=90.89%` 实际降低了速度；进入 `8 m` 后的动作样本为 `170`，其中 `12` 次制动，且有效率为 `10/12=83.33%`。
因此动作空间和运动模型确实能够执行减速，当前主要瓶颈不是“没有刹车动作”，而是策略在近目标区域主动制动的频率不足以及验证/测试策略没有泛化到目标附近。
训练集进入 `8 m` 为 `14/100`、最小距离约 `1.09 m`，但 validation/test 均为 `0/20` 进入 `8 m`，稳定到达仍为 `0/20`。
本结果仍属于单种子工程诊断，不能表述为导航已经收敛。

本轮新增或更新的关键产物为：

```text
src/simulation/motion_model.py
src/experiments/pilot.py
src/experiments/single_uav_diagnostics.py
src/experiments/single_uav_navigation_analysis.py
tests/test_motion_model.py
tests/test_single_uav_navigation_analysis.py
```

#### 单 UAV 制动课程与目标速度动作匹配验证（2026-09-10）

为解决“策略能否及时、持续、稳定地减速到目标”，在不改变单 UAV 固定场景和动作空间的前提下，
加入了两类可审计的训练信号：训练阶段交错采样 `3/5/8 m` 局部制动起点，以及将实际执行的 `Δv`
与距离相关目标速度对应的期望 `Δv` 进行匹配。验证与测试仍从原始远距离起点开始，稳定到达仍定义为
估计距离不超过 `1 m`、实际速度不超过 `0.5 m/s` 并连续保持 `5` 步。

最后一轮 `targetvelocity3` 结果仍未解决该问题：test 进入 `8 m` 为 `20/20`，但进入 `1 m` 为
`0/20`、稳定到达为 `0/20`；近目标动作样本中制动为 `0`，近目标平均速度约 `3.83 m/s`，
而目标速度约 `3.09 m/s`。因此本轮结论是：动作空间和运动模型支持制动，奖励和动作匹配信号也已记录，
但当前 PPO 策略仍未把近目标制动泛化到确定性验证/测试轨迹。本次按要求停止继续试验，不能把该版本称为
“已稳定到达”。

对应产物位于：

```text
data/synthetic/single_uav_nav_goal_v2_braking_distance_targetvelocity3_seed2026/
logs/preliminary/single_uav_nav_goal_v2_braking_distance_targetvelocity3_seed2026/
plots/preliminary/single_uav_nav_goal_v2_braking_distance_targetvelocity3_seed2026/
```

本轮关键实现位于 `src/simulation/environment.py`、`src/experiments/formal.py`、`src/cli.py`、
`src/schema.py` 和 `src/experiments/pilot.py`；新增制动课程测试为
`tests/test_single_uav_braking_curriculum.py`。针对性测试通过，`git diff --check` 无格式错误。

#### 单 UAV 透明制动安全投影验证（2026-09-10）

为验证“策略能否及时、持续、稳定地减速到目标”，在上一轮 `targetvelocity3` 配置基础上增加了
可选的低层制动安全投影。策略仍输出原始三维连续 `Δv`；当当前或下一步预测速度超过停车距离包络，
或物理位置进入目标半径时，执行层才把动作投影到允许的目标速度，并在 `actions.csv` 中同时保存
原始动作、实际动作、投影原因、目标速度和投影 Δv。该机制不增加离散 `brake` 动作，也不改变场景几何。

本轮固定 `uav_count=1`、`seed=2026`、`scenario_seed=2026`、100 步/回合和 100/20/20 个
训练/validation/test 回合。针对性单元测试为 `21 passed`，实验数据 schema 验证为 `scale_1=True`。
结果如下：

```text
test 进入 8 m：20/20
test 进入 1 m（按 EKF 估计距离）：3/20
test 稳定到达：3/20
test 近目标平均超速量：约 0.0026 m/s
test 近目标有效制动动作：61 次
```

这说明低层执行闭环已经把“进入目标附近后仍高速冲过”的问题显著压下去了；但是仍有 17/20 个
测试回合没有按估计距离进入 `1 m`，因此不能把本轮称为“策略已经稳定到达”。诊断还发现，部分
回合的物理位置已经进入目标半径并被执行层保持，但 EKF 估计距离仍大于 `1 m`，暴露出安全投影
状态源与成功判据状态源不一致的问题。其余主要瓶颈已从“不会刹车”转移为“远距离导航/末端目标捕获
和估计误差下的成功判定”。

本轮结果目录为：

```text
data/synthetic/single_uav_nav_goal_v2_braking_distance_safety_projection_seed2026/
logs/preliminary/single_uav_nav_goal_v2_braking_distance_safety_projection_seed2026/
plots/preliminary/single_uav_nav_goal_v2_braking_distance_safety_projection_seed2026/
```

下一次若继续，应先统一安全投影与成功判据的状态源，并在 `8 m` 目标捕获区增加“动作必须保持目标
方向进度”的末端约束，再做多随机种子验证；在此之前不应进入多 UAV 通信结论。

#### 单 UAV 独立边界惩罚与局部奖励接口（2026-09-10）

根据附件要求，在现有单 UAV 训练链路中加入了可选的独立边界惩罚，默认值为 `0.0`，本轮新实验使用
`boundary_penalty_weight=0.5`。边界物理处理保持不变：动作越界后位置投影回合法区域、速度按实际位移
修正、回合继续；`boundary_projected`、`safety_projected`、`action_clipped` 和 `collision` 仍是互相独立
的诊断量。边界惩罚按每架 UAV 每个 step 计算一次，同一步多轴越界不重复计罚：

```text
r_boundary(i,t) = -0.5, boundary_projected(i,t)=True
                  0.0, otherwise
r_learning(i,t) = base_learning_reward(t) + r_boundary(i,t)
```

该项在现有 learning reward 的最终裁剪前加入；单 UAV learning channel 继续沿用原有 bounded clip，
非单 UAV 的既有 `report_plus_feedback`/`report_direct` 路径不被额外改写。`reporting_rewards`、
`team_reward` 和通信/定位物理报告指标没有改变，只有传给 PPO 的 `learning_rewards` 增加了 per-UAV 局部项。
形式化 runner 仍通过 `next_info["learning_rewards"]` 传入 rollout buffer 和 PPO update，没有修改 MAPPO 网络或
actor observation。

新增的诊断包括 `base_learning_reward`、`boundary_reward`、`boundary_projected`、
`boundary_projection_count`、`learning_reward_preclip`、`boundary_events` 和
`boundary_penalty_sum`，并写入 episode reward CSV、episode summary、scenario manifest 和正式实验 manifest。
新增边界测试覆盖无越界、单轴越界、多轴单次计数、边界与碰撞/安全投影分离、回合不终止、PPO learning reward
变化、multi-UAV 局部独立性和零权重向后兼容。

为避免覆盖已有稳定基线，使用完全相同的单 UAV 固定场景配置新建：

```text
data/synthetic/single_uav_nav_goal_v2_boundary_seed2026/
logs/preliminary/single_uav_nav_goal_v2_boundary_seed2026/
plots/preliminary/single_uav_nav_goal_v2_boundary_seed2026/
```

与 `single_uav_nav_goal_v2_stable_seed2026` 对比：新实验训练集出现 `144` 次边界事件、涉及 `32/100`
个 episode，边界惩罚累计为 `-72.0`；validation 无边界事件；test 出现 `2` 次、涉及 `1/20` 个 episode，
累计为 `-1.0`。test 两组均无碰撞、稳定到达率仍为 `0/20`；基线 test 最小目标距离约 `7.60 m`，新实验约
`13.08 m`，训练平均 learning return 由约 `-51.49` 变为 `-56.25`，test 平均 learning return 由约
`-51.86` 变为 `-59.05`。报告 reward 基线与新实验均约为 `-69.74`，说明新增项确实只影响 learning channel，
但在当前固定权重下也使策略更倾向于避开边界，导航效果暂时变差。

新实验的 manifest/schema 校验通过；针对性边界测试 `22 passed`，全量回归测试 `357 passed, 5 skipped`。
当前结果属于单种子、固定场景的机制验证，下一步若要提升导航效果，应再做权重敏感性或更合理的训练场景，
但不应把这次 `0.5` 边界惩罚结果直接作为多 UAV 协同结论。

#### 单 UAV 稳定到达退出验证与双轨基线（2026-09-11）

本轮按照“单 UAV 稳定到达验证”方案，在固定场景、100 steps/回合、100/20/20 个
训练/validation/test 回合下，运行了完全相同配置的两条轨道。主线为原始 PPO 动作，
对照线仅额外启用低层制动安全投影；两条轨道均显式设置 `boundary_penalty_weight=0`，
因此没有把边界惩罚混入本轮基线。

```text
data/synthetic/single_uav_exit_raw_seed2026/
logs/preliminary/single_uav_exit_raw_seed2026/
plots/preliminary/single_uav_exit_raw_seed2026/

data/synthetic/single_uav_exit_safety_seed2026/
logs/preliminary/single_uav_exit_safety_seed2026/
plots/preliminary/single_uav_exit_safety_seed2026/
```

测试集结果为：

```text
                         原始 PPO       安全投影
进入 8 m                  20/20          20/20
进入 1 m                   0/20           3/20
稳定到达                   0/20           3/20
最小目标距离               3.230 m        0.811 m
近目标进入时平均速度       3.283 m/s      2.697 m/s
近目标平均速度             3.829 m/s      2.488 m/s
近目标平均实际减速         N/A            0.259 m/s
安全层制动投影次数         0              209
碰撞回合数                 0              0
```

结果说明：原始 PPO 已经能够稳定进入近目标区域，但没有自主进入 `1 m` 目标范围，
近目标平均速度反而高于进入时速度，说明 PPO 尚未学会持续减速。安全投影改善了末端捕获，
但测试中仍有 `17/20` 回合没有稳定到达，而且有 209 个测试动作经过安全层修正，不能把
`3/20` 的成功完全归因于 PPO。

本轮 smoke 实验、两条完整实验的 `validate-preliminary` 均通过；完整回归测试为
`357 passed, 5 skipped`。重新生成的导航分析已经给出有限的制动统计值，旧实验中出现的
NaN 汇总属于历史分析文件未重新生成，不是当前动作日志缺失。

由于原始 PPO 尚未达到“近目标进入率约 80%、稳定到达率约 70%--80%、至少 3 个种子可重复”
的退出门槛，本轮不继续运行多种子，也不进入 2 UAV 通信实验。当前研究结论是：动作空间和
执行层具备制动能力，但学习策略仍未形成可重复的“接近目标--减速--进入目标--稳定保持”闭环。

#### 单 UAV 制动课程训练执行（2026-09-11）

针对上述“原始 PPO 不自主产生制动动作”的问题，本轮启用已有的训练制动课程：训练阶段以 `0.4`
的比例交错加入距离目标 `3/5/8 m`、初始速度高于允许目标速度的局部制动回合；validation/test
仍保持原固定远距离起点。安全投影未启用，边界惩罚保持为 `0`，因此测试中的制动行为来自原始 PPO。
同时保留 `braking_distance` 目标速度、制动动作奖励和目标速度动作匹配奖励，并将稳定保持设为 `5` 步。

```text
data/synthetic/single_uav_braking_curriculum40_seed2026/
logs/preliminary/single_uav_braking_curriculum40_seed2026/
plots/preliminary/single_uav_braking_curriculum40_seed2026/
```

完整实验配置为固定 `seed=2026`、固定 `scenario_seed=2026`、100 steps/回合、100/20/20 个
训练/validation/test 回合。结果如下：

```text
                                      train        validation       test
进入 8 m                              44/100        19/20          19/20
进入 1 m                              40/100         0/20           1/20
稳定到达                              1/100         0/20           0/20
近目标制动动作比例                    59.3%         60.3%          65.7%
近目标有效制动比例                    97.8%         98.1%          97.2%
近目标平均速度下降                    0.102 m/s     0.044 m/s       0.046 m/s
测试最小目标距离                      --            --             0.717 m
安全投影                              关闭          关闭           关闭
```

结果表明，制动课程确实解决了“原始 PPO 完全不选择制动动作”的问题：测试近目标动作中约
`65.7%` 为制动动作，且其中约 `97.2%` 能产生实际速度下降；测试近目标平均速度也从进入时约
`2.42 m/s` 降至约 `1.86 m/s`。但是只有 `1/20` 回合进入 `1 m`，且没有回合同时满足速度
不超过 `0.5 m/s` 并连续保持 `5` 步。因此当前策略已经从“不会自主刹车”进展到“能够自主刹车”，
但仍未形成稳定的末端停车策略。

本轮 `validate-preliminary` 通过；制动课程与导航分析针对性测试为 `3 passed`。下一步应继续
针对“减速幅度不足和目标附近保持”进行单因素修改，不应把本轮结果作为单 UAV 稳定到达退出证据，
也不应据此进入 2 UAV 通信实验。

#### 单 UAV 末端制动课程与局部奖励执行（2026-09-11）

本轮执行上一阶段提出的末端制动方案：保留已有 `3/5/8 m` 制动课程，在原五回合调度中增加未占用的
`1.5/2/3 m` 训练起点；同时在目标最后 `3 m` 内新增两条可关闭的 learning-only 信号：实际降低速度时
给予 `terminal_braking_reward`，在超速状态继续沿目标方向加速时给予
`terminal_acceleration_penalty`。reporting reward、通信/定位物理指标、验证/测试起点和安全投影均未改变。
新增字段已经写入 `rewards.csv`、schema、episode summary 和 manifest，默认系数为 `0`，因此不影响历史配置。

本轮实验使用固定 `seed=2026`、固定 `scenario_seed=2026`、1 UAV、100 steps/回合、100/20/20 个
训练/validation/test 回合；新增末端课程比例为 `0.2`，实际减速奖励系数为 `3.0`，超速继续加速惩罚系数为
`2.0`，末端半径为 `3 m`。产物为：

```text
data/synthetic/single_uav_terminal_braking_seed2026/
logs/preliminary/single_uav_terminal_braking_seed2026/
plots/preliminary/single_uav_terminal_braking_seed2026/
```

结果如下：

```text
                                      train        validation       test
进入 8 m                              60/100         0/20           0/20
进入 1 m                              59/100         0/20           0/20
稳定到达                              3/100         0/20           0/20
最小目标距离                          0.103 m       10.162 m       10.155 m
末端奖励触发的 step                   1053           0              0
安全投影                              关闭          关闭           关闭
```

训练日志证明新增信号确实生效：末端区域内平均实际减速信号约 `0.136`，累计末端减速奖励约 `390.88`，
累计继续加速惩罚约 `85.62`。但 validation/test 完全没有进入 `3 m`，因此新信号没有机会在评估阶段直接
发挥作用；更重要的是，策略从原上一轮的 validation/test 进入 `8 m` 为 `19/20` 退化为 `0/20`。这说明
本轮“末端起点课程 + 末端局部奖励”的组合使策略过度适应局部制动起点，破坏了从原始远距离起点进行目标
导航的泛化，不能作为主线改进结果，也不能作为“已学会稳定停车”的证据。

因此本轮结论是：代码接口和日志链路正确，但算法结果不接受为改进。下一轮应采用严格的消融顺序，先固定
已有制动课程，只单独比较末端局部奖励开/关；若局部奖励本身无害，再单独降低末端课程比例并比较，不能
同时改变两项。当前输出保留为失败/消融对照，不覆盖上一轮基线，也不进入 2 UAV 通信实验。

本轮针对性测试为 `15 passed`，项目完整回归测试为 `359 passed, 5 skipped`；另修复了当前 Matplotlib 版本
下安全筛选图 `boxplot(labels=...)` 的兼容性参数，改为 `tick_labels=...`。`validate-preliminary` 返回
`ok: true`。

#### 单 UAV 超速误差下降奖励单因素实验（2026-09-11）

按照“先固定已有制动课程，再单独增强超速误差反馈”的方案，本轮只新增
`single_uav_braking_progress_reward_coef=3.0`。该项奖励计算
`max(0, v_pre-v_target)-max(0, v_post-v_target)`，因此只有实际降低距离相关超速误差时才得到正反馈；
超速误差变大时得到负反馈。已有 `3/5/8 m` 制动课程保持为 `0.4`，末端 `1.5/2/3 m` 课程保持为 `0`，
安全投影保持关闭，其他训练、场景和物理参数均与上一轮 `single_uav_braking_curriculum40_seed2026`
一致。新增前后超速误差、归一化信号和奖励字段写入 `rewards.csv`、schema 和 manifest。

正式运行命令为：

```text
run-preliminary --output data/synthetic/single_uav_braking_progress3_seed2026 --uav-counts 1 --seed 2026 --scenario-seed 2026 --steps 100 --warmup-episodes 10 --train-episodes 100 --validation-episodes 20 --test-episodes 20 --single-uav-learning-mode loc_goal_v2 --single-uav-speed-control-mode braking_distance --single-uav-near-goal-radius-m 8.0 --single-uav-near-goal-speed-penalty-weight 0.4 --single-uav-target-speed-far-mps 6.0 --single-uav-braking-action-reward-coef 3.0 --single-uav-target-velocity-action-reward-coef 3.0 --single-uav-braking-progress-reward-coef 3.0 --single-uav-stable-hold-steps 5 --single-uav-braking-curriculum-fraction 0.4 --learning-reward-clip 3.0 --device cpu --force
```

产物为：

```text
data/synthetic/single_uav_braking_progress3_seed2026/
logs/preliminary/single_uav_braking_progress3_seed2026/
plots/preliminary/single_uav_braking_progress3_seed2026/
```

与上一轮仅有制动课程的结果对比如下：

```text
                                      仅制动课程          新增超速误差下降奖励
测试进入 8 m                         19/20 (95%)        17/20 (85%)
测试进入 1 m                          1/20 (5%)          0/20 (0%)
测试稳定到达                         0/20 (0%)          0/20 (0%)
测试最小目标距离                     0.717 m             4.066 m
测试近目标平均速度                   1.863 m/s           1.254 m/s
测试近目标平均超速                   0.243 m/s           0.000 m/s
```

结论是：该项奖励确实降低了近目标速度并消除了测试集近目标超速，但同时削弱了继续向目标推进的能力，
因此没有改善 `1 m` 进入率或稳定到达率，不能作为主线改进结果。它证明了“速度反馈”和“目标推进”之间
存在权衡；下一步应在保持制动课程不变的前提下，将该系数降低到较小值做单因素灵敏度比较，或增加
一个独立的近目标进度保持项，不能再次同时加入末端课程。末端制动课程接口仍可单独用于后续消融，
但不应与本轮结果混合解释。

本轮 smoke 和正式实验的 `validate-preliminary` 均返回 `ok: true`；新增制动课程与奖励测试共 `16 passed`。
本轮仍未满足单 UAV 稳定到达退出条件，因此不进入 2 UAV 通信实验。

#### 单 UAV 超速误差奖励权重扫描（2026-09-11）

在保持其他参数完全不变的前提下，补充完成 `single_uav_braking_progress_reward_coef=0.5` 和 `1.0`
两组正式实验，并与已有的 `0`、`3.0` 两组统一比较。四组均为固定 `seed=2026`、固定
`scenario_seed=2026`、1 UAV、100 steps/回合、100/20/20 个训练/validation/test 回合；已有 `3/5/8 m`
制动课程为 `0.4`，末端停车课程为 `0`，安全投影关闭。

统一测试结果如下：

```text
coef       进入 8 m       进入 1 m       稳定到达       最小目标距离       近目标平均速度       平均超速
0          19/20 (95%)   1/20 (5%)     0/20 (0%)      0.717 m           1.863 m/s           0.243 m/s
0.5        19/20 (95%)   1/20 (5%)     0/20 (0%)      0.158 m           1.959 m/s           0.221 m/s
1.0        19/20 (95%)   0/20 (0%)     0/20 (0%)      1.127 m           1.595 m/s           0.240 m/s
3.0        17/20 (85%)   0/20 (0%)     0/20 (0%)      4.066 m           1.254 m/s           0.000 m/s
```

按照“先保持远距离导航，再改善近目标到达”的预先规则，当前单种子筛选选择 `coef=0.5`：它保持了
`19/20` 的 `8 m` 进入率和 `1/20` 的 `1 m` 进入率，同时将最小目标距离由 `0.717 m` 降至约 `0.158 m`，
并略微降低平均超速。`coef=1.0` 虽然近目标平均速度更低，但测试未进入 `1 m`；`coef=3.0` 则明显
损害远距离导航。因此 `0.5` 是当前较平衡的候选权重，但还不能称为最终最优参数，因为四组目前都为
单随机种子，稳定到达率仍为 `0/20`。

四组正式结果的统一汇总保存于：

```text
logs/preliminary/single_uav_braking_progress_sweep_seed2026.csv
```

本轮两组新增实验的 `validate-preliminary` 均返回 `ok: true`。下一步应固定 `coef=0.5`，使用至少
`2026、3035、4044` 三个随机种子重复验证；若多种子仍能保持导航并改善近目标距离，再单独研究停车保持
课程。当前仍不进入 2 UAV 通信实验。

#### 单 UAV 超速误差奖励权重补测（2026-09-11）

在保持上一轮权重扫描的所有条件不变的前提下，补测
`single_uav_braking_progress_reward_coef=0.25、0.75、1.5`。三组均为固定
`seed=2026`、固定 `scenario_seed=2026`、1 UAV、100 steps/回合、100/20/20 个训练/validation/test
回合；`3/5/8 m` 制动课程为 `0.4`，末端停车课程为 `0`，安全投影关闭。三组均独立写入各自目录，未覆盖
已有 `0、0.5、1.0、3.0` 结果。

新增三组的统一结果如下：

```text
coef       split       进入 8 m       进入 1 m       稳定到达       最小目标距离       近目标平均速度       平均超速
0.25       validation  19/20 (95%)   0/20 (0%)     0/20 (0%)      1.091 m           1.951 m/s           0.311 m/s
0.25       test        20/20 (100%)  1/20 (5%)     0/20 (0%)      0.468 m           1.896 m/s           0.308 m/s
0.75       validation  11/20 (55%)   0/20 (0%)     0/20 (0%)      2.273 m           2.055 m/s           0.028 m/s
0.75       test        9/20 (45%)    0/20 (0%)     0/20 (0%)      3.694 m           1.988 m/s           0.022 m/s
1.5        validation  1/20 (5%)     0/20 (0%)     0/20 (0%)      6.984 m           1.544 m/s           0.000 m/s
1.5        test        2/20 (10%)    0/20 (0%)     0/20 (0%)      2.897 m           1.780 m/s           0.236 m/s
```

结果表明，`coef=0.25` 是三组补测中最平衡的候选：测试进入 `8 m` 达到 `20/20`，进入 `1 m` 为
`1/20`，最小目标距离为 `0.468 m`。但它仍未产生稳定到达，因此不能单独证明已学会停车。`coef=0.75`
开始明显损害远距离目标导航，`coef=1.5` 更严重；虽然近目标超速在部分评估样本中降低，但主要原因是
策略很少进入近目标区域，不能把“超速变小”解释为制动能力真正改善。

三组新增实验的完整产物分别位于：

```text
data/synthetic/single_uav_braking_progress025_seed2026/
logs/preliminary/single_uav_braking_progress025_seed2026/
plots/preliminary/single_uav_braking_progress025_seed2026/

data/synthetic/single_uav_braking_progress075_seed2026/
logs/preliminary/single_uav_braking_progress075_seed2026/
plots/preliminary/single_uav_braking_progress075_seed2026/

data/synthetic/single_uav_braking_progress15_seed2026/
logs/preliminary/single_uav_braking_progress15_seed2026/
plots/preliminary/single_uav_braking_progress15_seed2026/
```

七个系数的 validation/test 统一汇总保存于：

```text
logs/preliminary/single_uav_braking_progress_sweep_extended_seed2026.csv
```

三组新增目录的 `validate-preliminary` 均返回 `ok: true`。当前仍不能把任何系数称为最终权重；下一步应
优先对 `0.25` 与已有 `0.5` 做多随机种子复验，随后再固定候选系数研究停车保持课程，而不是继续盲目扩大
超速误差奖励系数。

#### 单 UAV 终端接近/停车保持课程接口 smoke（2026-09-11，旧版调度）

本轮只在当前 `coef=0.25` 基线之上新增
`single_uav_terminal_braking_curriculum_fraction=0.30`，未启用 goal hold reward，未修改 braking reward、
`stable_hold_steps=5`、动力学、动作空间、安全投影或 PPO 参数，也未改变 validation/test。实现复用了现有的
`_apply_single_uav_braking_curriculum` 终端分支。

上一版 smoke 使用了 4 个 standard、3 个 terminal、3 个原始远距离槽位；这相当于把 terminal 预算叠加在原有
`0.4` 课程之上。该结果保留为历史接口验证记录，但不再作为当前调度方案。

terminal 起点按训练进度经过 `1.5–3 m`、`3–5 m`、`5–8 m` 三阶段，每阶段使用确定性的代表距离，不引入额外
随机起点因素。terminal 分支仅在 train phase 生效，warm-up、validation 和 test 不使用 terminal 起点。

导航分析复用了已有的 `single_uav_stable_steps`，新增每回合及 split 汇总指标
`maximum_consecutive_goal_hold_steps`，并保留 `maximum_stable_steps` 兼容旧输出。

针对性单元测试结果为 `17 passed`。短 smoke 输出位于：

```text
data/synthetic/single_uav_terminal_curriculum30_coef025_smoke_seed2026/
logs/preliminary/single_uav_terminal_curriculum30_coef025_smoke_seed2026/
plots/preliminary/single_uav_terminal_curriculum30_coef025_smoke_seed2026/
```

该 smoke 的 `validate-preliminary` 返回 `ok: true`；训练 30 回合实际记录为 12 个 standard、9 个 terminal、
9 个原始远距离回合，terminal 三个阶段均有样本。以上仅记录代码链路、课程分配和日志诊断是否正常，暂不据此
下研究结论。

#### 单 UAV 终端课程预算内调度修正 smoke（2026-09-11）

本轮只修改课程 episode 的调度分配逻辑：保持 `coef=0.25`、`braking_curriculum_fraction=0.4`、
`terminal_braking_curriculum_fraction=0.3` 以及其余奖励、动力学、动作空间、安全投影、PPO/MAPPO 参数不变。
终端课程不再叠加到原有课程之外，而是放入原有 `0.4` 总预算内部。

当前 10 回合槽位为：6 个 normal far-distance、1 个 standard braking、3 个 terminal；因此目标比例为
`60% / 10% / 30%`。30-train smoke 的实际数量为：

```text
normal far-distance       18/30 = 60%
standard braking           3/30 = 10%
terminal                   9/30 = 30%
```

terminal 三个距离阶段 `0/1/2` 均有样本，terminal 初速度仍为非静止主动制动起步。validation/test 均无课程元数据，
且与已有 `coef=0.25` 基线的 validation/test 初始位置、速度和目标逐字段一致；稳定保持判据仍为
`stable_hold_steps=5`，goal hold reward 仍为 `0`。

新增 hold 汇总字段已写入 navigation analysis：`mean_maximum_consecutive_goal_hold_steps`、
`maximum_consecutive_goal_hold_steps_max`、以及 `>=1`、`>=3`、`>=5` 的 episode 数量和 rate。本次 smoke 中
train 的 mean/max 均为 `0`，三个阈值均为 `0/30`（对应 rate 均为 `0`）；这些仅是诊断输出，不影响 reward 或
termination。

修正后的 smoke 输出位于：

```text
data/synthetic/single_uav_terminal_curriculum30_coef025_budgeted_smoke_seed2026/
logs/preliminary/single_uav_terminal_curriculum30_coef025_budgeted_smoke_seed2026/
plots/preliminary/single_uav_terminal_curriculum30_coef025_budgeted_smoke_seed2026/
```

针对性 pytest 为 `17 passed`，`validate-preliminary` 返回 `ok: true`，schema、CSV、summary 和诊断图均正常。
全量 pytest 为 `358 passed, 5 skipped, 2 failed`；两项失败来自未改动的
`safety_risk_screening.py` 使用当前 Matplotlib 不支持的 `tick_labels` 参数，与本轮课程调度无关。
本轮仍未运行正式 `100/20/20` 实验，也不据此 smoke 结果下研究结论。

#### 单 UAV 终端课程正式 `100/20/20` 实验（2026-09-11）

在预算内调度 smoke 通过后，使用完全相同的课程、奖励、动作空间、动力学和 PPO/MAPPO 参数运行正式单 UAV
实验。唯一课程配置为：每 10 个训练回合中 6 个 normal far-distance、1 个 standard braking、3 个 terminal；
terminal 阶段为 `1.5–3 m`、`3–5 m`、`5–8 m`。terminal curriculum 仅作用于 train，validation/test 继续使用
原始完整远距离起点。

本轮固定 `seed=scenario_seed=2026`、`steps=100`、`warmup/train/validation/test=10/100/20/20`、
`coef=0.25`、`stable_hold_steps=5`、goal hold reward 为 `0`、安全投影关闭。正式运行共生成 140 个 episode，
训练课程实际统计为：normal far-distance `60/100`、standard braking `10/100`、terminal `30/100`；terminal
阶段 `0/1/2` 分别为 `11/30`、`10/30`、`9/30`，terminal 初始速度范围为 `2.0–5.5 m/s`。

正式产物位于：

```text
data/synthetic/single_uav_terminal_curriculum30_coef025_seed2026/
logs/preliminary/single_uav_terminal_curriculum30_coef025_seed2026/
plots/preliminary/single_uav_terminal_curriculum30_coef025_seed2026/
```

`validate-preliminary` 返回 `ok: true`；schema、CSV、summary、checkpoint 和图表均完整。validation/test 的 20
个 episode 起始位置、初始速度和目标与 `single_uav_braking_progress025_seed2026` 基线逐字段一致，且均无课程元数据。

测试集与原 `coef=0.25` 基线对比如下：

```text
指标                         coef=0.25 baseline       terminal curriculum
进入 8 m                     20/20 (100%)              2/20 (10%)
进入 1 m                      1/20 (5%)                0/20 (0%)
稳定到达                      0/20 (0%)                0/20 (0%)
最小目标距离                  0.468 m                   6.664 m
进入近目标时速度              2.536 m/s                 3.082 m/s
近目标区域平均速度            1.896 m/s                 3.271 m/s
近目标平均超速                0.308 m/s                 0.000 m/s
```

终端课程训练集出现了局部保持信号：`stable_goal=6/100`，`mean_maximum_consecutive_goal_hold_steps=0.5`，
`maximum_consecutive_goal_hold_steps_max=5`，达到 `>=1/>=3/>=5` 步的回合数分别为 `13/10/6`（`13%/10%/6%`）。
但 validation/test 的上述保持指标全部为 `0`。因此这些训练集样本没有转化为未见远距离起点上的终端成功。

本轮结论是：terminal curriculum 没有达到“保持远距离导航并改善终端到达”的预期，且首先破坏了粗粒度远距离导航能力；
当前验证集和测试集的主要瓶颈仍是 `8 m` 之前的远距离导航迁移，而不是可以直接进入下一项 goal hold reward 实验的纯
`1 m` 内停车保持问题。暂不叠加 `goal_hold_reward=0.5`，以免把远距离导航退化和保持奖励的作用混在一起。

#### 冻结 terminal curriculum 模型的起点距离分档只读诊断（2026-09-11）

为定位当前 terminal curriculum 模型从哪个起点距离开始失效，新增了冻结 checkpoint 的只读评估入口
`evaluate-single-uav-terminal-starts`。该入口只加载 `final.pt` 的策略参数和训练阶段冻结的归一化参数，不调用
`MAPPOTrainer.update()`，不加入 rollout transition，不读取或写回 optimizer state，也不改变 reward、curriculum、
动力学、动作空间或 PPO/MAPPO 参数。

本轮使用正式 terminal curriculum 模型，在固定场景几何下生成四个独立 test-only 分组，每组 20 个 episode：
`1.5–3 m`、`3–5 m`、`5–8 m` 和原始 normal far-distance。前三档使用等间距起点距离，并沿目标方向设置与正式
terminal 起点一致的非零初速度；normal far-distance 使用原始完整远距离起点。四档均未生成 train/validation，
所有 episode 均为只读推理诊断。

分档结果如下：

```text
起点分组             进入 8 m       进入 1 m       stable       最小目标距离   平均速度     平均最大保持步数   >=1 / >=3 / >=5
1.5–3 m              20/20 (100%)  20/20 (100%)  6/20 (30%)  0.059 m       1.428 m/s    2.45              13 / 10 / 6
3–5 m                20/20 (100%)  20/20 (100%)  1/20 (5%)   0.123 m       1.450 m/s    0.95              6 / 4 / 1
5–8 m                20/20 (100%)  20/20 (100%)  0/20 (0%)   0.156 m       1.761 m/s    0.00              0 / 0 / 0
normal far-distance   2/20 (10%)    0/20 (0%)    0/20 (0%)   6.664 m       1.913 m/s    0.00              0 / 0 / 0
```

该诊断显示：从 `1.5–3 m` 和 `3–5 m` 起点出发，策略已经能高概率进入 `1 m`，并能在部分回合形成连续保持；
从 `5–8 m` 起点出发虽然都能进入 `1 m`，但没有任何回合达到 1 个稳定保持步，说明较长终端距离上的制动/保持链条
仍未形成。原始 normal far-distance 组只有 `2/20` 进入 `8 m`，因此正式模型当前最先失效的环节是远距离导航向
终端课程的迁移；不能把近距离分档的改善直接解释为完整任务成功。

诊断产物位于：

```text
data/synthetic/single_uav_terminal_curriculum30_coef025_start_distance_diagnostic_seed2026/
logs/preliminary/single_uav_terminal_curriculum30_coef025_start_distance_diagnostic_seed2026/
plots/preliminary/single_uav_terminal_curriculum30_coef025_start_distance_diagnostic_seed2026/
```

其中 `group_summary.csv` 提供四档汇总，`episode_metrics.csv` 提供每回合指标，
`start_distance_diagnostic.png` 提供分档图。schema 校验为 `ok=true`；checkpoint SHA-256 和策略参数在评估前后
均未变化。新增评估相关测试通过；全量 pytest 为 `361 passed, 5 skipped, 2 failed`，两项失败仍是未改动的
`safety_risk_screening.py` 使用当前 Matplotlib 不支持的 `tick_labels` 参数所致。

#### navigation-preserving terminal fine-tuning（2026-09-11）

根据起点距离诊断，本轮没有继续使用已经退化的 terminal curriculum checkpoint，而是先加载原始
`single_uav_braking_progress025_seed2026` 正式 checkpoint 做只读复现。baseline checkpoint 的 SHA-256 为：

```text
77d8702e500fe7d1750bcbba96b9aea238e9a982c0af88dd440881e1b7a9330c
```

只读复现得到正常远距离 test：进入 `8 m = 20/20`、进入 `1 m = 1/20`、stable `= 0/20`、最小目标距离
`0.468 m`、近目标平均超速 `0.308 m/s`，与原 baseline 一致；只读阶段 checkpoint 文件哈希和策略哈希均未变化。

随后从该 baseline 权重和 optimizer state 继续进行独立 fine-tuning。除课程调度外没有修改 reward、动作空间、动力学、
PPO/MAPPO 参数、安全投影、`stable_hold_steps` 或 goal hold reward；`goal_hold_reward` 保持 `0`，
`stable_hold_steps` 保持 `5`，且没有新增奖励项。课程调度明确为：

```text
90% normal full-distance train episodes
10% terminal train episodes
terminal distance: 5–8 m
standard braking curriculum: 0%
validation/test: 100% normal full-distance, no curriculum
```

smoke（10 train + 2 validation + 2 test）实际为 `9 normal + 1 terminal`，参数从 baseline policy hash 开始，
optimizer state 的 27 个条目成功恢复，参数发生更新，`validate-preliminary` 为 `ok=true`。正式轮（100 train + 20
validation + 20 test）实际为 `90 normal + 10 terminal`；10 个 terminal 起点覆盖 `5.0–8.0 m`，初速度均非零，
范围约为 `3.87–5.50 m/s`。validation/test 的实际起点均保持约 `19.729 m` 的正常远距离分布，curriculum 标记为
关闭；正式输出 `validate-preliminary` 为 `ok=true`。

正式 fine-tuning 与 baseline 的正常远距离 test 对比如下：

```text
指标                         coef=0.25 baseline       navigation-preserving fine-tuning
进入 8 m                     20/20 (100%)              20/20 (100%)
进入 1 m                     1/20 (5%)                0/20 (0%)
stable goal                 0/20 (0%)                0/20 (0%)
最小目标距离                 0.468 m                   3.033 m
进入近目标时速度             2.536 m/s                 5.114 m/s
近目标区域平均速度           1.896 m/s                 2.416 m/s
近目标平均超速               0.308 m/s                 0.563 m/s
平均最大连续保持步数         0.00                      0.00
最大连续保持步数             0                         0
保持步数 >=1 / >=3 / >=5     0 / 0 / 0                 0 / 0 / 0
```

因此本轮满足“保护远距离导航”的首要条件（进入 `8 m = 20/20`），但没有改善 `8 m → 1 m` 的精确接近，且
进入 `1 m` 从 baseline 的 `1/20` 下降为 `0/20`；停车保持的所有 hold-step 指标仍为零。本轮
`navigation-preserving terminal fine-tuning` 不支持直接进入 goal hold reward 实验，后续应先分析为什么从 baseline
继续更新后仍未把 5–8 m terminal 样本迁移到正常远距离 test。

本轮新增执行入口和产物：

```text
src/experiments/single_uav_navigation_preserving_finetune.py
data/synthetic/single_uav_braking_progress025_baseline_verify_seed2026/
data/synthetic/single_uav_navigation_preserving_finetune_smoke_seed2026/
data/synthetic/single_uav_navigation_preserving_finetune_seed2026/
logs/preliminary/single_uav_navigation_preserving_finetune_seed2026/
plots/preliminary/single_uav_navigation_preserving_finetune_seed2026/
```

针对性 fine-tuning/冻结评估测试通过；全量 pytest 为 `365 passed, 5 skipped, 2 failed`。两项失败与本轮无关，仍为
未改动的 `safety_risk_screening.py` 在当前 Matplotlib 环境中使用不支持的 `tick_labels` 参数。

#### baseline 与 navigation-preserving fine-tuned 的终端接近分段诊断（2026-09-11）

为定位 fine-tuned policy 在正常远距离任务中从 `8 m` 到 `1 m` 的失效位置，新增了只读配对诊断入口：
每个 checkpoint 使用相同的 `20` 个 normal far-distance test 场景、相同 test seed index 和相同场景对象，
不调用 `MAPPOTrainer.update`，不写回 checkpoint。距离分段采用明确的半开区间：
`8–5 m=[5,8)`、`5–3 m=[3,5)`、`3–1 m=[1,3)`，`[0,1)` 为目标区。

关键分段结果如下（推进速度为日志 `goal_progress_delta / dt`，制动比例以 action log 的二值制动事件计算）：

```text
指标                         coef=0.25 baseline       navigation-preserving fine-tuned
进入 8–5 m                   20/20                    20/20
继续进入 5–3 m               17/20                    20/20
继续进入 3–1 m               11/20                    0/20
继续进入目标区(<1 m)         1/11                     0/0
5–3 m 平均推进速度           0.0542 m/s               0.1745 m/s
5–3 m 平均速度               1.814 m/s                2.712 m/s
5–3 m 平均目标速度           2.057 m/s                2.193 m/s
5–3 m 制动/有效制动比例      0.730/0.718              0.755/0.752
3–1 m 平均推进速度           0.1125 m/s               无样本
3–1 m 平均速度               1.821 m/s                无样本
```

更细的 `3–3.5 m` 检查显示 fine-tuned policy 只有 `6/20` 个 episode 进入该窄区间，区间内平均速度约
`2.582 m/s`，而距离相关目标速度约 `1.148 m/s`；估计距离仍有局部下降，但没有一个 episode 跨过
`3 m` 进入 `3–1 m`。这与正式 fine-tuning test 的最小距离 `3.033 m` 一致，说明当前行为在约 `3 m`
附近形成了“接近—制动—回退/停滞”的局部策略切换点，而不是远距离 `8 m` 入口再次失效。

因此下一轮优先级应为 **C：针对约 3 m 附近的速度—推进奖励权衡做最小单因素诊断/修正**；暂时不先扩大
terminal 起点分布（A），也不从已退化的 fine-tuned policy 继续加大训练（B）。本轮没有修改 reward、
curriculum、动力学、动作空间或 PPO/MAPPO 参数，也没有加入 goal hold reward。

本轮产物：

```text
src/experiments/single_uav_terminal_approach_diagnostic.py
tests/test_single_uav_terminal_approach_diagnostic.py
data/synthetic/single_uav_terminal_approach_diagnostic_seed2026/
logs/preliminary/single_uav_terminal_approach_diagnostic_seed2026/
plots/preliminary/single_uav_terminal_approach_diagnostic_seed2026/
```

两份 checkpoint 的文件 SHA-256 和加载后策略参数哈希在评估前后均未变化；两边 schema 校验均为
`ok=true`，完整回归测试为 `373 passed, 5 skipped, 1 warning`。

#### 单 UAV scripted/oracle 终端停车可行性上限测试（2026-09-11）

为区分“当前任务在动力学上能否完成”和“RL 策略是否学会完成”，新增独立的只读
`diagnose-single-uav-scripted-feasibility` 入口。它不训练 neural policy、不调用 PPO 更新、不写回
checkpoint，也不改变 reward、curriculum、动作空间、动力学、稳定判据或正常 test 场景。scripted/oracle
直接读取仿真真值位置、速度和 goal，仅通过现有 `environment.step(delta_v)` 提交合法的三维速度增量，
因此只能作为物理可行性上限，不是 RL 性能结果。

当前任务定义记录为：正常起点到 goal 约 `19.7286 m`，`100` steps，`dt=0.2 s`，最大速度
`12 m/s`，最大加速度 `3 m/s²`，每步最大 `delta-v=0.6 m/s`，goal radius `1 m`，stable speed
threshold `0.5 m/s`，连续稳定 `5` steps。稳定条件仍为 `loc_goal_v2` 的 estimated goal distance
不超过 `1 m` 且实际速度不超过 `0.5 m/s`，连续满足 5 次；碰撞和稳定到达仍会提前结束，否则在
100 steps 截止。环境原有 safety projection 保持开启，braking safety projection 保持关闭。

scripted controller 使用四段式规则：远距离以 `6 m/s` 朝 goal 巡航；`8--3 m` 按带 1 m 控制裕量的
停止距离曲线降速；`3--1 m` 保持正径向低速推进；进入 `1 m` 后将目标速度压到 `0.35 m/s` 以下并
进行小幅误差修正。所有动作仍经过当前动力学和 action interface，没有位置/速度 teleport。

20 个相同 normal far-distance test 场景的只读结果为：

```text
指标                         coef=0.25 baseline RL   navigation-preserving fine-tuned RL   scripted/oracle
进入 8 m                     20/20                   20/20                                 20/20
进入 5 m                     17/20                   20/20                                 20/20
进入 3 m                     11/20                   0/20                                  20/20
进入 1 m                     1/20                    0/20                                  20/20
stable goal                 0/20                    0/20                                  20/20
最小目标距离                 0.468 m                 3.033 m                               0.295 m
平均最小目标距离             3.135 m                 3.584 m                               0.610 m
进入 1 m 时平均速度          1.239 m/s               NA                                    0.473 m/s
目标区内平均速度             1.459 m/s               NA                                    0.360 m/s
平均最大连续保持步数         0.00                    0.00                                  5.00
保持步数 >=1 / >=3 / >=5     0 / 0 / 0               0 / 0 / 0                             20 / 20 / 20
平均进入 8/5/3/1 m 步数      36.6/44.9/50.7/63.0     20.6/24.4/NA/NA                        15.0/18.0/21.0/28.2
```

oracle 20 个回合均在 `32--38` steps 内达到稳定成功，平均约 `32.75` steps；环境动作 clipping 为
`0`。控制器自身约 `45.9%` 的步使用了达到合法上限的 delta-v，最大请求制动约 `1.187 m/s`，
实际应用上限为 `0.600 m/s`，但该制动力仍足以完成任务。因此在当前 `100`-step horizon、动作空间和
动力学下，单 UAV 的“远距离导航 → 进入 1 m → 低速连续保持 5 steps”在物理上可完成；主要差距在
RL 的策略学习、信用分配和技能组合，而不是当前任务不可行或时间上限不足。fine-tuned policy 的
`3 m` 附近失效也由同一配对评估复现：它保住了进入 `8 m`，但没有越过 `3 m`。

正式产物：

```text
src/experiments/single_uav_scripted_feasibility.py
tests/test_single_uav_scripted_feasibility.py
data/synthetic/single_uav_scripted_feasibility_seed2026/
logs/preliminary/single_uav_scripted_feasibility_seed2026/
plots/preliminary/single_uav_scripted_feasibility_seed2026/
```

scripted/oracle、baseline RL 和 fine-tuned RL 三个角色的 schema 均为 `ok=true`；两个 RL checkpoint
在评估前后的 SHA-256 和策略参数哈希均未变化。使用项目本地 pytest 临时目录的全量回归为
`379 passed, 5 skipped, 1 warning`。

#### 单 UAV 约 3 m 速度控制单因素实验（2026-09-11）

本轮只新增 `single_uav_braking_distance_speed_profile=early_smooth`，使 braking-distance 模式在
更远处平滑降低目标速度；未修改 reward、terminal curriculum、动作空间、动力学、PPO/MAPPO 参数、
stable hold 判据或 validation/test 分布。当前节点值如下：

```text
距离              1 m     3 m     5 m     6 m     8 m
legacy            0.50    0.50    2.872   3.50    4.50 m/s
early_smooth      0.50    1.38    2.425   3.415   6.00 m/s
```

正式 `100/20/20` fine-tuning 从原始 coef=`0.25` baseline checkpoint 起步，训练结构保持
`90% normal + 10% terminal(5--8 m)`。正式 test 对比如下：

```text
指标                         baseline       old fine-tuned       early_smooth fine-tuned
进入 8 m                     20/20          20/20                20/20
进入 5 m                     17/20          20/20                20/20
进入 3 m                     11/20          0/20                 20/20
进入 1 m                     1/20           0/20                 2/20
stable goal                 0/20           0/20                 0/20
最小目标距离                 0.468 m        3.033 m              0.821 m
近目标区平均速度             1.896 m/s      2.416 m/s            2.181 m/s
近目标区平均目标速度         2.742 m/s      3.271 m/s            2.767 m/s
近目标区平均超速             0.308 m/s      0.563 m/s            0.667 m/s
平均最大连续保持步数         0.00           0.00                 0.00
保持步数 >=1 / >=3 / >=5     0/0/0          0/0/0                0/0/0
```

新方案保住了远距离 `8 m` 的 `20/20`，并把旧 fine-tuned 的 `5--3 m -> 3--1 m` 从 `0/20` 改善为
`20/20` 进入 `3--1 m`，最终进入 `1 m` 为 `2/20`。但是 stable goal 仍为 `0/20`，目标区内速度
仍明显高于目标速度。因此本轮解决了约 `3 m` 的穿越瓶颈，但没有解决最后 `1 m` 的低速停车保持；
不应据此启用 goal hold reward。

两份冻结诊断均使用完全相同的 `20` 个 normal far-distance 场景，三策略统一比较的场景哈希一致；
checkpoint 和 policy 参数在评估前后均未变化。early_smooth 的 `8--5 m / 5--3 m / 3--1 m`
继续进入下一段分别为 `20/20 / 20/20 / 2/20`，说明剩余主要瓶颈已转移到 `1 m` 内的减速和连续保持，
但尚未达到稳定成功。

本轮产物包括：

```text
src/simulation/motion_model.py
src/simulation/environment.py
src/experiments/formal.py
src/experiments/pilot.py
src/experiments/single_uav_navigation_analysis.py
src/experiments/single_uav_navigation_preserving_finetune.py
src/experiments/single_uav_terminal_approach_diagnostic.py
tests/test_motion_model.py
tests/test_single_uav_terminal_approach_diagnostic.py
data/synthetic/single_uav_speed_curve_early_smooth_finetune_seed2026/
logs/preliminary/single_uav_speed_curve_early_smooth_finetune_seed2026/
plots/preliminary/single_uav_speed_curve_early_smooth_finetune_seed2026/
logs/preliminary/single_uav_speed_curve_three_policy_comparison_seed2026/
plots/preliminary/single_uav_speed_curve_three_policy_comparison_seed2026/
```

三组正式实验的 `validate-preliminary` 均为 `ok=true`、schema 为 `episode-v1.3`；本轮完整回归为
`379 passed, 5 skipped, 1 warning`。两份冻结诊断的 schema、场景配对和只读检查也全部通过。

#### 单 UAV RL-equivalent observation Oracle 与 Actor action-output 审计（2026-09-11）

本轮暂停所有 reward、curriculum、target-speed 和 PPO 调整，新增只读诊断入口
`diagnose-single-uav-rl-equivalent-oracle-audit`。评估使用同一组 `20` 个正常远距离场景，
并行比较 truth scripted/oracle、仅使用 Actor 可见 EKF/node features 的 estimated-state oracle、
原始 coef=`0.25` baseline RL 和 navigation-preserving fine-tuned RL。四类角色均未训练、未更新
checkpoint、未加入 rollout transition，也未改变任务定义。

正式结果如下：

```text
指标                         truth oracle   estimated-state oracle   baseline RL   fine-tuned RL
进入 8 m                     20/20          20/20                    20/20         20/20
进入 5 m                     20/20          20/20                    17/20         20/20
进入 3 m                     20/20          20/20                    11/20         0/20
进入 1 m                     20/20          20/20                    1/20          0/20
stable goal                 20/20          20/20                    0/20          0/20
平均最小目标距离             0.610 m        0.575 m                  3.135 m       3.584 m
1 m 入口平均速度             0.473 m/s      0.595 m/s                1.239 m/s     --
1 m 内平均速度               0.360 m/s      0.386 m/s                1.459 m/s     --
平均最大连续保持步数         5.00           5.00                     0.00          0.00
保持步数 >=1 / >=3 / >=5     20/20/20       20/20/20                 0/0/0         0/0/0
超时回合                     0/20           0/20                    20/20         20/20
```

estimated-state oracle 只从 `node_features` 的自机节点读取 `[0:3]` 估计位置、`[3:6]` 估计速度和
`[30:33]` 估计目标差向量，未使用 simulator truth fallback；其相对 truth 的平均绝对误差为：
目标距离 `0.092 m`、目标方向角 `4.71°`、径向速度 `0.0738 m/s`。因此 RL 实际观测信息本身
足以支持当前任务，主要瓶颈不是 localization/observation insufficiency。

Actor action chain 的审计显示：baseline 的平均 scaled/applied action norm 为 `0.0937 m/s`、
最大 braking projection 为 `0.1164 m/s`；fine-tuned 分别为 `0.1573 m/s` 和 `0.2910 m/s`，
均显著低于合法动作约 `0.6 m/s` 的可用制动幅度。两者的 scale-to-applied residual 均约为
`1e-8 m/s`，action clipping 为 `0`，故没有发现 action scaling/mapping 缩小问题。按全局审计规则，
主要根因分类为 `C`（Actor 未充分使用可用动作/制动幅度）；距离分段还显示 baseline 在 `8--5 m`
和 `5--3 m` 的径向速度均值分别约 `-0.051` 和 `-0.095 m/s`，fine-tuned 在 `5--3 m`
约 `-0.334 m/s`，说明局部还存在径向推进与过早制动的策略耦合，但不改变本轮“只诊断、不修改”的结论。

配置审计确认 baseline 与 fine-tuned 的 test effective environment contract 相同；与先前 feasibility
实验的 `environment_safety_projection_enabled=true`、`braking_safety_projection_enabled=false` 一致。
两份 checkpoint SHA-256 未变：baseline 为
`77d8702e500fe7d1750bcbba96b9aea238e9a982c0af88dd440881e1b7a9330c`，fine-tuned 为
`083e07253f6d2f775509c46bc1959afae0f5e3bacf455fa476ad8df08e8cd795`。

正式产物包括：

```text
src/experiments/single_uav_rl_equivalent_oracle_audit.py
tests/test_single_uav_rl_equivalent_oracle_audit.py
data/synthetic/single_uav_rl_equivalent_oracle_audit_seed2026/
logs/preliminary/single_uav_rl_equivalent_oracle_audit_seed2026/
plots/preliminary/single_uav_rl_equivalent_oracle_audit_seed2026/
```

四个角色的 `validate_episode_benchmark` 均为 `ok=true`，`overall_schema_ok=true`，并生成 8 张诊断图。
新增诊断测试为 `5 passed`。本次全量 pytest 为 `382 passed, 5 skipped, 2 failed`；两个失败来自已有
`src/experiments/safety_risk_screening.py` 使用 `tick_labels` 与当前 Matplotlib 版本不兼容，未涉及本轮
诊断代码或研究机制，未在本轮扩大修改范围。

#### C 路线第二半部分：3--1 m 低速径向推进奖励（2026-09-11）

基于 early-smooth speed-control 正式实验，本轮只增加一个单因素奖励：
`single_uav_terminal_radial_progress_reward_coef=0.5`。该奖励仅在估计目标距离满足
`1 m < d <= 3 m` 时生效，使用非负径向进度和速度门控：

```text
radial_progress_m = max(0, d_pre - d_post)
speed_gate = 1 / (1 + (max(0, speed - target_speed) / 1.0)^2)
progress_signal = clip(radial_progress_m / 0.5, 0, 1) * speed_gate
terminal_reward = 0.5 * progress_signal
```

该项没有新增负惩罚，也没有改变 goal hold reward、stable hold steps、termination、动作空间、动力学、
PPO/MAPPO 参数或 terminal curriculum。逐步日志新增 active、径向进度、speed gate、signal 和 reward
字段；clip saturation 只作诊断，不参与 reward 或 termination。

正式 fine-tuning 仍从原始 coef=`0.25` baseline checkpoint 开始，保持 `90% normal full-distance +
10% terminal(5--8 m)`，完成 `100 train / 20 validation / 20 test`。训练审计为 `90/10/0`，validation
和 test 均为 normal far-distance，terminal 只出现在 train；smoke 和正式结果的
`validate-preliminary` 均为 `ok=true`。正式 D checkpoint 的源 checkpoint SHA-256 为
`77d8702e500fe7d1750bcbba96b9aea238e9a982c0af88dd440881e1b7a9330c`，输出 checkpoint SHA-256 为
`edeb876312e053c3fef4e6393e72aa56b275004f6b63f1f2153ef153f232f5e4`。

四策略在同一组 `20` 个 normal far-distance test 场景下的统一结果为：

```text
指标                         A baseline   B old fine   C early-smooth   D + radial reward
进入 8 m                     20/20        20/20        20/20             20/20
进入 5 m                     17/20        20/20        20/20             20/20
进入 3 m                     11/20        0/20         20/20             17/20
进入 1 m                     1/20         0/20         2/20              0/20
stable goal                 0/20         0/20         0/20              0/20
最小目标距离                 0.468 m      3.033 m      0.821 m            1.885 m
近目标区平均速度             1.896 m/s    2.416 m/s    2.181 m/s          2.545 m/s
近目标区平均目标速度         2.742 m/s    3.271 m/s    2.767 m/s          3.132 m/s
近目标区平均超速             0.308 m/s    0.563 m/s    0.667 m/s          0.639 m/s
平均最大连续保持步数         0.00         0.00         0.00               0.00
保持步数 >=1 / >=3 / >=5     0/0/0        0/0/0        0/0/0             0/0/0
```

D 仍保住了 `8 m=20/20` 和 `5 m=20/20`，但没有超过 C 的 `3 m=20/20`；D 的 `5--3 m -> 3--1 m`
继续比例为 `17/20`，且 `1 m=0/20`。因此 `coef=0.5` 的低速径向推进奖励没有改善最终进入目标区，当前瓶颈仍是
`3 m -> 1 m` 的可持续低速推进，尚未转移为 `1 m` 内停车保持问题。D 的新奖励在 `8--5 m`、`5--3 m`
均为 0，只有 `3--1 m` 激活；正式 train/validation/test 的平均 active fraction 分别为 `0.0376/0.0385/0.0430`，
平均 reward 分别为 `0.001044/0.001228/0.001296`，训练 clip saturation 为 `0.0016`，validation/test 为 `0`。

本轮新增或更新的主要产物：

```text
src/experiments/single_uav_four_policy_comparison.py
src/experiments/single_uav_navigation_preserving_finetune.py
src/experiments/single_uav_terminal_approach_diagnostic.py
tests/test_single_uav_terminal_approach_diagnostic.py
data/synthetic/single_uav_terminal_radial_progress05_early_smooth_seed2026/
data/synthetic/single_uav_terminal_radial_progress05_early_smooth_smoke_seed2026/
data/synthetic/single_uav_terminal_radial_progress05_early_smooth_terminal_approach_diagnostic_seed2026/
logs/preliminary/single_uav_terminal_radial_progress05_early_smooth_seed2026/
logs/preliminary/single_uav_terminal_radial_progress05_early_smooth_terminal_approach_diagnostic_seed2026/
logs/preliminary/single_uav_terminal_radial_progress05_early_smooth_four_policy_comparison_seed2026/
plots/preliminary/single_uav_terminal_radial_progress05_early_smooth_four_policy_comparison_seed2026/
```

smoke 定向审计测试为 `21 passed`，全量回归为 `386 passed, 5 skipped, 1 warning`。本轮没有加入
goal hold reward，也没有自动开始下一轮机制修改。

#### Actor-action-constrained estimated-state Oracle 可行性诊断（2026-09-11）

在不训练、不更新 checkpoint、不改变 reward、curriculum、动力学、动作映射或成功判据的前提下，
基于上一轮 `single_uav_rl_equivalent_oracle_audit_seed2026` 的 20 个正常远距离配对场景，
新增并运行了第五种只读角色：`actor_constrained_estimated_state_oracle`。其状态输入仍只来自
RL 可见的 estimated node features；期望动作经过当前 MAPPO Actor 的真实变换
`Normal -> tanh -> max_delta_v/sqrt(3) per-axis` 后才送入 `environment.step`。

当前动作集合为：

```text
Actor:       |delta_v_i| <= 0.346410 m/s, i=1,2,3; theoretical norm max = 0.600000 m/s
Environment: ||delta_v||_2 <= 0.600000 m/s
```

两者不是同一个集合；Actor 的 box 是 environment norm-ball 的子集。对单位目标方向 `u`，
Actor 沿该方向的最大径向动作为
`min(max_delta_v, (max_delta_v/sqrt(3))/max(abs(u_i)))`。

五类策略统一结果如下：

```text
策略                                      8m       5m       3m       1m       stable    mean min(m)
truth oracle                              20/20    20/20    20/20    20/20    20/20       0.610
estimated-state oracle                   20/20    20/20    20/20    20/20    20/20       0.575
Actor-constrained estimated oracle       20/20    20/20    20/20    20/20    20/20       0.610
coef=0.25 baseline RL                    20/20    17/20    11/20     1/20     0/20       3.135
navigation-preserving fine-tuned RL      20/20    20/20     0/20     0/20     0/20       3.584
```

Actor 约束在 `79.10%` 的 constrained-oracle step 中改变了期望动作；但约束后的 Oracle 仍然
20/20 进入 1 m、20/20 稳定保持 5 steps。正常远距离场景的平均径向可达动作约为 `0.454 m/s`
（范围 `0.375--0.556 m/s`）。这说明当前 per-axis Actor 参数化虽比环境动作空间更窄，仍足以完成任务；
主要差距在 RL 策略对已有动作能力的使用方式，而不是 observation 不足、定位不可行、动力学不可行、
episode horizon 不足或 Actor 动作集合本身不可行。

距离段动作幅度与成功 constrained Oracle 的 mean action norm 比值已写入：

```text
logs/preliminary/single_uav_actor_constrained_oracle_seed2026/action_amplitude_distance_summary.csv
```

正式只读诊断产物：

```text
data/synthetic/single_uav_actor_constrained_oracle_seed2026/
logs/preliminary/single_uav_actor_constrained_oracle_seed2026/
plots/preliminary/single_uav_actor_constrained_oracle_seed2026/
```

第五角色及四个缓存角色的 schema 校验均通过，`summary.json` 中 `overall_schema_ok=true`，
checkpoint 和 cached policy state 未改变；新增定向测试为 `11 passed`。本轮停止在可行性诊断，
没有自动执行 Behavior Cloning、重新训练或 Actor 结构修改。

#### Oracle Demonstration → Behavior Cloning → BC-only Evaluation（2026-09-12）

基于 Actor-constrained estimated-state Oracle 已在相同 observation/action 约束下完成任务的结论，
本轮没有继续修改 reward、target-speed、terminal curriculum、稳定判据、动力学、动作映射、episode horizon
或 PPO 参数。新增了独立的 Oracle demonstration 生成、Actor-only Behavior Cloning 和只读 BC evaluation
入口。teacher 只读取 RL Actor 可见的 estimated `node_features`，其期望动作经过当前 Actor 的真实
`Normal → tanh → scaled physical delta-v` 约束后写入 demonstration；未使用 simulator truth 控制信息。

正式 demonstration 使用独立的 train/validation 场景和 noise seed：`48` 个 train episodes、`20` 个
validation episodes，共 `68` 条 canonical episode；train/validation 的 scenario seed 与 noise seed 均不重叠，
curriculum 在 train/validation/test 均关闭。每步保存 observation、physical delta-v target、goal distance、
speed、径向/切向速度、goal direction、braking component 及 action projection 诊断。

BC 从原始 coef=`0.25` baseline checkpoint 初始化，只训练 Actor mean 相关参数；critic、normalizer、log_std
冻结，不进行 PPO 更新。损失为最终 post-tanh scaled physical delta-v 的 MSE，不拟合 pre-tanh target，
也没有加入额外 radial/tangential loss。训练 `2406` 个样本、验证 `973` 个样本、`12` epochs 后：

```text
loss                           first       final
train physical-action MSE      0.0361173    0.0185977
validation physical-action MSE 0.0373503    0.0219903
```

在相同的 `20` 个正常远距离 test episodes 上的统一结果为：

```text
指标                         baseline RL   navigation fine-tuned   BC Actor   Actor-constrained Oracle
进入 8 m                     20/20         20/20                   20/20      20/20
进入 5 m                     17/20         20/20                   20/20      20/20
进入 3 m                     11/20         0/20                    17/20      20/20
进入 1 m                     1/20          0/20                    2/20       20/20
stable goal                 0/20          0/20                    0/20       20/20
平均每回合最小距离           3.135 m       3.584 m                 1.819 m    0.610 m
1 m 内平均速度               1.459 m/s     --                      1.234 m/s  0.709 m/s
平均最大连续保持步数         0.00          0.00                    0.00       5.00
保持步数 >=1 / >=3 / >=5     0/0/0         0/0/0                   0/0/0      20/20/20
```

BC 同时保住了远距离进入 `8 m=20/20`，并把进入 `3 m` 从 baseline 的 `11/20` 提升到 `17/20`；
但进入 `1 m` 仅为 `2/20`，stable 仍为 `0/20`。因此 BC 已证明原始 PPO 的探索/信用分配确实是重要瓶颈，
但当前 BC-only Actor 仍未学会 Oracle 的终端低速保持行为。下一阶段可以设计低学习率
`PPO + BC imitation anchor`，但本轮在 BC-only evaluation 后停止，没有自动执行 PPO。

正式产物：

```text
src/experiments/single_uav_oracle_behavior_cloning.py
tests/test_single_uav_oracle_behavior_cloning.py
data/synthetic/single_uav_oracle_demonstrations_seed2026/
data/synthetic/single_uav_bc_oracle_seed2026/
data/synthetic/single_uav_bc_only_evaluation_seed2026/
logs/preliminary/single_uav_oracle_demonstrations_seed2026/
logs/preliminary/single_uav_bc_oracle_seed2026/
logs/preliminary/single_uav_bc_only_evaluation_seed2026/
plots/preliminary/single_uav_bc_oracle_seed2026/
plots/preliminary/single_uav_bc_only_evaluation_seed2026/
```

相关定向测试为 `18 passed`；demonstration 根目录及四个 evaluation role 根目录的
`validate-preliminary` 均为 `ok=true`，每个 evaluation role 均为 `40` 个 episode，schema 完整，
checkpoint 文件保持未被评估过程改写。

## 2026-09-12：E 路线向量修正实验

在原始 `coef=0.25` checkpoint 上完成了 E：early-smooth + terminal target-velocity-vector correction
（系数 `0.5`，仅作用于估计距离 `(1,3] m`，不启用 goal hold reward）。正式配置为
`100 train / 20 validation / 20 test`，训练调度为 `90% normal + 10% terminal(主要 5--8 m)`，
`validate-preliminary` 为 `ok=true`。

E 的正常远距离 test 为：进入 `8 m=20/20`、进入 `5 m=20/20`、进入 `3 m=20/20`，
进入 `1 m=0/20`、stable `0/20`，最小距离约 `1.062 m`。因此该实验保护了远距离和 3 m
接近能力，但尚未解决 1 m 内的稳定停车问题；本阶段未继续加入 goal hold reward。

统一只读比较产物位于：

```text
src/experiments/single_uav_acde_comparison.py
logs/preliminary/single_uav_acde_comparison_seed2026/
plots/preliminary/single_uav_acde_comparison_seed2026/
```

#### BC-v2 与 DAgger round-1（2026-09-12）

在上一阶段 BC-only 结果基础上，本阶段没有修改 reward、braking progress coef、target-speed、terminal
curriculum、goal hold reward、stable hold 判据、动作映射、动力学、episode horizon 或 PPO 参数。
BC-v2 仍从 baseline Actor 初始化，只训练 Actor mean；critic、normalizer、log_std 冻结，使用
post-tanh scaled physical delta-v MSE，不执行 PPO。正式 BC-v2 使用 60 epochs、patience=8，最佳验证
epoch 为 60，validation MSE 从 `0.0373503` 降至 `0.0166720`。

BC-v2 闭环 test 仍只有 stable `1/20`，但进入 `3 m` 提升到 `18/20`、进入 `1 m` 提升到 `8/20`，
因此满足“离线拟合已改善但闭环仍有分布偏移”的 DAgger 条件。DAgger round-1 只在 train 场景中运行
BC-v2 学生策略，再用 Actor-visible estimated-state Oracle 对学生访问状态重标注；没有读取 truth 控制，
没有使用 test 场景，也没有更新策略直到进入单独的 Actor-only BC 训练阶段。

正式 DAgger 数据覆盖 `24` 个独立 scenario seed 和 `24` 个 noise seed，收集 `2274` 条 correction，
与原始 `2406` 条 expert train 样本合并为 `4680` 条训练样本，validation 仍为原始 disjoint expert
validation 的 `973` 条样本。correction 距离分布为：`>8 m=919`、`8--5 m=521`、`5--3 m=519`、
`3--1 m=269`、`<=1 m=46`。DAgger Actor 从 BC-v2 最佳 checkpoint 初始化，39 epochs 后 early stop，
最佳验证 epoch 为 31，validation MSE 为 `0.0175382`；critic、normalizer、log_std 均冻结，未执行 PPO。

相同正常远距离 test 场景下的统一结果：

```text
指标                         baseline RL   old BC    BC-v2     DAgger r1   Oracle
进入 8 m                     20/20         20/20     20/20     20/20       20/20
进入 5 m                     17/20         20/20     20/20     20/20       20/20
进入 3 m                     11/20         17/20     18/20     20/20       20/20
进入 1 m                     1/20          2/20      8/20      20/20       20/20
stable goal                  0/20          0/20      1/20      20/20       20/20
平均每回合最小距离           3.135 m       1.819 m   1.549 m   0.449 m     0.610 m
1 m 内平均速度               1.459 m/s     1.234     0.958     0.541       0.709
平均最大连续保持步数         0.00          0.00      0.25      5.00        5.00
保持步数 >=1 / >=3 / >=5     0/0/0         0/0/0     1/1/1     20/20/20    20/20/20
```

DAgger round-1 同时保住了远距离导航（进入 `8 m=20/20`），并在正常远距离起点上达到进入 `1 m=20/20`
与 stable `20/20`。这表明 BC-v2 的主要剩余问题确实是 student visited-state 分布偏移，而不是当前
动力学或 Actor action 接口不可行；本阶段仍停止在 BC-only/DAgger evaluation，没有自动进入 PPO+BC anchor。

正式产物：

```text
src/experiments/single_uav_bc_v2_dagger.py
tests/test_single_uav_bc_v2_dagger.py
data/synthetic/single_uav_bc_v2_oracle_seed2026/
data/synthetic/single_uav_dagger_round1_seed2026/
data/synthetic/single_uav_dagger_actor_seed2026/
data/synthetic/single_uav_dagger_evaluation_seed2026/
logs/preliminary/single_uav_bc_v2_oracle_seed2026/
logs/preliminary/single_uav_dagger_round1_seed2026/
logs/preliminary/single_uav_dagger_actor_seed2026/
logs/preliminary/single_uav_dagger_evaluation_seed2026/
plots/preliminary/single_uav_bc_v2_oracle_seed2026/
plots/preliminary/single_uav_dagger_actor_seed2026/
plots/preliminary/single_uav_dagger_evaluation_seed2026/
```

相关定向测试为 `21 passed`。DAgger 数据根目录及五个 evaluation role 根目录的
`validate-preliminary` 均为 `ok=true`，checkpoint 在只读评估期间未被改写。

## F：early_smooth + 径向对齐门控终端速度向量奖励

在原始 `coef=0.25` baseline checkpoint 上完成了 F 单因素实验：保持
`early_smooth` 目标速度曲线，以及 `90% normal full-distance + 10% terminal`
的 navigation-preserving fine-tuning 调度；新增项仅为 `1–3 m` 区间的
径向对齐门控终端 target-velocity-vector reward，系数为 `0.5`。未启用
goal hold reward，未修改 braking reward、动作空间、动力学、PPO/MAPPO 参数、
stable hold 判据或 validation/test 分布。

正式 `100 train / 20 validation / 20 test` 结果（相同正常远距离 test 场景）：

```text
指标                         A baseline   C early-smooth   E ungated   F gated
进入 8 m                     20/20        20/20            20/20       20/20
进入 5 m                     17/20        20/20            20/20       20/20
进入 3 m                     11/20        20/20            20/20       20/20
进入 1 m                     1/20         2/20             0/20        9/20
stable goal                  0/20         0/20             0/20        0/20
最小目标距离                 0.468 m      0.821 m          1.062 m     0.180 m
1 m 内平均速度               1.896 m/s    2.181 m/s        2.057 m/s   2.273 m/s
平均最大连续保持步数         0.00         0.00             0.00        0.00
保持步数 >=1 / >=3 / >=5     0/0/0        0/0/0            0/0/0       0/0/0
```

F 的向量奖励审计通过：激活距离来自 action 前估计距离，正奖励与非正径向运动
的冲突计数为 `0`，奖励上界违规为 `0`，实际最大值约 `0.4394 <= 0.5`。
因此 F 保住了远距离导航，并把进入 `1 m` 从 C 的 `2/20` 提高到 `9/20`；
但 stable goal 仍为 `0/20`，当前瓶颈已转移到进入 `1 m` 后的低速连续保持，
不能据此宣称已经完成稳定停车。

F 正式产物：

```text
src/experiments/single_uav_acef_comparison.py
tests/test_single_uav_terminal_target_velocity_vector_reward.py
data/synthetic/single_uav_F_radial_gated_vector05_early_smooth_seed2026/
logs/preliminary/single_uav_F_radial_gated_vector05_early_smooth_seed2026/
plots/preliminary/single_uav_F_radial_gated_vector05_early_smooth_seed2026/
logs/preliminary/single_uav_acef_comparison_seed2026/
plots/preliminary/single_uav_acef_comparison_seed2026/
```

F 相关定向测试为 `15 passed`；全量测试为 `409 passed, 5 skipped`，
正式结果的 `validate-preliminary` 为 `ok=true`。本阶段到此停止，未自动加入
goal hold reward 或继续修改其他机制。

## 2026-09-12：hierarchical-dagger-v1 分层控制工程验证

本阶段基于已验证的 single-UAV DAgger round-1 checkpoint，新增独立的
`hierarchical-dagger-v1` 只读诊断协议，不替换原始 Local EKF + GAT + MAPPO
直接输出 delta-v 的 baseline。层级数据流为：Local EKF estimated state → GAT/MAPPO
高层通信与资源动作、临时 local waypoint → 冻结 DAgger low-level Actor → 合法
physical delta-v → 现有 environment dynamics。DAgger checkpoint 未重新初始化、
没有 optimizer、没有梯度更新；原 checkpoint 文件 SHA256 为
`f6c83f451ce1d1bc23e4f51e2d2b27ba615ad0283058137adda0620bd2e35690`。

当前 MAPPO action contract 保持不变：communication/measurement head、bandwidth/resource
head 和三维运动 `delta_vx/delta_vy/delta_vz` 均仍存在；hierarchical smoke 仅丢弃
高层 motion delta-v，通信动作继续由 MAPPO action heads 产生，低层三维 delta-v 由
冻结 DAgger 执行。temporary waypoint 通过环境接口同步更新 estimated relative goal
vector、goal distance、goal direction 和 braking-distance features；控制只读取
Actor 可见的 estimated local observation，不读取 truth state。

单 UAV waypoint 泛化正式评估覆盖 `30` 个 test episodes：三个距离段
`2--5 m`、`5--8 m`、`8--10 m` 各 `10` 个 episode，覆盖前/后/左/右/上/下及四种
斜向，共 `10` 个方向、`30` 个 geometry/noise seed 组合。三个距离段均为
进入 waypoint 半径 `10/10`、stable `10/10`；总体进入 `8/5/3/1 m` 和 stable 均为
`30/30`，collision 与 timeout 均为 `0/30`。总体平均每回合最小距离为 `0.6445 m`，
1 m 内平均速度为 `0.4572 m/s`，平均最大连续保持为 `5` 步，保持步数
`>=1/3/5` 均为 `30/30/30`。因此达到开发门槛 `reach >=95%`、`stable >=90%`。

补充的只读动作诊断为：平均 action norm `0.2416 m/s`，平均径向 action projection
`0.0628 m/s`，平均切向 action `0.1634 m/s`，平均 braking magnitude
`0.0962 m/s`（实际执行 `0.0918 m/s`），braking action fraction `0.6642`，
effective braking fraction `0.6292`，action clipping `0/30`、clipping fraction `0`。
waypoint 初始状态三类 `rest/toward_waypoint/tangential` 各 `10` 个；步级 CSV 同时
保留径向速度、目标速度、制动动作及有效制动等逐步记录。

由于单 UAV waypoint 泛化通过，随后执行了只用于接口验证的 2 UAV hierarchical smoke，
没有训练 MAPPO。两架 UAV 分别使用独立 Local EKF 和独立冻结 DAgger inference，
GAT 正常读取邻居图，通信动作未被 DAgger 覆盖，delta-v 合法，低层 checkpoint
文件哈希和 Actor 参数哈希前后一致；2 UAV episode 的核心 schema validator 为
`ok=true`。该 smoke 只证明层级接口与数据流可运行，不代表未训练高层 MAPPO 已具备
协同策略性能，也不等同于正式 2 UAV 训练结果。

正式产物：

```text
src/experiments/single_uav_hierarchical_waypoint.py
tests/test_single_uav_hierarchical_waypoint.py
data/synthetic/single_uav_hierarchical_waypoint_generalization_seed2026/
data/synthetic/two_uav_hierarchical_dagger_smoke_seed2026/
logs/preliminary/single_uav_hierarchical_waypoint_generalization_seed2026/
logs/preliminary/two_uav_hierarchical_dagger_smoke_seed2026/
plots/preliminary/single_uav_hierarchical_waypoint_generalization_seed2026/
plots/preliminary/two_uav_hierarchical_dagger_smoke_seed2026/
```

hierarchical 定向测试为 `14 passed`；正式单 UAV 输出的 `validate-preliminary`
为 `ok=true`，2 UAV 输出由 `validate_episode_benchmark(scale_2)` 检查为 `ok=true`。
`pytest tests` 的回归结果为 `410 passed, 5 skipped, 2 failed`；两项失败来自已有
`safety_risk_screening` 绘图代码向当前 Matplotlib 版本传入不支持的 `tick_labels`
参数，未涉及本阶段 hierarchical 代码。完整 `pytest -q` 另受项目根目录已有
`pytest-tmp-e-*` 权限目录影响，未能完成收集。

## 2026-09-13：Actor 障碍物观测可辨识性审计（只读）

基于单 UAV 静态障碍物 Stage 1 formal checkpoint 和原始 `coef=0.25` baseline，新增
`src/experiments/single_uav_obstacle_observation_audit.py`，构造六个固定距离、固定
UAV/目标/速度且只改变障碍物方向的 paired probe：`front`、`left-front`、
`right-front`、`left`、`right`、`behind`。审计只调用冻结 Actor inference，不执行
`env.step`、optimizer 或 PPO 更新，并在运行前后核对 policy 与 checkpoint hash。

runtime 结果显示当前 Actor 的 node feature 为 `39` 维，最终 actor context 为 `79` 维：
15 个 estimated EKF state、15 个 covariance diagonal、3 个 goal-relative 分量、4 个
障碍物标量聚合量（minimum clearance、box/sphere count、timestep）和 2 个制动描述量。
六个方向在 Stage 1 checkpoint 上的 normalized node feature、actor context 和确定性
Actor action 均完全相同；baseline 也得到相同结论。当前 Actor 没有 obstacle relative
`dx/dy/dz`、bearing、绝对几何、障碍物 identity、pre-projection swept clearance 或
unsafe flag。相对位置与完整障碍物几何仍由 centralized critic、safety layer 和
diagnostics 使用，说明当前主要问题是 Blocker A：Actor 无法从输入中辨识主动左右绕障
所需的方向信息，而不是 action space 缺少横向控制能力。

本轮完整 `pytest tests` 为 `422 passed, 5 skipped`。首次运行因审计脚本 feature-table
字段名错误留下的 3 个不完整 CSV 未被覆盖；修复后的完整产物写入独立的 `_rerun` 目录，
原有 checkpoint 与既有实验目录未修改。下一步应先在新的 opt-in observation contract
中增加 UAV-local obstacle relative direction/geometry feature，完成 observation-only
smoke 后，再生成近障碍物 oracle/DAgger 数据；本轮没有修改 reward、curriculum、动力学、
action space、safety projection 或 checkpoint。

审计产物：

```text
logs/preliminary/single_uav_obstacle_observation_audit_seed2026_rerun/
plots/preliminary/single_uav_obstacle_observation_audit_seed2026_rerun/
```

## 2026-09-13：Stage 2A UAV-relative 静态障碍物方向观测增强（只读）

在 Blocker A 审计基础上，新增可选的
`single_uav_obstacle_direction_features_enabled` observation contract。默认值为
`false`，因此历史 39 维 node observation 和旧 checkpoint 行为保持不变；启用后在旧
前缀之后追加 3 个 feature，node dimension 为 `42`。新增索引为 `39--41`，分别是
`relative_obstacle_dx_body_m`、`relative_obstacle_dy_body_m` 和
`relative_obstacle_dz_body_m`。

方向量使用 EKF estimated position/yaw，而不使用真实 UAV pose。坐标约定为 ENU 世界坐标
（`+x` east、`+y` north、`+z` up）转换到 UAV yaw/body frame（`+x` forward、`+y`
left、`+z` up）。最近物理障碍物按 point-to-surface minimum clearance 选择，
`obstacle_id` 作为等距时的确定性 tie-break；clearance 和方向向量因此来自同一障碍物。
无物理障碍物时方向量为有限的 `[0, 0, 0]`，由原有 box/sphere count 区分缺失与中心障碍物。

Stage 1 checkpoint 通过显式 migration 生成独立的 42 维兼容 checkpoint：旧 GAT 首层和
critic 首层的历史输入 slice 原样复制，新增输入列置零，ObservationNormalizer 的旧 39
维统计原样保留，新统计使用 `mean=0、scale=1、count=0`；central normalizer 从 `171`
维扩展到 `174` 维。迁移和 probe 全程没有调用 PPO/MAPPO update，Stage 1 与 baseline
源 checkpoint SHA256 保持不变。

六方向 probe 的左右与前后输入均已可辨识，旧 39 维前缀差异为 `0`；迁移初始状态下
actor context/action 差异为 `0`，符合新增权重置零的兼容预期。这只证明方向信息已经
进入 Actor observation，不代表 Actor 已经学会主动避障。

Stage 2A 产物：

```text
src/experiments/single_uav_obstacle_direction_observation_stage2a.py
tests/test_single_uav_obstacle_direction_observation_stage2a.py
data/synthetic/single_uav_obstacle_direction_observation_stage2a_seed2026/
logs/preliminary/single_uav_obstacle_direction_observation_stage2a_seed2026/
plots/preliminary/single_uav_obstacle_direction_observation_stage2a_seed2026/
```

Stage 2A 定向测试 `9 passed`，完整回归为 `431 passed, 5 skipped`；observation-only smoke
通过，`stage2a_passed=true`。Stage 2B near-obstacle Oracle + DAgger 仅具备设计前置条件，
本阶段未自动执行。

为补齐“原始 flag=false baseline checkpoint”兼容性审计，另生成了不覆盖原 smoke 的
独立复核目录：

```text
data/synthetic/single_uav_obstacle_direction_observation_stage2a_seed2026_rerun/
logs/preliminary/single_uav_obstacle_direction_observation_stage2a_seed2026_rerun/
plots/preliminary/single_uav_obstacle_direction_observation_stage2a_seed2026_rerun/
```

复核报告确认 baseline 39 维 checkpoint 在 flag=false 下精确加载，Stage 1 checkpoint
迁移后为 42 维；两者均通过 normal/static probe，未调用训练更新，源 checkpoint hash
保持不变。

## 2026-09-13：Stage 2B 近障碍物几何 Oracle + DAgger smoke

基于 Stage 2A 的 42 维方向观测，新增独立的
`src/experiments/single_uav_obstacle_dagger_stage2b.py`。该模块复用了现有环境动作接口、
EKF 可见状态提取、checkpoint 加载、Actor-only imitation trainer、episode trace writer
和 Stage 2A probe；没有修改动力学、动作空间、reward、PPO/MAPPO、safety mode 或已有
checkpoint。Oracle 只使用 EKF estimated position/velocity/yaw、Stage 2A 障碍物方向特征
和静态障碍物几何，通过正常 `environment.step(delta_v)` 产生绕行动作，不读取 truth state，
不 teleport。

smoke 运行顺序为 feature-scale preflight、统一 relevant-obstacle selector、Oracle
safety-off/on sanity、DAgger 数据收集、Actor-only BC 更新和冻结 holdout 对照。训练集
调度严格为 `21/30` normal far-distance 与 `9/30` static pressure，收集样本为 `2400`
train、`600` validation，总计 `3000`；未使用 test 场景。Oracle sanity 的 safety-off/on
均为 `0` collision、`9/10` passed obstacle，safety projection 为 `0`，说明 Oracle
本身通过了进入 DAgger 的技术门槛。

技术 smoke 流程完成：新增方向权重从零变为非零，Actor-only 训练 `8` 个 epoch，BC
updates 为 `8`，DAgger rollout updates 与 PPO updates 均为 `0`；新增边界投影步数和
episode 计数也写入汇总诊断。新增定向测试为 `8 passed`，完整回归为 `438 passed,
5 skipped`。

但行为验收未通过，不能进入 Stage 2B 正式实验：在小规模 holdout 中，Stage 2B normal
test 仍为进入 8 m `5/5`，但进入 3 m 从冻结 Stage 2A 的 `5/5` 降为 `3/5`，进入 1 m
为 `0/5`、stable 为 `0/5`；static unseen holdout 没有出现向目标的有效进展。方向 probe
虽然动作已不再完全相同，但 left 与 right 两类的 lateral action 仍同号（分别约
`-0.2700` 与 `-0.2682`），尚未形成正确的左右绕行符号。因此当前 blocker 是
`static holdout no learned progress`、`left/right bypass action sign not learned`，并伴有
小规模 normal navigation regression；这不是 observation blocker，也不能用 collision=0
代替主动避障证据。

最终 smoke 报告写入独立的 `rerun3` 目录。`stage2b_summary.json` 将技术流程完成与行为
验收分开记录：`technical_smoke_passed=true`、`behavioral_acceptance_passed=false`、
`stage2b_passed=false`；正式 100/20/20、multi-seed、reward 修改和下一阶段均未执行。

```text
src/experiments/single_uav_obstacle_dagger_stage2b.py
tests/test_single_uav_obstacle_dagger_stage2b.py
data/synthetic/single_uav_obstacle_dagger_stage2b_smoke_seed2026_rerun3/
logs/preliminary/single_uav_obstacle_dagger_stage2b_smoke_seed2026_rerun3/
plots/preliminary/single_uav_obstacle_dagger_stage2b_smoke_seed2026_rerun3/
```

该自定义 DAgger smoke 目录不包含 formal `scale_1` benchmark 根结构，因此通用
`validate-preliminary` 对它返回 `ok=false`（空 scale 集合），这属于验证器适用范围限制，
不是 episode trace 已损坏；本阶段以专用 summary、CSV、checkpoint identity、trace
writer 输出和 pytest 回归作为验收依据。

## 2026-09-13：Stage 2B-R 左右平衡迭代 DAgger smoke

针对上一轮 Stage 2B 的左右监督偏置，新增独立的
`src/experiments/single_uav_obstacle_dagger_stage2br.py`，并配套
`tests/test_single_uav_obstacle_dagger_stage2br.py`。本轮只使用 Stage 2A 的 42 维观测、
现有环境动作接口和 Actor-only BC；没有启用 PPO，也没有修改 reward、动力学、动作空间、
动作映射、safety mode 或已有 checkpoint。Oracle 使用 Actor 可见的 EKF 状态、方向观测
和静态几何，镜像场景严格平衡 left/right，并按 3 轮 rollout → aggregate → BC 迭代。

最终独立 smoke 目录为：

```text
data/synthetic/single_uav_obstacle_dagger_stage2br_smoke_seed2026_rerun4/
logs/preliminary/single_uav_obstacle_dagger_stage2br_smoke_seed2026_rerun4/
plots/preliminary/single_uav_obstacle_dagger_stage2br_smoke_seed2026_rerun4/
```

最终数据为 24 个 normal far-distance 与 12 个 static mirror-pressure episode，训练/验证
样本为 `2700/900`，left/right Oracle query 均为 `600`，无轨迹泄漏。Oracle 方向符号
probe 为 `4/4`，safety sanity 的 on/off 均为 `0/10` collision、`9/10` passed obstacle、
`0` safety projection；3 轮 checkpoint 链和标准 `validate-preliminary` 均通过，后者返回
`ok=true`、`summary_present=true`、`schema_version=episode-v1.3`。

技术流程 gate 已通过，但行为 gate 未通过：最终 Actor 的 left/left-front lateral sign
正确，而 right/right-front 仍为错误负号（`2/4`）；静态 mirror-pressure 的
`entered_3m` 与 `entered_1m` 均为 `0/4`，boundary drift 也未通过。因此按提示要求没有
运行 holdout，也没有进入正式 Stage 2B 实验。当前结论是监督数据和 Oracle 链路已修正，
但 Actor 仍存在 direction imitation / static trajectory execution blocker，不能把这次
smoke 视为主动避障成功证据。

本轮新增定向测试为 `13 passed`。在项目临时目录权限修正后，完整 pytest 为 `442 passed,
5 skipped`，另有 2 个与本轮无关的既有 `safety_risk_screening` Matplotlib API 失败，原因是
当前 Matplotlib 不接受该模块中的 `boxplot(labels=...)` 参数；本轮未修改该模块。

## 2026-09-13：Stage 2B-R2 Actor 左右避障学习瓶颈诊断

新增独立诊断入口
`src/experiments/single_uav_obstacle_stage2br2_learning_bottleneck.py` 和对应契约测试
`tests/test_single_uav_obstacle_stage2br2_learning_bottleneck.py`。本阶段从 Stage 2A 的
42 维观测兼容 checkpoint 开始，只使用估计状态、严格镜像静态障碍物和
Actor-only tiny BC；没有 PPO 更新，也没有修改 reward、动力学、动作空间、safety、
Stage 2A/2B-R 旧目录或 checkpoint。

独立结果写入：

```text
data/synthetic/single_uav_obstacle_stage2br2_learning_bottleneck_seed2026/
logs/preliminary/single_uav_obstacle_stage2br2_learning_bottleneck_seed2026/
plots/preliminary/single_uav_obstacle_stage2br2_learning_bottleneck_seed2026/
```

8 个严格镜像 pair（32 samples，train/validation 为 24/8，pair-level 无泄漏）的监督
目标审计通过；左右 Oracle target 符号正确，右侧样本梯度非零并朝目标方向，左右梯度未
显示主要抵消。四个 tiny-BC 范围均在受控镜像数据上达到 `4/4`：仅解冻 `39:42`、完整
第一 GAT 层、Actor 后端/action head、完整 Actor 均可学习正确符号。因此该问题不是
observation、Oracle target 或 BC/action indexing 的直接错误，而是 sequential rollout
分布与行为组合仍未被 tiny overfit 证明。

只有最小 `39:42` scope 被用于后续 4-episode static smoke。它虽表现为 4/4 左右自适应
横向动作，但仅 2/4 episode 通过 obstacle/post-bypass 判据，且 boundary projection
未改善；因此 `stage2br2_passed=false`、`eligible_for_next_stage=false`，不应直接进入
正式 100/20/20。R2 summary、梯度 CSV、ablation checkpoint 和诊断图位于上述独立目录。

本阶段新增测试 `7 passed`，与 Stage 2A/Stage 2B-R 的相关回归合计 `21 passed`；项目完整
pytest 为 `449 passed, 5 skipped, 2 failed`，两项失败均为本轮之前已存在的
`safety_risk_screening.py` Matplotlib `boxplot(labels=...)` API 兼容问题。

## 2026-09-13：Stage 2B-R2 Follow-up sequential bypass / post-bypass recovery

新增独立的 Follow-up 诊断入口
`src/experiments/single_uav_obstacle_stage2br2_followup.py`，配套契约测试
`tests/test_single_uav_obstacle_stage2br2_followup.py`。本阶段严格限制为冻结 scope
rollout、极小规模 Actor-only sequential DAgger 和只读诊断；没有运行正式 100/20/20，
没有运行 PPO，也没有修改 reward、observation、动力学、动作空间、safety/boundary 规则
或既有 gate。由于早期输出目录已有部分结果且不允许破坏性删除，最终完整结果写入独立
`rerun6` 目录，既有目录均保留。

```text
src/experiments/single_uav_obstacle_stage2br2_followup.py
tests/test_single_uav_obstacle_stage2br2_followup.py
data/synthetic/single_uav_obstacle_stage2br2_followup_seed2026_rerun6/
logs/preliminary/single_uav_obstacle_stage2br2_followup_seed2026_rerun6/
plots/preliminary/single_uav_obstacle_stage2br2_followup_seed2026_rerun6/
```

Step 1 在完全相同的 4 个 static smoke 场景中得到：Stage 2A frozen 为 direction `0/4`、
passed/post-progress `2/4`、safety projection `173`；R2 `39:42` 为 direction `4/4`、
passed/post-progress `2/4`、boundary projection `117`；actor-head 为 direction、
passed/post-progress `4/4`、boundary projection `88`；full Actor 为 direction/passed
`4/4`、post-progress `2/4`、boundary projection `140`。actor-head 是行为上更好的最小
候选，但仍未满足完整 gate，因此执行了极小 sequential DAgger。

DAgger 数据包含 8 episodes、800 samples、4 个 mirror pairs，left/right 各 400 samples，
类别 left/right/left-front/right-front 各 200 samples；train/validation 按 mirror pair
划分且无 trajectory 或 pair 泄漏。实际 phase 计数为
`decision=8、bypass=164、post_bypass=600、recovery=4、boundary_onset=24`，其中
`approach` 没有样本，phase imbalance ratio 为 `150`，这本身构成后续数据覆盖审计的证据。
BC 仅训练 actor-head，critic、normalizer、log_std 冻结，PPO updates 为 `0`，44 个 epoch
后最佳 validation MSE 为 `0.018655`。

post-DAgger 同场景复评中 sequential policy 为 direction `3/4`、passed/post-progress
`1/4`、recovery `0/4`、boundary projection `204`、safety projection `0`、collision `0`；
相较 R2 的 `117` boundary steps 没有改善。按阶段拆分后，post-DAgger candidate 的 `204`
个 boundary steps 全部发生在 obstacle passed 之前，说明候选策略在完成绕障前就发生了
侧向漂移；而 sequential DAgger 数据收集阶段的 `228/228` 个 boundary steps 发生在
obstacle passed 之后，直接暴露出 post-bypass lateral persistence。`safety projection
174→0` 不能解释为 autonomous avoidance 改善，而与干预转移到 boundary layer 相容。
normal-navigation smoke 通过代码的弱保护门，但候选 mean minimum goal distance 约
`18.80 m`，明显差于 Stage 2A 的约 `8.87 m`，因此不能宣称无退化；最终 summary 还明确
记录 entered-8m 为 Stage 2A `1/4`、候选 `0/4`。

最终标记为 `stage2br2_followup_passed=false`、`eligible_for_next_stage=false`，当前唯一
下一步为先做只读的 post-bypass recovery / boundary-onset state-coverage audit，之后再决定
是否重新收集 DAgger 数据；暂不进行正式训练、PPO 或新的奖励改动。新增 Follow-up 定向测试
为 `10 passed`，Stage 2A/2B/2B-R/2B-R2 相关回归合计 `43 passed`。全量 pytest 为
`459 passed, 5 skipped, 2 failed`；两项仍是未被本阶段修改的
`safety_risk_screening.py` 中 Matplotlib `boxplot(labels=...)` 兼容性失败。

## 2026-09-13：Stage 2B-R2-R3 post-bypass recovery / boundary-onset coverage audit

新增隔离的 coverage-audit 入口
`src/experiments/single_uav_obstacle_stage2br2_r3_coverage_audit.py`，配套测试
`tests/test_single_uav_obstacle_stage2br2_r3_coverage_audit.py`。本阶段只修正诊断阶段判定、
coverage 场景和 episode-phase capped dataset 视图，没有进行 BC/PPO，也没有修改 reward、
动力学、动作空间、observation、safety/boundary projection 或 collision 判据。

最终独立结果写入：

```text
data/synthetic/single_uav_obstacle_stage2br2_r3_coverage_audit_seed2026_rerun4/
logs/preliminary/single_uav_obstacle_stage2br2_r3_coverage_audit_seed2026_rerun4/
plots/preliminary/single_uav_obstacle_stage2br2_r3_coverage_audit_seed2026_rerun4/
```

16 个 episode、8 个 mirror pairs，train/validation 按 pair 划分且无泄漏。通过 state-only
decision classifier 保留了真实 approach 状态，全部六个阶段均被多个 episode 覆盖；原始
post_bypass 样本占比过高，但按 episode-phase 限额后的 phase imbalance 从 `159.875`
降至 `16`。post-bypass lateral persistence 仍出现在 `12/16` 个 episode，boundary
projection 共 `299` steps，其中 `296` steps 发生在 post-bypass；这说明 coverage 问题
已经被单独暴露和缓解，但策略行为本身尚未解决。

正常导航审计仍有退化警告：Stage2A frozen entered-8m 为 `1/4`，actor-head 为 `0/4`，
candidate mean minimum goal distance 从约 `8.87 m` 变为约 `18.18 m`。因此本阶段不授予
BC 或 PPO 资格，下一步仍需先审查 balanced DAgger 数据和 post-bypass 恢复动作。
R3 及相关回归测试为 `48 passed`；全量 pytest 为 `464 passed, 5 skipped, 2 failed`，
两个失败仍是未修改的 `safety_risk_screening.py` Matplotlib 兼容性问题。验证详情见
`r3_execution_checks.json`。

## 2026-09-14：单 UAV 静态障碍物 post-bypass recovery / goal DAgger + BC

针对“绕障持续过久 → 横向漂移 → 边界层介入 → 无法恢复目标径向运动”的问题，新增独立入口
`src/experiments/single_uav_static_obstacle_dagger_recovery_goal.py` 和契约测试
`tests/test_single_uav_static_obstacle_dagger_recovery_goal.py`。本阶段使用 Actor 可见的
EKF 估计状态、Stage 2A 方向观测和静态地图几何 Oracle 采集 DAgger demonstration，再从
baseline 的 42 维迁移 checkpoint 做 Actor-only BC；critic、normalizer、log_std 冻结，未执行
PPO，未修改 reward、动力学、动作空间、safety/boundary 机制或已有 checkpoint。

训练 demonstration 使用多场景 seed、噪声 seed、左右镜像和不同起点/目标几何；train/validation
按 trajectory/pair 隔离，并按 episode-phase 平衡采样。正式独立产物位于：

```text
data/synthetic/single_uav_static_obstacle_dagger_recovery_goal_seed2026_rerun2/
logs/preliminary/single_uav_static_obstacle_dagger_recovery_goal_seed2026_rerun2/
plots/preliminary/single_uav_static_obstacle_dagger_recovery_goal_seed2026_rerun2/
```

正式 demonstration 原始样本为 `train=6645、validation=1239`，平衡后为 `2826/612`，
train/validation 分别覆盖 `72/16` 个 episode，trajectory/pair leakage 均为 `false`。
BC-only 的 20-episode 正常远距离评估中，原始 baseline 为进入 8 m `20/20`、进入 1 m `1/20`；
新 BC Actor 为进入 8 m `5/20`、进入 1 m `0/20`、stable `0/20`。静态 holdout 中 Oracle
为通过障碍 `20/20`、进入 1 m `20/20`、stable `20/20`；BC Actor 为通过障碍 `13/20`、
post-bypass recovery `13/20`，但进入 1 m 和 stable 均为 `0/20`，且发生 `319` 个
pre-risk/safety-projection steps。因 BC 未满足 normal entered-8m `>=19/20` 的首要保护条件，
本阶段不进入 PPO+BC anchor；下一步应先检查 demonstration coverage、BC 动作幅度/径向推进
误差和训练样本权重，不能把 collision=0 或通过障碍代替稳定到达证据。

smoke6 的 schema、checkpoint provenance、train/validation 隔离和 `ppo_updates=0` 均通过；
正式 rerun2 的 12 个角色/split benchmark 目录 `validate-preliminary` 均返回 `ok=true`。
项目测试目录回归为 `469 passed, 5 skipped`；直接扫描根目录会额外触及已有权限受限的
`pytest-tmp-e-20260912`，该环境问题不属于本阶段代码。

## 2026-09-14：Actor-action-constrained teacher 与 BC demo kind balance 复验

针对前一轮发现的 teacher 几何欧氏范数约束与 MAPPO Actor 分量盒约束不一致问题，
`src/experiments/single_uav_static_obstacle_dagger_recovery_goal.py` 增加了
`ActorConstrainedBoundaryAwareGeometricObstacleOracleController`。该适配器复用原有
估计状态几何绕障路线，仅调用已有 `actor_action_transform` 生成 Actor 可达的 teacher
delta-v；静态 teacher sanity 为 `4/4` 进入 1 m、`4/4` stable、最大 hold 均为 5，且
没有 pre-risk、safety projection、boundary projection 或 action clipping。

随后发现按 episode-phase 独立截断会使 BC 样本偏向静态场景（正式 rerun3 为 static
`2108` 对 normal `864`），导致远距离行为遗忘。新增确定性的 phase-stratified per-episode
cap 后 kind equal-total balance，并增加对应契约测试；BC 仍为 post-tanh scaled physical
delta-v 的简单 MSE，未修改 reward、动力学、动作空间或 PPO。新增回归为 `471 passed,
5 skipped`，目标模块测试为 `6 passed`。

独立 smoke10 和正式 balanced rerun4 产物分别位于：

```text
data/synthetic/single_uav_static_obstacle_dagger_recovery_goal_seed2026_smoke10/
data/synthetic/single_uav_static_obstacle_dagger_recovery_goal_seed2026_actorconstrained_balanced_rerun4/
logs/preliminary/single_uav_static_obstacle_dagger_recovery_goal_seed2026_smoke10/
logs/preliminary/single_uav_static_obstacle_dagger_recovery_goal_seed2026_actorconstrained_balanced_rerun4/
plots/preliminary/single_uav_static_obstacle_dagger_recovery_goal_seed2026_smoke10/
plots/preliminary/single_uav_static_obstacle_dagger_recovery_goal_seed2026_actorconstrained_balanced_rerun4/
```

正式 rerun4 的 balanced demo 为 normal/static 各 `864` 条，trajectory/pair leakage 均为
`false`。normal test 的 BC Actor 进入 8 m 为 `14/20`，静态 holdout 的 BC Actor 进入
8 m、1 m、stable 均为 `0/20`；同一评估中 actor-constrained Oracle 为静态 `20/20`
进入 1 m、`20/20` stable、最大连续 hold 为 5，且无安全层介入。故当前目标“学习到
稳定静态绕障并恢复目标径向运动”仍未完成，且 BC 未达到 normal `>=19/20` 保护门；
当前应继续检查 demonstration 状态覆盖/可观测 Markov 性与 BC 表达能力，不进入 PPO+BC
anchor，也不加入 goal-hold reward。

## 2026-09-14：Stage 2B-R3 Actor observation identifiability audit

针对 rerun4 中 Oracle 能完成静态绕障而 BC Actor 不能完成 normal→bypass→recovery 闭环的
问题，新增只读诊断模块
`src/experiments/single_uav_stage2br3_observation_identifiability.py` 及其契约测试。
本阶段没有执行 PPO/MAPPO、DAgger、BC 正式重训，也没有修改 reward、curriculum、动力学、
动作空间、safety 或 stable 判据。输入严格锁定为
`single_uav_static_obstacle_dagger_recovery_goal_seed2026_actorconstrained_balanced_rerun4`，
train/validation 按 episode/scenario 保持独立。

最终只读审计产物写入：

```text
data/synthetic/single_uav_stage2br3_observation_identifiability_seed2026_rerun3/
logs/preliminary/single_uav_stage2br3_observation_identifiability_seed2026_rerun3/
plots/preliminary/single_uav_stage2br3_observation_identifiability_seed2026_rerun3/
```

结果显示，42D Actor-visible normalized node observation 对 bypass/recovery 总体具有较强
可辨识性：BYPASS_LEFT↔RECOVERY 与 BYPASS_RIGHT↔RECOVERY 的 held-out 线性 probe 准确率
分别为 `0.980` 和 `0.970`，左右 bypass 直接 probe 为 `1.000`。Obstacle-geometry feature
消融后线性 probe 从 `0.904` 降至 `0.771`，说明 Stage-2A 障碍物几何特征确实提供了重要
信息；但没有显式 boundary-clearance、radial/tangential velocity 或 passed-state feature。

近邻审计仍发现局部相似 observation 对应不同 teacher action：BYPASS_LEFT↔RECOVERY 的
强冲突率约 `25.2%`，BYPASS_RIGHT↔RECOVERY 约 `19.1%`，合计记录到 `312/1433` 个强冲突
候选。与此同时，balanced train 中 recovery 有 `157` 条、`23` 个 episode，检测到
`42` 个 bypass→post/recovery transition；tiny-overfit 在当前 mean-path scope 的
physical delta-v MSE 可降至约 `2.63e-5`，actor-head-only 则明显较弱。这表明“全局 phase
不可辨识”或“当前 Actor 完全没有容量”都不成立，但局部 transition 条件冲突、时序闭环
分布和 BC 泛化之间的关系仍未被本审计单独证明。

因此本阶段按要求标记为 `INCONCLUSIVE`，Primary/Secondary/Tertiary cause 暂不强行排序，
不自动进入 PPO+BC anchor；下一步应针对 transition-window 的局部冲突做受控诊断，而不是
直接继续调 PPO 或加入 goal-hold reward。R3 targeted pytest 为 `8 passed`；全量回归为
`438 passed, 5 skipped, 2 failed, 43 errors`，失败/错误来自既有项目的 pytest 临时目录
权限和 `safety_risk_screening.py` 与当前 Matplotlib `boxplot(labels=...)` API 兼容性，
不是 R3 诊断模块测试。

## 2026-09-14：外部开源方法对照与 post-bypass recovery Teacher

针对单 UAV 静态障碍物“绕障动作持续时间过长、横向漂移、边界层持续介入、无法恢复目标径向运动”
的问题，新增独立入口 `src/experiments/single_uav_external_recovery_method_study.py`。
该入口参考 DAgger 的 student-visited-state 标注、静态障碍物局部影响衰减和 sigmoid soft fusion，
但没有复制外部任务结构，也没有修改 reward、动力学、动作空间、动作映射、PPO/MAPPO 或安全语义。
Teacher 使用 42D Actor 可见估计状态与静态地图几何，绕障通过后以带迟滞的连续权重释放 bypass，
并对估计切向速度施加 Teacher 标签侧阻尼；所有最终动作仍通过现有 Actor componentwise transform。

正式输出位于：

```text
data/synthetic/single_uav_external_recovery_method_study_seed2026/
logs/preliminary/single_uav_external_recovery_method_study_seed2026/
plots/preliminary/single_uav_external_recovery_method_study_seed2026/
```

20 个静态 Teacher-only 场景中，8/3/1 m 进入率均为 `20/20`，post-bypass recovery 为 `20/20`，
stable goal 为 `16/20`，最大连续保持步数均值为 `4.75`、最大为 `5`。随后完成的 Actor-only
DAgger/BC 仅作为冻结学习诊断：normal test 进入 8 m 为 `15/20`，未达到 `>=19/20` 的导航保护门槛；
因此本阶段停止在 BC，不自动进入 PPO+BC anchor。所有 12 个角色/分割评估目录的标准 schema
校验均为 `ok=true`，全量测试为 `483 passed, 5 skipped`。详细证据见
`logs/preliminary/single_uav_external_recovery_method_study_seed2026/` 下的
`external_method_comparison.md`、`current_failure_root_cause.md`、`teacher_design_report.md`、
`dagger_collection_report.md` 和 `final_evaluation_report.md`。

## 2026-09-14：Stage 2B-R4 BC / DAgger policy-preservation audit

界面重启恢复检查确认本阶段实验进程已结束，未重复启动训练或评估。R4 的 E0 control、E1
distance×phase sampling、E2 四种 Actor mean 解冻范围和统一 normal/static holdout 评估均已
写入 `data/synthetic/single_uav_stage2br4_bc_policy_preservation_seed2026/`、
`logs/preliminary/single_uav_stage2br4_bc_policy_preservation_seed2026/` 和对应 `plots/`。
E0 复现、20 个评估根目录的 schema 校验均通过；E3 因没有候选同时满足 normal 与 static 核心
门槛而按协议跳过，未进入 PPO。E1/E2 中最接近 normal 导航门槛的是 E1 与 E2-C（均为
`19/20` 进入 8 m），但 static holdout 均未达到既定保持/恢复门槛，因此当前停止在 R4。
本轮相关 targeted pytest 为 `7 passed`；全量回归的剩余失败来自既有 pytest 临时目录权限和
`safety_risk_screening.py` 的 Matplotlib `boxplot(labels=...)` 兼容性，不属于 R4 变更。

## 2026-09-16：可开关 Goal Recovery Correction Layer

新增独立低层控制模块 `src/control/goal_recovery_correction.py`，用于冻结 Actor 评估中的
post-bypass 目标恢复。模块只接收 Actor 可见的 EKF 估计位置、估计速度、场景目标位置和
SoftFusion Teacher 已有的估计状态 `release_latched/pass_progress/route_phase` 信号；不接收
simulator truth state，也不修改 UAV position、velocity、yaw 或 next state。

配置集中在 `FormalExperimentConfig`，默认严格关闭：

- `goal_recovery_correction_enabled=False`
- `goal_recovery_radial_gain=0.8`
- `goal_recovery_tangential_gain=0.7`
- `goal_recovery_max_delta_v_mps=0.6`
- `goal_recovery_activation_mode="post_bypass_only"`
- `goal_recovery_alpha_ramp_enabled=True`

冻结 evaluation 可通过 Python 参数 `goal_recovery_correction_enabled=True/False`，或 CLI
`--goal-recovery-correction` / `--no-goal-recovery-correction` 做同 checkpoint 的 ON/OFF
对照。未指定运行时覆盖时沿用配置值；旧 checkpoint 缺少新字段时使用 dataclass 默认值 OFF。
Teacher/bootstrap、DAgger 采集与训练标签始终不经过该层。

激活链为：`Actor physical delta-v + alpha * bounded correction` → 现有 Actor componentwise
transform → braking supervisor → motion action/speed clipping 与积分 → boundary projection →
scene safety projection。OFF、未 release、非 post-bypass 或 `alpha=0` 时结构性旁路矫正与
二次 transform，原 Actor 动作逐元素直接进入旧执行链。step 日志补充 Actor 原始动作、径向/
切向 correction、alpha、transform 前后动作、最终 applied 动作和 safety/boundary 标志，便于
之后做因果归因。快速测试入口：

```text
python -m pytest tests/test_goal_recovery_correction.py -q
```

本次只完成代码、配置与单元测试；没有修改 reward、动力学、observation dimension、Actor/GAT/
Critic、PPO/MAPPO/BC/DAgger 训练目标，也没有运行正式训练或 rollout。

## 2026-09-16：Goal Recovery Correction 独立冻结 ON/OFF 评估入口

新增 `src/experiments/single_uav_goal_recovery_correction_evaluation.py`。该入口只加载现有
BC Actor checkpoint，并在同一份有序 static holdout manifest 上比较 correction OFF 与 ON；
不会调用 external recovery 完整实验、DAgger/BC runner、Teacher action、bootstrap 或任何
PPO/MAPPO/BC 更新。

默认是 paired smoke，两种条件各使用相同的前 2 个 canonical static holdout 场景、seed、顺序、
safety、boundary、max steps 和 checkpoint。只有显式传入 `--formal` 才选择 20 个场景。输出根
必须尚不存在；paired 模式会在每个 data/logs/plots 根下创建独立 `off/`、`on/` 子目录，并在
logs 根保存 `goal_recovery_correction_comparison.json/.csv`。

PyCharm 可直接配置：

```text
Script path: D:\research project\src\experiments\single_uav_goal_recovery_correction_evaluation.py
Working directory: D:\research project
Parameters: --paired
```

单独模式可使用 `--goal-recovery-correction` 或 `--no-goal-recovery-correction`。默认 checkpoint
是外部恢复研究已经生成的固定 `bc_final/scale_1/checkpoints/best_validation.pt`，也可通过
`--checkpoint` 显式覆盖；脚本会记录解析后的路径、SHA-256、policy hash、observation dimension、
normalizer 来源和有序 scenario IDs/seeds。release gate 在 `environment.step` 之前只读取当前
Actor 可见 EKF 估计、场景 goal 与静态地图，并复用既有 SoftFusion pass-progress、clearance、
recovery-buffer 常量；不读取 Teacher 状态或动作。

## 2026-09-16：near-goal stable failure 离线诊断

新增 `src/experiments/single_uav_near_goal_stable_failure_diagnostic.py`，只读取已完成的
Goal Recovery Correction formal paired evaluation 中 correction ON 的 episode/step 日志，不加载
checkpoint、不创建环境、不执行 rollout、Teacher 或训练，也不修改原始 formal 输出。脚本按 CSV
字段签名定位 ON 日志，只选择 `entered_1m=1 && stable_goal=0`，并围绕首次进入 1 m 的 step 默认
截取前 15、后 20 步。

正式 `stable_goal` 判据来自 `src/simulation/environment.py`：估计目标距离不超过 1.0 m、simulator
actual speed 不超过 0.5 m/s，并连续保持 5 步。现有 formal step log 没有保存 post-step actual speed、
`single_uav_goal_speed_ok` 或稳定保持计数，因此工具不会声称精确重算判据；它把
`goal_distance_after_m[t]` 与同 episode 连续下一行 `t+1` 的估计径向/切向速度对齐，明确标记为
derived proxy。分类规则版本为 `near_goal_failure_rules_v1`，非互斥报告 `HIGH_ENTRY_SPEED`、
`CORRECTION_OVERSHOOT`、`TANGENTIAL_RESIDUAL`、`ACTOR_CORRECTION_CONFLICT`、
`INSUFFICIENT_BRAKING`、`GOAL_REGION_OSCILLATION` 和 `STABLE_HOLD_INTERRUPTED`；每类均保存阈值、
可评估性、缺失字段、替代来源与证据 step，primary label 仅按固定优先级汇总。

运行示例：

```text
python src/experiments/single_uav_near_goal_stable_failure_diagnostic.py \
  --evaluation-root logs/preliminary/single_uav_goal_recovery_correction_eval_formal20_seed2026 \
  --pre-steps 15 --post-steps 20
```

可用 `--no-plots` 跳过可选图片，`--quiet` 关闭逐 episode 输出。CSV/JSON 写入新的
`logs/diagnostics/single_uav_near_goal_stable_failure_seed2026/`；图片默认写入对应 `plots/diagnostics/`
目录。任一目标目录已存在时拒绝覆盖。

## 2026-09-16：真实 stable 判据日志与 frozen ON-only 验证入口

`single_uav_goal_recovery_correction_evaluation.py` 新增 `--actual-stable-validation` 专用模式，供后续
在 PyCharm 手动运行同一 frozen BC Actor、同一 formal20 场景和 correction ON。该模式强制
`formal=True`、`condition=on`，拒绝 OFF、paired 和 smoke 参数，并使用独立的新 data/logs/plots
根目录；目标根已存在时仍会拒绝覆盖。

运行前会只读核对上一轮 paired formal 的 `evaluation_summary.json`：checkpoint 绝对路径、文件
SHA-256、policy hash、scenario manifest hash，以及 20 个 episode 的 ID、scenario/noise seed、
scenario hash 和顺序必须完全一致。任何不一致都会在创建输出目录及 episode rollout 前失败。

step 日志直接复制 environment 同一次 stable 判定放入 `next_info` 的 canonical 字段：post-step
simulator actual speed、stable 使用的 estimated goal distance、distance/speed threshold、distance/speed
布尔条件、candidate、`_single_uav_stable_steps` counter 和 stable flag。它们不是 estimated-speed
proxy，也不在 evaluator 中重新计算 stable 判据。成功达到 hold threshold 的终止 transition 会先写入
step row，再结束 episode。未来手动运行参数为：

```text
--actual-stable-validation
```

默认新输出根为：

```text
data/synthetic/single_uav_goal_recovery_actual_stable_validation_seed2026/
logs/preliminary/single_uav_goal_recovery_actual_stable_validation_seed2026/
plots/preliminary/single_uav_goal_recovery_actual_stable_validation_seed2026/
```

本阶段只完成日志、身份锁定、CLI 和纯 helper/mock 测试；Codex 未执行 smoke、formal 或任何 UAV
rollout。
## Near-goal Terminal Capture v1（代码准备，未执行实验）

Goal Recovery Correction 现已支持默认关闭的近目标捕获子模式。它只在原 correction 已满足 post-bypass/released eligibility 且估计目标距离不大于配置半径时激活；默认半径为 `8.0 m`。该模式以现有 braking-distance desired speed 为上限，在接近 `1.0 m` stable 边界时平滑降至径向 floor `0.25 m/s`，并把径向 correction 分成推进与制动：只对推进分量做距离 fade，径向制动和切向阻尼不会随推进 fade 一起削弱。切向增益由原 `0.7` 平滑过渡到默认 `1.0`，合成 correction 仍使用原 `0.6 m/s` norm budget，之后仍经过原 action transform、safety projection、boundary projection 和环境动力学。

专用入口为 `src/experiments/single_uav_goal_recovery_correction_evaluation.py --terminal-capture-validation`。它固定使用 Goal Recovery Correction ON、formal20、原 checkpoint 和原 ordered scenario identity，并引用 `logs/preliminary/single_uav_goal_recovery_actual_stable_validation_seed2026/evaluation_summary.json` 作为只读 baseline。默认输出分别写入 `data/synthetic/single_uav_terminal_capture_validation_seed2026`、`logs/preliminary/single_uav_terminal_capture_validation_seed2026` 和 `plots/preliminary/single_uav_terminal_capture_validation_seed2026`；任一目录已存在时拒绝覆盖。可用 `--terminal-capture-radius`、`--terminal-radial-speed-floor`、`--terminal-radial-brake-gain`、`--terminal-tangential-gain` 定义候选 treatment，非默认参数会在 summary 中标为 non-canonical 并进入 candidate config hash。

本阶段只完成控制代码、日志、summary、CLI 与 pure/mock/AST 测试准备；未执行 `--terminal-capture-validation`、smoke、formal、任何 episode、`environment.step()` 闭环、Teacher rollout、PPO/MAPPO、BC 或 DAgger。

### Single-UAV Terminal Capture v2（2026-09-17，待人工冻结评估）

在现有 `GoalRecoveryCorrection` 内新增了默认关闭的 speed-dependent stopping governor 与 goal-boundary anti-stall。v2 使用 EKF 估计距离/径向速度，固定初始经验参数 `a_eff=1.697726 m/s²`（75 个 saturated-clean braking steps、11 个独立 episode 的 per-episode median p10，并由 actual-speed 交叉检查）和 `stopping_margin=1.328009 m`；速度包络为 `sqrt(0.25² + 2*a_eff*max(distance-1.0-margin, 0))`，且只会降低原 base desired radial speed。进入 1 m 后 desired radial speed 切换为 0；在 1 m 外且 radial error 为正时保留正常 radial propulsion authority，避免 v1 fade 导致边界外失速。8 m 半径仅保留给 v1 兼容、terminal gain blend 与分析日志，不再门控 v2 governor。

未来人工入口为 `src/experiments/single_uav_goal_recovery_correction_evaluation.py --terminal-capture-v2-validation`。该入口锁定 correction ON、Terminal Capture ON、v2 ON、formal20，并以 `logs/preliminary/single_uav_terminal_capture_validation_seed2026/terminal_capture_validation_summary.json` 对应的 v1 formal20 结果作为 checkpoint、policy、scenario manifest 和 ordered episode identity 的唯一基线。新结果写入独立的 `single_uav_terminal_capture_v2_validation_seed2026` data/log/plot roots，已存在即拒绝覆盖；日志目录包含 `terminal_capture_v2_validation_summary.json`、`terminal_capture_v2_validation_episodes.csv`、`steps.csv` 和 `episodes.csv`。本次只完成代码与纯单元测试准备，尚未执行该入口、smoke、formal 或任何 rollout。

### Terminal Capture v2 Extended-Horizon Diagnostic（2026-09-17，待人工执行）

新增独立入口 `--terminal-capture-v2-horizon-diagnostic`，用于将已经完成的 v2 formal20 原样复现，只把单 episode horizon 从 `max_steps=100` 扩展到 `120`。入口以 `logs/preliminary/single_uav_terminal_capture_v2_validation_seed2026` 为冻结基线，逐项校验 checkpoint 路径与 SHA-256、policy hash、20 个 ordered episode/scenario identity、controller config、stable criterion 以及除 horizon 外的完整 runtime config；8 m、a_eff、stopping margin、terminal tangential gain、0.6 correction cap 和全部 v2 控制公式均不变。

诊断输出写入独立的 `single_uav_terminal_capture_v2_horizon120_diagnostic_seed2026` data/log/plot roots，已存在即拒绝覆盖。日志至少包含 `terminal_capture_v2_horizon120_summary.json`、`terminal_capture_v2_horizon120_episodes.csv`、`steps.csv`、`episodes.csv`，并额外生成 recovery episode 对照表。step100/110/120 snapshot 分别表示执行第 100/110/120 步后的状态，即零基日志索引 99/109/119；extension window 从索引 100 开始。本阶段仅实现入口、身份审计和日志，没有执行该诊断或任何 UAV rollout，也不预判延长 horizon 是否足以完成 stable goal。

### Terminal Capture v2.1 Near-Goal Radial Conflict Arbitration（2026-09-17，待人工验证）

Stage 0 只读审计使用既有 horizon120 `steps.csv`，筛选 passed obstacle、post-bypass recovery、v2 anti-stall、距离大于 1 m 且 radial error 为正的样本。480 个样本中 Actor 径向反向 441 个（91.875%）；step≥100 的 140 个样本中为 132 个（94.286%）。六个仍未进入 1 m 的重点 episode 在 extension window 共 119 个样本，其中 118 个反向；平均 Actor radial 为 `-0.130714 m/s`，平均 inward correction 为 `+0.133947 m/s`，平均 pre-transform net radial 仅 `+0.003233 m/s`。该证据满足“重复出现 Actor–correction 径向抵消”的实现 gate，但不构成 v2.1 有效性的实验结论。

v2.1 新增默认关闭的 `terminal_radial_conflict_arbitration_enabled`，第一版 gain 固定为 `1.0`。仅当 v2、v2.1、既有 correction eligibility/post-bypass/release/state gate、anti-stall、距离大于 stable threshold、positive radial error 全部成立且 raw pre-transform Actor radial 小于零时，使用既有 `motion.goal_direction` 计算 `gain × max(0, -actor_radial) × goal_direction`。补偿作为 external Goal Recovery correction 的一部分，在既有 combined correction `0.6 m/s` norm cap 之前加入；之后仍按原有 cap、alpha、Actor fusion、action transform、safety、boundary、dynamics 顺序执行。Actor raw action 从不重写，Actor tangential action和 v2 braking/governor 均不修改；v2.1 OFF 或任一 gate 不满足时保持原 v2 arithmetic exact-equivalent。

未来人工入口为 `src/experiments/single_uav_goal_recovery_correction_evaluation.py --terminal-capture-v21-validation`。该入口锁定 correction/Terminal Capture/v2/v2.1 ON、formal20、`max_steps=100`，并以已完成的 v2 formal100 为冻结 checkpoint、policy、scenario 和 ordered episode identity 基线；允许的唯一行为配置差异是 arbitration enable 从 false 变为 true，gain 仍为 1.0。输出写入独立 `single_uav_terminal_capture_v21_validation_seed2026` data/log/plot roots，包含 `terminal_capture_v21_validation_summary.json`、`terminal_capture_v21_validation_episodes.csv`、`steps.csv` 和 `episodes.csv`，已存在即拒绝覆盖。本阶段只完成离线审计、代码和纯单元测试，不执行 v2.1 formal、smoke 或任何 UAV rollout。

## 2026-09-17：Terminal Capture v1 有效制动能力离线审计

`src/experiments/single_uav_terminal_braking_authority_audit.py` 只读取已经存在的 Terminal Capture v1
formal20 CSV/JSON，不导入或调用 simulator、controller、Actor、Teacher 或训练代码。它按 episode 保留重复
测量结构，恢复 braking sample 的 goal radial direction，并分别审计 raw request、0.6 correction cap、
Actor fusion、action transform 和最终 dynamics-applied velocity increment。默认命令为：

```text
python src/experiments/single_uav_terminal_braking_authority_audit.py
```

默认输入为 `logs/preliminary/single_uav_terminal_capture_validation_seed2026/`，输出写入全新的
`logs/preliminary/single_uav_terminal_braking_authority_audit_final_seed2026/`。输出目录已存在时拒绝覆盖；脚本
生成 `braking_authority_summary.json`、`braking_authority_steps.csv`、
`braking_authority_episodes.csv` 和 `braking_authority_report.md`。该入口只做离线日志分析，不执行
`environment.step()`、任何 episode、smoke/formal rollout、训练或控制参数修改。

## 2026-09-17：Terminal Capture v2.1 目标边界抖动离线审计

新增 `src/experiments/single_uav_terminal_v21_boundary_chattering_audit.py`，只读取已经完成的 v2.1
formal100 的 `steps.csv`、两个 episode 汇总 CSV 和验证 summary；它不导入在线控制、仿真、策略或训练
模块，也不执行 `environment.step()` 或任何新 rollout。脚本逐步重建 canonical stable 判据（估计目标距离
不超过 1.0 m、simulator actual speed 不超过 0.5 m/s，连续 5 步），并在生成结果前要求重建 counter
与日志逐行一致。

审计将 counter reset 区分为 distance-only、speed-only、both 和 unexplained，按首次进入 1 m 后的原始
轨迹识别双向 crossing 与连续的 `distance > 1 m` outward excursion。enter-radius screen 使用
`1.00/0.98/0.96/0.94/0.92/0.90/0.88/0.85 m`：从首次达到候选半径的当行开始（inclusive），检查后续
是否存在连续 5 行原始 canonical candidate；这 5 行不要求继续处于候选 enter radius 内。release-radius
screen 固定使用原始 1 m excursion 分段，仅当 excursion peak 严格大于候选 release radius 时计为触发，
不会为每个候选半径重新分段。两种 screen 都只是既有轨迹上的观察性描述，不能解释为改变控制切换后的
反事实 rollout。

默认命令为：

```text
python src/experiments/single_uav_terminal_v21_boundary_chattering_audit.py
```

结果写入全新的
`logs/preliminary/single_uav_terminal_v21_boundary_chattering_audit_seed2026/`，包含 JSON summary、episode、
crossing、excursion、radius screen CSV 和 Markdown 报告；目标目录已存在时拒绝覆盖。报告明确区分未来
controller mode 的 enter/release switching boundary 与保持不变的 1.0 m evaluation stable boundary。
当前日志没有独立 actual goal distance，因此 estimated-vs-actual distance noise 归因固定报告为
`NOT_ASSESSABLE`，不据此猜测噪声机制。本阶段不修改 v2.1 或其他控制逻辑，也不执行 smoke、formal、
训练或 UAV rollout。

本次冻结日志审计已完成：20 个 episode、1968 个 step 的 canonical stable counter 逐行重建完全一致。
entered 的 S/F 组共有 10 次 reset，全部为 distance-only；首次进入 1 m 之后记录到 10 次
inside→outside 和 9 次 outside→inside recrossing。四个 F episode 中，0015 仅归类为
`DISTANCE_CHATTER`，0003/0016/0019 同时具有与随后出界直接关联的 hold-latched inward-region
连续负径向速度证据，归类为 `MIXED`。固定网格给出的下一阶段候选是 enter 半开区间
`[0.98, 1.00) m`（区间内部值尚未逐点扫描）和 release `1.05–1.08 m`；这只是下一轮实验设计范围，
不是参数冻结或有效性结论。最终离线 verdict 为 `AUDIT_READY_FOR_V22`，但 hysteresis 单独使用不被
当前证据判为充分，restorative inward hold 仅标记为值得后续受控实验检验。

## 2026-09-18：Terminal Capture v2.2 Hold Latch（代码准备，未执行实验）

在既有 Goal Recovery Correction 内加入默认关闭的 `terminal_capture_v22_enabled`。v2.2 仅在
Terminal v2、v2.1 arbitration、post-bypass/release gate 均有效时使用 episode-local hold latch：
估计距离不大于 `0.98 m` 时进入，锁存后在 canonical `1.0 m` 内以零径向目标继续制动；漂移到
`(1.0, 1.05) m` 时复用原 `0.25 m/s` radial floor，按
`0.25 * clip((distance - 1.0) / (1.05 - 1.0), 0, 1)` 生成小幅向内恢复目标；达到
`1.05 m` 时释放并在同一调用恢复原 v2/v2.1 capture。restorative radial error 继续走原 v2.1
Actor 径向冲突仲裁与既有 `0.6 m/s` correction cap，之后的 action transform、safety、boundary、
dynamics 均未改变。每次 episode 开始显式 reset latch，v2.2 OFF 保持 v2.1 输出兼容。

未来人工验证的 PyCharm 配置如下（本次没有运行）：

```text
Script path: D:\research project\src\experiments\single_uav_goal_recovery_correction_evaluation.py
Working directory: D:\research project
Parameters: --terminal-capture-v22-validation
```

该入口锁定 correction/Terminal Capture/v2/v2.1/v2.2 ON、formal20、`max_steps=100`、enter
`0.98 m`、release `1.05 m`、restorative ON，并以已完成的 v2.1 formal100 为只读身份基线。
本阶段仅完成控制逻辑、日志、summary、CLI、pure/helper/mock 测试与静态检查；没有执行 validation、
formal、smoke、`environment.step()`、任何 UAV rollout 或训练。

## 2026-09-18：Terminal Capture v2.3 Target-Radius Guard（代码准备，未执行实验）

Stage 0 已修复 v2.2 evaluation aggregate 中“每 episode 最大值的均值被命名为全局最大值”的
schema 错误；新字段明确区分 global maximum 与 mean episode maximum，未重新运行 v2.2 formal。
Stage 1 随后只读扫描既有 v2.2 formal100 的 20 个 episode、1,968 个 step：9 个 distance-only
counter reset 中有 8 个存在 1–2 步 outward precursor。固定候选扫描结果为：`0.94 m` 覆盖 8/9
但成功 episode 介入率为 58.82%，`0.95 m` 覆盖 7/9、介入率 47.06%，`0.96 m` 覆盖 7/9、
介入率 35.29%，`0.97 m` 只覆盖 6/9。按“覆盖至少 7/9 后选择介入更少的最大 target”规则，
离线 verdict 为 `AUDIT_READY_FOR_V23`，选择 `0.96 m`。该扫描只是冻结轨迹上的 counterfactual
command 计算，不代表改变 controller 后的因果轨迹预测。审计产物位于
`logs/preliminary/single_uav_terminal_capture_v23_preboundary_audit_seed2026/`。

在 v2.2 episode-local hold latch 之上新增默认关闭的
`terminal_capture_v23_enabled`。完整 v2/v2.1/v2.2/v2.3 gate 打开且 latch 保持时，
以冻结目标半径 `0.96 m` 和 release 半径 `1.05 m` 使用统一向内目标
`0.25 * clip((distance - 0.96) / (1.05 - 0.96), 0, 1) m/s`；目标半径以内为零，
`0.96–1.0 m` 记录为 pre-boundary，`1.0–1.05 m` 记录为 post-boundary restore。
达到 `1.05 m` 时仍由 v2.2 同调用 release/capture 逻辑处理。v2.3 复用 v2.1
径向冲突仲裁和既有 correction cap，不修改 Actor、action transform、reward、dynamics、
obstacle 或 stable criterion；v2.3 OFF 保持 v2.2 算术路径不变。

未来人工验证入口（本次没有运行）：

```text
Script path: D:\research project\src\experiments\single_uav_goal_recovery_correction_evaluation.py
Working directory: D:\research project
Parameters: --terminal-capture-v23-validation
```

该入口锁定 correction/Terminal Capture/v2/v2.1/v2.2/v2.3 ON、formal20、`max_steps=100`、
target `0.96 m`，并以 v2.2 formal100 的 checkpoint、policy、manifest、ordered episode
artifacts 为只读基线；输出使用新的 v2.3 validation roots 和
`terminal_capture_v23_validation_summary.json` / `terminal_capture_v23_validation_episodes.csv`。
本阶段仅完成控制逻辑、日志、summary、CLI、pure/helper/mock 测试与静态检查；没有执行 validation、
formal、smoke、`environment.step()`、任何 UAV rollout 或训练。

## 2026-09-18：Terminal Capture v2.3 Pre-Boundary Radial Authority Repair（代码准备，未执行实验）

Stage 0 对既有 v2.3 formal100 日志进行了只读复核：20 个 pre-boundary active step 中有 11 个
`terminal_v23_radial_error_mps > 1e-9`，这 11 个 step 的 radial propulsion norm 全部
`<= 1e-9 m/s`，非零计数为 0，因此确认实现缺陷。根因是 v2.3 在 `0.96–1.0 m` 内给出了正的
径向跟踪误差，但沿用了 v1 的近目标 propulsion fade；该 fade 在这一距离带为零，而 v2 anti-stall
只会在 canonical `1.0 m` 边界之外恢复 authority。

修复仍复用原 radial propulsion 分支、原 radial gain 和原 goal direction。只有在 v2.3 明确启用、
hold latch 已锁存、v2.3 guard 与 pre-boundary 同时有效且 radial error 严格大于 `1e-9` 时，
才把现有 propulsion scale 恢复为完整 authority；其他路径继续使用原 v1 fade。没有新增第二套控制器，
也没有修改 braking/propulsion 互斥、`<=0.96 m` 零目标、v2.1 仲裁、`0.6 m/s` correction cap、
action transform、reward、dynamics 或 stable criterion。新增的 authority 日志字段只读取已计算结果，
用于区分 requested/executed radial propulsion、允许/抑制状态、原因以及 v2.1 前后的净径向跟踪量。

为后续人工复验准备了独立入口（本次严禁且未执行）：

```text
Script path: D:\research project\src\experiments\single_uav_goal_recovery_correction_evaluation.py
Working directory: D:\research project
Parameters: --terminal-capture-v23-authority-repair-validation
```

入口锁定既有 v2.3 formal20/max100 的 checkpoint、policy、scenario manifest、ordered episode、
controller 参数、action transform 与 stable criterion，并写入独立的
`single_uav_terminal_capture_v23_authority_repair_validation_seed2026` data/log/plot roots；结果文件名为
`terminal_capture_v23_authority_repair_validation_summary.json` 和
`terminal_capture_v23_authority_repair_validation_episodes.csv`。当前状态为 prepared-only；本次只进行
静态编译、pure/helper 单元测试、差异检查和旧日志哈希复核，没有运行该入口、smoke、formal、
`environment.step()`、任何 UAV rollout 或训练。

## 2026-09-18：Single-UAV Near-Goal Capture Latency Offline Audit（已完成）

新增只读审计入口 `src/experiments/single_uav_near_goal_capture_latency_audit.py`。它只读取冻结的
authority-repaired v2.3 formal100 `steps.csv`、episode CSV 和 summary JSON，距离进展统一使用
`goal_distance_after_m` 的 post-step 序列，实际速度使用 `post_step_actual_*`；不导入 controller、
simulator、policy、training 或 environment，也不调用 rollout、`environment.step()`、smoke 或 formal。

审计锁定 20 个 episode、1,955 个 step，检查 episode/scenario/noise identity、连续 step、重复行、
NaN 分布、checkpoint/policy/manifest/config hash 与 frozen source 一致性。目标失败 episode 为
`test_r99_episode_0001`、`test_r99_episode_0008`、`test_r99_episode_0017`：三者均在 100 步内未
进入 1 m，但未观察到 late safety/boundary projection。逐 episode 诊断为：0001 为
`MIXED + HORIZON_LIMITED_APPROACH + BRAKING_PROPULSION_OSCILLATION`，0008 为
`HORIZON_LIMITED_APPROACH`，0017 为 `MIXED + HORIZON_LIMITED_APPROACH + ACTOR_RADIAL_CONFLICT`。
这些是冻结轨迹上的探索性证据，不是反事实因果结论；没有证据要求重新打开 Terminal Hold v2.3。

规范输出位于 `logs/preliminary/single_uav_near_goal_capture_latency_audit_seed2026/`，包括 summary、
episode diagnostics、late-window steps、reference comparison、phase timeline 和
`near_goal_capture_latency_report.md`。报告状态为 `CAPTURE_LATENCY_AUDIT_READY`；唯一建议的后续
实验是另行授权后仅将 `max_steps=100→120` 的 horizon-only diagnostic，本轮未执行，也未修改任何
controller、Actor、governor、action transform、hold、source logs 或训练/rollout 状态。新增纯离线
审计测试 4 项全部通过。

## 2026-09-18：Single-UAV Near-Goal Capture Horizon-Only Diagnostic（准备完成，未执行）

HORIZON120_DIAGNOSTIC_READY

已新增独立的 `--terminal-capture-v23-authority-repair-horizon120-diagnostic` 入口，严格固定
`formal=true`、`condition=on`、20 个 canonical static-holdout episode，并只把
`max_steps=100` 改为 `120`。入口不会复用或覆盖
`--terminal-capture-v23-authority-repair-validation`；data/log/plot 根分别为：

```text
data/synthetic/single_uav_near_goal_capture_horizon120_diagnostic_seed2026/
logs/preliminary/single_uav_near_goal_capture_horizon120_diagnostic_seed2026/
plots/preliminary/single_uav_near_goal_capture_horizon120_diagnostic_seed2026/
```

候选运行前会读取并严格校验只读基线
`logs/preliminary/single_uav_terminal_capture_v23_authority_repair_validation_seed2026/`，
包括 baseline summary SHA-256 `4048b73c42423d56eb95f006d8177b205752b4d5d4580c28d65d699e551ef2e2`、checkpoint SHA-256
`b7ed3ba79cf3055906bd0425cf4a7c4e613b206a8680ebf17f3b586fbc31b3c6`、policy hash
`c423ad25f7832ad74fc1782ac56aabfaeacb2f004e217cb5a02b225cd04a78b5`、manifest hash
`becf1bba9fb868787356889f3825c71645788ff766e6ead1506886ec0fbf2114` 和 ordered episode
identity。fail-fast identity lock 覆盖 checkpoint、policy、observation、scenario/order、reward、
dynamics、obstacle/safety/boundary、action transform、stable criterion、max correction、
`a_eff`、stopping margin、v2 governor、v2.1 arbitration、v2.2 parameters、v2.3 parameters、
authority-repair behavior 与 controller config；最终要求 `only_horizon_changed=true`。

后处理明确以零基索引 `0–99` 为 first100、`100–119` 为 extension（自然语言 step 101–120），
并输出 `entered1_*`/`stable_*` 六类分类、step100/110/120 snapshots、扩展区距离减少、径向/切向
速度、Actor opposing fraction、v2.1 conflict fraction、propulsion/braking switch，以及 collision、
safety projection、boundary projection、action clipping、terminal saturation 的 first100/extension-only
计数。恢复对照表和 focus 表保留 `test_r99_episode_0001`、`0008`、`0017`。

本轮只修改 evaluation harness、README 和纯/mock 测试；`src/control/goal_recovery_correction.py`、
v2/v2.1/v2.2/v2.3 arithmetic、authority repair、Actor/GAT/Critic、reward、dynamics、action
transform、stable criterion、source logs 均未修改。训练、formal/smoke、environment.step、rollout 和
新 UAV trajectory 均未执行；新输出根也未创建。待用户人工运行该入口后再读取结果，本轮在准备完成处停止。

准备状态的单独报告见 `sections/single_uav_near_goal_capture_horizon120_diagnostic_preparation.md`。

## 2026-09-18：Single-UAV Upstream Obstacle-Bypass Failure Offline Audit（已完成）

UPSTREAM_BYPASS_AUDIT_READY

本阶段对既有 horizon120 authority-repaired v2.3 日志执行了只读离线审计，输入固定为
`logs/preliminary/single_uav_near_goal_capture_horizon120_diagnostic_seed2026/`，共 20 个 episode、
2,182 个 step。身份门控确认目标失败为 `0002/0004/0005/0006/0011/0012/0013/0014/0020`，成功参考为
`0001/0003/0007/0008/0009/0010/0015/0016/0017/0018/0019`；两组的
`passed_obstacle`/`post_bypass_recovery` 标签与请求完全一致，`only_horizon_changed=true`，源文件审计前后
SHA-256 未变化。

审计只读取 CSV/JSON，不导入 simulator、environment、controller、policy 或 training runtime，不调用
`environment.step()`，没有 rollout、训练、formal、smoke，也没有修改任何源日志或控制器。正径向定义为
朝向 goal，切向字段是速度大小而非带符号分量；日志缺少 UAV/障碍横向几何与坐标手性，因此报告
`OBSTACLE_GEOMETRY_PARTIALLY_OBSERVABLE`，左右绕行方向保持 `NOT_IDENTIFIABLE`。

现有日志的安全投影共 5 步（0013=4、0014=1），边界投影共 41 步（0013=7、0014=34），动作裁剪与两者
分开统计。九个失败的 Goal Recovery active steps 均为 0，符合 `activation_mode=post_bypass_only`，因此
terminal Goal Recovery 不是这些 upstream failures 的直接来源。冻结轨迹上的主诊断簇为：6 个
`EXCESSIVE_LATERAL_PERSISTENCE`、1 个 `RADIAL_PROGRESS_COLLAPSE`、1 个
`SAFETY_LAYER_DOMINATED`、1 个 `BOUNDARY_DRIFT_AFTER_AVOIDANCE`；这些是证据标签，不是反事实因果结论。

规范产物位于 `logs/preliminary/single_uav_upstream_obstacle_bypass_failure_audit_seed2026/`，包括机器可读
summary、episode diagnostics、phase timeline、obstacle-window steps、success-IQR comparison、projection
breakdown 以及 `upstream_bypass_report.md`。本阶段在报告生成后停止，不启动下一项实验。

## 2026-09-18：Single-UAV Signed Bypass Observability Diagnostic（准备完成，未执行）

SIGNED_BYPASS_OBSERVABILITY_READY

已完成 Stage 0 静态审计并加入有符号绕障观测旁路。底层动作/状态/障碍 contract 仍是 3D，
但冻结 static holdout 的 start、goal、obstacle center 均位于 `z=20` 水平面；因此采用
`goal_projected_xy_left` 冻结横向轴（正号为 goal-frame-left，负号为 goal-frame-right），并单独记录
ENU vertical 分量。XY 退化时严格输出 `SIGNED_LATERAL_AXIS_NOT_WELL_DEFINED`。

新增独立纯 helper、可选 CLI flag
`--single-uav-signed-bypass-observability-diagnostic`、轨迹 fingerprint fail-fast
`LOGGING_PERTURBED_TRAJECTORY` 与独立输出根：

```text
logs/preliminary/single_uav_signed_bypass_observability_diagnostic_seed2026/
```

未来运行会生成 `signed_bypass_summary.json`、`signed_bypass_episodes.csv`、
`signed_bypass_steps.csv`、`signed_bypass_focus_failures.csv` 和
`signed_bypass_direction_transitions.csv`。旁路只读取既有 estimated/actual state、tracker obstacle
geometry 和动作流水线，不进入 observation、reward、controller、RNG 或 reset；不调用 Teacher/Oracle。

当前 baseline horizon120 summary/current evaluation config 没有独立持久化的 reward、dynamics、
action-transform、safety-projection 或 boundary-projection identity digest；五项 identity 因此均 fail-closed
为 `NOT_INDEPENDENTLY_VERIFIED`，原有 `same_*` 聚合检查只作为 supporting context。step-level trajectory
fingerprint 仍提供行为等价保护，并保留 `LOGGING_PERTURBED_TRAJECTORY` fail-fast。

本次只完成静态编译与 51 项纯离线测试，未运行该 CLI、environment.step、rollout、formal、smoke 或训练，
未修改历史日志、Actor/GAT/Critic、Goal Recovery、reward/dynamics、safety/boundary、action transform。
准备报告见 `sections/single_uav_signed_bypass_observability_diagnostic_preparation.md`，等待人工运行。

## 2026-09-18：Single-UAV Encounter-Relative Lateral Exit Authority Offline Audit（已完成）

LATERAL_EXIT_AUTHORITY_AUDIT_READY

本轮只读取既有 `single_uav_signed_bypass_observability_diagnostic_seed2026` 的五个冻结
CSV/JSON artifact，对 20 个 episode、2,182 个 step 做 encounter-relative 离线审计。身份门确认
`only_logging_changed=true`、`logging_perturbed_trajectory=false`，checkpoint、policy、observation、
ordered episode、scenario manifest、max steps 与 frozen controller identity 全部匹配；审计前后五个
源文件 SHA-256 完全一致。

审计固定使用 `step >= signed_encounter_step`，5 步窗口至少 4 步反向 command 定义 sustained
counter-steer，连续 3 步 clearance increase 与连续 3 个 actual lateral speed 下降 transition；near-zero
阈值直接复用现有 `stable_speed_threshold_mps=0.5`。原 whole-episode
`LATERAL_SIGN_OSCILLATION` 不参与 primary diagnosis。11 个成功 episode 使用相同规则建立 median/IQR
参考，0013/0014 作为 safety/boundary 特殊案例排除于主 cluster。

七个主要 failure 均出现 sustained counter-steer，因而不支持统一解释为“Actor 不知道何时停止横移”。
0002、0004、0020 为 `LATERAL_MOMENTUM_DECAY_TOO_SLOW`；0005、0011、0012 为
`COUNTERSTEER_NOT_SUSTAINED`；0006 为两者混合。七个主案例的 Actor→final retention 均为 1.0，
没有 `PIPELINE_ATTENUATION` 或 `COUNTERSTEER_TOO_WEAK` 主案例。主要 failure 的 extra drift median
为 2.416 m，而成功参考为 0 m；steps-to-near-zero median 为 14，而成功参考为 0。当前没有证据要求
先修改 safety、boundary 或 Terminal controller。唯一建议的下一步是冻结 Actor/场景/其余 controller，
只准备 encounter 后 minimal lateral-exit braking assistance OFF/ON 配对因果诊断；本轮未实现或运行。

派生产物位于：

```text
logs/preliminary/single_uav_lateral_exit_authority_audit_seed2026/
```

其中包含 summary JSON、episode diagnostics、counter-steer windows、成功参考、failure-vs-reference
CSV 与 `lateral_exit_report.md`。新增审计脚本不导入 simulator/environment/controller/policy/training
runtime；31 项纯 unit tests 通过。未执行 environment.step、rollout、formal、smoke 或训练，也未修改
Actor、Critic、GAT、reward、dynamics、action transform、safety、boundary、Goal Recovery 或 Terminal
controller。

## 2026-09-19：Single-UAV Minimal Lateral-Exit Braking Assistance OFF/ON（准备完成，未执行）

LATERAL_EXIT_BRAKING_DIAGNOSTIC_VALIDITY_FIXED

已准备独立的 paired causal diagnostic。OFF 保持 authority-repaired v2.3、`max_steps=120`、
Goal Recovery ON 的 signed-observability 基线；ON 唯一增加固定 cap `0.075 m/s per step` 的
deterministic lateral-exit damping，不决定 bypass side，也不使用 simulator actual state 控制。

全部 gate 使用 pre-step EKF estimated state、goal、static map、raw Actor action 与 encounter-frozen
lateral frame：valid encounter、estimated past closest approach、连续三个 clearance increase、5 步中至少
4 步 sustained counter-steer、estimated lateral velocity 仍为原方向、Goal Recovery inactive。动作路径为互斥
fusion：Goal Recovery active 时完整保留旧路径；仅在其 inactive 且 ON gates 全部通过时，对
`raw Actor + assistance` 调用一次既有 transform，之后继续经过 safety、boundary、dynamics；共享 `0.6 m/s`
cap 不变。

encounter frame 明确使用 `goal - EKF estimated position` 的 XY 投影，随后调用既有 fail-closed horizontal
frame helper；非零 estimated z 偏差不再误触发“goal ray is not horizontal”。若 XY 投影退化，则不使用
actual state 或 scenario initial position 补救，而是记录 `horizontal_goal_frame_degenerate` 并令 assistance
unavailable。signed helper 本身及其非水平 ray 拒绝契约保持不变。

paired 入口先执行 OFF，并在任何 ON reset/artifact 前用版本化 step trace 与冻结 signed-observability
artifact 精确比较；失败输出 `LATERAL_EXIT_OFF_BASELINE_MISMATCH`。Actor/Critic/GAT/normalizer 在条件边界
做三点哈希，20 个 episode、seeds、顺序、checkpoint、policy、observation、controller config 与 horizon
均由实际数据比较。reward、dynamics、action transform、safety 与 boundary 缺少独立 digest 时报告
`NOT_INDEPENDENTLY_VERIFIED`，不参与字符串真值判断；`only_lateral_exit_braking_assistance_changed` 由
paired identity、OFF fingerprint 与精确 treatment config diff 派生。

endpoint 采用 all-20 ITT，并以 OFF 首个 eligible step 定义 secondary cohort 与共同 ON anchor。轨迹型
secondary 统一截取到 `min(OFF last, ON last)`，新增 anchor-relative `0.5 m/s` threshold time 与明确右删失；
system-level passed/recovery/stable/collision/timeout/projection endpoints 仍使用完整 episode。primary 最大侧移
按既有实际计算语义改名为 initial-direction excursion，未改变计算。active preregistration hash 为
`a93837621a04c9fa7895881155a90878b664eeac318fbcb0cf7dded5afedd2c2`；旧 hash 仅标记为 pre-run superseded。

人工入口为：

```text
python -m src.experiments.single_uav_lateral_exit_braking_assistance --condition paired
```

未来独立日志根为
`logs/preliminary/single_uav_lateral_exit_braking_assistance_diagnostic_seed2026/`，fresh-output 拒绝覆盖。
本轮没有运行该 CLI、environment.step、rollout、formal、smoke、simulation 或训练，也没有创建新输出根。
准备详情见 `sections/single_uav_lateral_exit_braking_assistance_preparation.md`。

人工首次启动在 rollout 前发现 frozen runtime normalizer hash 无法直接序列化嵌套 `FeatureNormalizer`。
现已将 normalizer identity 改为稳定、只读、递归 JSON snapshot：数组显式记录 dtype/shape/values，
非有限浮点使用稳定标签，自引用对象 fail closed，callable 字段不进入 snapshot。该修复不调用或修改
normalizer，不影响 normalization arithmetic；Actor/Critic/GAT/policy hash、assistance、endpoint、预注册与
OFF baseline fingerprint 均保持不变。

## 2026-09-19：Post-Lateral-Exit Reorientation 离线审计（已准备，未运行）

已新增纯离线审计脚本与合成单元测试，固定使用 active preregistration、OFF 首个 eligible anchor、共同窗口及三步 `|actual lateral velocity| <= 0.5 m/s` local-exit 定义；审计同时检查 frame/action 投影、精确 pass gate、ON 成功参考 IQR 和主 failure 分类。输出固定为 `logs/preliminary/single_uav_post_lateral_exit_reorientation_audit_seed2026/` 下八个文件。唯一后续实验建议为冻结 Actor 与其余控制组件后的 paired post-local-exit/pre-pass radial-intent-persistence intervention。本轮未运行审计、测试或仿真。

## 2026-09-20：Pre-pass Reorientation 成功参考校正（已准备，未运行）

新增纯离线审计入口 `src/experiments/single_uav_prepass_reorientation_reference_audit.py` 与合成单元测试。审计复用冻结 OFF/ON 身份与 frame 投影校验，将 11 个固定 ON 成功回合的参考窗口截断在首次精确 pass gate，并对 7 个固定失败回合提供 whole/pre-best/post-best 分段；旧参考只用于证据变更对照。目标输出目录固定为 `logs/preliminary/single_uav_prepass_reorientation_reference_audit_seed2026/`，恰含六个文件。没有运行审计、测试、py_compile 或仿真，也没有实现任何干预。

## 2026-09-20：Single-UAV Pre-pass Radial-Intent-Persistence Diagnostic（代码已准备，未执行）

新增 radial-only assistance evaluator opt-in、离线 cap / OFF-baseline 准备模块与 unittest 覆盖。固定 all-20 paired ITT；两臂均保持 Goal Recovery、signed observability、lateral-exit assistance 开启，radial assistance 为唯一处理差异。机制 gate 使用冻结 encounter frame、estimated-only 连续 lateral exit 与 Actor 五步历史；cap 只从固定七个 prior-ON failure episode 派生，fail closed，不做 sweep。准备契约及命令见 `sections/single_uav_radial_intent_persistence_diagnostic_preparation.md`。

本轮已经运行纯离线 `--prepare-only`，物化且校验了 7 个 canonical preparation artifacts。最终 consistency fix 把 cap derivation 的歧义 `fail_closed=false` 拆为 `fail_closed_policy=true` 与 `fail_closed_triggered=false`，并在 preparation summary、cap artifact、runtime lock、preregistration lock 中保持一致；preregistration policy 仍为 `fail_closed=true`。由于 canonical payload 锁定 diagnostic 源码 hash，旧 `5a2b175b...` 在任何 rollout 前被新 active preregistration hash `bf70fb95188ca73c1ba87997a090d22331a3ac76a27daaaea1d86da09f59e40b` 正式取代。OFF baseline fingerprint 仍为 `9d10d1529c4a0126899d72428c227a710cb00a2293205d32a113308e4819787f`，冻结 cap 仍为 `0.075 m/s`。`py_compile` 与 40 项纯 `unittest` 均通过；没有运行 OFF/ON diagnostic、formal、smoke、rollout、仿真、训练或 `environment.step()`。没有新增论文符号，也没有修改历史数据或控制器实现。

## 2026-09-20：Single-UAV Pre-Pass Handoff / Pass-Progress Regression 离线审计（已完成）

PREPASS_HANDOFF_PASS_PROGRESS_AUDIT_READY

本轮只读取最新 radial-intent paired run、active preregistration/runtime lock 与既有 pre-pass reference；审计前后所有输入 SHA-256 完全一致，OFF/ON 仍为 11/20 与 12/20。固定六个普通 main failures 均在 best margin 后出现 regression，best-row clearance 均已超过 2.5 m，限制项全部为 `MARGIN_LIMITED`；0002、0004、0006 的 assistance 分别晚于 best pass window 20、17、28 step，0005、0011、0020 虽较早介入，仍普遍出现 goal progress 与 pass geometry 方向不一致。0012 在 ON 中于 step 63 正式 pass，best margin 为 0.324 m，assist-active frame angle 平均仅 5.43°，作为独立 rescue control 保留。

描述性 PASS_COMMIT 条件为 6/6 支持，达到预设至少 4/6 gate；唯一下一步建议为 `PREPARE_MINIMAL_PASS_COMMIT_CAUSAL_DIAGNOSTIC`，本轮未设计或实现控制律。0013 的 safety projection 从 OFF 4 step 增至 ON 9 step，其中 5 step 与 radial assistance 时间重叠，但仅报告关联；0014 两臂均有 34 个 boundary projection step，继续作为 boundary-dominated special case 排除主判断。

八个产物位于 `logs/preliminary/single_uav_prepass_handoff_pass_progress_audit_seed2026/`。新增模块仅使用 Python 标准库读取冻结 CSV/JSON，52 项纯单元测试、`py_compile` 与 diff 检查通过；未运行 environment、rollout、formal、smoke、simulation 或训练，未修改任何 controller、Actor/Critic/GAT、reward、dynamics、action transform、safety、boundary、Goal Recovery、Terminal 或 pass gate。

## 2026-09-20：Single-UAV PASS_COMMIT Diagnostic（原 attempt 已中止）

新增 estimated-only pass-progress non-regression helper、Evaluator opt-in 与只读准备入口。固定 all-20 paired ITT；两臂均启用 lateral-exit braking、关闭 radial intent assistance，仅 ON 启用 PASS_COMMIT。离线 cap 只使用六个固定 prior-ON failure episode；逐源行输出纳入/排除原因，冻结规则为 Type-7 Q3 与 `0.075 m/s` 上限的较小者。OFF 基线采用独立 PASS_COMMIT fingerprint，在 ON 布局/reset 前进行严格 identity/trace 校验。准备契约与人工运行门控见 `sections/single_uav_pass_commit_diagnostic_preparation.md`。

原 canonical preparation bundle 已物化，cap 为 0.075 m/s，父 active preregistration hash 为 `74b999edb1158fc960374025f418d33c7812a57653c07f6d0e73fab8b69e02ac`，OFF baseline fingerprint 为 `de53e383264db1c46ea793519f7b627f3ef83fbb447ca33d9d44cf72a603c97e`。之后人工 attempt 完成 OFF 20/20 并通过 OFF fingerprint、逐 episode trace/outcome 与 aggregate 检查；ON episode 0001 遇到 `ACTION_DECOMPOSITION_ASSERTION` 后中止，没有保存 ON step rows。该 attempt 保留在 `logs/preliminary/single_uav_pass_commit_diagnostic_seed2026/run/`，并标记为不计入正式结果。

## 2026-09-20：PASS_COMMIT Action-Decomposition Residual 修复（bookkeeping only，等待新 paired run）

Root cause 是 pre-transform reconstruction residual 漏加实际 applied/capped PASS_COMMIT vector。旧式分解为 `||Actor + GoalRecoveryApplied + LateralApplied + RadialApplied - combined_pre_transform||₂`，ON active action 实际已在 Actor 上加了 PASS_COMMIT applied vector，故这是结构性残差；tolerance 保持 `1e-9`。失败 ON step 没有持久化 row，精确 residual 数值不可恢复；纯 synthetic case 的 0.075 m/s applied vector 会使旧式 residual 为 0.075 m/s，修复后为 0。

修复属于 `BOOKKEEPING_ONLY_FIX`：控制律、trigger、direction、cap、clearance gate 与实际 correction 均未改变。加入纯 action-decomposition helper、应用向量互斥检查和每步 transform call-count assertion；PASS_COMMIT OFF / ON inactive / ON active 的 pure tests 均验证 transform 恰好一次（raw Actor 原有 bypass 保持不变）。原 OFF contract 文件逐字节未变，重新核对失败 run 的 2185 step rows 与 20 episodes 均匹配目标 fingerprint `de53e383264db1c46ea793519f7b627f3ef83fbb447ca33d9d44cf72a603c97e`。

修复版 preparation 位于 `logs/preliminary/single_uav_pass_commit_diagnostic_seed2026/preparation_action_decomposition_fix_v1/`。父 hash 为 `74b999edb1158fc960374025f418d33c7812a57653c07f6d0e73fab8b69e02ac`，新 active preregistration hash 为 `ddd08b7fc729c759b51eff091bfc25b6eb273aef2368e40dea4d57c317dee73a`；新运行将写到独立的 `run_action_decomposition_fix_v1/`。runtime-lock 与 OFF contract offline verification、69 项 targeted pure tests、`py_compile`、`git diff --check` 均通过。没有重新运行 OFF/ON 或任何环境步骤；下一 attempt 必须从完整 OFF 20 + ON 20 开始。

## 2026-10-10：PASS_COMMIT 离线汇总闭合与独立单 UAV PPO 工程准备

后续磁盘证据表明 `run_action_decomposition_fix_v1/` 已保存完整 OFF/ON 各 20 回合、2,185 行逐步记录。最终汇总缺失的代码缺陷是 evaluator 两次引用未绑定的 `ALL_EPISODE_IDS`；本次将该块替换为 `src/pass_commit_finalization.py` 中的纯汇总函数，并以冻结预注册总体显式提供 ID。没有找到原退出堆栈，因此没有把历史退出原因补写为已证实的同一异常。

只读取原 CSV/JSON 完成了离线恢复，独立输出位于 `logs/preliminary/single_uav_pass_commit_diagnostic_seed2026/offline_recovery_action_decomposition_fix_v1_20261010/`，包含 `evaluation_summary.json`、`recovery_provenance.json`、`recovery_checks.csv` 和 `paired_summary.csv`。两臂均为 11/20 通过、10/20 稳定到达、0 碰撞，PASS_COMMIT active steps 为 OFF 0、ON 73；安全投影各 5 步、边界投影各 41 步。OFF fingerprint、锁定预注册、三点 Actor/Critic/GAT/normalizer 身份及输入前后哈希均通过核查。原轨迹、中止尝试、冻结文件和 ZIP 归档保留；修复后的代码哈希另行登记，旧 source lock 沿用历史含义。

新增独立配置 `data/configs/single_uav_ppo_baseline_v1.json` 及实现 `src/experiments/single_uav_ppo_baseline.py`，从头初始化单 UAV PPO，五类辅助控制 PASS_COMMIT、lateral-exit、radial-intent、Goal Recovery、Terminal Capture 全部关闭。保留既有 `legacy` 基础实体安全投影及边界处理；额外目标制动 safety projection 关闭。固定决策周期 0.2 s、最大速度 12 m/s、最大加速度 3 m/s²、速度增量范数上限 0.6 m/s、安全距离 2 m、边界 `(0,0,2)..(200,200,60)` m、评估时域 120 步/24 s。

严格成功使用动作后的真实任务位置与实际状态速度：全程无碰撞、已完成实体障碍通过、目标距离不超过 1 m、实际速度不超过 0.5 m/s，连续保持 5 个决策步。独立环境统一成功奖励、终止和日志；碰撞永久锁存，条件破坏时保持计数清零，回合 reset 清空全部状态。原环境与历史配置保留原行为。

`ppo_action_logprob_mode="sampled"` 显式固定。默认新训练器的 reconciliation 路径自动写入采样动作、实际执行动作、采样 old log-probability、总修正与安全修正向量/范数，并区分动作限幅、速度限幅、安全投影和边界原因。未来真实执行日志默认写到 `logs/preliminary/single_uav_ppo_baseline_v1_seed2026/sampled_execution.csv`；构造时可以传入新的独立路径，已有日志拒绝覆盖。本阶段仅保存独立的预检动作行。

Actor 节点特征为 42 维，估计速度、目标相对位置、机体系实体障碍相对方向的零基切片分别为 `3:6`、`30:33`、`39:42`。上游状态扰动在原始及归一化输入中均有效，实际 Actor encoder 收到完整的归一化 42 维特征。真实未训练策略的采样/执行修正检查确认 old log-probability 不变、PPO ratio=1、buffer 保留采样动作。训练归一化和奖励 potential 校准保持待生成；临时 reset probe 统计没有写成训练 artifact，正式 factory 需核对训练 artifact 路径/哈希、划分、特征顺序和实际统计，并以冻结 view 拒绝 fit/update。

最终工程预检与配置锁位于 `logs/diagnostics/single_uav_ppo_pretraining_20261010_v1/preflight_final/`；配置 SHA-256 为 `bd34f163de64fc814cd21f705029343c4c4d45efa0ff7ace8a3ef6acbf07c609`。PASS_COMMIT 相关 75 项测试及 3 个子测试、PPO 相关 27 项测试通过。预检环境推进、backward 和 optimizer step 均为 0；未启动正式训练、重跑历史轨迹、提交或推送。完整协议见 `sections/single_uav_ppo_baseline_pretraining_v1.md`。此前 `preflight/` 是保留的中间检查记录，最终状态以 `preflight_final/` 为准。

只读恢复及工程预检入口为：

```text
python main.py recover-pass-commit-summary
python main.py prepare-single-uav-ppo --config data/configs/single_uav_ppo_baseline_v1.json --output logs/diagnostics/single_uav_ppo_pretraining_20261010_v1/preflight_final
```

上述默认输出已完成，入口拒绝覆盖；需要再次检查时使用新的输出目录。

## 2026-10-10：单 UAV 静态绕障 PPO 两组 smoke 已执行

基于前阶段冻结的统一环境新增 opt-in smoke 入口 `src/experiments/single_uav_ppo_smoke.py`，配套初始化、课程及只读产物审计模块。A 使用既有 42 维迁移 PPO，B 使用既有 BC/DAgger Actor 并进行了真实在线 PPO 微调。两组各保留 18 个兼容参数张量；因训练归一化语义改变，完整首层 GAT 与 Critic 使用相同 fresh constructor 初始化，旧 optimizer、rollout 与教师样本均未导入。42 维 Actor 中目标、估计速度和障碍相对方向的实际归一化输入及输出敏感性均通过检查。

独立配置 `data/configs/single_uav_static_ppo_smoke_v1_seed2026.json`，两组相同三阶段课程与场景划分。训练池 18 场（各 6 normal/single/full），实际 PPO 接触 12 场，其余 6 场仅参与训练池校准；验证池独立为 2/4/4 场。无障碍归航停车、左右平衡单障碍、完整绕障后归航依次训练，并在后阶段回放早期场景。历史 20 回合未用于此次训练、统计、验证或模型选择。真实无障碍场景显式 `obstacles=[]`，避免环境对空 `obstacle_specs` 的历史默认障碍回退。

每组执行 288 个校准步及 2160 个 PPO 在线步，总训练交互 2448；18 次 update、4 epochs、minibatch 64，最终 144 次实际 Adam step。校准流相同，仅拟合一份共享 train-only 统计，所有方向槽计数均为 306，训练/验证期间冻结。每组另有 3360 个 deterministic 验证步、28 个验证 episode，成本独立记录。采样动作/old log-prob 与执行动作、安全修正、总修正分别记录；每组两个动作/step CSV 各 5520 行逐条对齐，PPO buffer 没有校准、验证或教师样本。

最终 10 场验证：A/B 稳定到达均 0/10，碰撞均 0/10、失败超时均 10/10；实体绕障 A 0/8、B 8/8。B 初始化时已是 8/8，通过率没有观察到本次 PPO 带来的提升。原始动作实体危险与基础安全投影均 0；边界投影 A 239/1200、B 765/1200，B 大量停在远离目标的边界，归航停车尚未建立。Actor/GAT/Critic 最终参数变化均非零、损失及梯度有限，B 的最终 Adam state 独立复核为 144 步，工程链路确认通过，不能据此宣称性能、泛化或一般安全性。

B 最终 BC 的已知成本为 45 epochs、每轮 1728 样本，即 77760 次样本呈现、2430 个推导 batch step；前置 round0 另有 34560 次呈现、1080 batch step，已知 BC 工作流合计 112320/3510。教师完整采集/推理成本与墙钟时间缺少完整记录，保持未知，不计为本次在线 PPO。详细成本账本与标签条数口径分开保留。

独立数据/checkpoint、CSV/报告、诊断图分别位于 `data/synthetic/`、`logs/preliminary/`、`plots/preliminary/` 下的 `single_uav_static_ppo_smoke_v1_seed2026/`，保存初始、三阶段、最终 checkpoint。统一实际 trainer 配置见 `effective_smoke_training_config.json`；总结见 `logs/preliminary/single_uav_static_ppo_smoke_v1_seed2026/smoke_summary.md`，协议见 `sections/single_uav_static_ppo_smoke_v1.md`。49 项相关测试、20 项独立产物检查与符号审计通过，71 个历史/冻结输入 SHA 一致。未覆盖历史结果，未启动多种子训练，未提交或推送。

正式多种子性能比较前需先解决基础停车失败与边界依赖，再冻结实验版本、登记独立 seed、等量扩大的 PPO 预算和完整成本账本。运行入口如下，现有输出拒绝覆盖：

```text
python -m src.experiments.single_uav_ppo_smoke --config data/configs/single_uav_static_ppo_smoke_v1_seed2026.json
python -m src.analysis.single_uav_ppo_smoke_audit data/synthetic/single_uav_static_ppo_smoke_v1_seed2026 logs/preliminary/single_uav_static_ppo_smoke_v1_seed2026
```
