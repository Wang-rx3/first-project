# 硕士课题进度审计（2026-10-10）

## 1. 当前研究位置

课题是“面向复杂遮挡环境的多无人机协同定位与通信联合优化”。已经完成较大规模的仿真平台、EKF/GAT/MAPPO 实现、工程预实验和机制诊断；尚未完成当前最终方法的多随机种子、跨规模、消融和独立泛化证据，也尚未写成完整论文。

最近的实际工作重点已转到**冻结单 UAV 策略的静态绕障—回正—近目标稳定捕获诊断**。最后一个有完整配对轨迹的尝试是 `single_uav_pass_commit_diagnostic_seed2026/run_action_decomposition_fix_v1/`。两臂各 20 回合，PASS_COMMIT 在 ON 的六个主要失败回合中实际介入 73 步，但两臂都是 11/20 通过障碍、10/20 稳定到达，主要成功指标未改善。

更精确的停点是：**修复版 OFF/ON rollout 已保存，效果尚未支持；最终验收汇总没有闭合。** 该目录缺少当前代码应写出的 `evaluation_summary.json`。当前 evaluator 的 PASS_COMMIT 汇总代码引用未定义的全局名称 `ALL_EPISODE_IDS`，正常执行到该处会发生 `NameError`。这能解释产物停在汇总前的现象，但没有找到该次运行的终端异常堆栈，不能把历史退出原因写成已证实的同一异常。

README 末尾仍写“等待新 paired run”，已经落后于磁盘证据。不能直接照 README 的最后一句重新开始全部实验。

## 2. 审计范围与可信程度

本次采用全目录文件清单、全部项目 Python 文件语法/结构盘点、论文和进度材料阅读、实验 summary/manifest/report 与关键逐步 CSV 交叉核验、参考文献 PDF 可读性检查、答辩 PPT 文本检查以及当前测试复核。

- 文件元数据快照共 256,315 项，包含 Git、虚拟环境、第三方依赖、缓存和临时产物；无文件 stat 错误。测试过程中缓存数量会变化，这不是恒定的研究文件数。
- `data/` 208,465 文件，约 33.56 GB；`logs/` 原有 1,969 文件，约 2.89 GB；`plots/` 1,393 文件，约 258.76 MB。容量均按十进制计算。
- `src/` 152 个 Python 文件、155,029 行；`tests/` 135 个 Python 文件、24,027 行。287 个 Python 文件均通过本次 AST 语法解析。
- `data/synthetic/` 202 个一级目录、`logs/preliminary/` 191 个一级目录。目录包含 smoke、正式配置、失败尝试、重跑和诊断，不等于独立有效实验数。
- `knowledge/` 原有 18 篇 PDF、3 篇 Markdown；18 篇 PDF 共 296 页，全部完成文本提取且没有读取错误。这是本地文件可读性/主题盘点，不等于重新审定每篇论文的科学结论或 DOI。
- `sections/` 15 个 Markdown 和 1 张图片；根目录答辩 PPT 为 22 页，检查了全部页的文本。PPT 临时工作区主要是编辑脚本、依赖、审阅图和另一份历史演示文稿。
- 7 个数据/日志 ZIP 已检查目录结构，没有展开或覆盖原文件。

**范围边界：**没有逐行人工阅读 19 万余份 CSV 的每个数值，没有逐张检查全部 1,198 张 plots PNG，没有把第三方依赖逐行审成研究代码，也没有重新训练全部历史模型。全量清单与针对研究结论的内容审计是不同层次；本报告不声称进行了完整数值重演或形式化代码证明。

审计索引位于 `logs/diagnostics/research_progress_audit_20261010/`：

- `file_inventory.csv` / `inventory_summary.json`：全量文件元数据与统计；
- `python_source_inventory.csv`：全部研究/测试 Python 文件行数、函数数、语法状态和哈希；
- `experiment_directory_index.csv`：202 个数据运行目录及同名日志/图目录对应情况；
- `literature_pdf_inventory.json` / `proposal_slide_text.json`：文献盘点和答辩文字；
- `archive_inventory.json`：ZIP 目录盘点；
- `latest_pass_commit_recheck.json` / `latest_run_identity_excerpt.json`：最新运行独立复核；
- `pytest_workspace_result.txt`：本次完整测试结果。

## 3. 研究目标和现有技术体系

研究的原始因果链是：遮挡和链路变化影响消息交互，继而影响协同定位；通过感知通信质量、选择协同测量源、分配带宽并调整运动，改善定位与通信的综合表现。

当前论文主方法名称为 **Local-EKF-GAT-MAPPO (Proposed Method)**，内部 ID 仍是 `baseline_mappo`。此处 baseline 是历史兼容名称，不能据此误认为它只是弱基线。

| 部分 | 已有工作 | 仍需区分的边界 |
| --- | --- | --- |
| 三维仿真 | 运动学、实体障碍、通信遮挡、随机流、传感器和信道、场景配置 | 纯 Python 统计/运动学仿真，不是实飞或完整六自由度飞控模型 |
| 本地与协同 EKF | 本机观测融合、邻居消息协同更新、协方差和延迟消息处理 | 实现存在不等于已证明所有噪声/失效条件下稳定有效 |
| 动态图与 GAT | 节点/边特征、动态图编码、邻居注意力 | 图表示不等于已经实现显式拓扑重构或中继规划 |
| MAPPO | Actor/Critic、候选测量选择、带宽与速度动作、PPO 更新和日志 | 仍缺当前最终版本的跨训练种子稳定性和最终比较 |
| 实验工具 | warm-up、train/validation/test、checkpoint、schema、指标、配对与哈希审计 | schema/audit 通过只证明结构或契约，不自动证明算法优越 |
| 单 UAV 外部控制 | Goal Recovery、Terminal Capture、横向制动、径向意图、PASS_COMMIT | 最近运行冻结学习策略，性能属于混合控制系统，不能归因成 MAPPO 自主学会 |

数据来源已经明确为自建 Python 仿真；论文 PDF 只用于理论参考。Flightmare/PX4/Gazebo 是历史兼容或未来迁移选项，不是现阶段主实验依赖。`data/README.md` 明确说明物理/传感器频率主要记录为元数据，当前决策步并未实现完整异步多频子步；因此不能把协议里的 100 Hz/10 Hz 标注直接当成已经运行的多速率仿真。

主要依据：`sections/2_system_model.md:5,382,410,1599`；`sections/3_methods.md`；`knowledge/dataset_selection.md`；`data/README.md`；`src/simulation/`、`src/learning/`。

## 4. 多 UAV 主线做到哪里

### 4.1 已经做过的工作

历史上已做过 3 UAV 小规模闭环、EKF/观测校准、测量密度/邻居选择/带宽干预、信息收益和通信开销审计、训练反馈与安全投影诊断，以及多种 proxy/quality/通信约束分支。

早期 3/5/7 UAV 工程预实验也确实存在：`data/synthetic/python_sim_preliminary_fixed/manifest.json` 索引三个规模，各有 140 个 episode，明确标记 `engineering_preliminary` 和 `preliminary_only`。因此不能说“从未做过 5/7 架实验”；准确说法是：**已有规模工程试运行，但尚不能作为当前最终主方法的跨规模确认实验。**

当前活动比较集是 Proposed 与 `local_only_fixed`、`random_cooperative`、`budgeted_greedy_ig` 三个规则参照。MLP 不在当前主线，是否额外增加应由最终实验设计决定，不能误记为一个已承诺却遗漏的必需分支。`communication_constrained_proxy_guided_mappo` 已从当前论文主线停用。

### 4.2 现存可核验的开发结果

v11 的当前目录为：

`logs/preliminary/mainline_core_framework_v11_comm_occluder_seed42_retry2/development/reference/`

其中 `audit_report.json` 记录 `audit_ok=true`、`status=complete`，同时明确 `cross_seed_stability=not_evaluated`、`candidate_selected=false`。`method_means.csv` 是每方法 20 个回合、seed 42 的开发比较：

| 方法 | 位置 RMSE（m） | P95（m） | 实际测量消息负载（kB，1000 bytes） |
| --- | ---: | ---: | ---: |
| Local | 0.44815 | 0.76168 | 0 |
| Random | 0.42450 | 0.70658 | 39.014 |
| Greedy | 0.41060 | 0.70221 | 39.168 |
| Proposed | 0.40190 | 0.68184 | 61.030 |

在这次配置中，Proposed 的 RMSE/P95 最低，但测量消息负载约为 Greedy 的 1.56 倍。通信综合指标与测量字节数不是同一指标，不能简化成“所有通信指标都更差”，也不能说“定位和通信各指标全面最优”。

另一现存 v9 开发结果位于 `logs/preliminary/mainline_core_framework_v9_action_selection_seed42/development/action-head-v3/method_means.csv`：Proposed RMSE 0.40358，Greedy 0.39885；Proposed 测量负载 69.888 kB，Greedy 43.034 kB。两版配置不同，不能直接把差值当成单因素改进；它们说明目前结论依赖具体版本与场景。

**主线阶段判断：核心框架已实现，单训练 seed 的开发比较与机制排错已完成多轮；最终算法选择和论文级确认实验未完成。**

### 4.3 历史证据缺口

README 记录 2026-09-09 清理了约 235,186 个文件、54.11 GB，随后另清理 37 个版本化结果目录。若干 v3/v4 目录如今为空，部分 v6/v7 产物缺失。README 留下的历史数值可以帮助理解研究过程，但不能代替当前存在的 checkpoint、manifest 和原始轨迹。必须把“当时记录运行完成”和“现在本地可复现”分开。

依据：`README.md:37`；v11、v9 的 `audit_report.json` / `method_means.csv`；早期多规模根 manifest。

## 5. 为什么后来集中研究单 UAV

单 UAV 是为了把问题拆小：先判断策略能否可靠地从远处飞向目标，绕过静态障碍，再在目标附近减速并稳定停留。它不包含 UAV 间的协同通信与 UWB 更新，不能用来证明多 UAV 通信—定位联合优化成功。

研究历程大致如下。日期以项目文字记录的研究阶段为准，文件复制/解压后的修改时间不作为独立的实验执行时间证据。

| 阶段 | 具体完成内容 | 主要发现/未完成点 |
| --- | --- | --- |
| 9 月 9–11 日 | 单 UAV sanity、长回合、导航奖励、速度/制动奖励与课程 | 能接近目标不等于能够停车；需要同时检查距离、实际速度和连续保持 |
| 9 月 11–12 日 | scripted/oracle 可行性、Actor 动作限制、BC、DAgger、分层控制 | 教师能完成不代表 Actor 能复现；不同训练方式会损伤远距离导航或末端能力 |
| 9 月 13 日 | 静态障碍阶段 1、障碍方向观测、Stage 2A/2B/2B-R | 零碰撞可能主要来自 safety projection；主动绕障和绕障后恢复仍不稳 |
| 9 月 14 日 | post-bypass recovery、教师约束、BC 保持性和诊断图 | 学习策略与教师在恢复/末端阶段有明显差距；需要区分“通过障碍”和“回到目标” |
| 9 月 16 日以后 | Goal Recovery、真实 stable 判据、Terminal Capture 多次修复 | 通过外部控制层改善近目标运动，但有速度不足、过冲、边界抖动和保持失败 |
| 9 月 18 日阶段 | v2.3 authority repair、延长 horizon、上游绕障诊断 | 对已通过障碍的回合，末端捕获明显改善；剩余失败更多集中在尚未完成通过障碍 |
| 9 月 19–20 日阶段 | 有符号绕障日志、lateral-exit braking、radial intent、pass progress 审计 | 横向停止并不保证通过障碍；向目标接近也可能与通过障碍所需几何进展冲突 |
| 最后保留的运行 | PASS_COMMIT action-decomposition-fix-v1 配对轨迹 | 实际介入但没有新增成功，且最终验收汇总缺失 |

9 月 14 日可视化报告显示，在静态 holdout 上，BC Actor 18/20 通过障碍、18/20 进入 recovery，但只有 8/20 进入 8 m、0/20 进入 1 m、0/20 stable；同份报告的 Oracle/Teacher 为 19/20 stable。其主要漏斗损失在通过障碍后的归航。普通无障碍场景下某 DAgger Actor 则达到 20/20 stable。这说明不能把“某个无障碍场景训练成功”推广成“静态障碍场景成功”。

来源：`plots/visualization/current_run_20260914_01/visualization_diagnostic_report.md:24`；该报告同时明确此次是诊断复现、PPO 更新为 0，图中的 PPO loss 来自历史 baseline 训练，不是本次 BC/DAgger 评估新产生的 PPO 训练曲线。

目前保留的后续控制诊断还可以还原出以下进展。表中每组都是 20 回合，但控制组合/时域不同，跨行不能直接解释为单因素因果效应；只有各自冻结协议内的 OFF/ON 才是对应配对。

| 阶段/控制设置 | 通过障碍 | 进入 1 m | stable | 具体含义 |
| --- | ---: | ---: | ---: | --- |
| Goal Recovery formal20 OFF | 11 | 0 | 0 | 原冻结 Actor 的归航/末端不足 |
| Goal Recovery formal20 ON | 11 | 11 | 0 | 能进入目标附近，但没有稳定停住 |
| v2.3 authority repair ON | 11 | 8 | 7 | 终端控制已有部分成功 |
| authority repair + horizon120 | 11 | 11 | 11 | 延长回合后，通过障碍的 11 回合都完成稳定到达；仍有 9 回合未通过障碍 |
| lateral-exit assistance OFF | 11 | 11 | 11 | 对照保持上一步结果 |
| lateral-exit assistance ON | 11 | 11 | 10 | 未增加通过，且有一个下游 stable 退化案例 |
| radial-intent OFF | 11 | 11 | 10 | 使用开启 lateral assistance 的新基线 |
| radial-intent ON | 12 | 11 | 10 | 多救回一个通过案例，但未增加 stable；安全投影由 5 步升至 10 步 |
| 最新 PASS_COMMIT OFF/ON | 11 / 11 | 11 / 11 | 10 / 10 | 两臂关闭 radial intent；不能称为在 radial ON 的 12 次成功基础上继续增加 |

上述运行来源依次为 `single_uav_goal_recovery_correction_eval_formal20_seed2026`、`single_uav_terminal_capture_v23_authority_repair_validation_seed2026`、`single_uav_near_goal_capture_horizon120_diagnostic_seed2026`、`single_uav_lateral_exit_braking_assistance_diagnostic_seed2026`、`single_uav_radial_intent_persistence_diagnostic_seed2026/run` 和最新 PASS_COMMIT 目录，均位于 `logs/preliminary/` 下。某些 JSON 的 `*_executed_by_codex=false` 只描述执行主体/准备时声明，不能用它否认已经保存的实际 episode 与 step 轨迹。

## 6. 最后一步的证据和真正卡点

### 6.1 冻结的是什么

最新控制诊断使用以下冻结 checkpoint：

`data/synthetic/single_uav_external_recovery_method_study_seed2026/dagger_bc/bc_final/scale_1/checkpoints/best_validation.pt`

checkpoint SHA-256 为 `b7ed3ba79cf3055906bd0425cf4a7c4e613b206a8680ebf17f3b586fbc31b3c6`，观测维度为 42。最新 JSON 中 Actor、Critic、GAT、normalizer 在 OFF 前、OFF 后、ON 后三次哈希一致，`training_performed=false`、PPO/MAPPO 更新数为 0。

两臂都开启 Goal Recovery、终端控制和 lateral braking，关闭 radial-intent assistance，只有 ON 开启 PASS_COMMIT。因此它是冻结 Actor 上外部辅助控制的配对机制诊断，不是新一轮 MAPPO 训练。

### 6.2 两个 PASS_COMMIT 尝试必须分开

1. 原 `run/`：`attempt_status.json` 明确 `ABORTED_IN_ON_EPISODE_0001`，异常是 `ACTION_DECOMPOSITION_ASSERTION`。OFF 完成 20 回合且 fingerprint 检查通过；ON 没有保存 step rows。该次不能计为成功完成的配对结果。
2. `run_action_decomposition_fix_v1/`：OFF/ON 均保存 20 个 episode、各 2,185 个 step rows，还有通用比较表。控制分解的修复没有阻止本次全部轨迹落盘，但最终专属验收未落盘。

当前修复版配对结果：

| 指标 | OFF | ON |
| --- | ---: | ---: |
| 通过障碍 | 11/20 | 11/20 |
| 通过后恢复 | 11/20 | 11/20 |
| 进入 1 m | 11/20 | 11/20 |
| 稳定到达 | 10/20 | 10/20 |
| 碰撞回合 | 0/20 | 0/20 |
| 超时回合 | 10/20 | 10/20 |
| safety projection 步数 | 5 | 5 |
| boundary projection 步数 | 41 | 41 |
| PASS_COMMIT active 步数 | 0 | 73 |

ON 的 73 步分布为：0002=6、0004=1、0005=17、0006=15、0011=30、0020=4。这六个主失败回合仍未通过障碍。不能把结果解释为“开关没有打开”，也不能因为局部动作改变就宣称任务性能改善。

### 6.3 当前最具体的科学问题

剩余问题主要在**通过障碍前的运动衔接**：已经开始退出横移，但向前/朝目标的动作没有稳定地转化为 pass geometry 改善，甚至出现先接近通过边界、随后又退回的现象。下游 Goal Recovery 尚未释放，是因为上游通过条件未满足；不应把这种正常门控状态直接诊断为 Goal Recovery 的错误。

上一轮 radial-intent paired run 的离线审计记录 OFF 11/20、ON 12/20 通过，0012 是单独救回案例；但六个固定主失败仍有 best margin 后回退。其成功并没有自动迁移到本次 PASS_COMMIT：本次两臂明确关闭 radial-intent，不能将两次数字直接拼接成同一控制器的连续提升。

依据：`logs/preliminary/single_uav_prepass_handoff_pass_progress_audit_seed2026/prepass_handoff_report.md`；最新运行 `off/summary.json`、`on/summary.json`、`comparison_episodes.csv`、`off/steps.csv`、`on/steps.csv`。

### 6.4 本次审计新发现的工程阻塞

`src/experiments/single_uav_goal_recovery_correction_evaluation.py:6713` 在 PASS_COMMIT 最后汇总阶段遍历 `ALL_EPISODE_IDS`，`:6780` 再次引用该名称；AST/symtable 检查显示它是全局引用，不是局部定义或导入，模块也无对应绑定。应写 `evaluation_summary.json` 的位置在 `:6822`，位于该问题之后。

因此本次结论应限定为“配对轨迹与通用比较已经存在”，不能直接引用 `pass_commit_diagnostic_executed=true` 将整个验收流程算作完成。这个字段在更早的通用比较 JSON 中就会被写出。

本轮只报告问题，没有修改控制器、修复代码、重新运行研究实验或覆盖冻结锁。

## 7. 论文实际写到哪里

| 文件 | 现状 | 判断 |
| --- | --- | --- |
| `sections/0_abstract_keywords.md` | 0 字节 | 摘要和关键词未写 |
| `sections/1_introduction.md` | 0 字节 | 引言未写成章节 |
| `sections/2_system_model.md` | 2,146 行，约 134 KB | 系统模型、约束、变量、符号和实现边界已有大量材料 |
| `sections/3_methods.md` | 1,244 行，约 79 KB | 方法和实验版本记录丰富，但仍混有历史分支/开发协议，需整理成唯一最终方法 |
| `sections/4_experimental_protocol.md` | E1–E6、3/5/7 UAV、300/60/60 等协议 | 有实验设计，不代表各项已经完成 |
| `sections/5_results_template.md` | 指标表为空，结果待填 | 实验结果章未完成 |
| `sections/6_discussion_limitations_template.md` | 简短局限与未来工作 | 有边界材料，尚非围绕最终结果展开的讨论章 |
| `sections/5_conclusion.md` | 0 字节 | 结论未写 |
| `sections/advisor_report_seed42_v10.md` | 阶段汇报 | 是旧版本汇报材料，不是最新实验总结 |
| 五份单 UAV preparation 文档 | 诊断准备/冻结契约 | 适合追溯实验设计，状态可能落后于实际运行 |

符号登记源本次审计通过：300 个规范符号、10 个历史符号，无重复、畸形行或缺失含义。符号审计通过不代表所有公式与当前代码已逐式证明等价。

22 页开题 PPT 已明确研究背景、系统框架、方法和工作计划，仍属于开题/方案材料。它没有覆盖最近的单 UAV 控制诊断证据，也不能作为毕业论文完成度的证据。PPT 文本中的若干简化公式仍应在正式写作时与唯一符号表重新核对。

文献方面，有本地资料储备，但 `knowledge/` 只有数据路线、仿真设计和下载状态三份 Markdown，尚未形成清晰的逐篇综述矩阵或完整 related-work 章节。下载状态清单是 7 月的历史快照，与当前文件集合不能逐字照搬。

## 8. 当前工程健康度与复现风险

本次执行 `.venv/Scripts/python.exe -m pytest tests -q -p no:cacheprovider`，将 TEMP/TMP 与 pytest basetemp 放在本项目审计目录中后，结果为 **1,147 passed，5 skipped，3 failed，1 warning，242.97 秒**。

失败项：

1. `tests/test_safety_risk_screening.py` 两项：`src/experiments/safety_risk_screening.py:411` 仍向 `Axes.boxplot()` 传 `labels`，当前 Matplotlib 3.11.1 不接受该参数。属于绘图接口兼容问题，不能据此判定所有控制算法失效。
2. `tests/test_single_uav_pass_commit_diagnostic.py:247`：仍断言修复版输出根不存在，实际该运行目录已经生成。它把“运行前现场状态”写入了测试，与当前研究进度冲突。

本次环境版本为 NumPy 2.5.1、Torch 2.13.0+cpu、Pandas 2.3.3、Matplotlib 3.11.1。`requirements.txt` 只约束较宽的版本范围，未冻结完整运行环境。首次测试的临时目录权限错误经调整本次进程 TEMP/TMP 后消失，不计入研究代码缺陷。

另一个未被现有测试捕获的问题是上文 PASS_COMMIT 汇总中的未定义全局名称。大量单元测试通过不能代替完整运行到最终汇总的集成检查。

Git 当前最后提交是 `9146c12`（2026-09-16），但之后还有已修改的 README/源代码和大量未跟踪的控制器、测试、准备材料与图目录。`data/`、`logs/`、PDF 和 checkpoint 在 `.gitignore` 中被忽略；因此“代码已提交”也不等于实验数据/模型已备份。本轮没有提交、推送或运行 `.local-sync` 脚本。

## 9. 后续恢复研究的顺序

以下为本次审计建议，不代表已执行的新实验或已获通过的研究方案。

1. **先闭合最新运行的记录。** 保留当前 OFF/ON CSV，核对最终汇总代码的未定义名称，先评估能否仅离线从现有轨迹恢复缺失汇总；不要先覆盖输出或重复训练。若要修复源码，要处理冻结 source hash/版本登记，不能静默改动原 preregistration。
2. **解释 PASS_COMMIT 为什么有 73 步动作却没有新增通过。** 逐一核对六个主失败的触发时刻、剩余负向分量、cap 饱和、通过余量变化、与 lateral assistance 的互斥窗口和执行后的动作；现有证据不足以直接宣布“只需加大 cap”或“奖励一定有错”。
3. **明确单 UAV 分支的停止条件。** 若作为多 UAV 研究的工程前置，应冻结一个行为和边界清晰的版本，避免无限叠加辅助层。若打算把混合控制作为论文贡献，则须明确方法归属，并补足 Actor-only 与各控制层的消融。
4. **回到原课题的多 UAV 联合优化证据。** 选定唯一方法/环境/协议，完成多个训练 seed、规则基线、公平通信预算、必要组件消融和当前版本 3/5/7 UAV/未见场景验证。旧分支的五种子结果不能直接继承给新主线。
5. **为最终结论保留真正独立的测试。** 当前 20 个固定场景已反复用于诊断和控制设计，即使文件路径名为 test，也不能继续当作完全未触碰的最终测试。现有结果适合开发和机制描述；最终方法冻结后需要独立确认数据。
6. **整理写作。** 将系统模型与方法材料缩成统一版本，补写引言/相关工作，再用真正冻结的实验填结果章、讨论和结论。现在直接润色摘要或宣称全面优越会早于证据。

## 10. 建议首先重新打开的文件

1. 本报告：恢复全局研究位置。
2. `logs/preliminary/single_uav_pass_commit_diagnostic_seed2026/run_action_decomposition_fix_v1/goal_recovery_correction_comparison.csv`：最新任务指标。
3. 同目录 `on/steps.csv` 与 `off/steps.csv`：六个失败回合和介入时刻。
4. `logs/preliminary/single_uav_prepass_handoff_pass_progress_audit_seed2026/prepass_handoff_report.md`：为什么选择 PASS_COMMIT。
5. `sections/single_uav_pass_commit_diagnostic_preparation.md`：冻结契约；注意“尚未运行”文字已过时。
6. `logs/preliminary/mainline_core_framework_v11_comm_occluder_seed42_retry2/development/reference/method_means.csv`：原多 UAV 主线当前可核验的一组比较。
7. `sections/2_system_model.md` 与 `sections/3_methods.md`：恢复原始模型和方法定义。

本次新增内容仅为审计报告和审计输出；没有替用户推进控制器修改、训练或论文性能宣称。测试临时目录已清理，原研究产物保留。
