# 第三阶段：固定导航下 MAPPO 协同测量选择训练准备

日期：2026-10-12。当前状态是代码准备与短接口预检完成；真实训练、完整校准、验证和基线实验由用户在 PyCharm 执行。本轮没有运行 MAPPO smoke 或任何 PPO/优化器更新。

## 已实现的动作与信息边界

继续使用 `motion_action_mode="fixed_navigation_comm_v1"`，增加显式 `communication_learning_scope="measurement_only"`。其默认值仍为 `all`，保留第二阶段通信联合动作和旧 joint 行为。

新分支只采样四类测量动作：按 `uav_00/uav_01/uav_02/none` 排列，自身类别禁止，none 永远合法。链路固定为当前潜在出链，带宽各0.5，空闲份额不转移；固定导航仍使用原控制器。学习样本中的 bandwidth、link、delta 向量为空，不构造 Bernoulli、Dirichlet 或运动 Normal，不计算其概率、熵或策略损失，相关头冻结且不进入新模式优化器。GAT、Actor 测量网络和 Critic 可训练。

导航仅取本机原始 EKF 位置/速度和任务目标。保留位置增益0.6、速度反馈1.6、巡航目标2.5 m/s、航点半径0.8 m；dt=0.2 s、速度上限12 m/s、加速度上限3 m/s²、速度增量范数0.6 m/s、逐轴0.6/sqrt(3)，120步和原边界不变。基础2 m安全投影开启，五类辅助全关闭。本轮场景不含障碍，未增加导航或避障算法。

Actor 只接收本机 EKF、预算/剩余时间及本机已交付邻居消息。37维节点和11维边保持不变；本地测量掩码共同用于 MAPPO、Random、Greedy 和轮询，交集为当前物理/UWB候选、周期、预算、有效已交付消息。当前缓存可能已消费，但未来是否有新包未知；最终门控失败只影响收益和实际成本，不重写原采样动作或旧log-prob。

Critic 仅在集中训练中接收估计中央图。当前无障碍基线435维，加剩余时间1维及三机预算比例3维，总计439维。中央估计、其他机器私有状态和真值误差不进入 Actor。有限时域设定为 `episode_objective="finite_horizon"`、`include_remaining_time=true`、`bootstrap_time_limits=false`。

修复仅测量分支的候选协方差描述：旧v3描述器读取节点12:15，实际是偏置；新分支改读本机及已交付消息的原始位置协方差。历史默认分支保持原行为以保护结果与checkpoint兼容性。

## 奖励的含义和成本记账

现有相对EKF在邻机交叉相关未知时保守恢复更新前的最终协方差。因此不能用最终报告协方差下降作为有效奖励。本分支对实际接受的更新使用“线性更新相对信息信用代理”：before与after-linear的非负相对trace差，裁剪至0–1，再以训练专用正值p95缩放。明确标注为代理，不声称实际最终协方差或真实定位精度得到同等改善。after-final与真实误差单独诊断。

代价为0.05乘实际 `load_packets_applied`，另以0.05乘“预算已消费而没有生成overhead的部分”作为浪费机会惩罚。512字节表示现有等价队列负载，不是精确RF能耗。运动后链路失效会先耗预算、随后丢弃动作，没有overhead包；不得为此虚构512字节发送。普通状态广播另行记录，不混入测量动作成本。训练使用三机局部效用的均值作为共享团队奖励，忽略原环境legacy学习奖励。

Greedy 从当前本机/已交付邻机原始协方差、估计相对方向与现有未知交叉相关上界计算预期线性信用，减去固定选择成本；none得分0。测距噪声方差必须来自有效冻结的 `SimulationConfig.sensor.uwb_los_std_m`，不得使用EKF默认值或硬编码回退。有效SensorConfig及方差记录在配置、校准和checkpoint中。当前配置经静态核实为0.08 m，方差0.0064 m²。

## 场景、初始化和校准准入

固定4个TRAIN、3个VALIDATION场景；含两个稳定多候选和两个动态75 m邻居门槛训练场景。长直线目标约45–55 m，三机航路分道或分高度，保持持续运动。验证为平移后的保留几何，坐标/ID/噪声种子分离；这不是分布外几何泛化设计。所有控制器参数均继承前阶段。

种子：校准训练噪声2026116100–2026116103；验证噪声2026116400–2026116402；基线动作2026116500（各场景加索引）；网络初始化2026116200；在线rollout为2026116300加回合索引。

校准只运行4个TRAIN场景，各完整120步，共480联合步，不创建Trainer、不收集PPO rollout。拟合既有37/11 Actor标准化与439维Critic标准化，并计算正值信用p95。保存完整原始观测银行、统计、常量通道、裁剪记录、配置/源码/场景哈希。此后训练和验证禁止再次fit。

校准准入规则在运行前固定，失败写出 `ready=false` 并拒绝训练：

- 每个TRAIN场景，三机各自实际速度超过0.5 m/s的步比例最低值至少0.5。
- 四场景合计至少30个实际接受的正值信用事件，p95有限且为正；至少一个Greedy正净效用决策。
- 四场景合计至少30个决策点同时有两个合法非none邻居，而且未缩放的可观测预期信用最大差超过1e−6；记录数量、分母、比例和最大差。
- 至少一个预先声明的TRAIN动态场景出现物理潜在图变化，各场景均报告计数。
- 归一化统计、维度和来源完整且有限。

各向同性EKF和相同LOS噪声下，几何方向本身可能不能产生候选收益差。这些门槛尚未用完整轨迹验证；不得为通过门槛添加传感器异质性、修改EKF/噪声或使用验证集调参。

Actor/GAT/Critic全部鲜初始化，Adam与rollout为空；不加载A/B或旧joint权重，无resume选项。验证只加载本阶段checkpoint；先校验JSON sidecar的模式、scope、37/11/439、架构、配置、校准/归一化、场景、源码和文件SHA，再允许 `torch.load(weights_only=True)`，并复核内部metadata。

## PyCharm 设置与实际运行顺序

Python解释器：`D:\research project\.venv\Scripts\python.exe`。

Working directory：`D:\research project`。使用Script path方式，路径取下表；Parameters取命令中脚本后的参数。五个入口也支持对应 `python -m src.experiments.stage3_...` 模块方式，不在import时运行实验。

| 顺序 | 实际脚本（相对工作目录） | 用途 |
|---|---|---|
| 0 | `src/experiments/stage3_precheck_measurement_selection.py` | 可选重新检查接线，仅24联合步，零训练 |
| 1 | `src/experiments/stage3_calibrate_measurement_selection.py` | TRAIN-only全时域统计与任务准入 |
| 2 | `src/experiments/stage3_train_measurement_selection.py` | 鲜初始化单seed小规模真实MAPPO smoke |
| 3 | `src/experiments/stage3_evaluate_measurement_selection.py` | 固定checkpoint和统计的验证集评估 |
| 4 | `src/experiments/stage3_baselines_measurement_selection.py` | 同验证场景/噪声的Random、Greedy、None对照 |

在PyCharm Terminal按顺序执行，或分别建立Python运行配置：

```powershell
.\.venv\Scripts\python.exe src/experiments/stage3_precheck_measurement_selection.py --run-id precheck_pycharm_s2026
.\.venv\Scripts\python.exe src/experiments/stage3_calibrate_measurement_selection.py --run-id calibration_s2026
.\.venv\Scripts\python.exe src/experiments/stage3_train_measurement_selection.py --run-id smoke_s2026 --calibration-id calibration_s2026
.\.venv\Scripts\python.exe src/experiments/stage3_evaluate_measurement_selection.py --run-id validation_smoke_s2026 --calibration-id calibration_s2026 --checkpoint-run smoke_s2026
.\.venv\Scripts\python.exe src/experiments/stage3_baselines_measurement_selection.py --run-id baselines_s2026 --calibration-id calibration_s2026
```

步骤0本轮已经以 `prep_precheck_20261012` 完成，可省略；上述新ID用于用户手动复核。步骤1未通过时停止，不执行步骤2。首次运行不修改配置，不以验证结果选择奖励或初始化。所有命令均对应项目中真实脚本，均已通过直接和模块 `--help` 检查。

配置统一为 `data/configs/three_uav_measurement_selection_v3_seed2026_20261012.json`，无参数时自动使用此文件。

每个run-id对应下列三类独立目录，任何目录已存在都拒绝运行，不覆盖、不自动添加后缀：

```text
data/synthetic/three_uav_measurement_selection_v3_seed2026_20261012/<run-id>/
logs/preliminary/three_uav_measurement_selection_v3_seed2026_20261012/<run-id>/
plots/preliminary/three_uav_measurement_selection_v3_seed2026_20261012/<run-id>/
```

校准生成data目录的 `observation_bank.json`、`calibration.json`，logs目录的 `calibration_readiness.json`。失败也保留银行和诊断，退出码2；训练拒绝其ready=false。

训练保存data目录的 `checkpoint_initial.pt`、`checkpoint_update_001.pt`等及sidecar、`checkpoint_final.pt`、各update的rollout JSON；保存logs目录的 `training_curve.csv`、`ppo_records.csv`、`ppo_post_update_records.csv`、动作/门控/预算/开销/误差/裁剪CSV，plots目录的 `training_curve.png`。post-update记录是最终更新策略在保存在线样本上的诊断比率，不冒充每个minibatch中的比率；原生PPO minibatch统计另在训练曲线中记录。

验证生成 `evaluation_summary.json` 及相同物理/通信/误差日志，不fit或update。Random/Greedy/None共用校准和预算，在相同验证场景与indexed噪声种子下比较；闭环轨迹可因估计反馈分离，不声称后续状态相同。

## 首次 smoke 规模和停止条件

固定12回合、每回合120联合步，总1440联合交互步；每回合调用一次原生MAPPO update，共12次。每update有360个agent-time样本，4个epoch、minibatch120，最多144个实际Adam step。PPO配置为gamma0.99、GAE0.95、clip0.2、学习率1e−4、熵0.005、max-grad0.5，不做超参数搜索。

立即停止并保留已有输出：NaN/Inf、无效策略/元数据、碰撞、同步间隔低于2 m、边界/安全投影或环境裁剪、实际加速度越界、预算/步计数越界、冻结头发生变化、缺失update或rollout未清空。只有完成含合法非none样本、有限非零学习信号且累计非零Actor梯度的更新后，Actor参数仍完全不变，才报告参数停滞；全none或数值零梯度批次不据此失败。数值零判别1e−12仅用于检查，不改变优化器或奖励。

奖励低、Greedy优于smoke或单次失败不自动增加训练量，也不能证明MAPPO不可行。请返回校准readiness、run_status、训练曲线、验证与基线汇总，再决定后续修复或扩大研究。

## 本轮已执行与未执行

新分支及工作流纯测试、旧模式精确回归、十个入口帮助、静态编译通过。本轮唯一环境预检为三场景各8步，共24联合步；36次非none选择全部产生真实预算/等价开销和接受相对EKF更新，计36个预算单位、36个等价512字节负载包。最小同步间隔11.6132 m；没有碰撞、边界/安全投影或动作裁剪。没有Trainer或网络在环境中运行。

真实PPO更新、Adam step、checkpoint加载、PPO rollout加入、神经策略执行、真实轨迹归一化fit均为0；没有运行校准、smoke、完整验证或基线实验。合成测试中的标准化和backward仅为接口夹具，不属于训练数据或参数更新。

短预检未覆盖120步持续运动、动态图门槛、重复候选收益差或学习性能，不能产生ready=true训练准入。真正可学习性以用户TRAIN-only校准结果为准。EKF仍用真实初态初始化，运行中控制不直接读真值；在线CI仍未验证，本阶段不改变TTL/周期。A/B、单机与第二阶段全部产物保留，不提交或推送。

本协议使用代码字段名与既有术语，不新增论文数学符号；收尾运行 `main.py audit-symbols`。
