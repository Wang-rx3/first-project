# 2 系统模型

## 2.1 研究场景、参与实体与系统边界

### 2.1.1 研究场景

本文考虑一个由多架无人机组成的三维空中协同系统。无人机在部分 GNSS 拒止、动态遮挡和无线链路时变的环境中执行协同导航与定位任务。每架无人机能够利用本机惯性测量、视觉里程计、GNSS（若可用）以及与邻居之间的 UWB 相对测距进行状态估计；无人机之间通过时变无线网络交换定位信息和任务相关信息。

本文研究的核心闭环为：

```text
无人机运动与环境变化
        ↓
IMU / UWB / 视觉里程计 / GNSS / 无线链路观测
        ↓
通信约束下的 Distributed EKF
        ↓
估计状态、定位协方差与动态通信图
        ↓
GAT 邻居关系编码
        ↓
MAPPO 联合选择协同测量、通信资源和三维轨迹动作
        ↓
下一时刻的运动、通信和定位状态
```

该闭环中的因果关系必须保持一致：通信状态决定邻居信息能否被接收和使用，定位状态与定位不确定度影响后续协作对象和轨迹决策，动作又会通过改变无人机位置、邻居关系和通信资源影响下一时刻的通信与定位性能。

本文不把“通信优化”和“定位优化”视为两个彼此独立的子问题，而是研究二者通过相对几何、测量可用性和时变通信图产生的耦合关系。

当前论文主线实现将完整框架统一命名为 **Local-EKF-GAT-MAPPO (Proposed Method)**。代码中的 `baseline_mappo` 仅作为历史兼容键，不表示去除邻居信息或退化为独立控制器；代理教师与执行期通信约束分支属于历史探索，不纳入当前主线的实现和图表比较。

在当前 `core-framework-v3` 实验协议中，主方法只训练和报告
**Local-EKF-GAT-MAPPO (Proposed Method)**，其内部兼容键仍为 `baseline_mappo`；
`local_only_fixed`、`random_cooperative` 和 `budgeted_greedy_ig` 作为不训练的规则参照。
历史 `mappo_mlp` 仅保留在旧架构对照结果中，`core-framework-v3` 不训练、不读取也不绘制该方法。
该协议的前三张主图只描述主方法训练过程，后两张性能图才与三个规则参照比较。

### 2.1.2 参与实体

系统包含以下实体：

1. **无人机集合 $\mathcal{U}$**：系统中的协同飞行节点。令

   $$
   \mathcal{U}
   =
   \left\{
   \mathrm{UAV}_{i}
   \mid
   i\in\{1,\ldots,N_u\}
   \right\},
   \qquad
   |\mathcal{U}|=N_u .
   $$

   其中，$\mathrm{UAV}_{i}$ 表示第 $i$ 架无人机，$N_u$ 表示无人机总数，$i$ 和 $j$ 表示无人机索引。除非特别说明，$i\neq j$。无人机具有三维运动能力、机载传感器、无线通信能力和本地状态估计能力。
2. **障碍物集合 $\mathcal{O}_t$**：包括建筑物、树木、车辆或行人等静态和动态障碍物。定义

   $$
   \mathcal{O}_t
   =
   \left\{
   o_m
   \mid
   m\in\{1,\ldots,N_o(t)\}
   \right\},
   \qquad
   |\mathcal{O}_t|=N_o(t).
   $$

   其中，$o_m$ 表示第 $m$ 个有效障碍物，$N_o(t)$ 表示时刻 $t$ 被环境模型纳入碰撞、遮挡或观测计算的障碍物数量。障碍物既影响碰撞安全，也可能遮挡无线链路和视觉观测。
3. **传感器集合**：包括 IMU、UWB、视觉里程计和 GNSS。不同无人机可以配置不同的传感器可用性、测量噪声和更新频率。
4. **无线通信链路**：由无人机之间的可用通信边组成。每条边具有相对距离、LOS/NLOS 状态、RSSI、SNR、丢包率、时延、带宽和 AoI 等属性。
5. **任务目标或导航参考**：用于确定无人机的期望运动方向、任务区域或终点。目标信息是策略输入的一部分，但不改变协同定位的观测模型。

在时刻 $t$，无人机 $i$ 的有效通信邻居集合定义为：

$$
\mathcal{N}_{i,t}
\subseteq
\mathcal{U}\setminus\{\mathrm{UAV}_{i}\}.
$$

若 $\mathrm{UAV}_{j}\in\mathcal{N}_{i,t}$，表示无人机 $i$ 在时刻 $t$ 能够按照当前通信条件接收来自无人机 $j$ 的有效信息。该关系不只由几何距离决定，还必须满足通信距离、LOS/NLOS、链路质量、丢包和时延等条件。后续通信图 $G_t$ 的边集合由这些有效邻居关系确定。

为避免“异构”概念被过度使用，本文采用如下定义：若无人机之间的传感器配置、通信资源、噪声水平或运动约束不同，则称其为异构多无人机；若仅有初始位置和运动轨迹不同，则称其为同构多无人机。主模型采用统一的状态向量维度，以便进行 Distributed EKF 和 MAPPO 建模；异构性通过传感器可用性、观测噪声、通信能力和个体约束参数体现，而不是通过随意改变状态维度体现。

### 2.1.3 系统输入、状态和输出

系统在 episode 开始时接收初始条件、场景配置和模型参数；在每个决策时刻接收当前环境和通信观测。将系统输入统一表示为：

$$
\mathcal{I}_t
=
\left(
\mathcal{I}^{\mathrm{init}},
\mathcal{I}^{\mathrm{env}}_t,
\mathcal{I}^{\mathrm{sens}}_t,
\mathcal{I}^{\mathrm{com}}_t,
\mathcal{I}^{\mathrm{exp}}
\right).
$$

其中，$\mathcal{I}^{\mathrm{init}}$ 表示初始状态，$\mathcal{I}^{\mathrm{env}}_t$ 表示环境和任务配置，$\mathcal{I}^{\mathrm{sens}}_t$ 表示传感器配置和观测条件，$\mathcal{I}^{\mathrm{com}}_t$ 表示通信配置和链路条件，$\mathcal{I}^{\mathrm{exp}}$ 表示实验配置。具体包括：

- 无人机初始位置、速度和姿态；
- 目标位置或目标轨迹；
- 障碍物位置、速度和几何包络；
- 传感器噪声、更新频率和失效模式；
- 通信半径、总带宽、链路带宽、发射功率、载波频率、噪声参数和链路统计参数；
- GNSS 退化、丢包、时延和 LOS/NLOS 场景参数；
- 实验随机种子和场景布局。

初始运动状态写为：

$$
\mathcal{I}^{\mathrm{init}}
=
\left\{
\mathbf{p}_{i,0},
\mathbf{v}_{i,0},
\boldsymbol{\varphi}_{i,0}
\right\}_{i\in\mathcal{U}}.
$$

目标可以是固定位置，也可以是随时间变化的参考轨迹。固定目标表示为 $\mathbf{p}^{\mathrm{goal}}_i$，目标轨迹表示为：

$$
\tau^{\mathrm{goal}}_i
=
\left\{
\mathbf{p}^{\mathrm{goal}}_{i,t}
\right\}_{t=0}^{T}.
$$

传感器配置和观测条件可概括为：

$$
\mathcal{I}^{\mathrm{sens}}_t
=
\left\{
\mathbf{Q}_{i,t},
\mathbf{R}_{i,t},
f^{\mathrm{sens}}_i,
\mu^{\mathrm{sens}}_{i,t}
\right\}_{i\in\mathcal{U}},
$$

其中，$\mathbf{Q}_{i,t}$ 和 $\mathbf{R}_{i,t}$ 分别表示过程噪声和观测噪声协方差，$f^{\mathrm{sens}}_i$ 表示传感器更新频率，$\mu^{\mathrm{sens}}_{i,t}$ 表示传感器在时刻 $t$ 的可用或失效模式。以 GNSS 为例：

$$
\mu^{\mathrm{gnss}}_{i,t}
\in
\{
\mathrm{normal},
\mathrm{degraded},
\mathrm{denied}
\}.
$$

通信配置和链路条件可表示为：

$$
\mathcal{I}^{\mathrm{com}}_t
=
\left\{
d_{\max},
B_i^{\mathrm{tot}},
P^{\mathrm{tx}}_{ij,t},
f_c,
d_0,
N_0,
F_{\mathrm{nf}},
\mathrm{RSSI}_{\min},
\mathrm{SNR}_{\min},
P_{\mathrm{loss},\max},
D_{\mathrm{com},\max}
\right\}_{(i,j)} ,
$$

其中，$d_{\max}$、$B_i^{\mathrm{tot}}$、$P^{\mathrm{tx}}_{ij,t}$、$f_c$、$d_0$、$N_0$、$F_{\mathrm{nf}}$ 和各质量阈值是环境或通信配置参数，不由执行期策略直接伪造。带宽 $b_{ij,t}$ 由当前动作产生，队列 $Q_{ij,t}$ 和信息年龄 $\mathrm{AoI}_{ij,t}$ 是信道内部状态，链路统计量 $\boldsymbol{\chi}_{ij,t}$ 是信道观测与模型输出，而不是外生输入。链路统计量为：

$$
\boldsymbol{\chi}_{ij,t}
=
\left[
P^{\mathrm{rx}}_{ij,t},
\mathrm{RSSI}_{ij,t},
\mathrm{SNR}_{ij,t},
R_{ij,t},
P_{\mathrm{loss},ij,t},
D_{\mathrm{com},ij,t},
\mathrm{AoI}_{ij,t}
\right].
$$

为避免把模型输出误写成外部输入，$\boldsymbol{\xi}_t$ 仅作为由当前真实场景和传感器模式派生的场景描述量，不再作为独立的在线输入元组。当前定义为：

$$
\boldsymbol{\xi}_t
=
\left\{
\mu^{\mathrm{gnss}}_{i,t},
\ell_{ij,t}
\right\}_{i,j}.
$$

其中，$\ell_{ij,t}$ 由无人机位置和三维障碍物几何计算，$P_{\mathrm{loss},ij,t}$ 与 $D_{\mathrm{com},ij,t}$ 则由 $\boldsymbol{\chi}_{ij,t}$ 所代表的信道计算链得到，二者不能再作为 $\mathcal{I}^{\mathrm{com}}_t$ 的外生输入。真正的通信配置输入只包括距离、带宽、发射功率、载波频率、噪声和候选阈值等固定或实验配置参数。

随机种子和场景布局属于实验复现配置，记录在 $\mathcal{I}^{\mathrm{exp}}$ 中，但不作为在线定位器或执行期 actor 的物理观测输入。

为避免输入、状态和输出之间产生循环，系统在每个决策窗口按照以下因果顺序运行。$G^{\mathrm{cand}}_t$ 和当前动作掩码属于动作产生前的观测；本次动作产生的带宽只影响下一窗口的链路结果：

$$
\mathbf a_t
\longrightarrow
\left\{b_{ij,t},\Delta\mathbf v_{i,t}\right\}
\longrightarrow
\text{在 }[t,t+\Delta t_{\mathrm{dec}})\text{ 内执行运动、障碍物、传感器和信道状态转移}
\longrightarrow
\left\{\boldsymbol{\chi}_{ij,t+1},G_{t+1},\hat{\mathbf x}_{i,t+1},\boldsymbol{\Sigma}_{i,t+1}\right\}.
$$

这里的动作在当前时刻产生，并在 $[t,t+\Delta t_{\mathrm{dec}})$ 内执行；下一时刻的通信观测、定位估计和局部策略观测由执行结果产生。$\operatorname{reset}$ 阶段使用一次中性的均匀带宽探测，只用于形成初始候选边观测，不计作第一步策略动作。

系统状态需要严格区分以下五类：

| 状态类别 | 典型内容 | 是否可由执行期策略直接获得 |
| --- | --- | --- |
| 真实物理状态 | 真实位置、速度、姿态和偏置 | 否，仅用于仿真生成和评估 |
| 传感器观测 | IMU、UWB、视觉和 GNSS 原始观测 | 仅能获得当前实际到达的数据 |
| 滤波估计状态 | $\hat{\mathbf{x}}_{i,t}$、$\boldsymbol{\Sigma}_{i,t}$ | 是，由本机 EKF 输出 |
| 图网络状态 | 节点特征、边特征、邻居集合和注意力表示 | 是，由本机可见信息构造 |
| 强化学习决策状态 | 本机及邻居可见观测、目标和障碍物信息 | 是，但受局部观测范围约束 |

为严格满足后续 MDP 定义，本文进一步区分三个层次的状态：

1. **完整环境状态 $s^{\mathrm{env}}_t$**：包含真实无人机运动状态、障碍物状态、传感器模式、队列、AoI、消息缓冲、邻居信念和信道内部状态，用于环境状态转移；
2. **中心状态 $s^{\mathrm c}_t$**：训练阶段提供给 centralized critic 的固定结构信息，包括估计状态、协方差、全局图摘要、障碍物摘要和通信内部状态的允许摘要；
3. **局部执行观测 $\mathbf{o}_{i,t}$**：无人机 $i$ 在执行期实际可获得的本机观测、成功接收的邻居消息、局部链路质量、UWB 辅助观测、目标/障碍物特征和动作掩码。

其中，$s^{\mathrm{env}}_t$ 是用于声明马尔可夫性的完整环境状态，$s^{\mathrm c}_t$ 不是环境真值的同义词，$\mathbf{o}_{i,t}$ 也不能包含测试真值或未来状态。

在当前输入和系统状态作用下，系统输出统一表示为：

$$
\mathcal{Y}_t
=
\left(
\mathcal{Y}^{\mathrm{est}}_t,
\mathcal{Y}^{\mathrm{graph}}_t,
\mathcal{Y}^{\mathrm{act}}_t,
\mathcal{Y}^{\mathrm{eval}}_t
\right).
$$

其中，$\mathcal{Y}^{\mathrm{est}}_t$ 表示定位估计输出，$\mathcal{Y}^{\mathrm{graph}}_t$ 表示通信图输出，$\mathcal{Y}^{\mathrm{act}}_t$ 表示 MAPPO 决策输出，$\mathcal{Y}^{\mathrm{eval}}_t$ 表示评价指标输出。具体包括：

- 各无人机的估计位置、速度和姿态；
- 各无人机的定位协方差和定位质量指标；
- 当前可用通信图及链路统计量；
- 协同测量对象和通信资源分配结果；
- 三维轨迹和安全约束执行结果；
- 定位误差、丢包率、时延、AoI、连通性、能耗和碰撞统计。

Distributed EKF 的估计状态和协方差输出为：

$$
\mathcal{Y}^{\mathrm{est}}_t
=
\left\{
\hat{\mathbf{x}}_{i,t},
\boldsymbol{\Sigma}_{i,t}
\right\}_{i\in\mathcal{U}}.
$$

当前通信图及其链路统计量输出为：

$$
\mathcal{Y}^{\mathrm{graph}}_t
=
\left(
G_t,
\left\{
\boldsymbol{\chi}_{ij,t}
\right\}_{(i,j)\in\mathcal{E}_t}
\right).
$$

MAPPO 的动作和轨迹输出为：

$$
\mathcal{Y}^{\mathrm{act}}_t
=
\left\{
\mathbf{a}_{i,t}
\right\}_{i\in\mathcal{U}},
\qquad
\tau_i
=
\left\{
\mathbf{p}_{i,t}
\right\}_{t=0}^{T}.
$$

其中，$\mathbf{a}_{i,t}$ 表示无人机 $i$ 在时刻 $t$ 的协同决策，$\tau_i$ 表示其实际执行轨迹。安全约束执行结果由碰撞事件和最小安全距离共同判定。

系统评价指标可以组成向量：

$$
\mathbf{m}_t
=
\left[
E_{\mathrm{loc},t},
P_{\mathrm{loss},t},
D_{\mathrm{com},t},
\mathrm{AoI}_t,
C_{\mathrm{net},t},
E_{\mathrm{energy},t},
E_{\mathrm{coll},t}
\right].
$$

其中，定位不确定度可由位置协方差的迹表示：

$$
E_{\mathrm{uncert},t}
=
\frac{1}{N_u}
\sum_{i=1}^{N_u}
\operatorname{tr}
\left(
\boldsymbol{\Sigma}^{p}_{i,t}
\right),
$$

真实位置误差用于仿真或离线评价：

$$
E_{\mathrm{loc},t}
=
\frac{1}{N_u}
\sum_{i=1}^{N_u}
\left\|
\hat{\mathbf{p}}_{i,t}
-
\mathbf{p}_{i,t}
\right\|_2.
$$

其中，$E_{\mathrm{uncert},t}$ 是滤波器内部的不确定度，$E_{\mathrm{loc},t}$ 是相对于真实位置的实际误差，二者在实验中分别报告。

系统状态转移和观测生成统一写为：

$$
s^{\mathrm{env}}_{t+1}
\sim
\mathcal{T}
\left(
\cdot
\mid
s^{\mathrm{env}}_t,
\mathbf{a}_t
\right),
$$

$$
\left(
\mathcal{Y}_{t+1},
\{\mathbf{o}_{i,t+1}\}_{i\in\mathcal{U}}
\right)
=
\mathcal{H}
\left(
s^{\mathrm{env}}_{t+1}
\right).
$$

其中，$\mathcal{T}$ 由运动、障碍物、传感器、信道、消息缓冲和定位状态转移共同构成，$\mathcal{H}$ 负责从下一时刻环境状态生成系统输出、中心状态和局部观测。策略参数 $\boldsymbol{\theta}$ 只出现在动作生成分布中，不再把当前输出图 $G_t$ 同时作为系统映射的外部输入。

### 2.1.4 系统边界

本文的系统边界限定如下。

**纳入系统模型的部分：**

1. 多无人机三维运动学；
2. 动态障碍物和碰撞安全约束；
3. IMU、UWB、视觉里程计和 GNSS 观测；
4. 通信距离、LOS/NLOS、链路质量、丢包、时延、带宽和 AoI；
5. 通信约束下的 Distributed EKF；
6. 由定位状态、协方差和通信边构成的动态图；
7. GAT 图表示；
8. MAPPO 的集中训练、分散执行；
9. 协同测量选择、通信资源和三维轨迹的联合决策。

**不纳入本文核心优化的部分：**

1. 电机转速、螺旋桨气动特性和完整六自由度飞行动力学；
2. 无线通信协议栈的 MAC 层和物理层波形设计；
3. 新型调制、编码、波束赋形和射频电路设计；
4. 无人机机体结构、能源系统和电池化学模型；
5. 大规模真实飞行数据采集；
6. 任务分配、目标识别和目标跟踪本身；
7. 直接控制真实飞控执行器的安全认证。

其中，飞控或仿真器负责将高层轨迹动作转换为低层控制量；通信统计模型负责产生可校准的链路质量；定位模块负责产生状态估计和协方差；GAT 和 MAPPO 只负责图关系提取和高层协同决策。

### 2.1.5 信息可见性和运行方式

本文采用集中训练、分散执行（CTDE）方式。

在训练阶段，中央 critic 可以访问所有无人机的联合状态、完整通信图、障碍物状态和全局奖励，用于降低多智能体训练的不稳定性。真实物理状态可以在仿真训练中用于环境转移和离线误差计算，但不能作为执行期定位器或 actor 的输入。

在执行阶段，无人机 $i$ 只能使用：

- 本机传感器当前实际到达的观测；
- 本机 Distributed EKF 输出的估计状态和协方差；
- 通过当前可用通信边成功接收到的邻居信息；
- 本机能够观测到的障碍物和目标信息；
- 本机的资源、能量和通信状态。

丢失、过期或未经通信接收的邻居信息不能直接进入执行期 actor。该约束用于保证仿真策略与真实分布式无人机系统之间的一致性。

### 2.1.6 潜在边、候选边、有效边和 UWB 测量边

本文将通信资源分配前的几何可达边、满足统计链路阈值的候选边和数据包成功传输后的有效边严格区分。UWB 物理测量的可用性单独建模：

$$
G^{\mathrm{pot}}_t=(\mathcal U,\mathcal E^{\mathrm{pot}}_t),
\qquad
G^{\mathrm{cand}}_t=(\mathcal U,\mathcal E^{\mathrm{cand}}_t),
\qquad
G_t=(\mathcal U,\mathcal E_t).
$$

其中，$G^{\mathrm{pot}}_t$ 用于确定资源分配的潜在出边，且由环境内部使用真实几何构造；执行期 actor 只接收潜在出边的节点索引、可观测链路特征和带宽掩码，不接收真实距离。$G^{\mathrm{cand}}_t$ 是上一窗口产生的候选图，用于当前协同测量动作掩码；当前动作产生的带宽将在下一窗口重新生成候选图。$G_t$ 只包含当前成功传输后能够交换有效邻居信息的边。UWB 可用边集合记为 $\mathcal E^{\mathrm{uwb}}_t$；它不等同于 $\mathcal E_t$。UWB 物理测量可以独立记录，但当前版本只有在对应邻居状态消息通过有效通信边成功到达时，才能进入 Distributed EKF 和 GAT 的协同邻居更新。

## 2.2 数据角色

本课题不使用公开数据集作为训练、验证、测试或参数校准输入。所有实验数据均由 `src/simulation/` 的纯 Python 环境和 `src/synthetic/` 的 episode 生成器产生，并保存生成配置、随机种子、模型版本和数据 schema。

- 纯 Python episode benchmark：定位、通信、GAT、MAPPO 和联合优化的统一数据源；
- `python_sim_pilot`：环境—EKF—GAT—MAPPO—日志接口联通性检查；
- `python_sim_preliminary`：3/5/7 架无人机规模扩展和绘图预实验；
- `python_sim_benchmark`：正式训练、验证、测试和消融实验的主数据源。

`knowledge/` 中的论文只用于理论依据、符号对齐和模型设计，不被读取为实验数据，也不用于拟合通信或传感器参数。历史 Flightmare 导入适配器仅保留旧文件兼容性，不属于当前研究数据路线。

## 2.3 系统组成与信息流

系统由无人机集合、动态障碍物、机间无线链路、多源传感器、Distributed EKF、GAT 和 MAPPO 决策器组成。整体信息流为：

```text
真实运动状态和环境
        ↓
IMU / UWB / 视觉里程计 / GNSS / 链路观测
        ↓
时间同步、坐标转换和质量标记
        ↓
Distributed EKF：输出估计状态和协方差
        ↓
GAT：构造动态通信图和邻居表征
        ↓
MAPPO：选择通信、拓扑和轨迹动作
        ↓
飞行动力学、通信环境和下一时刻观测
```

真实状态只用于仿真生成、离线评估和训练期 critic；在线定位和策略执行只能使用传感器观测、滤波估计量及可接收的邻居信息。

## 2.4 坐标系、时间基准与数据同步

项目统一采用 ENU 世界坐标系：第一、第二、第三坐标轴分别指向东、北和上。无人机机体系用于表达 IMU 观测，传感器数据在进入融合模块前转换到世界坐标系。若 PX4 使用 NED 坐标系，则在接口层完成 NED 到 ENU 的显式转换，不能在实验结果中混用两套坐标约定。

所有数据使用 episode 内的相对时间戳。当前纯 Python `episode-v1.2` 基线按决策频率推进：$\Delta t=\Delta t_{\mathrm{dec}}$，运动、传感器、链路和 EKF 在一次 `env.step()` 中完成一次更新。`physics_hz`、各传感器频率和 `link_hz` 作为场景配置和未来异步调度的元数据记录，当前版本不宣称已经执行物理子步。若导入真实异步数据，仍通过时间戳匹配到最近的决策窗口；超出容许窗口的观测标记为缺失，不进行无依据插值。

## 2.5 基本假设和建模约定

为使通信、定位、图学习和强化学习模块能够在同一仿真环境中闭合，本文采用以下基本假设。

### 2.5.1 几何和运动假设

1. 所有无人机均在有限三维飞行区域内运动，世界坐标系采用 ENU 约定。
2. 无人机的高层运动由离散时间运动学模型描述，低层姿态和推力控制由飞控或仿真器完成。
3. 每架无人机使用一个具有有限几何包络的安全体进行碰撞检测。
4. 无人机之间必须保持最小安全距离；无人机与障碍物之间也必须保持最小安全距离。
5. 动态障碍物的真实运动轨迹在环境内部生成，但执行期策略只能使用本机实际可观测或通过通信得到的障碍物信息。

### 2.5.2 传感器和定位假设

1. IMU、UWB、视觉里程计和 GNSS 的观测时间戳和噪声参数由 Python 场景配置和传感器模型生成，并通过重复种子、敏感性分析和消融实验检查。视觉里程计还可使用当前窗口可见的 tracking quality 触发 warm-up 拟合的保守协方差映射；该映射只校准观测可信度，不向执行期引入真实位置或未来状态。
2. GNSS 可以在 episode 内处于正常、退化或失效状态；当前基线中的退化表现为噪声增大和间歇无效，未单独加入系统性偏差或独立更新调度。
3. UWB 测距在 LOS 条件下主要表现为零均值噪声，在 NLOS 条件下允许存在正偏差。
4. 只有当前时刻有效到达且未超过时间容许窗口的观测，才能进入 Distributed EKF 更新。
5. Distributed EKF 不直接访问真实位置；真实位置仅用于仿真环境状态转移和离线误差评估。
6. 定位协方差表示滤波器对自身估计不确定度的内部表征，不等同于真实位置误差；二者必须分别报告。当前运动环境的真实状态转移不额外采样过程噪声，过程不确定性由 EKF 过程协方差表示。

### 2.5.3 通信和拓扑假设

1. 无人机之间的通信图是时变图，其边由相对距离、遮挡、链路质量和资源分配共同决定。
2. 通信丢包和时延可能导致邻居信息缺失或过期；超过 `stale_message_limit_s` 的消息不能被当作当前时刻同步观测使用。
3. RSSI、SNR、速率、丢包率、时延和 AoI 由第 2.9 节的显式统计信道模型产生；参数由版本化 Python 配置、参数扫描、重复随机种子和消融实验确定。
4. 定位与通信字段均来自同一 Python 环境，因此共享坐标系、时间基准、动作和场景状态，不需要拼接外部数据源。
5. 通信链路是否可用由环境和系统约束决定，MAPPO 不能直接创建违反通信距离或遮挡条件的物理链路；带宽动作只在潜在出边上生效。
6. GAT 使用当前可获得的节点和边信息进行图表示，不使用测试集真值或未来链路状态。
7. 第一版信道模型不启用隐式自动重传；数据包失败直接进入丢包和 AoI 更新，不把重传延迟重复计入通信时延。

### 2.5.4 强化学习假设

1. MAPPO 采用 CTDE；训练期 critic 可以访问联合状态，执行期 actor 只能使用局部观测。
2. 动作在决策时刻产生，并在一个决策窗口内保持或由低层控制器跟踪。
3. 第一版模型将拓扑变化作为环境和通信动作的结果，不额外设置一个可以任意修改邻接矩阵的独立动作。
4. 在线奖励优先使用 EKF 协方差、通信状态和安全事件等可获得量；真实位置误差主要用于仿真评价和辅助训练分析。
5. 所有奖励分量在加权前进行归一化，权重、随机种子、场景布局和训练/验证/测试划分固定记录。

### 2.5.5 数据和复现假设

1. 纯 Python 自建 benchmark 同时承担定位、测距、异构传感器、通信质量、图结构和联合优化实验。
2. 参数不从公开数据集估计；通过 Python 配置版本、控制变量实验、敏感性分析和多随机种子重复确定。
3. 训练、验证和测试场景在布局、障碍物、初始编队、GNSS 状态和通信参数上进行隔离。
4. 每个实验保留 episode 标识、随机种子、坐标系、时间基准、生成器版本和模型配置，以保证结果可重复。

## 2.6 动态环境与障碍物模型

时刻 $t$ 的障碍物集合为：

$$
\mathcal{O}_t
=
\{o_m\mid m=1,\ldots,N_o(t)\}.
$$

第 $m$ 个障碍物的状态为 $\mathbf{o}_{m,t}$，至少包含位置、速度和几何包络。若障碍物为静态障碍物，则其位置在 episode 内保持不变；若障碍物为动态障碍物，则其状态按照预设轨迹或随机运动模型更新。当前 Python 基线使用固定障碍物列表和 `active` 标记，不在 episode 内自动生成或删除障碍物；只有在场景配置显式改变有效性时，$N_o(t)$ 才发生变化。

将障碍物状态展开为：

$$
\mathbf{o}_{m,t}
=
\left[
\left(\mathbf{p}^{o}_{m,t}\right)^{\mathsf T},
\left(\mathbf{v}^{o}_{m,t}\right)^{\mathsf T},
\left(\mathbf{s}^{o}_{m}\right)^{\mathsf T}
\right]^{\mathsf T},
$$

其中，$\mathbf{p}^{o}_{m,t}\in\mathbb{R}^{3}$ 和 $\mathbf{v}^{o}_{m,t}\in\mathbb{R}^{3}$ 分别表示障碍物位置和速度，$\mathbf{s}^{o}_{m}\in\mathbb{R}^{3}_{+}$ 表示障碍物几何包络的长、宽和高。本文将每个障碍物近似为以 $\mathbf{p}^{o}_{m,t}$ 为中心、以 $\mathbf{s}^{o}_{m}$ 为尺寸的轴对齐包围盒：

$$
\mathcal{B}_{m,t}
=
\left\{
\mathbf q\in\mathbb{R}^{3}
\ \middle|\
\left|
q_d-p^{o}_{m,t,d}
\right|
\leq
\frac{s^{o}_{m,d}}{2},
\ d\in\{1,2,3\}
\right\}.
$$

对于静态障碍物，其状态更新为：

$$
\mathbf{p}^{o}_{m,t+1}
=
\mathbf{p}^{o}_{m,t},
\qquad
\mathbf{v}^{o}_{m,t}
=
\mathbf 0.
$$

对于动态障碍物，本文保留带有随机扰动的离散恒速模型作为一般形式：

$$
\mathbf{p}^{o}_{m,t+1}
=
\mathbf{p}^{o}_{m,t}
+
\mathbf{v}^{o}_{m,t}\Delta t,
$$

$$
\mathbf{v}^{o}_{m,t+1}
=
\mathbf{v}^{o}_{m,t}
+
\boldsymbol{\epsilon}^{o}_{m,t},
\qquad
\boldsymbol{\epsilon}^{o}_{m,t}
\sim
\mathcal{N}
\left(
\mathbf 0,
\mathbf Q^{o}_{m}
\right).
$$

仿真中对动态障碍物速度进行裁剪：

$$
\left\|
\mathbf{v}^{o}_{m,t}
\right\|_2
\leq
v^{o}_{\max},
$$

并在障碍物到达场景边界或预设运动区域边界时采用反射、停留或重新采样规则。具体边界规则作为场景配置记录，不改变上述状态转移模型。

当前 Python 基线取 $\mathbf Q^{o}_{m}=\mathbf 0$，即默认动态车辆采用匀速运动并在场景边界反射；随机速度扰动属于可配置扩展，不作为当前正式基线的默认行为。

令障碍物包围盒与无人机中心之间的距离为：

$$
d^{o}_{im,t}
=
\min_{\mathbf q\in\mathcal{B}_{m,t}}
\left\|
\mathbf{p}_{i,t}-\mathbf q
\right\|_2.
$$

无人机与障碍物或其他无人机之间的最小安全距离记为 $d_{\mathrm{safe}}$。碰撞事件定义为：

$$
E_{\mathrm{coll},t}
=
\mathbb{I}
\left[
\min
\left\{
\min_{i,m}d^{o}_{im,t},
\min_{i\neq j}d_{ij,t}
\right\}
<
d_{\mathrm{safe}}
\right].
$$

障碍物还会影响无人机间链路的 LOS/NLOS 状态。若无人机 $i$ 与 $j$ 之间的视线线段未与任一有效障碍物包围盒相交，则定义为 LOS；否则定义为 NLOS：

$$
\ell_{ij,t}
=
\begin{cases}
1,
&
\overline{\mathbf{p}_{i,t}\mathbf{p}_{j,t}}
\cap
\displaystyle\bigcup_{m=1}^{N_o(t)}
\mathcal{B}_{m,t}
=
\varnothing,
\\[4pt]
0,
&
\text{otherwise}.
\end{cases}
$$

其中，$\ell_{ij,t}=1$ 表示 LOS，$\ell_{ij,t}=0$ 表示 NLOS。该状态同时用于通信质量模型、UWB 测距偏差模型和协同定位可用性判断。

## 2.7 多无人机运动与状态模型

无人机 $i$ 的真实状态向量由位置、速度、姿态和 IMU 偏置组成：

$$
\mathbf{x}_{i,t}
=
\left[
\mathbf{p}_{i,t}^{\mathsf T},
\mathbf{v}_{i,t}^{\mathsf T},
\boldsymbol{\varphi}_{i,t}^{\mathsf T},
\mathbf{b}^{a\mathsf T}_{i,t},
\mathbf{b}^{\omega\mathsf T}_{i,t}
\right]^{\mathsf T}.
$$

其离散时间演化写为：

$$
\mathbf{x}_{i,t+1}
=
f\!\left(\mathbf{x}_{i,t},\mathbf{u}_{i,t},\Delta t\right)
+
\boldsymbol{\epsilon}^{\mathrm{proc}}_{i,t}.
$$

MAPPO 输出高层动作 $\mathbf{a}_{i,t}$，其中连续分量 $\Delta\mathbf v_{i,t}$ 在下一决策窗口由运动执行器执行。为使纯 Python 环境中的轨迹转移可复现，本文采用三维平移运动学模型，不建立完整六自由度气动动力学：

$$
\mathbf v_{i,t+1}
=
\operatorname{clip}_{v_{\max}}
\left(
\mathbf v_{i,t}+\Delta\mathbf v_{i,t}
\right),
$$

$$
\mathbf p_{i,t+1}
=
\mathbf p_{i,t}
+
\mathbf v_{i,t}\Delta t
+
\frac{1}{2}\Delta\mathbf v_{i,t}\Delta t.
$$

其中，$\operatorname{clip}_{v_{\max}}(\cdot)$ 对速度向量进行范数裁剪；安全投影还要检查飞行区域、障碍物安全距离和最小机间距。低层控制量 $\mathbf u_{i,t}$ 是仿真环境内部由上述速度指令生成的执行量，不作为 MAPPO 的直接动作输出。运动约束至少包括：

$$
\|\mathbf{v}_{i,t}\|_2\leq v_{\max},\qquad
\|\Delta\mathbf{v}_{i,t}\|_2\leq a_{\max}\Delta t.
$$

其中，$\Delta\mathbf v_{i,t}$ 的单位是 m/s，表示一个决策窗口内的速度总增量，$a_{\max}$ 通过 $\|\Delta\mathbf v_{i,t}\|_2\leq a_{\max}\Delta t$ 对其进行约束；它不是加速度本身。当前基线令 $\boldsymbol{\epsilon}^{\mathrm{proc}}_{i,t}=\mathbf 0$，不额外向真实运动状态注入过程噪声。$\mathbf{a}_{i,t}$ 是包含离散和连续分量的 MAPPO 高层动作，不能直接当作物理加速度使用。三维姿态在当前简化环境中仅将滚转角和俯仰角置为零，并根据水平速度方向更新偏航角；完整姿态动力学属于暂缓扩展。

正式安全消融另设独立的场景池，将场景文件按 train、validation 和 test 分组并固定写入清单。训练阶段按排序后的场景列表和 episode 顺序读取 train 场景，验证和测试阶段复用各自分组，三个观测/策略分支使用相同的场景序列。既有十个安全风险场景作为冻结 hold-out 集，不参与场景池生成后的参数拟合或模型选择。该划分只改变实验组织方式，不改变环境状态转移、通信模型或碰撞判定。

## 2.8 多源传感器观测模型

无人机 $i$ 的多源观测集合为：

$$
\mathcal{Z}_{i,t}
=
\left\{
\mathbf{f}^{\mathrm{imu}}_{i,t},
\boldsymbol{\omega}^{\mathrm{imu}}_{i,t},
\{z^{\mathrm{uwb}}_{ij,t}\}_{j:(i,j)\in\mathcal{E}^{\mathrm{uwb}}_t},
z^{\mathrm{vo}}_{i,t},
z^{\mathrm{gnss}}_{i,t}
\right\}.
$$

当前纯 Python 环境使用世界坐标系的简化 IMU 观测模型，不展开机体系旋转、重力补偿和完整六自由度惯性动力学：

$$
\mathbf{f}^{\mathrm{imu}}_{i,t}
=
\mathbf{f}_{i,t}
+
\mathbf{b}^{a}_{i,t}
+
\boldsymbol{\epsilon}^{\mathrm{meas}}_{i,t},
$$

$$
\boldsymbol{\omega}^{\mathrm{imu}}_{i,t}
=
\boldsymbol{\omega}_{i,t}
+
\mathbf{b}^{\omega}_{i,t}
+
\boldsymbol{\epsilon}^{\mathrm{meas}}_{i,t}.
$$

其中，$\mathbf f_{i,t}$ 和 $\boldsymbol\omega_{i,t}$ 均按世界坐标系表达；当前实现中 $\mathbf f_{i,t}$ 是运动学加速度代理量，重力和机体系到世界系的旋转不单独建模。

UWB 协同测距模型为：

$$
z^{\mathrm{uwb}}_{ij,t}
=
\|\mathbf{p}_{i,t}-\mathbf{p}_{j,t}\|_2
+
b^{\mathrm{nlos}}_{ij,t}
+
\epsilon^{\mathrm{meas}}_{ij,t}.
$$

LOS 条件下 $b^{\mathrm{nlos}}_{ij,t}=0$，NLOS 条件下该项表示遮挡引入的正偏差。UWB 测量是否存在由 $\mathcal E^{\mathrm{uwb}}_t$ 和传感器状态决定，不直接由通信有效边 $\mathcal E_t$ 决定；但对应的邻居状态消息仍必须通过成功通信边才能进入协同更新。当前实现中的视觉里程计是决策窗口位移增量，GNSS 是带模式相关噪声的位置观测：

$$
\Delta\mathbf p^{\mathrm{vo}}_{i,t}
=
\mathbf p_{i,t}-\mathbf p_{i,t-1}
+
\boldsymbol{\epsilon}^{\mathrm{meas}}_{i,t},
\qquad
z^{\mathrm{gnss}}_{i,t}
=
\mathbf p_{i,t}
+
\boldsymbol{\epsilon}^{\mathrm{meas}}_{i,t}.
$$

当前滤波器直接使用相对位移观测，不把上一时刻估计位置当作独立、无噪声的绝对位置锚点：

$$
z^{\mathrm{vo}}_{i,t}
=
\Delta\mathbf p^{\mathrm{vo}}_{i,t},
\qquad
z^{\mathrm{gnss}}_{i,t}
=
\mathbf p_{i,t}
+
\boldsymbol{\epsilon}^{\mathrm{meas}}_{i,t}.
$$

历史实现曾将上一时刻估计位置 $\mathbf p^{\mathrm{ref}}_{i,t}$ 与位移相加作为伪位置观测；该符号保留用于追溯，不代表当前更新模型。当前实现保存预测前状态及协方差，并用状态转移矩阵传播前后时刻交叉协方差，联合计算位移创新及其协方差。以下矩阵仅为单时刻位置选择矩阵；相对位移对前后状态的雅可比具有相反符号，不能按独立绝对位置更新理解。

$$
\mathbf H^{\mathrm{vo}}_{i,t}
=
\left[\mathbf I_3\ \ \mathbf 0_{3\times 12}\right].
$$

GNSS 退化通过噪声标准差增大和间歇无效实现；无效观测不进入 EKF。更一般的视觉和 GNSS 观测方程仍可写为：

$$
z^{m}_{i,t}=h^{m}(\mathbf{x}_{i,t})+\boldsymbol{\epsilon}^{m}_{i,t},
\qquad
m\in\{\mathrm{vo},\mathrm{gnss}\}.
$$

不同传感器的噪声协方差由 Python 传感器配置指定，并在敏感性实验中报告。Distributed EKF 输出 $\hat{\mathbf{x}}_{i,t}$ 和 $\boldsymbol{\Sigma}_{i,t}$，二者是 GAT 和 MAPPO 的输入，不得直接替代真实状态。

## 2.9 动态通信拓扑与链路质量模型

通信网络表示为：

$$
G_t=(\mathcal{V}_t,\mathcal{E}_t).
$$

本文采用面向联合优化的统计有效信道模型。该模型描述位置、障碍物、带宽和通信配置对链路质量的影响，但不展开 OFDM 波形、MIMO、波束赋形和具体 MAC 协议。通信边为有向边，因为发送功率、带宽分配和队列状态可能具有方向性。

### 2.9.1 带宽分配和信道输入

设 $B_i^{\mathrm{tot}}$ 为无人机 $i$ 的总可用发送带宽，$a^{\mathrm{bandwidth}}_{ij,t}$ 为无人机 $i$ 对潜在出边 $i\!\to\!j$ 分配的带宽比例，则：

$$
b_{ij,t}
=
a^{\mathrm{bandwidth}}_{ij,t}B_i^{\mathrm{tot}},
\qquad
0\leq a^{\mathrm{bandwidth}}_{ij,t}\leq 1.
$$

当无人机 $i$ 存在潜在出边时，带宽比例由 Dirichlet 动作头生成，并满足

$$
\sum_{j:(i,j)\in\mathcal E^{\mathrm{pot}}_t}
a^{\mathrm{bandwidth}}_{ij,t}=1.
$$

当不存在潜在出边时，带宽动作返回空集合，不能用虚拟节点补足总带宽；因此上述等式只适用于潜在出边非空的窗口。

当潜在出边非空时，带宽分配满足：

$$
\sum_{j:(i,j)\in\mathcal{E}^{\mathrm{pot}}_t}b_{ij,t}
= B_i^{\mathrm{tot}}.
$$

当潜在出边为空时，$b_{ij,t}=0$ 对所有 $j$ 成立，且该窗口不生成通信带宽资源边。

潜在边集合先由距离得到：

$$
\mathcal{E}^{\mathrm{pot}}_t
=
\left\{
(i,j)\in\mathcal{U}\times\mathcal{U}
\ \middle|
i\neq j,
d_{ij,t}\leq d_{\max}
\right\}.
$$

随后策略只对潜在出边分配带宽，再根据 LOS/NLOS、RSSI、SNR、丢包率和时延阈值生成候选边。当前基线策略对潜在出边均匀分配带宽；MAPPO 可以通过逐边的 $a^{\mathrm{bandwidth}}_{ij,t}$ 改变后续链路的 $b_{ij,t}$。发射功率 $P^{\mathrm{tx}}_{ij,t}$ 在第一版中作为通信配置输入，不作为独立强化学习动作。

在当前实现中 $P^{\mathrm{tx}}_{ij,t}=P^{\mathrm{tx}}$ 为统一固定配置；若后续改为逐边功率，必须同步扩展动作和配置接口。潜在边和候选边的真实距离、遮挡判断由环境内部使用，actor 只能使用观测到的链路特征、节点索引和带宽掩码。

### 2.9.2 对数距离路径损耗和 LOS/NLOS 衰落

定义 $\ell_{ij,t}=1$ 表示链路为 LOS，$\ell_{ij,t}=0$ 表示链路为 NLOS。参考距离 $d_0$ 处的自由空间路径损耗为：

$$
\mathrm{PL}_0
=
20\log_{10}
\left(
\frac{4\pi d_0 f_c}{c_{\mathrm{prop}}}
\right).
$$

路径损耗采用 LOS/NLOS 条件化的对数距离模型：

$$
L^{\mathrm{path}}_{ij,t}
=
\mathrm{PL}_0
+
10n_{\ell_{ij,t}}
\log_{10}
\left(
\frac{\max(d_{ij,t},d_0)}{d_0}
\right),
$$

其中：

$$
n_{\ell_{ij,t}}
=
\ell_{ij,t}n_{\mathrm{LOS}}
+
(1-\ell_{ij,t})n_{\mathrm{NLOS}}.
$$

阴影衰落和遮挡附加损耗分别表示为 $X^{\mathrm{sh}}_{ij,t}$ 和 $L^{\mathrm{obs}}_{ij,t}$。总链路损耗和接收功率为：

$$
L_{ij,t}
=
L^{\mathrm{path}}_{ij,t}
+
X^{\mathrm{sh}}_{ij,t}
+
L^{\mathrm{obs}}_{ij,t},
$$

$$
P^{\mathrm{rx}}_{ij,t}
=
P^{\mathrm{tx}}_{ij,t}-L_{ij,t}.
$$

在 LOS 条件下 $L^{\mathrm{obs}}_{ij,t}=0$；在 NLOS 条件下，$L^{\mathrm{obs}}_{ij,t}\geq 0$，由三维障碍物遮挡模型和版本化 Python 场景配置确定。阴影衰落在仿真中按 LOS/NLOS 条件采样；正式实验通过参数扫描、重复随机种子和敏感性分析检查其影响，不使用外部通信数据拟合。

第一版将阴影衰落建模为条件高斯变量：

$$
X^{\mathrm{sh}}_{ij,t}
\sim
\mathcal N\left(0,\sigma^2_{\mathrm{sh},\ell_{ij,t}}\right),
$$

相邻链路更新之间默认独立采样；若后续 Python 敏感性实验需要评估时间相关阴影衰落，可将其替换为带相关系数的过程，不改变接收功率和 SNR 接口。

### 2.9.3 RSSI、噪声、SNR 和有效速率

RSSI 是接收功率的带测量噪声观测：

$$
\mathrm{RSSI}_{ij,t}
=
P^{\mathrm{rx}}_{ij,t}
+
\epsilon^{\mathrm{meas}}_{ij,t}.
$$

当 $b_{ij,t}>0$ 时，接收机噪声功率和 SNR 为：

$$
N_{ij,t}
=
N_0
+
10\log_{10}(b_{ij,t})
+
F_{\mathrm{nf}},
$$

$$
\mathrm{SNR}_{ij,t}
=
P^{\mathrm{rx}}_{ij,t}-N_{ij,t}.
$$

其中，$N_0$ 是噪声功率谱密度，单位为 dBm/Hz；$F_{\mathrm{nf}}$ 是噪声系数，单位为 dB；$N_{ij,t}$ 的单位为 dBm。对于 $b_{ij,t}>0$，将 SNR 转换为线性量后，采用带物理层效率系数的 Shannon 型速率近似：

$$
R_{ij,t}
=
\eta_{\mathrm{phy}}b_{ij,t}
\log_2
\left(
1+10^{\mathrm{SNR}_{ij,t}/10}
\right).
$$

式中 $R_{ij,t}$ 的单位为 bit/s。该速率是有效链路容量近似，不等同于某一具体无线协议的应用层吞吐量。

当 $b_{ij,t}=0$ 时，不使用 $\log_{10}(b_{ij,t})$ 计算噪声，直接定义 $R_{ij,t}=0$、$P_{\mathrm{loss},ij,t}=1$，并将链路标记为 `no_bandwidth`；对应链路不能成为候选边。

### 2.9.4 丢包、时延和 AoI

在给定 SNR、LOS/NLOS 状态和队列长度 $Q_{ij,t}$ 后，单链路丢包率采用可由数据拟合的 Logistic 模型：

$$
P_{\mathrm{loss},ij,t}
=
\frac{1}{
1+
\exp\left[
-\left(
\beta_0-
\beta_1\mathrm{SNR}_{ij,t}+
\beta_2\mathbb{I}[\ell_{ij,t}=0]+
\beta_3Q_{ij,t}
\right)
\right]
}.
$$

设 $L_{\mathrm{pkt}}$ 为单个数据包的比特数，$c_{\mathrm{prop}}$ 为传播速度。当 $R_{ij,t}>0$ 且链路在通信范围内时，不启用隐式自动重传的未截断物理时延为：

$$
D^{\mathrm{raw}}_{ij,t}
=
\frac{d_{ij,t}}{c_{\mathrm{prop}}}
+
\frac{L_{\mathrm{pkt}}}{R_{ij,t}}
+
\frac{Q_{ij,t}L_{\mathrm{pkt}}}{R_{ij,t}}.
$$

当 $R_{ij,t}=0$ 或链路超出通信范围时，不计算上式，而是定义 $D^{\mathrm{raw}}_{ij,t}=D_{\mathrm{com},\max}$ 表示不可用状态。日志和观测接口中的 $D_{\mathrm{com},ij,t}$ 定义为：

$$
D_{\mathrm{com},ij,t}
=
\min\left(D^{\mathrm{raw}}_{ij,t},D_{\mathrm{com},\max}\right).
$$

候选边阈值使用 $D^{\mathrm{raw}}_{ij,t}\leq D_{\mathrm{com},\max}$，避免截断后的显示时延掩盖真实超时。

令 $t^{\mathrm{last}}_{ij}$ 表示最近一次成功接收的时间戳，则：

$$
\mathrm{AoI}_{ij,t}=t-t^{\mathrm{last}}_{ij}.
$$

在离散链路更新周期中，等价地使用：

$$
\mathrm{AoI}_{ij,t+1}
=
\begin{cases}
D_{\mathrm{com},ij,t}, & \delta^{\mathrm{rx}}_{ij,t}=1,\\
\min\left(\mathrm{AoI}_{ij,t}+\Delta t_{\mathrm{link}},\mathrm{AoI}_{\max}\right), & \delta^{\mathrm{rx}}_{ij,t}=0.
\end{cases}
$$

链路队列采用离散到达—服务模型：

$$
Q_{ij,t+1}
=
\max\left\{
0,
Q_{ij,t}+A_{ij,t}-S_{ij,t}
\right\},
$$

其中，$A_{ij,t}$ 为当前窗口新到达的数据包数量，$S_{ij,t}$ 为由 $R_{ij,t}$ 和窗口长度决定的可服务数据包数量。$Q_{ij,t}$ 表示服务前队列，$Q_{ij,t+1}$ 表示服务后的队列。第一版环境采用固定消息需求生成 $A_{ij,t}$，不启用隐式自动重传。

为使协同测量动作能够真实地影响通信资源，本研究在正式训练中可以开启一个明确记录的等效开销开关 `measurement_overhead_enabled`。当接收方 $i$ 在当前窗口选择非 `none` 的测量动作 $a^{\mathrm{meas}}_{i,t}=j$，且 $j\to i$ 是当前潜在通信边时，当前窗口的到达包数按如下规则增加一个等效数据包：

$$
A_{ji,t}\leftarrow A_{ji,t}+1.
$$

该等效数据包沿 $j\to i$ 方向进入队列，并在相同的速率、丢包、时延和 AoI 计算链路中处理；它不改变无人机运动、UWB 物理测量或 EKF 更新公式。实现中复用已有 $L_{\mathrm{pkt}}$，规定该开销包大小为 $512$ bytes（即 $4096$ bit），并要求通信配置的 `packet_size_bytes=512`。因此，测量次数增加会通过队列服务和链路统计量反映到 $J_{\mathrm{com},t}$，而不是额外添加一个通信惩罚项。该机制是面向联合优化的等效队列负载，不代表完整的 MAC、请求—响应握手或自动重传协议；开关关闭时不产生测量诱导队列负载，历史实验的运动、通信和奖励结果保持可复现。

### 2.9.5 候选边和有效边

上述信道量计算完成后，通信边分为“潜在边”“候选边”和“当前有效边”三层。潜在边用于资源分配，候选边由带宽和链路质量阈值决定：

$$
\mathcal{E}^{\mathrm{cand}}_t
=
\left\{
(i,j)\in\mathcal{E}^{\mathrm{pot}}_t
\ \middle|\
i\neq j,\ 
d_{ij,t}\leq d_{\max},\
b_{ij,t}>0,\
\mathrm{RSSI}_{ij,t}\geq \mathrm{RSSI}_{\min},\
\mathrm{SNR}_{ij,t}\geq \mathrm{SNR}_{\min},\
P_{\mathrm{loss},ij,t}\leq P_{\mathrm{loss},\max},\
D^{\mathrm{raw}}_{ij,t}\leq D_{\mathrm{com},\max}
\right\}.
$$

定义 $\delta^{\mathrm{rx}}_{ij,t}$ 表示链路 $i\!\to\!j$ 的数据包是否在有效时间窗口内成功到达：

$$
\delta^{\mathrm{rx}}_{ij,t}
=
\begin{cases}
1, & \text{packet arrives successfully before the deadline},\\
0, & \text{otherwise}.
\end{cases}
$$

只有成功到达的数据包才能形成当前有效通信边：

$$
\mathcal{E}_t
=
\left\{
(i,j)\in\mathcal{E}^{\mathrm{cand}}_t
\ \middle|\
\delta^{\mathrm{rx}}_{ij,t}=1
\right\},
\qquad
[\mathbf A_t]_{ij}
=
\mathbb{I}\!\left[(i,j)\in\mathcal{E}_t\right].
$$

因此，$G_t=(\mathcal{V}_t,\mathcal{E}_t)$ 是当前能够实际交换有效信息的通信图；丢包或超时链路不进入 Distributed EKF 和 GAT 的有效邻居消息，但仍保留在候选边统计中用于计算丢包率、时延和 AoI。

系统级通信指标由候选链路聚合得到，以避免把已经丢失或超时的链路从丢包率统计中删除，例如：

$$
P_{\mathrm{loss},t}
=
\operatorname{mean}_{(i,j)\in\mathcal{E}^{\mathrm{cand}}_t}
P_{\mathrm{loss},ij,t},
$$

$$
D_{\mathrm{com},t}
=
\operatorname{mean}_{(i,j)\in\mathcal{E}^{\mathrm{cand}}_t}
D_{\mathrm{com},ij,t}.
$$

丢包、时延和链路统计参数全部由 Python `ChannelConfig`、队列模型和场景配置产生；通过参数扫描、重复随机种子、敏感性分析和消融实验检查其影响。纯 Python 自建 benchmark 产生具有统一坐标、时间、动作和通信字段的联合优化数据；历史 Flightmare 适配器不属于本研究环境或数据源。

## 2.10 协同定位状态与估计信息模型

本节定义协同定位模块在系统中的状态、输入、输出和信息可用条件；Distributed EKF 的预测、更新、线性化及协方差交叉融合算法放在第 3.2 节。无人机 $i$ 的定位模块输出后验估计状态 $\hat{\mathbf{x}}_{i,t}$ 和后验协方差 $\boldsymbol{\Sigma}_{i,t}$。前者是对第 2.7 节真实状态 $\mathbf{x}_{i,t}$ 的在线估计，后者描述滤波器内部的不确定度，不等同于真实定位误差。

定位模块的输入包括上一时刻估计状态和协方差、本机 IMU、视觉里程计、GNSS 和 UWB 当前有效观测、观测噪声与质量标记，以及通过有效通信边成功到达的邻居消息。其系统级输入输出关系写为：

$$
\left(
\hat{\mathbf{x}}_{i,t-1},
\boldsymbol{\Sigma}_{i,t-1},
\mathcal Z_{i,t},
\{\mathfrak m_{j\rightarrow i,t}\}_{j\in\mathcal N_{i,t}}
\right)
\longmapsto
\left(
\hat{\mathbf{x}}_{i,t},
\boldsymbol{\Sigma}_{i,t}
\right).
$$

该映射只规定定位模块与其他系统模块之间的数据依赖，不在本节展开滤波算法的计算步骤。真实状态仅供仿真转移和离线误差评价使用，不进入执行期定位器。

对于 UWB 相对测距，观测函数不能只依赖本机状态。若无人机 $i$ 收到邻居 $j$ 的状态消息

$$
\mathfrak m_{j\rightarrow i,t}
=
\left(
\hat{\mathbf x}_{j,t},
\boldsymbol{\Sigma}_{j,t},
\tau^{\mathrm{msg}}_{j,t}
\right),
$$

且该消息对应的通信边有效、消息年龄不超过最大允许消息陈旧时间 $\Delta t_{\mathrm{stale}}^{\max}$，则该消息可以进入本机协同定位更新。若 $(j,i)\in\mathcal E_t$，表示消息沿有向边 $j\!\to\!i$ 成功到达，接收方 $i$ 使用发送方 $j$ 的状态。UWB 相对测距模型同时依赖本机位置和收到的邻居位置估计：

$$
h_{ij}(\mathbf x_{i,t};\hat{\mathbf x}_{j,t})
=
\left\|
\mathbf p_{i,t}-\hat{\mathbf p}_{j,t}
\right\|_2,
$$

### 2.10.1 延迟消息缓冲与时间对齐

通信时延不再只作为事后统计量，而是决定邻居消息何时进入接收方的消息缓冲。对在决策时刻 $t$ 发送的消息，定义离散到达窗口：

$$
k^{\mathrm{arr}}_{ji,t}
:=
t+
\left\lceil
\frac{D_{\mathrm{com},ji,t}}
{\Delta t_{\mathrm{dec}}}
\right\rceil .
$$

消息缓冲记录发送时的估计状态、协方差、生成时间和预计到达窗口：

$$
\mathfrak b_{j\rightarrow i,t}
:=
\left(
\hat{\mathbf x}^{\mathrm{send}}_{j,t},
\boldsymbol\Sigma^{\mathrm{send}}_{j,t},
\tau^{\mathrm{send}}_{j,t},
k^{\mathrm{arr}}_{ji,t}
\right).
$$

只有同时满足以下条件的消息才能进入接收方的 Distributed EKF 和 GAT：数据包成功发送、当前窗口已经到达、消息字段和协方差有限且有效、消息年龄不超过 $\Delta t_{\mathrm{stale}}^{\max}$，并且对应协同测量动作允许使用该消息。到达后，若发送时刻与当前时刻不一致，则使用第 3.2 节的状态传播模型将邻居估计传播至当前更新时刻，并同步传播协方差；无法可靠传播或已经过期的消息直接丢弃。

邻居估计的不确定度必须并入相对测距观测的不确定度，不能把 $\hat{\mathbf p}_{j,t}$ 当作无噪声真值。当同一邻居状态来自交叉相关未知的多个信息源时，系统使用协方差交叉融合（CI）形成保守的邻居位置信念，再执行本机定位更新。CI 只针对同一邻居状态的多个估计源；它不被表述为自动消除本机与邻居估计之间全部未知交叉相关的方法。相对测距线性化、等效观测协方差、CI 均值与协方差以及权重选择均在第 3.2 节给出。

分布式特征体现在：每架无人机只融合本机可获得的观测和通过当前通信图成功接收的邻居信息；链路中断或观测过期时，相关观测不进入更新步骤。

本文使用 $\boldsymbol{\Sigma}_{i,t}$ 表示无人机 $i$ 的状态估计协方差。相关协同定位和 EKF 文献通常使用 $\mathbf{P}_{i,t}$ 表示同一类协方差对象，因此二者在本文的语义对应关系为：

$$
\boldsymbol{\Sigma}_{i,t}
\ \longleftrightarrow\ 
\mathbf{P}_{i,t}\quad\text{(literature covariance notation)}.
$$

正文和代码统一使用 $\boldsymbol{\Sigma}_{i,t}$；$\mathbf{P}_{i,t}$ 仅作为文献对照别名，不作为本项目的主符号。这样既保留与 EKF 文献的可读对应关系，又避免与链路丢包率 $P_{\mathrm{loss},ij,t}$ 混用。

## 2.11 图结构状态与局部观测表示

本节只定义 GAT 的图输入、信息可见性和输出接口；注意力系数的具体计算、特征归一化和张量实现放在第 3.4 节。系统采用有向通信边约定：$(j,i)$ 表示无人机 $j$ 向无人机 $i$ 发送信息，因此 GAT 在接收节点 $i$ 处聚合有效入邻居：

$$
\mathcal{N}_{i,t}
=
\left\{
j\in\mathcal{U}
\mid
(j,i)\in\mathcal{E}_t
\right\}.
$$

其中，只有有效通信图 $G_t$ 中的边参与邻居消息聚合；潜在图 $G_t^{\mathrm{pot}}$ 和候选图 $G_t^{\mathrm{cand}}$ 只提供当前可观测链路质量、资源分配特征和动作掩码，不直接提供 GAT 的邻居状态消息。GAT 的虚拟自环仅用于保留本机特征，不计入物理通信图和链路统计。

### 2.11.1 节点特征

GAT 的节点输入仅由定位估计、定位不确定度以及本机可见的目标和障碍物状态组成：

$$
\mathbf{h}^{(0)}_{i,t}
=
\left[
\hat{\mathbf{x}}_{i,t}^{\mathsf T},
\operatorname{diag}(\boldsymbol{\Sigma}_{i,t})^{\mathsf T},
\mathbf{q}^{\mathrm{goal}\mathsf T}_{i,t},
\mathbf{q}^{\mathrm{obs}\mathsf T}_{i,t}
\right]^{\mathsf T}.
$$

在默认配置中，局部障碍物特征 $\mathbf{q}^{\mathrm{obs}}_{i,t}$ 仍为四维聚合量，依次表示当前最小障碍物间隙、包围盒数量、球体数量和决策时间步，因此节点特征为 $15+15+3+4=37$ 维。为使安全场景中的动态障碍物运动趋势能够被执行期策略利用，安全运动特征配置在该四维量之后追加四个聚合标量：

$$
\mathbf{q}^{\mathrm{obs}}_{i,t}
\supset
\left[
\hat d^{\mathrm{dyn}}_{i,t},
\hat c^{\mathrm{dyn}}_{i,t},
\hat d^{\mathrm{dyn}}_{i,t+1\mid t},
\mu^{\mathrm{dyn}}_{i,t}
\right]^{\mathsf T}.
$$

四个量的固定顺序分别为：当前动态障碍物最小间隙、动态障碍物相对无人机的最大非负接近速率、使用当前估计运动状态对下一决策窗口得到的最小预测间隙，以及动态运动观测有效性标记。接近速率取非负值，零表示当前聚合观测没有显示接近趋势；有效性标记为 $1$ 仅表示至少存在可用的动态运动估计，不能被解释为碰撞标签或安全保证。若没有可用动态运动信息，代码写入有限的保守占位值并将有效性标记置为 $0$，不把缺失观测当作安全。

这些特征只保留局部聚合结果，不传递障碍物 ID、绝对坐标、原始速度、真实定位误差、未来实际位置或碰撞结果。一步预测是基于当前估计和当前窗口可见运动信息的模型推断，并不读取下一窗口真值。默认开关关闭时，上述追加量不进入图输入，历史 37 维检查点和实验结果的输入契约保持不变；开关开启时，节点特征为 $15+15+3+8=41$ 维。

### 2.11.2 边特征与辅助观测

对于发送节点 $j$ 和接收节点 $i$，中心图使用的链路边特征定义为：

$$
\mathbf e_{ji,t}
=
\left[
(\Delta\hat{\mathbf p}_{ji,t})^{\mathsf T},
\hat d_{ji,t},
\ell_{ji,t},
\mathrm{RSSI}_{ji,t},
\mathrm{SNR}_{ji,t},
P_{\mathrm{loss},ji,t},
D_{\mathrm{com},ji,t},
\mathrm{AoI}_{ji,t},
Q_{ji,t}
\right]^{\mathsf T}.
$$

其中，$\Delta\hat{\mathbf p}_{ji,t}$ 和 $\hat d_{ji,t}$ 由已接收的邻居估计、UWB 观测或允许的地图推断得到；若候选边尚未成功传输邻居状态，则相应几何字段使用可用 UWB 观测或显式缺失标记，不得回填仿真真实距离。UWB 测距 $z^{\mathrm{uwb}}_{ji,t}$ 及其可用性属于独立辅助观测；当前基线不把它们强行拼入上述 11 维链路边特征。

### 2.11.3 Actor/Critic 可见性边界

中心 critic 可以使用由估计状态构造的全局有效图和候选图，并在训练阶段使用物理 LOS/NLOS 指示量和其他真值摘要；执行期 actor 只使用本机局部观测、已经到达且成功接收的邻居估计、当前可观测链路统计量、UWB 辅助观测和动作掩码。actor 不直接使用仿真真实距离 $d_{ij,t}$、物理 LOS 标签、未来链路状态或测试集真值；当前纯 Python 基线没有独立 LOS 传感器，因此 actor 局部边特征的 LOS 项记为显式缺失标记 $-1$。候选或潜在出边的质量只能用于资源分配和动作选择，不能把失败边或尚未到达的消息伪装成有效邻居消息。零向量和 $-1$ 仅表示占位或缺失，不表示真实物理状态；具体有效性掩码和归一化流程在第 3.4 节规定。

### 2.11.4 GAT 输出接口

对无人机 $i$，GAT 根据本机节点特征、有效入邻居节点特征和对应边特征生成图表征：

$$
\mathbf g_{i,t}
=
\operatorname{GAT}_i
\left(
\mathbf h^{(0)}_{i,t},
\left\{
\mathbf h^{(0)}_{j,t},\mathbf e_{ji,t}
\right\}_{j\in\mathcal{N}_{i,t}}
\right).
$$

$\mathbf g_{i,t}$ 不是物理状态估计，而是供 MAPPO actor 或 centralized critic 使用的图表示。actor 使用局部 $\mathbf g_{i,t}$ 以及候选/潜在出边质量和动作掩码；critic 可以使用全局节点和边图表示。图表示随位置、障碍物、传感器可用性和链路状态在每个决策窗口更新，不预先固定为完全图。

## 2.12 多智能体决策过程与动作约束

联合决策过程定义为：

$$
\mathcal{M}
=
(\mathcal{S},\mathcal{A},\mathcal{T},\mathcal{R},\gamma_{\mathrm{disc}}).
$$

完整环境状态包括真实运动、估计、通信、消息和障碍物信息；用于集中训练的中心状态则只保留训练阶段允许访问的固定结构摘要：

$$
s^{\mathrm{env}}_t
=
\left[
\{\mathbf{x}_{i,t},
\hat{\mathbf{x}}_{i,t},
\boldsymbol{\Sigma}_{i,t}\}_{i\in\mathcal{U}},
\mathcal{O}_t,
\{\mathbf{p}^{\mathrm{goal}}_{i,t}\}_{i\in\mathcal{U}},
\{\mu^{\mathrm{sens}}_{i,t}\}_{i\in\mathcal{U}},
\boldsymbol{\xi}_t,
\{Q_{ij,t},\mathrm{AoI}_{ij,t}\}_{(i,j)},
\mathcal{B}^{\mathrm{msg}}_t,
G^{\mathrm{pot}}_t,
G^{\mathrm{cand}}_t,
G_t
\right].
$$

$$
s^{\mathrm c}_t
:=
\left[
\{\hat{\mathbf{x}}_{i,t},\boldsymbol{\Sigma}_{i,t}\}_{i\in\mathcal{U}},
\{Q_{ij,t},\mathrm{AoI}_{ij,t}\}_{(i,j)},
G^{\mathrm{pot}}_t,
G^{\mathrm{cand}}_t,
G_t,
\mathcal{O}_t
\right].
$$

完整环境状态 $s^{\mathrm{env}}_t$ 用于环境状态转移，中心状态 $s^{\mathrm c}_t$ 用于 centralized critic。执行期无人机 $i$ 只能获得局部观测 $\mathbf o_{i,t}$，其内容包括本机估计状态和协方差、本机可见的目标与障碍物特征、已经到达且有效的邻居消息、当前可观测的潜在/候选链路质量、UWB 辅助观测、队列与 AoI 状态、固定测量预算的剩余量以及对应动作掩码。局部观测不包含仿真真实位置、未来链路、未成功接收的邻居状态或测试集真值。因此，完整环境按 $s^{\mathrm{env}}_t$ 建模为马尔可夫决策过程，而分散执行策略面对的是局部部分可观测信息。

每架无人机的动作由协同测量选择、逐边带宽分配和轨迹控制组成：

$$
\mathbf{a}_{i,t}
=
\left[
a^{\mathrm{meas}}_{i,t},
\left\{a^{\mathrm{bandwidth}}_{ij,t}\right\}_{j:(i,j)\in\mathcal{E}^{\mathrm{pot}}_t},
\Delta\mathbf{v}_{i,t}
\right].
$$

离散测量动作选择指向本机的当前候选入邻居中同时具有有效 UWB 测量的源节点，并通过动作掩码 $\mathbf M^{\mathrm{act}}_{i,t}$ 排除无效节点；带宽动作以潜在出边为索引，使用带宽掩码并经过归一化后满足总带宽约束。若没有可行节点，离散动作空间保留显式的 none 选项，不用虚假的邻居填充。

本文将接收方 $i$ 的测量选择解释为已经到达且有效的邻居 $j$ 状态消息/UWB 伪观测是否进入本地 EKF；在开启 `measurement_overhead_enabled` 的训练环境中，同一非空选择还对应第 2.9.4 节定义的一个等效 $L_{\mathrm{pkt}}$ 队列负载。当前版本不设置独立中继动作，也不引入隐式多跳转发；多跳中继必须在未来建立明确的路由、转发队列和端到端 AoI 模型后再扩展。动作 $\mathbf a_{i,t}$ 使用时刻 $t$ 的观测和掩码产生，带宽与速度在下一窗口执行，下一窗口的候选图和有效图再由环境重新计算。

在预算约束实验中，为每个随机种子 $\nu$ 和无人机 $i$ 仅由 baseline 的训练划分确定固定测量预算：

$$
K^{\mathrm{meas}}_{\nu,i}
=
\operatorname{round}
\left(
\frac{1}{|\mathcal{D}_{\mathrm{train}}|}
\sum_{e\in\mathcal{D}_{\mathrm{train}}}
M^{\mathrm{baseline}}_{\nu,i,e}
\right),
$$

其中 $M^{\mathrm{baseline}}_{\nu,i,e}$ 是训练 episode $e$ 中 baseline 对无人机 $i$ 的非 `none` 测量次数。执行期只复用该预算，不从 validation 或 test 重新拟合。窗口 $t$ 的剩余预算为：

$$
B^{\mathrm{rem}}_{\nu,i,t}
=
\max\left(
0,
K^{\mathrm{meas}}_{\nu,i}
-
\sum_{\tau=0}^{t-1}
\mathbb{I}\left[a^{\mathrm{meas}}_{i,\tau}\neq\mathrm{none}\right]
\right).
$$

当 $B^{\mathrm{rem}}_{\nu,i,t}=0$ 时，动作掩码仅保留 `none`；每次合法的非空测量最多消耗一个预算单位。该预算约束只限制测量选择，不改变带宽、速度、EKF、GAT、信道公式或核心目标函数。

为稳定跨随机种子的测量行为，新增的 rate-controlled quality-aware 分支只在训练期更新中加入测量率约束。对随机种子 $\nu$ 和无人机 $i$，目标测量率仅由 baseline 的训练回合统计得到：

$$
\kappa^{\mathrm{target}}_{\nu,i}
=
\frac{\sum_{e\in\mathcal{D}_{\mathrm{train}}}M^{\mathrm{baseline}}_{\nu,i,e}}
{\sum_{e\in\mathcal{D}_{\mathrm{train}}}W_{\nu,i,e}},
$$

其中 $W_{\nu,i,e}$ 是训练 episode $e$ 中无人机 $i$ 的实际决策窗口数。当前 minibatch 中该无人机的期望非 `none` 测量率记为 $\bar\kappa_{\nu,i}$，由 $1-\pi_{\boldsymbol{\theta}}(a^{\mathrm{meas}}_{i,t}=\mathrm{none}\mid\mathbf o_{i,t})$ 在该 minibatch 的均值计算。令 $\tau_{\mathrm{rate}}$ 为固定容差，$\lambda_{\mathrm{rate}}$ 为固定约束系数，则训练更新使用：

$$
L^{\mathrm{rate}}
=
\frac{\lambda_{\mathrm{rate}}}{|\mathcal I|}
\sum_{i\in\mathcal I}
\left[
\operatorname{ReLU}\!\left(\bar\kappa_{\nu,i}-\kappa^{\mathrm{target}}_{\nu,i}-\tau_{\mathrm{rate}}\right)^2
+
\operatorname{ReLU}\!\left(\kappa^{\mathrm{target}}_{\nu,i}-\tau_{\mathrm{rate}}-\bar\kappa_{\nu,i}\right)^2
\right].
$$

这里 $\mathcal I$ 只表示当前 minibatch 中实际出现的无人机集合。该项不进入环境奖励、不读取 validation 或 test，不使用真实误差或未来通信结果；baseline 和旧 budgeted 分支的约束项关闭，只有 rate-controlled 分支启用。它只用于训练期稳定测量动作率，非空测量仍按本节前述规则产生一个等效 $L_{\mathrm{pkt}}$ 队列负载。

为检验固定系数不足时的稳定性，另设不改变旧分支行为的自适应测量率分支
`adaptive_budgeted_quality_aware_mappo`。对每个随机种子 $\nu$ 和无人机 $i$，定义当前 minibatch 的线性率偏差：

$$
v^{\mathrm{rate}}_{\nu,i}
=
\operatorname{ReLU}\!\left(\bar\kappa_{\nu,i}-\kappa^{\mathrm{target}}_{\nu,i}-\tau_{\mathrm{rate}}\right)
+
\operatorname{ReLU}\!\left(\kappa^{\mathrm{target}}_{\nu,i}-\tau_{\mathrm{rate}}-\bar\kappa_{\nu,i}\right).
$$

该分支为每架无人机维护独立的非负乘子 $\lambda^{\mathrm{dual}}_{\nu,i}$，其训练更新为：

$$
L^{\mathrm{rate,dual}}
=
\frac{1}{|\mathcal I|}
\sum_{i\in\mathcal I}
\lambda^{\mathrm{dual}}_{\nu,i}v^{\mathrm{rate}}_{\nu,i},
$$

$$
\lambda^{\mathrm{dual}}_{\nu,i}\leftarrow
\operatorname{clip}_{[0,\lambda^{\mathrm{dual}}_{\max}]}
\left(
\lambda^{\mathrm{dual}}_{\nu,i}
+
\eta^{\mathrm{dual}}_{\mathrm{rate}}v^{\mathrm{rate}}_{\nu,i}
\right).
$$

其中 $\eta^{\mathrm{dual}}_{\mathrm{rate}}=0.25$、$\lambda^{\mathrm{dual}}_{\max}=5$，乘子只在 train 的 `trainer.update()` 完成后以 `detach()` 状态更新，validation 和 test 不更新。该附加项仍不进入环境奖励，不改变动作掩码、固定测量预算、通信队列、EKF 或 $J_t$；旧 baseline、budgeted 和 fixed rate-controlled 分支的默认行为保持不变。

为检验“定位收益—通信开销”代理是否能够被测量动作头利用，进一步定义执行期可见的信息增益和通信代价代理。对候选邻居 $j$，仅使用当前估计位置、协方差、UWB 测量噪声、消息年龄和当前可见链路质量，得到归一化代理 $\widehat G^{\mathrm{IG}}_{ij,t}\in[0,1]$ 与预测通信代价 $\widehat C^{\mathrm{com}}_{ij,t}\in[0,1]$。代理尺度只由同一随机种子的 baseline `train` 数据拟合，validation 与 test 复用该尺度，不参与拟合。定义测量代理效用为：

$$
U^{\mathrm{proxy}}_{ij,t}
=
\widehat G^{\mathrm{IG}}_{ij,t}
-
w^{\mathrm{proxy}}_{\mathrm{com}}
\widehat C^{\mathrm{com}}_{ij,t},
\qquad
U^{\mathrm{proxy}}_{i,\mathrm{none},t}=0.
$$

其中 $w^{\mathrm{proxy}}_{\mathrm{com}}=1$。该效用只构造 measurement head 的辅助 teacher 分布；对当前动作掩码后的可行集合使用温度 $T^{\mathrm{proxy}}=0.25$ 的 softmax，并以 teacher 与 actor 分布的 KL 散度提供辅助损失：

$$
L^{\mathrm{proxy}}
=
\alpha_{\mathrm{proxy}}
\operatorname{KL}
\left(
\pi^{\mathrm{proxy}}_t
\middle\|
\pi_{\boldsymbol{\theta},t}
\right),
\qquad
\alpha_{\mathrm{proxy}}=0.25.
$$

teacher 目标使用 `detach()`；actor body 输出在该辅助路径中停止梯度，因此 $L^{\mathrm{proxy}}$ 只更新 measurement head，不改变带宽头、速度头、GAT 或 centralized critic。候选无效时保留 `none`，预算耗尽时仍只允许 `none`。该代理不读取真实位置、真实误差、真实 LOS/NLOS、NIS 或未来通信结果，也不修改 $J_{\mathrm{loc}}$、$J_{\mathrm{com}}$、EKF 和环境奖励。

在通信约束代理分支中，通信约束已经在真实队列路径上计入完整的测量代价，因此 teacher 不再次惩罚 baseline 训练目标以内的固定负载。此时仅将超过同一随机种子、同一无人机 baseline `train` 目标的部分计入 teacher：

$$
U^{\mathrm{proxy,rel}}_{ij,t}
=
\widehat G^{\mathrm{IG}}_{ij,t}
-
w^{\mathrm{proxy}}_{\mathrm{com}}
\max\left(0,\widehat C^{\mathrm{com}}_{ij,t}-c^{\mathrm{target}}_{\nu,i}\right),
\qquad
U^{\mathrm{proxy,rel}}_{i,\mathrm{none},t}=0.
$$

该目标相对代价只改变通信约束分支的 detached teacher，不改变真实 $J_{\mathrm{com}}$、512 字节队列负载、硬测量预算或通信乘子；baseline 和历史 proxy 分支继续使用原有 teacher 语义。

为显式约束代理分支在状态分布变化下产生的实际通信开销，定义包含固定测量包负载的执行期候选代价：

$$
\widehat c^{\mathrm{meas}}_{ij,t}
=0.5+0.5\left(1-q_{ij,t}\right)\in[0.5,1],
\qquad
\widehat c^{\mathrm{meas}}_{i,\mathrm{none},t}=0.
$$

其中 $q_{ij,t}$ 仅由当前 RSSI、SNR、预测丢包率、时延、AoI、队列和链路状态构造；常数项表示每次有效测量必然进入队列的一个 512 字节等效包。每个随机种子 $\nu$、无人机 $i$ 的目标 $c^{\mathrm{target}}_{\nu,i}$ 只由 baseline train 的实际测量动作及其对应 $\widehat c^{\mathrm{meas}}_{ij,t}$ 求均值得到。当前 minibatch 的策略期望通信代价为 $\bar c_{\nu,i}$。采用单侧违反量和自适应乘子：

$$
v^{\mathrm{com}}_{\nu,i}
=\operatorname{ReLU}\!\left(\bar c_{\nu,i}-c^{\mathrm{target}}_{\nu,i}-\tau_{\mathrm{com}}\right),
\qquad
L^{\mathrm{com,dual}}
=\frac{1}{|\mathcal I|}\sum_{i\in\mathcal I}\lambda^{\mathrm{com}}_{\nu,i}v^{\mathrm{com}}_{\nu,i},
$$

$$
\lambda^{\mathrm{com}}_{\nu,i}\leftarrow
\operatorname{clip}_{[0,\lambda^{\mathrm{com}}_{\max}]}
\left(\lambda^{\mathrm{com}}_{\nu,i}+\eta^{\mathrm{com}}v^{\mathrm{com}}_{\nu,i}\right).
$$

当前固定 $\tau_{\mathrm{com}}=0.05$、$\eta^{\mathrm{com}}=0.25$、$\lambda^{\mathrm{com}}_{\max}=5$。该分支关闭原双侧测量率约束，保留训练集硬预算上限；乘子只在 train 的策略更新后变化，validation/test 不更新。约束梯度通过停止梯度的 actor 上下文只进入 measurement head，不改变环境奖励、EKF、GAT、带宽头或速度头。

为针对单次协同更新收益在后续预测中衰减的问题，新增 `persistent_proxy_guided_quality_mappo` 分支。该分支不替换实际 EKF，而是用当前 actor 可见的本机协方差和候选邻居估计构造固定预测视界的测量选择代理。令 $H_{\mathrm{persist}}=3$，以当前配置中的名义状态转移矩阵 $\mathbf F$ 和过程噪声协方差 $\mathbf Q$，分别对“未进行该候选测量”和“仅进行线性相对测距更新”两种假设传播：

$$
\widetilde{\boldsymbol{\Sigma}}^{(0)}_{i,t+H_{\mathrm{persist}}}
=
\mathbf F^{H_{\mathrm{persist}}}
\boldsymbol{\Sigma}^{\mathrm{cur}}_{i,t}
\left(\mathbf F^{H_{\mathrm{persist}}}\right)^{\mathsf T}
+
\sum_{\ell=0}^{H_{\mathrm{persist}}-1}
\mathbf F^\ell\mathbf Q(\mathbf F^\ell)^{\mathsf T},
$$

$$
\widetilde{\boldsymbol{\Sigma}}^{(1)}_{i,t+H_{\mathrm{persist}}}
=
\mathbf F^{H_{\mathrm{persist}}}
\boldsymbol{\Sigma}^{\mathrm{lin}}_{i,t}(j)
\left(\mathbf F^{H_{\mathrm{persist}}}\right)^{\mathsf T}
+
\sum_{\ell=0}^{H_{\mathrm{persist}}-1}
\mathbf F^\ell\mathbf Q(\mathbf F^\ell)^{\mathsf T}.
$$

其中 $\boldsymbol{\Sigma}^{\mathrm{lin}}_{i,t}(j)$ 是候选邻居 $j$ 的标量相对测距观测对应的假设线性协方差，不会写回实际滤波器。持久信息收益使用位置协方差迹的固定视界减少量并截断为非负，再用同一随机种子的 baseline `train` 分位尺度归一化为 $\widehat G^{\mathrm{persist}}_{ij,t}\in[0,1]$。通信质量单独作为代价 $\widehat C^{\mathrm{com}}_{ij,t}$，不乘入定位收益；测量效用为：

$$
U^{\mathrm{persist}}_{ij,t}
=
\widehat G^{\mathrm{persist}}_{ij,t}
-
w^{\mathrm{proxy}}_{\mathrm{com}}\widehat C^{\mathrm{com}}_{ij,t},
\qquad
U^{\mathrm{persist}}_{i,\mathrm{none},t}=0.
$$

该效用只用于 measurement head 的 detached teacher 分布；teacher 温度为 $0.25$、系数为 $0.25$，并继续保留当前动作掩码、硬预算和固定 512 字节负载。持久代理只读取执行期可见的估计量、UWB 噪声、消息年龄和当前链路字段，不读取真实位置、真实误差、NIS、LOS/NLOS、未来通信结果或测试数据。其辅助梯度只进入 measurement head，不改变 GAT、带宽动作、速度动作、centralized critic、EKF、奖励或 $J_t$。

状态转移算子 $\mathcal T$ 不是额外的黑箱动力学，而是由第 2.6 节障碍物转移、第 2.7 节无人机运动、第 2.8 节观测生成、第 2.9 节信道与拓扑更新、消息缓冲更新以及第 2.10 节协同定位状态更新共同组成。该定义保证动作、物理状态、通信状态和下一时刻观测之间保持一致的因果顺序。

策略目标为：

$$
\max_{\pi_{\boldsymbol{\theta}}}
\mathbb{E}_{\pi_{\boldsymbol{\theta}}}
\left[
\sum_{t=0}^{T-1}\gamma_{\mathrm{disc}}^t r_t
\right].
$$

其中，奖励函数 $\mathcal R$ 由第 2.13 节的即时联合代价、碰撞惩罚和能耗惩罚定义，并且所有无人机共享同一个团队奖励 $r_t$。训练阶段 critic 使用中心状态 $s^{\mathrm c}_t$，执行阶段 actor 只使用局部观测 $\mathbf o_{i,t}$。Actor/Critic 网络、混合动作分布、优势估计和 PPO 参数更新放在第 3.5 节。

本文不设置能够任意修改邻接矩阵或执行多跳转发的独立拓扑动作。实际通信拓扑由无人机位置、通信距离、障碍物遮挡、链路质量、带宽分配和协同测量选择共同决定；具体动作执行和策略训练过程见第 3 章。

## 2.13 通信—定位联合优化问题

本文以定位不确定度、丢包、信息年龄、通信时延和网络连通性为主要优化因素。核心系统目标只包含定位代价和通信代价；碰撞与能耗不并入 $J_t$，而作为强化学习训练阶段的额外奖励惩罚。所有连续指标的归一化尺度只由训练集统计得到，验证集和测试集复用训练阶段保存的尺度。

定位代价和通信代价将在下式中分别定义。

$$
J_{\mathrm{loc},t}
:=
w_{\mathrm{unc}}\bar E_{\mathrm{uncert},t}.
$$

$$
J_{\mathrm{com},t}
:=
w_{\mathrm{loss}}\bar P_{\mathrm{loss},t}
+
w_{\mathrm{AoI}}\overline{\mathrm{AoI}}_t
+
w_{\mathrm{delay}}\bar D_{\mathrm{com},t}
-
w_{\mathrm{net}}\bar C_{\mathrm{net},t}.
$$

核心联合代价为：

$$
J_t
:=
J_{\mathrm{loc},t}+J_{\mathrm{com},t}.
$$

当候选边集合为空时，$\bar P_{\mathrm{loss},t}$、$\bar D_{\mathrm{com},t}$ 和 $\overline{\mathrm{AoI}}_t$ 分别使用预设的最差归一化值；不把空集合当作零代价。系统级 AoI 定义为：

$$
\mathrm{AoI}_t
:=
\operatorname{mean}_{(i,j)\in\mathcal E_t^{\mathrm{cand}}}
\mathrm{AoI}_{ij,t}.
$$

丢包率和通信时延也在同一候选边集合上聚合。对任意需要归一化的非负指标 $x_t$，只使用训练集统计的95%分位数：

$$
\bar x_t
:=
\operatorname{clip}
\left(
\frac{x_t}{q^{95}_{x,\mathrm{train}}},
0,1
\right).
$$

验证集和测试集复用训练集尺度，不重新拟合。网络连通性定义为有效有向图无向化后的最大连通分量比例：

$$
C_{\mathrm{net},t}
=
\frac{|\mathcal C_t^{\max}|}{N_u}.
$$

其中，先将有效有向图 $G_t$ 的边忽略方向得到无向图，再令 $\mathcal C_t^{\max}$ 为该无向图的最大连通分量；因此 $C_{\mathrm{net},t}$ 表示最大连通分量中的无人机比例，而不是“至少有一个邻居”的节点比例。

强化学习训练使用共享团队奖励。团队能耗代理采用安全投影后的实际速度增量：

$$
E_{\mathrm{energy},t}
:=
\frac{1}{N_u}
\sum_{i\in\mathcal U}
\frac{\left\|\Delta\mathbf v^{\mathrm{exec}}_{i,t}\right\|_2}
{a_{\max}\Delta t},
\qquad
r_t
:=-\left(
J_t
+\lambda_{\mathrm{coll}}E_{\mathrm{coll},t}
+\lambda_{\mathrm{energy}}E_{\mathrm{energy},t}
\right),
\qquad
r_{i,t}:=r_t.
$$

其中，$\Delta\mathbf v^{\mathrm{exec}}_{i,t}$ 是安全投影后的实际执行速度增量，$\lambda_{\mathrm{coll}}$ 和 $\lambda_{\mathrm{energy}}$ 只属于强化学习奖励惩罚，不并入核心系统代价 $J_t$。所有无人机共享同一团队奖励，轨迹平滑度仍作为独立评价指标记录，暂不加入奖励。碰撞事件由三维安全检测器直接计算，不能在数据生成阶段固定为零。

为避免碰撞惩罚和协方差代价的少量异常值主导 PPO 的数值更新，稳定化协议 `stabilized-v3` 采用双通道奖励记录。`reward_total` 始终保存上式定义的原始团队奖励，用于论文报告、奖励分解和与历史实验的兼容比较；`learning_reward_total` 仅作为 PPO 的优化输入。后者对原始联合代价采用有界裁剪，对碰撞惩罚只使用不超过 `learning_collision_penalty_cap` 的学习惩罚，并加入基于当前联合代价的有界势差项 `learning_potential`。该处理不改变环境状态转移、真实碰撞事件、$J_t$、$J_{\mathrm{loc}}$、$J_{\mathrm{com}}$ 或报告指标，也不把碰撞从数据中删除。未启用 `stabilized-v3` 时，学习通道等于原始团队奖励，以保持旧检查点和旧实验可复现。

`core-framework-v3` 延续报告奖励与 PPO 学习奖励的双通道定义，并单独修正动作概率的语义。
其新实验使用 `ppo_action_logprob_mode=sampled`：安全投影、动作裁剪和预算约束只改变
环境实际执行动作，不回写策略已经采样的动作及其旧对数概率；采样动作和执行动作均写入
诊断日志。这样，PPO 的重要性采样比率对应采样分布，执行投影的影响则通过环境奖励和
通信/定位结果体现。历史 `executed` 模式继续保留，仅用于兼容旧实验和旧检查点。

在 `core-framework-v4-delta-scale` 中，PPO 的定位反馈进一步使用同一 episode 内的相邻
不确定度变化。令 $\Delta u^{\mathrm{raw}}_t=u_{t-1}-u_t$，只对 episode 内相邻窗口求差分，并以 warm-up 中
这些原始差分的稳健尺度
$s_{\Delta u}=\max(1.4826\,\operatorname{MAD}(\Delta u^{\mathrm{raw}}_t),10^{-3})$
进行归一化，得到裁剪反馈
$\Delta\tilde u_t=\operatorname{clip}(\Delta u^{\mathrm{raw}}_t/s_{\Delta u},-3,3)$。
该训练反馈不改变上式报告奖励、$J_{\mathrm{loc}}$、$J_{\mathrm{com}}$ 或任何物理状态转移；
validation/test 只读取 warm-up 冻结的尺度，且缺失尺度时拒绝运行。这样明确区分了
“绝对协方差水平的尺度”和“相邻窗口变化的尺度”，避免跨 episode 或错误分母造成反馈异常。

`core-framework-v10-safety` 在不改变上述奖励和状态模型的前提下，使用
`safety_execution_mode=joint_swept_v1` 统一处理一个决策窗口内的安全执行。环境先保存时刻
$t$ 的全部 UAV 与动态障碍物状态，基于所有请求动作计算候选下一位置，并预测障碍物的
时刻 $t+1$ 位置；随后联合检查 UAV--UAV 端点距离、线段交叉、UAV--动态障碍物扫掠路径
和边界越界，对冲突 UAV 进行对称安全投影，统一提交最终状态后再进行一次物理碰撞检查。
`legacy` 模式保留历史逐 UAV 执行语义。日志区分投影前风险、投影后物理碰撞、保护距离
违反和不可避免动态碰撞，且训练图的终止归因优先使用可靠的 `termination_reason`、
`terminated`、`truncated` 和碰撞标志：碰撞归为 `collision`，非碰撞终止归为
`early_termination`，时间上限归为 `time_limit`，无法判断的短回合归为 `unknown_short`。
因此正常时间上限回合不被当作异常，碰撞和非碰撞提前终止也不会被重复分类。

### 通信遮挡区域与实体障碍物的语义边界

在 `core-framework-v11-comm-occluder` 中，`screening_moving_nlos_screen` 被定义为纯通信
遮挡区域，而不是实体障碍物。该几何区域仍参与链路的 `line_of_sight` 判断，并可改变
LOS/NLOS、丢包、时延和 AoI 等通信量；但不参与物理碰撞、保护距离违规、安全投影或
动态实体障碍物风险特征。真实建筑物和动态车辆仍参与上述物理安全检查。这样可以避免
把“为了研究通信遮挡而接近屏蔽区域”误记为 UAV 撞上建筑物或车辆。

该语义通过场景字段 `physical_collision_enabled=false` 版本化记录，并写入每回合的
`obstacles.csv`、`scenario.json` 和集成报告。历史 `legacy`/v10 场景缺少该字段时仍按
旧的实体障碍物语义读取，不覆盖历史结果。v11 的联合安全执行仍对所有实体 UAV 和
动态障碍物使用一步预测与扫掠检查；同时开启实体动态障碍物运动特征，使策略能够利用
物理障碍物的运动趋势，但不会把通信遮挡区域当成可碰撞对象。

整体通信—定位联合优化写为受约束的策略优化问题：

$$
\begin{aligned}
\min_{\pi_{\boldsymbol{\theta}}}\quad
J(\pi_{\boldsymbol{\theta}})
&=
\mathbb{E}_{\pi_{\boldsymbol{\theta}}}
\left[
\sum_{t=0}^{T-1}\gamma_{\mathrm{disc}}^tJ_t
\right] \\
\text{s.t.}\quad
&\mathbf{x}_{i,t+1}
=
f\!\left(\mathbf{x}_{i,t},\mathbf{u}_{i,t},\Delta t\right)
+
\boldsymbol{\epsilon}^{\mathrm{proc}}_{i,t},
\qquad i\in\mathcal{U},\quad t=0,\ldots,T-1,\\
&\|\mathbf{v}_{i,t}\|_2\leq v_{\max},\qquad
\|\Delta\mathbf{v}_{i,t}\|_2\leq a_{\max}\Delta t,
\qquad i\in\mathcal{U},\quad t=0,\ldots,T-1,\\
&0\leq a^{\mathrm{bandwidth}}_{ij,t}\leq 1,\qquad
\sum_{j:(i,j)\in\mathcal E_t^{\mathrm{pot}}}
a^{\mathrm{bandwidth}}_{ij,t}=1,
\quad\text{if }\{j:(i,j)\in\mathcal E_t^{\mathrm{pot}}\}\neq\varnothing,\\
&a^{\mathrm{bandwidth}}_{ij,t}=0\quad\text{for all }j,
\quad\text{if }\{j:(i,j)\in\mathcal E_t^{\mathrm{pot}}\}=\varnothing,\\
&\hspace{6em} i\in\mathcal U,\quad t=0,\ldots,T-1,\\
&\|\mathbf p_{i,t}-\mathbf p_{j,t}\|_2\geq d_{\mathrm{safe}},
\qquad i\neq j,\\
&\mathbf p_{i,t}\notin\bigcup_m\mathcal B_{m,t},
\qquad i\in\mathcal U.
\\
&\sum_{\tau=0}^{t-1}
\mathbb{I}\left[a^{\mathrm{meas}}_{i,\tau}\neq\mathrm{none}\right]
\leq K^{\mathrm{meas}}_{\nu,i},
\qquad i\in\mathcal U,\quad\text{(仅预算约束分支)},
\end{aligned}
$$

其中最后一项仅在预算约束分支启用时生效；无预算的 baseline 和原 quality-aware 分支不受该项限制。

其中，测量对象必须满足第 2.12 节的动作掩码，不能选择候选集合之外的节点；当前不设置中继动作。碰撞安全在系统模型中由最小机间距和障碍物避让约束表达，训练阶段再通过安全投影和 $\lambda_{\mathrm{coll}}E_{\mathrm{coll},t}$ 惩罚帮助策略满足约束。能耗只通过 $\lambda_{\mathrm{energy}}E_{\mathrm{energy},t}$ 进入强化学习奖励，不改变核心目标 $J_t$；轨迹平滑度仍作为辅助评价指标。第 3 章只说明归一化量的实现、即时奖励计算和 MAPPO 求解流程，不重新定义联合目标。

## 2.14 模型与数据文件映射

| 模型对象 | 主要字段或文件 | 用途 |
| --- | --- | --- |
| $\mathbf{x}_{i,t}$ 与真实运动状态 | `ground_truth.csv` | 仿真真值和定位误差评估 |
| $\mathbf{f}^{\mathrm{imu}}_{i,t}$、$\boldsymbol{\omega}^{\mathrm{imu}}_{i,t}$ | `imu.csv` | IMU 预测和 EKF 更新 |
| $z^{\mathrm{uwb}}_{ij,t}$ | `uwb_ranges.csv` | 协同测距和 UWB 更新 |
| $z^{\mathrm{vo}}_{i,t}$ | `visual_odometry.csv` | 视觉里程计更新 |
| $z^{\mathrm{gnss}}_{i,t}$ | `gnss.csv` | GNSS 正常、退化和失效实验 |
| $G_t$、$\mathcal{E}^{\mathrm{cand}}_t$、$\mathrm{RSSI}_{ij,t}$、$\mathrm{SNR}_{ij,t}$、$b_{ij,t}$、$R_{ij,t}$、$P^{\mathrm{rx}}_{ij,t}$、$Q_{ij,t}$、$D^{\mathrm{raw}}_{ij,t}$ | `links.csv` | 候选链路、有效通信图、信道中间量和通信质量建模 |
| $\delta^{\mathrm{rx}}_{ij,t}$、丢包与时延事件 | `packet_events.csv` | 数据包到达、链路可靠性和 AoI 计算 |
| $\mathbf{a}_{i,t}$ | `actions.csv` | MAPPO 动作记录 |
| $r_t$ 和目标分量 | `rewards.csv` | 奖励复现和消融分析 |
| $\mathcal{O}_t$、$\mathbf{o}_{m,t}$ | `obstacles.csv` | 遮挡、碰撞和泛化测试 |
| $\tau_i$ | `trajectory.csv` | 轨迹、任务成功率和能耗评估 |
| $\hat{\mathbf{x}}_{i,t}$、$\boldsymbol{\Sigma}_{i,t}$ | `estimates.csv` | Distributed EKF 在线估计和协方差记录 |
| $\mathbf{h}^{(0)}_{i,t}$、$\mathbf e_{ij,t}$ | `graph_features.csv` | GAT 节点、边特征和动作掩码复现 |

其中，`links.csv` 只记录进入无线通信拓扑计算的潜在出边；超出无线通信范围但仍有物理 UWB 测量的无人机对只记录在 `uwb_ranges.csv`。`links.csv` 中的 `delay` 是截断后的观测时延，候选边判定同时使用模型内部的未截断 $D^{\mathrm{raw}}_{ij,t}$。`graph_features.csv` 保存本机局部图、潜在出边特征、带宽掩码以及单独标记的 `central_state` 快照；中心 critic 可使用该快照，actor 只能使用局部字段。`trajectory.csv` 是由真值、动作和链路日志派生的兼容性汇总文件，不替代原始状态、估计和链路文件。

在 `links.csv` 中，`channel_bandwidth_hz` 对应物理带宽 $b_{ij,t}$，`rate_bps` 对应有效速率 $R_{ij,t}$，`rx_power_dbm` 对应 $P^{\mathrm{rx}}_{ij,t}$，`queue_length` 对应 $Q_{ij,t}$，`raw_delay` 对应未截断的 $D^{\mathrm{raw}}_{ij,t}$；`potential_edge`、`candidate_edge`、`effective_edge` 和 `uwb_available` 分别对应潜在边、候选边、成功通信边和 UWB 物理测量可用性。保留的 `bandwidth_kbps` 仅作为旧 schema 的兼容别名，定义为 `rate_bps / 1000`，不再作为论文中的物理带宽字段。`actions.csv` 保存逐边带宽 JSON 和动作掩码，`estimates.csv` 保存 EKF 输出，`graph_features.csv` 保存 GAT 输入特征。

所有正式实验数据必须由 Python 生成器写入项目统一 schema；由模型推导的字段必须在 `scenario.json` 和 manifest 中标记为模型生成，不能伪造为实测观测。

## 2.15 符号表（Notation Registry）

本表是本课题数学符号的唯一登记源。后续在论文、代码注释、实验配置和图表中出现的新符号，必须先在本表中查重；确认未占用后再登记和使用。符号比较同时考虑：大小写、粗体/花体、上下标、字体和语义。例如 `P_loss` 与 `P_i` 虽然字面相似，也不得在同一公式体系中不加限定地混用。

### Table I. Notation

| Group | Symbol | Meaning | Status / first use |
| --- | --- | --- | --- |
| System / UAV set | $\mathcal{U}$ | 无人机集合 | 已登记；系统模型 |
| System / UAV set | $N_u$ | 无人机数量，$N_u=\lvert\mathcal{U}\rvert$ | 已登记；系统模型 |
| System / indices | $i,j$ | 无人机索引；通信边 $(i,j)$ 默认表示 $i\to j$，GAT 局部公式另明确 $j$ 为发送节点、$i$ 为接收节点 | 已登记；全局约定；2026-08-09 |
| System / entity | $\mathrm{UAV}_{i}$ | 第 $i$ 架无人机的实体标识 | 已登记；参与实体；2026-07-31 |
| System / time | $t$ | 离散决策时刻或统一时间索引 | 已登记；全局约定 |
| System / time | $\Delta t_{\mathrm{dec}}$ | MAPPO 决策时间间隔 | 已登记；实验协议 |
| System / horizon | $T$ | 单个 episode 的决策时域长度 | 已登记；联合优化问题；2026-07-31 |
| System / remaining time | $\tau^{\mathrm{remain}}_t$ | 固定任务时域内归一化剩余时间，取值为 max(0, 1-t/T) | 有限时域目标对齐；2026-09-06 |
| System / trajectory | $\tau_i$ | 无人机 $i$ 的完整轨迹序列 | 已登记；轨迹建模 |
| System / input tuple | $\mathcal{I}_t$ | 时刻 $t$ 的系统输入元组 | 已登记；系统输入输出定义；2026-07-27 |
| System / input component | $\mathcal{I}^{\mathrm{init}}$ | 无人机初始状态和初始编队配置 | 已登记；系统输入输出定义；2026-07-27 |
| System / input component | $\mathcal{I}^{\mathrm{env}}_t$ | 时刻 $t$ 的环境和任务配置 | 已登记；系统输入输出定义；2026-07-27 |
| System / input component | $\mathcal{I}^{\mathrm{sens}}_t$ | 时刻 $t$ 的传感器配置和观测条件 | 已登记；系统输入输出定义；2026-07-27 |
| System / input component | $\mathcal{I}^{\mathrm{com}}_t$ | 时刻 $t$ 的通信配置和链路条件 | 已登记；系统输入输出定义；2026-07-27 |
| System / input component | $\mathcal{I}^{\mathrm{exp}}$ | 实验随机种子、场景布局和复现实验配置 | 已登记；系统输入输出定义；2026-07-27 |
| System / target | $\mathbf{p}^{\mathrm{goal}}_i$ | 无人机 $i$ 的固定目标位置 | 已登记；系统输入输出定义；2026-07-27 |
| System / target trajectory | $\tau^{\mathrm{goal}}_i$ | 无人机 $i$ 的目标轨迹序列 | 已登记；系统输入输出定义；2026-07-27 |
| System / target position | $\mathbf{p}^{\mathrm{goal}}_{i,t}$ | 无人机 $i$ 在时刻 $t$ 的目标位置 | 已登记；系统输入输出定义；2026-07-27 |
| System / output tuple | $\mathcal{Y}_t$ | 时刻 $t$ 的系统输出元组 | 已登记；系统输入输出定义；2026-07-27 |
| System / output component | $\mathcal{Y}^{\mathrm{est}}_t$ | 时刻 $t$ 的定位估计输出 | 已登记；系统输入输出定义；2026-07-27 |
| System / output component | $\mathcal{Y}^{\mathrm{graph}}_t$ | 时刻 $t$ 的通信图和图表示输出 | 已登记；系统输入输出定义；2026-07-27 |
| System / output component | $\mathcal{Y}^{\mathrm{act}}_t$ | 时刻 $t$ 的 MAPPO 动作和执行结果 | 已登记；系统输入输出定义；2026-07-27 |
| System / output component | $\mathcal{Y}^{\mathrm{eval}}_t$ | 时刻 $t$ 的定位、通信和安全评价指标 | 已登记；系统输入输出定义；2026-07-27 |
| System / mapping | $\mathcal{F}_{\boldsymbol{\theta}}$ | 由各物理模型、滤波器、GAT 和 MAPPO 组成的参数化系统映射 | 已登记；系统输入输出定义；2026-07-27 |
| System / evaluation vector | $\mathbf{m}_t$ | 时刻 $t$ 的定位、通信、能耗和安全评价指标向量 | 已登记；系统输入输出定义；2026-07-27 |
| State / position | $\mathbf{p}_{i,t}$ | 无人机 $i$ 在时刻 $t$ 的三维位置 | 已登记；状态变量 |
| State / velocity | $\mathbf{v}_{i,t}$ | 无人机 $i$ 的三维速度 | 已登记；状态变量 |
| State / attitude | $\boldsymbol{\varphi}_{i,t}$ | 无人机 $i$ 的姿态角向量（roll、pitch、yaw） | 已登记；状态变量 |
| State / true state | $\mathbf{x}_{i,t}$ | 无人机 $i$ 的真实状态向量 | 已登记；EKF |
| State / estimate | $\hat{\mathbf{x}}_{i,t}$ | 无人机 $i$ 的估计状态向量 | 已登记；EKF |
| State / estimated position | $\hat{\mathbf{p}}_{i,t}$ | 无人机 $i$ 在时刻 $t$ 的估计位置 | 已登记；系统输入输出定义；2026-07-27 |
| State / covariance | $\boldsymbol{\Sigma}_{i,t}$ | 无人机 $i$ 的状态估计协方差；对应相关 EKF 文献常用的 $\mathbf{P}_{i,t}$ | 已登记；EKF；项目正文统一使用 Sigma |
| State / position covariance | $\boldsymbol{\Sigma}^{p}_{i,t}$ | 无人机 $i$ 位置估计的协方差子矩阵 | 已登记；系统输入输出定义；2026-07-27 |
| State / control | $\mathbf{u}_{i,t}$ | 无人机 $i$ 的低层控制输入或动力学控制量 | 已登记；系统动力学 |
| State / bias | $\mathbf{b}^{a}_{i,t}$ | 无人机 $i$ 的加速度计零偏 | 已登记；IMU 模型 |
| State / bias | $\mathbf{b}^{\omega}_{i,t}$ | 无人机 $i$ 的陀螺仪零偏 | 已登记；IMU 模型 |
| State / process noise | $\boldsymbol{\epsilon}^{\mathrm{proc}}_{i,t}$ | 无人机 $i$ 的过程噪声 | 已登记；状态转移 |
| State / measurement noise | $\boldsymbol{\epsilon}^{\mathrm{meas}}_{i,t}$ | 传感器观测噪声 | 已登记；观测模型 |
| Observation / scalar measurement noise | $\epsilon^{\mathrm{meas}}_{ij,t}$ | UWB 或链路标量观测的测量噪声 | 已登记；UWB 和通信模型；2026-07-31 |
| Mathematical / point | $\mathbf q$ | 障碍物包围盒中的任意三维点 | 已登记；障碍物模型；2026-07-31 |
| Mathematical / component index | $d$ | 三维位置或几何尺寸的分量索引，$d\in\{1,2,3\}$ | 已登记；障碍物模型；2026-07-31 |
| Mathematical / indicator | $\mathbb{I}[\cdot]$ | 条件成立时取 1、否则取 0 的指示函数 | 已登记；碰撞和通信图模型；2026-07-31 |
| Observation / set | $\mathcal{Z}_{i,t}$ | 无人机 $i$ 在时刻 $t$ 的多源观测集合 | 已登记；观测模型 |
| Environment / obstacle set | $\mathcal{O}_t$ | 时刻 $t$ 的有效障碍物集合 | 已登记；环境模型 |
| Environment / obstacle | $o_m$ | 第 $m$ 个障碍物 | 已登记；障碍物模型 |
| Environment / obstacle count | $N_o(t)$ | 时刻 $t$ 被环境模型纳入计算的有效障碍物数量，$N_o(t)=\lvert\mathcal{O}_t\rvert$ | 已登记；障碍物集合定义；2026-07-27 |
| Environment / obstacle state | $\mathbf{o}_{m,t}$ | 障碍物 $o_m$ 在时刻 $t$ 的位置、速度和几何状态 | 已登记；障碍物模型 |
| Environment / obstacle position | $\mathbf{p}^{o}_{m,t}$ | 障碍物 $o_m$ 在时刻 $t$ 的三维位置 | 已登记；动态障碍物模型；2026-07-27 |
| Environment / obstacle velocity | $\mathbf{v}^{o}_{m,t}$ | 障碍物 $o_m$ 在时刻 $t$ 的三维速度 | 已登记；动态障碍物模型；2026-07-27 |
| Environment / obstacle geometry | $\mathbf{s}^{o}_{m}$ | 障碍物 $o_m$ 的长、宽和高几何尺寸 | 已登记；动态障碍物模型；2026-07-27 |
| Environment / obstacle envelope | $\mathcal{B}_{m,t}$ | 障碍物 $o_m$ 在时刻 $t$ 的轴对齐包围盒 | 已登记；动态障碍物模型；2026-07-27 |
| Environment / obstacle noise | $\boldsymbol{\epsilon}^{o}_{m,t}$ | 动态障碍物速度状态转移噪声 | 已登记；动态障碍物模型；2026-07-27 |
| Environment / obstacle covariance | $\mathbf{Q}^{o}_{m}$ | 动态障碍物速度状态转移噪声协方差 | 已登记；动态障碍物模型；2026-07-27 |
| Environment / obstacle speed limit | $v^{o}_{\max}$ | 动态障碍物速度上限 | 已登记；动态障碍物模型；2026-07-27 |
| Environment / obstacle distance | $d^{o}_{im,t}$ | 无人机 $i$ 与障碍物 $o_m$ 包围盒之间的最小距离 | 已登记；动态障碍物模型；2026-07-27 |
| Observation / dynamic clearance | $\hat d^{\mathrm{dyn}}_{i,t}$ | 无人机 $i$ 基于执行期可见聚合运动信息得到的动态障碍物最小间隙估计 | 已登记；动态障碍物局部观测；2026-08-28 |
| Observation / dynamic closing rate | $\hat c^{\mathrm{dyn}}_{i,t}$ | 动态障碍物相对无人机的最大非负接近速率估计；正值表示间隙正在缩小 | 已登记；动态障碍物局部观测；2026-08-28 |
| Observation / predicted dynamic clearance | $\hat d^{\mathrm{dyn}}_{i,t+1\mid t}$ | 使用当前估计运动状态预测至下一决策窗口的动态障碍物最小间隙 | 已登记；动态障碍物局部观测；2026-08-28 |
| Observation / dynamic motion validity | $\mu^{\mathrm{dyn}}_{i,t}$ | 动态障碍物运动聚合观测是否有效的二值标记；不表示安全或碰撞 | 已登记；动态障碍物局部观测；2026-08-28 |
| Graph / communication graph | $G_t=(\mathcal{V}_t,\mathcal{E}_t)$ | 时刻 $t$ 的无人机通信图 | 已登记；GAT |
| Graph / potential graph | $G^{\mathrm{pot}}_t=(\mathcal{U},\mathcal{E}^{\mathrm{pot}}_t)$ | 时刻 $t$ 由几何通信范围决定的潜在图，仅供环境和资源分配接口使用 | 已登记；通信拓扑；2026-08-02 |
| Graph / candidate graph | $G^{\mathrm{cand}}_t=(\mathcal{U},\mathcal{E}^{\mathrm{cand}}_t)$ | 时刻 $t$ 满足带宽和链路阈值的候选图，用于动作掩码和资源选择 | 已登记；通信拓扑；2026-08-02 |
| Graph / node set | $\mathcal{V}_t$ | 时刻 $t$ 仿真中处于 active 状态的无人机节点集合；当前基线为 $\mathcal{V}_t=\mathcal{U}$ | 已登记；GAT |
| Graph / candidate edge set | $\mathcal{E}^{\mathrm{cand}}_t$ | 时刻 $t$ 满足距离、带宽和链路质量阈值的候选有向边集合 | 已登记；通信拓扑；2026-07-31 |
| Graph / edge set | $\mathcal{E}_t$ | 时刻 $t$ 数据包成功到达后可实际交换有效信息的有向边集合 | 已登记；GAT；2026-07-31 |
| Graph / adjacency | $\mathbf{A}_t$ | 时刻 $t$ 的邻接矩阵 | 已登记；GAT |
| Graph / neighborhood | $\mathcal{N}_{i,t}$ | GAT 中时刻 $t$ 无人机 $i$ 的有效入邻居集合；$j\in\mathcal{N}_{i,t}$ 当且仅当 $(j,i)\in\mathcal{E}_t$ | 已登记；GAT；2026-08-03 |
| Graph / distance | $d_{ij,t}$ | 时刻 $t$ 无人机 $i$ 与 $j$ 的相对距离 | 已登记；通信模型 |
| Graph / link state | $\ell_{ij,t}$ | 链路 $i\!\to\!j$ 的 LOS 指示量；$\ell=1$ 表示 LOS，$\ell=0$ 表示 NLOS；链路是否可用另由候选/有效边表示 | 已登记；通信模型；2026-08-02 |
| Sensor / IMU | $\mathbf{f}^{\mathrm{imu}}_{i,t}$ | IMU 加速度计测得的比力向量 | 已登记；传感器模型 |
| Sensor / IMU | $\boldsymbol{\omega}^{\mathrm{imu}}_{i,t}$ | IMU 陀螺仪测得的角速度向量 | 已登记；传感器模型 |
| Sensor / UWB | $z^{\mathrm{uwb}}_{ij,t}$ | UWB 对无人机对 $i,j$ 的测距观测 | 已登记；传感器模型 |
| Sensor / vision | $z^{\mathrm{vo}}_{i,t}$ | 无人机 $i$ 的视觉里程计观测 | 已登记；传感器模型 |
| Sensor / visual displacement | $\Delta\mathbf p^{\mathrm{vo}}_{i,t}$ | 无人机 $i$ 在当前决策窗口内的视觉里程计局部位移观测 | 已登记；传感器模型；2026-08-09 |
| Sensor / visual reference | $\mathbf p^{\mathrm{ref}}_{i,t}$ | 历史伪位置VO模型中的预测前估计位置；当前使用相关相对位移更新 | 历史兼容，不用于当前观测公式；2026-09-03修订；原首次使用2026-08-09 |
| Sensor / GNSS | $z^{\mathrm{gnss}}_{i,t}$ | 无人机 $i$ 的 GNSS 位置观测 | 已登记；传感器模型 |
| Sensor / configuration | $f^{\mathrm{sens}}_i$ | 无人机 $i$ 的传感器更新频率 | 已登记；系统输入输出定义；2026-07-27 |
| Sensor / availability | $\mu^{\mathrm{sens}}_{i,t}$ | 无人机 $i$ 的传感器可用或失效模式 | 已登记；系统输入输出定义；2026-07-27 |
| Sensor / GNSS mode | $\mu^{\mathrm{gnss}}_{i,t}$ | 无人机 $i$ 在时刻 $t$ 的 GNSS 状态 | 已登记；系统输入输出定义；2026-07-27 |
| Sensor / GNSS Jacobian | $\mathbf H^{\mathrm{gnss}}_{i,t}$ | GNSS 位置观测对 15 维滤波状态的雅可比矩阵 | 已登记；EKF 方法；2026-08-09 |
| Sensor / visual Jacobian | $\mathbf H^{\mathrm{vo}}_{i,t}$ | 视觉位置增量观测对滤波状态的雅可比矩阵 | 已登记；EKF 方法；2026-08-09 |
| Sensor / GNSS covariance | $\mathbf R^{\mathrm{gnss}}_{i,t}$ | GNSS 位置观测噪声协方差 | 已登记；EKF 方法；2026-08-09 |
| Sensor / visual covariance | $\mathbf R^{\mathrm{vo}}_{i,t}$ | 视觉位置增量观测噪声协方差 | 已登记；EKF 方法；2026-08-09 |
| Sensor / visual interval | $\Delta t_{\mathrm{vo}}$ | 视觉里程计位置增量对应的时间间隔 | 已登记；EKF 方法；2026-08-09 |
| EKF / process model | $f(\cdot)$ | 状态转移或运动模型 | 已登记；EKF |
| EKF / measurement model | $h(\cdot)$ | 观测模型 | 已登记；EKF |
| EKF / process covariance | $\mathbf{Q}_{i,t}$ | 过程噪声协方差 | 已登记；EKF |
| EKF / measurement covariance | $\mathbf{R}_{i,t}$ | 观测噪声协方差 | 已登记；EKF |
| EKF / Kalman gain | $\mathbf{K}_{i,t}$ | 无人机 $i$ 在时刻 $t$ 的 Kalman 增益 | 已登记；EKF |
| EKF / identity matrix | $\mathbf I$ | 与状态维度一致的单位矩阵 | 已登记；EKF 协方差更新；2026-07-31 |
| EKF / linearization | $\mathbf{F}_{i,t}$ | 状态转移模型 $f(\cdot)$ 在时刻 $t$ 的线性化矩阵 | 已登记；Distributed EKF |
| EKF / linearization | $\mathbf{H}_{i,t}$ | 观测模型 $h(\cdot)$ 在时刻 $t$ 的线性化矩阵 | 已登记；Distributed EKF |
| EKF / delayed control sequence | $\mathbf u_{j,t:t'}$ | 从发送时刻到消息处理时刻的邻居控制输入序列 | 已登记；延迟消息传播；2026-08-09 |
| EKF / delayed transition | $\mathbf F_{j,t:t'}$ | 延迟邻居状态从发送时刻传播到处理时刻的状态转移雅可比 | 已登记；延迟消息传播；2026-08-09 |
| EKF / delayed process covariance | $\mathbf Q_{j,t:t'}$ | 延迟邻居状态传播期间累计的过程噪声协方差 | 已登记；延迟消息传播；2026-08-09 |
| EKF / innovation gate | $\chi^2_{1,1-\alpha_{\mathrm{gate}}},\alpha_{\mathrm{gate}}$ | 标量 UWB 创新门限及其显著性参数 | 已登记；EKF 异常观测拒绝；2026-08-09 |
| EKF / predicted position | $\hat{\mathbf p}^{-}_{i,t}$ | 无人机 $i$ 在当前观测更新前的预测位置 | 已登记；Distributed EKF 方法；2026-08-09 |
| EKF / arrived neighbor position | $\hat{\mathbf p}^{\mathrm{arr}}_{j,t}$ | 消息到达后传播到当前时刻的邻居 $j$ 位置估计 | 已登记；延迟消息传播；2026-08-09 |
| EKF / predicted UWB range | $\hat z^{\mathrm{uwb}}_{ij,t}$ | 根据本机预测位置和已到达邻居估计得到的 UWB 预测距离 | 已登记；Distributed EKF 方法；2026-08-09 |
| EKF / position NEES | $\mathrm{NEES}^{p}_{i,t}$ | 离线位置估计一致性统计量 | 已登记；EKF 评价；2026-08-09 |
| EKF / relative NIS | $\mathrm{NIS}^{\mathrm{rel}}_{ij,t}$ | 相对测距创新一致性统计量 | 已登记；EKF 评价；2026-08-09 |
| Observation / fused measurement | $\mathbf{z}_{i,t}$ | 进入 EKF 更新步骤的本机或邻居融合观测向量 | 已登记；Distributed EKF |
| Observation / modality | $m$ | 传感器观测类型索引，$m\in\{\mathrm{vo},\mathrm{gnss}\}$ | 已登记；观测模型 |
| State / specific force | $\mathbf{f}_{i,t}$ | 世界坐标系中的真实比力向量 | 已登记；IMU 模型 |
| State / angular velocity | $\boldsymbol{\omega}_{i,t}$ | 无人机 $i$ 的真实角速度向量 | 已登记；IMU 模型 |
| State / NLOS bias | $b^{\mathrm{nlos}}_{ij,t}$ | 链路 $i\!\to\!j$ 在 NLOS 条件下的 UWB 测距偏差 | 已登记；UWB 模型 |
| System / time | $\Delta t$ | 仿真或状态传播使用的基础时间步长 | 已登记；运动模型 |
| Environment / safety | $d_{\mathrm{safe}}$ | 无人机与障碍物或其他无人机之间的最小安全距离 | 已登记；环境约束 |
| Communication / range | $d_{\max}$ | 通信链路的最大有效距离 | 已登记；拓扑模型 |
| Communication / threshold | $\mathrm{RSSI}_{\min}$ | 候选链路允许的最小接收信号强度 | 已登记；通信拓扑；2026-07-31 |
| Communication / threshold | $\mathrm{SNR}_{\min}$ | 候选链路允许的最小信噪比 | 已登记；通信拓扑；2026-07-31 |
| Communication / threshold | $P_{\mathrm{loss},\max}$ | 候选链路允许的最大丢包率 | 已登记；通信拓扑；2026-07-31 |
| Communication / threshold | $D_{\mathrm{com},\max}$ | 候选链路允许的最大通信时延 | 已登记；通信拓扑；2026-07-31 |
| Communication / reference | $d_0$ | 对数距离信道模型的参考距离 | 已登记；信道模型；2026-08-02 |
| Communication / carrier | $f_c$ | 通信载波频率 | 已登记；信道模型；2026-08-02 |
| Communication / propagation | $c_{\mathrm{prop}}$ | 无线信号传播速度 | 已登记；信道模型；2026-08-02 |
| Communication / path loss | $\mathrm{PL}_0$ | 参考距离 $d_0$ 处的自由空间路径损耗 | 已登记；信道模型；2026-08-02 |
| Communication / path loss | $n_{\mathrm{LOS}},n_{\mathrm{NLOS}}$ | LOS 和 NLOS 条件下的路径损耗指数 | 已登记；信道模型；2026-08-02 |
| Communication / path loss | $n_{\ell_{ij,t}}$ | 根据链路 LOS/NLOS 状态选择的路径损耗指数 | 已登记；信道模型；2026-08-02 |
| Communication / path loss | $L^{\mathrm{path}}_{ij,t}$ | 链路 $i\!\to\!j$ 的对数距离路径损耗 | 已登记；信道模型；2026-08-02 |
| Communication / path loss | $X^{\mathrm{sh}}_{ij,t}$ | 链路 $i\!\to\!j$ 的阴影衰落项 | 已登记；信道模型；2026-08-02 |
| Communication / path loss | $L^{\mathrm{obs}}_{ij,t}$ | 障碍物遮挡造成的附加链路损耗 | 已登记；信道模型；2026-08-02 |
| Communication / path loss | $L_{ij,t}$ | 链路 $i\!\to\!j$ 的总传播损耗 | 已登记；信道模型；2026-08-02 |
| Communication / signal | $\mathrm{RSSI}_{ij,t}$ | 链路 $i\!\to\!j$ 的接收信号强度 | 已登记；通信模型 |
| Communication / signal | $\mathrm{SNR}_{ij,t}$ | 链路 $i\!\to\!j$ 的信噪比 | 已登记；通信模型 |
| Communication / signal | $P^{\mathrm{rx}}_{ij,t}$ | 链路 $i\!\to\!j$ 的接收功率 | 已登记；信道模型；2026-08-02 |
| Communication / noise | $N_{ij,t}$ | 链路 $i\!\to\!j$ 在分配带宽下的接收噪声功率 | 已登记；信道模型；2026-08-02 |
| Communication / noise | $N_0$ | 接收机噪声功率谱密度 | 已登记；信道模型；2026-08-02 |
| Communication / noise | $F_{\mathrm{nf}}$ | 接收机噪声系数 | 已登记；信道模型；2026-08-02 |
| Communication / bandwidth | $b_{ij,t}$ | 链路 $i\!\to\!j$ 的可用带宽 | 已登记；通信模型 |
| Communication / bandwidth | $B_i^{\mathrm{tot}}$ | 无人机 $i$ 的总可用发送带宽 | 已登记；信道模型；2026-08-02 |
| Communication / transmit power | $P^{\mathrm{tx}}_{ij,t}$ | 链路 $i\!\to\!j$ 在时刻 $t$ 的发射功率 | 已登记；系统输入输出定义；2026-07-27 |
| Communication / rate | $R_{ij,t}$ | 链路 $i\!\to\!j$ 的有效数据速率 | 已登记；信道模型；2026-08-02 |
| Communication / efficiency | $\eta_{\mathrm{phy}}$ | 有效速率相对于 Shannon 型容量的物理层效率系数 | 已登记；信道模型；2026-08-02 |
| Communication / packet | $L_{\mathrm{pkt}}$ | 单个数据包的比特数 | 已登记；信道模型；2026-08-02 |
| Communication / queue | $Q_{ij,t}$ | 链路 $i\!\to\!j$ 在时刻 $t$ 的排队数据包数量 | 已登记；信道模型；2026-08-02 |
| Communication / link statistics | $\boldsymbol{\chi}_{ij,t}$ | 链路 $i\!\to\!j$ 在时刻 $t$ 的通信统计量向量 | 已登记；系统输入输出定义；2026-07-27 |
| Communication / scenario condition | $\boldsymbol{\xi}_t$ | 时刻 $t$ 的通信—定位场景条件 | 已登记；系统输入输出定义；2026-07-27 |
| Communication / packet loss | $P_{\mathrm{loss},ij,t}$ | 链路 $i\!\to\!j$ 在时刻 $t$ 的单链路丢包率 | 已登记；链路模型 |
| Communication / packet loss | $P_{\mathrm{loss},t}$ | 时刻 $t$ 的系统丢包率 | 已登记；目标函数 |
| Communication / packet reception | $\delta^{\mathrm{rx}}_{ij,t}$ | 链路 $i\!\to\!j$ 的数据包是否在有效时间窗口内成功到达 | 已登记；通信拓扑；2026-07-31 |
| Communication / delay | $D_{\mathrm{com},ij,t}$ | 链路 $i\!\to\!j$ 在时刻 $t$ 的单链路通信时延 | 已登记；链路模型 |
| Communication / raw delay | $D^{\mathrm{raw}}_{ij,t}$ | 未截断的物理通信时延，用于候选边时延阈值判定 | 已登记；信道模型；2026-08-02 |
| Communication / delay | $D_{\mathrm{com},t}$ | 时刻 $t$ 的通信时延 | 已登记；目标函数 |
| Communication / link interval | $\Delta t_{\mathrm{link}}$ | 链路状态和 AoI 的更新周期 | 已登记；信道模型；2026-08-02 |
| Communication / AoI | $\mathrm{AoI}_{ij,t}$ | 链路 $i\!\to\!j$ 在时刻 $t$ 的单链路信息年龄 | 已登记；链路模型 |
| Communication / AoI cap | $\mathrm{AoI}_{\max}$ | 信息年龄记录和状态更新的最大截断值 | 已登记；信道模型；2026-08-02 |
| Communication / timestamp | $t^{\mathrm{last}}_{ij}$ | 链路 $i\!\to\!j$ 最近一次成功传输的时间戳 | 已登记；AoI 模型 |
| Communication / message buffer | $\mathcal B^{\mathrm{msg}}_t$ | 时刻 $t$ 尚未处理或已经到达的邻居消息缓冲 | 已登记；延迟消息模型；2026-08-09 |
| Communication / buffered message | $\mathfrak b_{j\rightarrow i,t}$ | 消息发送时状态、协方差、时间戳和预计到达窗口 | 已登记；延迟消息模型；2026-08-09 |
| Communication / arrival window | $k^{\mathrm{arr}}_{ji,t}$ | 消息 $j\!\to\!i$ 的离散预计到达窗口 | 已登记；延迟消息模型；2026-08-09 |
| Communication / send timestamp | $\tau^{\mathrm{send}}_{j,t}$ | 邻居消息发送时的时间戳 | 已登记；延迟消息模型；2026-08-09 |
| Communication / information age | $\mathrm{AoI}_t$ | 时刻 $t$ 的信息年龄 | 已登记；目标函数 |
| Communication / connectivity | $C_{\mathrm{net},t}$ | 时刻 $t$ 的网络连通性指标 | 已登记；目标函数 |
| Communication / connected component | $\mathcal C_t^{\max}$ | 有效通信图无向化后的最大连通分量 | 已登记；联合目标；2026-08-09 |
| Communication / loss parameter | $\beta_0,\beta_1,\beta_2,\beta_3$ | Logistic 丢包模型的可校准参数 | 已登记；信道模型；2026-08-02 |
| Constraint / speed | $v_{\max}$ | 无人机速度上限 | 已登记；运动约束 |
| Constraint / acceleration | $a_{\max}$ | 无人机加速度上限 | 已登记；运动约束 |
| MAPPO / MDP | $\mathcal{M}$ | 多智能体决策过程 | 已登记；MAPPO |
| MAPPO / state space | $\mathcal{S}$ | 联合状态空间 | 已登记；MAPPO |
| MAPPO / action space | $\mathcal{A}$ | 联合动作空间 | 已登记；MAPPO |
| MAPPO / transition | $\mathcal{T}$ | 状态转移算子 | 已登记；MAPPO；避免使用 $\mathcal{P}$ 与丢包率混淆 |
| MAPPO / reward | $\mathcal{R}$ | 奖励函数 | 已登记；MAPPO |
| MAPPO / environment state | $s^{\mathrm{env}}_t$ | 时刻 $t$ 的完整环境马尔可夫状态，包含运动、通信、消息和障碍物内部状态 | 已登记；多智能体决策过程；2026-08-09 |
| MAPPO / centralized state | $s^{\mathrm c}_t$ | 时刻 $t$ 提供给 centralized critic 的中心状态摘要 | 已登记；多智能体决策过程；2026-08-09 |
| MAPPO / legacy joint state | $s_t$ | 旧版未区分环境状态与中心状态的联合状态别名 | 已弃用；由 $s^{\mathrm{env}}_t$ 和 $s^{\mathrm c}_t$ 替代；2026-08-09 |
| MAPPO / action | $\mathbf{a}_{i,t}$ | 无人机 $i$ 在时刻 $t$ 的动作 | 已登记；MAPPO |
| MAPPO / action component | $a^{\mathrm{meas}}_{i,t},a^{\mathrm{bandwidth}}_{ij,t},\Delta\mathbf{v}_{i,t}$ | 协同测量选择、逐边带宽分配和速度增量动作 | 已登记；MAPPO 动作定义；2026-08-09 |
| MAPPO / joint action | $\mathbf{a}_t$ | 时刻 $t$ 的联合动作集合 | 已登记；MAPPO 更新；2026-07-31 |
| MAPPO / team reward | $r_t$ | 时刻 $t$ 的共享团队奖励 | 已登记；MAPPO；2026-08-09 |
| MAPPO / team reward copy | $r_{i,t}$ | 无人机 $i$ 接收的共享团队奖励副本，满足 $r_{i,t}=r_t$ | 已登记；MAPPO 奖励；2026-08-09 |
| MAPPO / policy | $\pi_{\boldsymbol{\theta}}$ | 参数为 $\boldsymbol{\theta}$ 的策略 | 已登记；MAPPO |
| GAT / node embedding | $\mathbf{h}^{(0)}_{i,t}$ | GAT 的无人机 $i$ 初始节点特征 | 已登记；GAT |
| GAT / goal feature | $\mathbf{q}^{\mathrm{goal}}_{i,t}$ | 无人机 $i$ 在时刻 $t$ 的目标方向或目标状态特征 | 已登记；GAT 方法；2026-07-31 |
| GAT / obstacle feature | $\mathbf{q}^{\mathrm{obs}}_{i,t}$ | 无人机 $i$ 在时刻 $t$ 的局部障碍物特征 | 已登记；GAT 方法；2026-07-31 |
| GAT / relative estimate | $\Delta\hat{\mathbf{p}}_{ji,t}$ | 接收节点 $i$ 对发送节点 $j$ 的相对位置估计（发送节点减接收节点） | 已登记；GAT 方法；2026-08-09 |
| GAT / edge feature | $\mathbf{e}_{ij,t}$ | 链路 $i\!\to\!j$ 在时刻 $t$ 的 GAT 边特征向量 | 已登记；GAT 方法；2026-07-31 |
| GAT / hidden node feature | $\mathbf{h}^{(l)}_{i,t}$ | GAT 第 $l$ 层的无人机 $i$ 节点表示 | 已登记；GAT 方法；2026-07-31 |
| GAT / graph embedding | $\mathbf{g}_{i,t}$ | 无人机 $i$ 在时刻 $t$ 由 GAT 输出的图表示，供 actor 或 centralized critic 使用 | 已登记；GAT 输出接口；2026-08-03 |
| GAT / attention coefficient | $\alpha^{(l)}_{ij,t}$ | GAT 第 $l$ 层对邻居 $j$ 的注意力系数 | 已登记；GAT 方法；2026-07-31 |
| GAT / attention vector | $\boldsymbol{\eta}^{(l)}$ | GAT 第 $l$ 层的注意力参数向量 | 已登记；GAT 方法；2026-07-31 |
| GAT / node transform | $\mathbf{W}^{(l)}$ | GAT 第 $l$ 层的节点特征变换矩阵 | 已登记；GAT 方法；2026-07-31 |
| GAT / edge transform | $\mathbf{W}^{(l)}_e$ | GAT 第 $l$ 层的边特征变换矩阵 | 已登记；GAT 方法；2026-07-31 |
| GAT / activation | $\sigma$ | GAT 节点更新使用的非线性激活函数 | 已登记；GAT 方法；2026-07-31 |
| MAPPO / discount | $\gamma_{\mathrm{disc}}$ | 回报折扣因子 | 已登记；MAPPO；不得单独写作 $\gamma$ |
| MAPPO / probability ratio | $\rho_{i,t}(\boldsymbol{\theta})$ | 共享 actor 在局部观测下对无人机 $i$ 动作的新旧策略概率比 | 已登记；MAPPO 更新；2026-08-09 |
| MAPPO / old policy parameter | $\boldsymbol{\theta}_{\mathrm{old}}$ | 更新前的策略参数 | 已登记；MAPPO 更新；2026-07-31 |
| MAPPO / centralized critic | $V_{\boldsymbol{\phi}}(s^{\mathrm c}_t),\boldsymbol{\phi}$ | 唯一 centralized critic 的团队价值及其参数 | 已登记；MAPPO 更新；2026-08-09 |
| MAPPO / terminal indicator | $d_t$ | 时刻 $t$ 的 transition 是否终止的指示量 | 已登记；MAPPO/GAE；2026-08-09 |
| MAPPO / TD residual | $\delta_t$ | 基于共享团队奖励和中心价值的 TD 残差 | 已登记；MAPPO/GAE；2026-08-09 |
| MAPPO / GAE parameter | $\lambda_{\mathrm{GAE}}$ | GAE 优势估计的时间平滑参数 | 已登记；MAPPO/GAE；2026-08-09 |
| MAPPO / shared advantage | $\hat A_t$ | 基于共享团队奖励的时刻 $t$ 团队优势估计 | 已登记；MAPPO 更新；2026-08-09 |
| MAPPO / team return target | $\hat R_t$ | centralized critic 使用的共享团队回报目标 | 已登记；MAPPO/GAE；2026-08-09 |
| MAPPO / clip range | $\epsilon_{\mathrm{clip}}$ | PPO 概率比裁剪范围 | 已登记；MAPPO 更新；2026-07-31 |
| MAPPO / actor objective | $L^{\mathrm{clip}}$ | PPO clipped surrogate actor 目标 | 已登记；MAPPO 更新；2026-07-31 |
| Objective / localization | $E_{\mathrm{loc}}$ | 定位误差指标 | 已登记；联合目标 |
| Objective / localization uncertainty | $E_{\mathrm{uncert},t}$ | 时刻 $t$ 的平均位置估计不确定度 | 已登记；系统输入输出定义；2026-07-27 |
| Objective / localization cost | $J_{\mathrm{loc},t}$ | 时刻 $t$ 的定位代价 | 已登记；联合目标；2026-08-09 |
| Objective / communication cost | $J_{\mathrm{com},t}$ | 时刻 $t$ 的通信代价 | 已登记；联合目标；2026-08-09 |
| Objective / objective | $J$ | 通信—定位联合优化目标 | 已登记；联合目标 |
| Objective / instantaneous objective | $J_t$ | 时刻 $t$ 的定位—通信核心联合代价，不含碰撞和能耗奖励惩罚 | 已登记；联合目标；2026-08-09 |
| Objective / weights | $w_{\mathrm{unc}},w_{\mathrm{loss}},w_{\mathrm{AoI}},w_{\mathrm{delay}},w_{\mathrm{net}}$ | 定位不确定度、丢包、AoI、时延和连通性权重 | 已登记；联合目标；2026-07-31 |
| Objective / normalized metrics | $\bar E_{\mathrm{uncert},t},\bar P_{\mathrm{loss},t},\overline{\mathrm{AoI}}_t,\bar D_{\mathrm{com},t},\bar C_{\mathrm{net},t}$ | 使用训练集95%分位尺度归一化后的即时评价指标 | 已登记；联合目标；2026-08-09 |
| Objective / training scale | $q^{95}_{x,\mathrm{train}}$ | 指标 $x$ 在训练集上的95%分位归一化尺度 | 已登记；联合目标；2026-08-09 |
| Objective / penalty | $E_{\mathrm{coll},t}$ | 时刻 $t$ 的碰撞事件或碰撞惩罚项 | 已登记；碰撞约束和 MAPPO 奖励；2026-07-31 |
| Objective / collision multiplier | $\lambda_{\mathrm{coll}}$ | 碰撞事件的奖励惩罚系数 | 已登记；MAPPO 奖励实现；2026-07-31 |
| Objective / legacy energy proxy | $E_{\mathrm{energy},i,t}$ | 旧版按无人机分别计算的能耗代理 | 已弃用；由团队能耗 $E_{\mathrm{energy},t}$ 替代；2026-08-09 |
| Objective / team energy proxy | $E_{\mathrm{energy},t}$ | 时刻 $t$ 所有无人机安全投影后速度增量的归一化平均值 | 已登记；MAPPO 奖励；2026-08-09 |
| Objective / energy multiplier | $\lambda_{\mathrm{energy}}$ | 速度增量能耗代理的奖励惩罚系数 | 已登记；MAPPO 奖励；2026-08-02 |
| Dataset / episode | $e$ | episode 索引 | 已登记；数据 schema |
| Dataset / dataset | $\mathcal{D}_{\mathrm{py}}$ | 纯 Python 环境生成的数据集集合 | 已登记；数据管理；2026-08-09 |
| Dataset / split | $\mathcal{D}_{\mathrm{train}},\mathcal{D}_{\mathrm{val}},\mathcal{D}_{\mathrm{test}}$ | 训练、验证和测试数据划分 | 已登记；实验协议 |
| Graph / potential edge set | $\mathcal{E}^{\mathrm{pot}}_t$ | 时刻 $t$ 满足几何通信范围、可用于资源分配的潜在有向边集合 | 已登记；通信带宽分配；2026-08-02 |
| Graph / UWB edge set | $\mathcal{E}^{\mathrm{uwb}}_t$ | 时刻 $t$ 物理 UWB 测量可用的有向边集合 | 已登记；传感器观测；2026-08-02 |
| Communication / edge action | $a^{\mathrm{bandwidth}}_{ij,t}$ | 无人机 $i$ 对潜在出边 $i\!\to\!j$ 的带宽分配比例 | 已登记；通信带宽分配；2026-08-02 |
| Communication / packet arrival | $A_{ij,t}$ | 当前通信窗口链路 $i\!\to\!j$ 新到达的数据包数量 | 已登记；队列状态转移；2026-08-02 |
| Communication / packet service | $S_{ij,t}$ | 当前通信窗口链路 $i\!\to\!j$ 可服务的数据包数量 | 已登记；队列状态转移；2026-08-02 |
| Communication / estimated distance | $\hat d_{ij,t}$ | 执行期由估计状态或 UWB 观测得到的相对距离 | 已登记；GAT 边特征；2026-08-02 |
| EKF / relative covariance | $\mathbf R^{\mathrm{rel}}_{ij,t}$ | 邻居状态不确定度并入后的相对测距观测协方差 | 已登记；Distributed EKF；2026-08-02 |
| EKF / neighbor message | $\mathfrak m_{j\rightarrow i,t}$ | 无人机 $j$ 向无人机 $i$ 成功传输的状态、协方差和时间戳消息 | 已登记；Distributed EKF；2026-08-02 |
| EKF / message timestamp | $\tau^{\mathrm{msg}}_{j,t}$ | 邻居 $j$ 的状态消息生成或发送时间戳 | 已登记；Distributed EKF；2026-08-02 |
| EKF / stale-message threshold | $\Delta t_{\mathrm{stale}}^{\max}$ | 邻居状态消息允许进入协同定位更新的最大信息年龄 | 已登记；协同定位信息条件；2026-08-09 |
| EKF / relative innovation | $\tilde z_{ij,t}$ | UWB 实际测距与本机—邻居预测距离之差 | 已登记；Distributed EKF 方法；2026-08-09 |
| EKF / relative Jacobian | $\mathbf H^{\mathrm{rel}}_{ij,t}$ | 本机状态对相对测距观测函数的线性化雅可比矩阵 | 已登记；Distributed EKF；2026-08-02 |
| EKF / UWB covariance | $R^{\mathrm{uwb}}_{ij,t}$ | UWB 原始相对测距观测噪声方差 | 已登记；Distributed EKF；2026-08-02 |
| EKF / position Jacobian | $\mathbf H^{p}_{j,t}$ | 邻居位置对相对测距观测的线性化矩阵 | 已登记；Distributed EKF；2026-08-02 |
| EKF / CI position covariance | $\boldsymbol{\Sigma}^{p,\mathrm{CI}}_{j,t}$ | 协方差交叉融合后的保守邻居位置协方差 | 已登记；Distributed EKF；2026-08-02 |
| EKF / CI position estimate | $\hat{\mathbf p}^{\mathrm{CI}}_{j,t}$ | 协方差交叉融合后的保守邻居位置均值 | 已登记；Distributed EKF 方法；2026-08-09 |
| EKF / CI source covariance | $\boldsymbol{\Sigma}^{p,(r)}_{j,t}$ | CI 第 $r$ 个输入源提供的邻居位置协方差，$r\in\{1,2\}$ | 已登记；Distributed EKF；2026-08-02 |
| EKF / CI source estimate | $\hat{\mathbf p}^{(r)}_{j,t}$ | CI 第 $r$ 个输入源提供的邻居位置估计，$r\in\{1,2\}$ | 已登记；Distributed EKF；2026-08-02 |
| EKF / CI weight | $\omega_{\mathrm{CI}}$ | 协方差交叉融合的权重 | 已登记；Distributed EKF；2026-08-02 |
| MAPPO / action mask | $\mathbf M^{\mathrm{act}}_{i,t}$ | 无人机 $i$ 在时刻 $t$ 离散动作的可行性掩码 | 已登记；MAPPO 交互接口；2026-08-02 |
| MAPPO / local observation | $\mathbf o_{i,t}$ | 无人机 $i$ 执行期可获得的局部观测结构 | 已登记；MAPPO 交互接口；2026-08-02 |
| MAPPO / seed index | $\nu$ | 随机种子或对应独立实验实例的索引 | 已登记；固定测量预算定义；2026-08-26 |
| MAPPO / measurement budget | $K^{\mathrm{meas}}_{\nu,i}$ | 随机种子 $\nu$ 下无人机 $i$ 由 baseline 训练集测量次数拟合得到的固定测量预算 | 已登记；固定测量预算定义；2026-08-26 |
| MAPPO / remaining measurement budget | $B^{\mathrm{rem}}_{\nu,i,t}$ | 随机种子 $\nu$ 下无人机 $i$ 在时刻 $t$ 尚可使用的测量预算 | 已登记；固定测量预算定义；2026-08-26 |
| MAPPO / baseline measurement count | $M^{\mathrm{baseline}}_{\nu,i,e}$ | 随机种子 $\nu$ 的 baseline 训练 episode $e$ 中无人机 $i$ 的非 `none` 测量次数 | 已登记；固定测量预算定义；2026-08-26 |
| MAPPO / decision-window count | $W_{\nu,i,e}$ | 随机种子 $\nu$ 的训练 episode $e$ 中无人机 $i$ 实际产生动作的决策窗口数 | 已登记；训练期测量率约束；2026-08-28 |
| MAPPO / target measurement rate | $\kappa^{\mathrm{target}}_{\nu,i}$ | 由 baseline 训练集非 `none` 测量次数除以决策窗口数得到的无人机测量率目标 | 已登记；训练期测量率约束；2026-08-28 |
| MAPPO / expected measurement rate | $\bar\kappa_{\nu,i}$ | 当前训练 minibatch 中无人机 $i$ 的期望非 `none` 测量率 | 已登记；训练期测量率约束；2026-08-28 |
| MAPPO / active UAV set | $\mathcal I$ | 当前 minibatch 中实际出现样本的无人机索引集合 | 已登记；训练期测量率约束；2026-08-28 |
| MAPPO / rate constraint loss | $L^{\mathrm{rate}}$ | 训练期使期望测量率接近 baseline 目标区间的附加损失项 | 已登记；训练期测量率约束；2026-08-28 |
| MAPPO / rate constraint multiplier | $\lambda_{\mathrm{rate}}$ | 测量率约束损失的固定权重系数 | 已登记；训练期测量率约束；2026-08-28 |
| MAPPO / rate tolerance | $\tau_{\mathrm{rate}}$ | 测量率目标允许的固定容差 | 已登记；训练期测量率约束；2026-08-28 |
| MAPPO / adaptive rate violation | $v^{\mathrm{rate}}_{\nu,i}$ | 随机种子 $\nu$、无人机 $i$ 的期望测量率超出目标容差区间的线性偏差 | 已登记；自适应测量率约束；2026-08-29 |
| MAPPO / adaptive rate loss | $L^{\mathrm{rate,dual}}$ | 使用每架无人机自适应乘子加权的训练期测量率约束损失 | 已登记；自适应测量率约束；2026-08-29 |
| MAPPO / adaptive rate multiplier | $\lambda^{\mathrm{dual}}_{\nu,i}$ | 随机种子 $\nu$、无人机 $i$ 的自适应测量率约束乘子 | 已登记；自适应测量率约束；2026-08-29 |
| MAPPO / adaptive multiplier step | $\eta^{\mathrm{dual}}_{\mathrm{rate}}$ | 自适应测量率乘子的训练后更新步长 | 已登记；自适应测量率约束；2026-08-29 |
| MAPPO / adaptive multiplier bound | $\lambda^{\mathrm{dual}}_{\max}$ | 自适应测量率约束乘子的上界 | 已登记；自适应测量率约束；2026-08-29 |
| MAPPO / proxy normalized information gain | $\widehat G^{\mathrm{IG}}_{ij,t}$ | 由执行期可见估计状态、协方差、测量噪声和链路质量得到的归一化信息增益代理 | 已登记；代理教师测量选择；2026-08-29 |
| MAPPO / proxy normalized communication cost | $\widehat C^{\mathrm{com}}_{ij,t}$ | 候选测量产生的预测归一化通信代价代理，仅使用执行期可见链路字段 | 已登记；代理教师测量选择；2026-08-29 |
| MAPPO / proxy utility | $U^{\mathrm{proxy}}_{ij,t}$ | 信息增益代理减去加权预测通信代价后的测量效用；`none` 效用为零 | 已登记；代理教师测量选择；2026-08-29 |
| MAPPO / target-relative proxy utility | $U^{\mathrm{proxy,rel}}_{ij,t}$ | 通信约束分支中，信息增益代理减去超过 baseline 训练目标部分的加权通信代价；`none` 效用为零 | 已登记；通信约束代理 teacher；2026-09-01 |
| MAPPO / proxy communication-cost weight | $w^{\mathrm{proxy}}_{\mathrm{com}}$ | 信息增益—通信代价代理中的通信代价权重 | 已登记；代理教师测量选择；2026-08-29 |
| MAPPO / proxy teacher coefficient | $\alpha_{\mathrm{proxy}}$ | measurement head 辅助 teacher KL 损失的系数 | 已登记；代理教师测量选择；2026-08-29 |
| MAPPO / proxy teacher temperature | $T^{\mathrm{proxy}}$ | 将掩码效用转换为 teacher 分布的 softmax 温度 | 已登记；代理教师测量选择；2026-08-29 |
| MAPPO / proxy teacher loss | $L^{\mathrm{proxy}}$ | 执行期代理 teacher 分布与 actor measurement 分布之间的 KL 辅助损失 | 已登记；代理教师测量选择；2026-08-29 |
| MAPPO / measurement communication cost | $\widehat c^{\mathrm{meas}}_{ij,t}$ | 候选测量的执行期归一化通信代价，包含固定 512 字节负载与可见链路风险；`none` 为零 | 已登记；通信代价约束；2026-08-30 |
| MAPPO / communication-cost target | $c^{\mathrm{target}}_{\nu,i}$ | 仅由同一随机种子 baseline train 拟合的每 UAV 平均测量通信代价目标 | 已登记；通信代价约束；2026-08-30 |
| MAPPO / expected communication cost | $\bar c_{\nu,i}$ | 当前训练 minibatch 中策略测量分布的期望通信代价 | 已登记；通信代价约束；2026-08-30 |
| MAPPO / communication-cost tolerance | $\tau_{\mathrm{com}}$ | baseline-train 通信代价目标允许的单侧容差 | 已登记；通信代价约束；2026-08-30 |
| MAPPO / communication-cost violation | $v^{\mathrm{com}}_{\nu,i}$ | 期望通信代价超过目标与容差之和的非负违反量 | 已登记；通信代价约束；2026-08-30 |
| MAPPO / communication dual loss | $L^{\mathrm{com,dual}}$ | 使用每 UAV 通信乘子加权的单侧通信代价约束损失 | 已登记；通信代价约束；2026-08-30 |
| MAPPO / communication multiplier | $\lambda^{\mathrm{com}}_{\nu,i}$ | 随机种子 $\nu$、无人机 $i$ 的自适应通信代价约束乘子 | 已登记；通信代价约束；2026-08-30 |
| MAPPO / communication dual step | $\eta^{\mathrm{com}}$ | 自适应通信代价乘子的训练后更新步长 | 已登记；通信代价约束；2026-08-30 |
| MAPPO / communication dual bound | $\lambda^{\mathrm{com}}_{\max}$ | 自适应通信代价约束乘子的上界 | 已登记；通信代价约束；2026-08-30 |
| MAPPO / persistent proxy horizon | $H_{\mathrm{persist}}$ | 持久信息收益代理向前传播的固定预测步数；当前取 $3$ | 已登记；持久信息收益代理；2026-08-31 |
| MAPPO / current proxy covariance | $\boldsymbol{\Sigma}^{\mathrm{cur}}_{i,t}$ | 持久代理使用的无人机 $i$ 当前执行期可见状态估计协方差 | 已登记；持久信息收益代理；2026-08-31 |
| MAPPO / linear proxy covariance | $\boldsymbol{\Sigma}^{\mathrm{lin}}_{i,t}(j)$ | 假设采用邻居 $j$ 的标量相对测距后得到的线性协方差，不写回实际 EKF | 已登记；持久信息收益代理；2026-08-31 |
| MAPPO / no-update propagated covariance | $\widetilde{\boldsymbol{\Sigma}}^{(0)}_{i,t+H_{\mathrm{persist}}}$ | 不采用候选测量时按固定预测视界传播的假设协方差 | 已登记；持久信息收益代理；2026-08-31 |
| MAPPO / linear-update propagated covariance | $\widetilde{\boldsymbol{\Sigma}}^{(1)}_{i,t+H_{\mathrm{persist}}}$ | 采用候选测量线性更新后按固定预测视界传播的假设协方差 | 已登记；持久信息收益代理；2026-08-31 |
| MAPPO / persistent normalized gain | $\widehat G^{\mathrm{persist}}_{ij,t}$ | 固定预测视界后位置协方差迹减少量的训练集尺度归一化代理 | 已登记；持久信息收益代理；2026-08-31 |
| MAPPO / persistent proxy utility | $U^{\mathrm{persist}}_{ij,t}$ | 持久信息收益代理减去加权预测通信代价后的测量效用；`none` 效用为零 | 已登记；持久信息收益代理；2026-08-31 |
| MAPPO / sampled action | $\mathbf a^{\mathrm{samp}}_{i,t}$ | 策略在时刻 $t$ 按当前动作分布采样、并用于计算 PPO 旧对数概率的混合动作 | 已登记；core-framework-v3 PPO action semantics；2026-09-04 |
| MAPPO / executed action | $\mathbf a^{\mathrm{exec}}_{i,t}$ | 经过环境有效性检查、安全投影、裁剪或预算约束后实际执行的混合动作 | 已登记；core-framework-v3 PPO action semantics；2026-09-04 |
| MAPPO / candidate measurement score | $s_{ij,t}$ | 接收 UAV $i$ 对候选邻居 $j$ 的逐候选测量动作得分；由本机 GAT 表示、邻居可见表示、入边特征、可行性标记和预算特征共同计算 | 已登记；Local-EKF-GAT-MAPPO action-conditioned measurement head；2026-09-04 |
| MAPPO / candidate mask vector | $\mathbf m_{ji,t}$ | 候选边 $j\to i$ 的可见性、有效性、动作可行性、节点可见性和剩余预算等执行期标记向量；不包含真实误差或未来结果 | 已登记；Local-EKF-GAT-MAPPO action-conditioned measurement head；2026-09-04 |
| MAPPO / candidate action descriptor | $\boldsymbol{\psi}_{ij,t}$ | v9 逐候选动作头使用的 actor 可见特征向量；包含当前估计协方差置信度、测距方差代理、链路质量、AoI、队列和可行性标记；不包含真值、未来 NIS 或未来通信结果 | 已登记；core-framework-v9 action-conditioned measurement head；2026-09-07 |
| MAPPO / candidate quality summary | $q^{\mathrm{cand}}_{ij,t}$ | 由 $\boldsymbol{\psi}_{ij,t}$ 的当前可见质量分量形成的有界候选质量摘要，仅用于 `none` 上下文诊断和动作头输入 | 已登记；core-framework-v9 action-selection diagnostic；2026-09-07 |
| MAPPO / none quality baseline | $q^{\mathrm{none}}_{i,t}$ | 接收 UAV $i$ 在未执行邻居测量时的训练反馈基准；由 warm-up 的 accepted-update feedback 统计冻结，不改变报告奖励 | 已登记；core-framework-v9 accepted-update feedback；2026-09-07 |
| MAPPO / accepted-update feedback location | $\mu_q$ | warm-up 中 accepted-update 候选质量相对 `none` 基准差分的稳健位置统计量 | 已登记；core-framework-v9 accepted-update feedback；2026-09-07 |
| MAPPO / accepted-update feedback scale | $s_q$ | warm-up 中 accepted-update 候选质量相对 `none` 基准差分的稳健归一化尺度 | 已登记；core-framework-v9 accepted-update feedback；2026-09-07 |
| MAPPO / action-linked accepted feedback | $r^{\mathrm{act}}_{i,t+\delta}$ | 仅对当前实际选择且被接受的邻居测量释放的有界延迟训练反馈；与 `none` 基准差分并按 warm-up 的 $\mu_q,s_q$ 归一化 | 已登记；core-framework-v9 accepted-update feedback；2026-09-07 |
| MAPPO / action-linked feedback coefficient | $\alpha_{\mathrm{act}}$ | accepted-update 反馈进入 PPO 学习通道的固定系数；不进入报告奖励 | 已登记；core-framework-v9 accepted-update feedback；2026-09-07 |
| MAPPO / uncertainty progress statistic | $u_t$ | 时刻 $t$ 由各 UAV 位置协方差对数行列式平均得到的训练进度统计量；与控制向量 $\mathbf u_{i,t}$ 区分 | 已登记；core-framework-v2 training-only progress potential；2026-09-04 |
| MAPPO / learning potential | $\Phi_t$ | 由归一化不确定度统计量和通信目标 $J_{\mathrm{com},t}$ 构造的有界训练势函数；只用于 PPO 学习通道 | 已登记；core-framework-v2 training-only progress potential；2026-09-04 |
| MAPPO / learning reward | $R_t^{\mathrm{learn}}$ | 在不改变环境报告奖励的前提下，由基础有界奖励和势差 shaping 构成的 PPO 训练奖励 | 已登记；core-framework-v2 training-only progress potential；2026-09-04 |
| MAPPO / uncertainty progress increment | $\Delta u_t$ | 相邻决策窗口位置协方差对数行列式的归一化下降量；正值表示不确定度下降 | 已登记；core-framework-v4 training-only feedback；2026-09-05 |
| MAPPO / raw within-episode uncertainty decrease | $\Delta u^{\mathrm{raw}}_t$ | 同一 episode 内相邻决策窗口位置协方差对数行列式的原始下降量 $u_{t-1}-u_t$；不跨 episode 求差分；避免与 GAE 的终止指示 $d_t$ 重名 | 已登记；core-framework-v4-delta-scale training-only feedback；2026-09-06 |
| MAPPO / robust uncertainty-difference scale | $s_{\Delta u}$ | 仅由 warm-up 的同一 episode 相邻 $d_t$ 按 $\max(1.4826\,\mathrm{MAD}(d_t),10^{-3})$ 拟合的稳健归一化尺度 | 已登记；core-framework-v4-delta-scale training-only feedback；2026-09-06 |
| MAPPO / clipped normalized uncertainty decrease | $\Delta\tilde u_t$ | $d_t/s_{\Delta u}$ 经 $[-3,3]$ 裁剪后的训练反馈；正值表示相邻窗口不确定度下降 | 已登记；core-framework-v4-delta-scale training-only feedback；2026-09-06 |
| Communication / measurement payload | $B_t^{\mathrm{meas}}$ | 时刻 $t$ 实际执行的非 `none` 协同测量所对应的 512 bytes 等效负载总量 | 已登记；core-framework-v4 training-only feedback；2026-09-05 |
| Communication / normalized measurement load | $b_t$ | $B_t^{\mathrm{meas}}/(512N_u)$，按 UAV 数归一化的当前窗口实际测量负载 | 已登记；core-framework-v4 training-only feedback；2026-09-05 |
| Communication / load excess | $c_t^{\mathrm{load}}$ | 当前归一化实际测量负载超过 baseline-train 固定目标 $b^\ast$ 的非负超额 | 已登记；core-framework-v4 training-only feedback；2026-09-05 |
| Motion / velocity projection | $\operatorname{clip}_{v_{\max}}(\cdot)$ | 将速度向量投影到最大速度约束内的裁剪算子 | 已登记；三维运动模型；2026-08-02 |
| Motion / lower flight bound | $\mathbf p_{\min}$ | 三维飞行区域的位置下界 | 已登记；EKF 后验有界投影；2026-08-09 |
| Motion / upper flight bound | $\mathbf p_{\max}$ | 三维飞行区域的位置上界 | 已登记；EKF 后验有界投影；2026-08-09 |
| Motion / position projection | $\operatorname{clip}_{[\mathbf p_{\min},\mathbf p_{\max}]}(\cdot)$ | 将 EKF 位置估计投影到三维飞行区域内的裁剪算子 | 已登记；EKF 后验有界投影；2026-08-09 |
| Motion / executed increment | $\Delta\mathbf v^{\mathrm{exec}}_{i,t}$ | 无人机 $i$ 经安全投影后在时刻 $t$ 实际执行的速度增量 | 已登记；强化学习奖励；2026-08-09 |

### 2.15.1 已弃用或仅允许作为历史别名的符号

| Legacy symbol | Canonical replacement | Reason |
| --- | --- | --- |
| $\mathbf{P}_{i,t}$ | $\boldsymbol{\Sigma}_{i,t}$ | 相关 EKF 文献中表示状态估计协方差的常用符号；项目正文统一使用 Sigma，避免与 $P_{\mathrm{loss},ij,t}$ 混用 |
| $N$ | $N_u$ | 明确表示无人机数量，避免与样本数、时间步数混淆 |
| $k$（时间下标） | $t$ | 统一离散时间索引，避免与 Kalman 增益 $\mathbf{K}$ 和测量索引混淆 |
| $\alpha,\beta,\gamma,\delta,\eta$（联合目标权重） | $w_{\mathrm{unc}},w_{\mathrm{loss}},w_{\mathrm{AoI}},w_{\mathrm{delay}},w_{\mathrm{net}}$ | 避免与 GAT 注意力系数和 MAPPO 折扣因子冲突 |
| $G_t=(V_t,E_t)$ | $G_t=(\mathcal{V}_t,\mathcal{E}_t)$ | 用花体明确集合，避免与误差指标和普通变量混淆 |
| $g(\cdot)$ | $L^{\mathrm{path}}_{ij,t},L^{\mathrm{obs}}_{ij,t}$ | 旧版未展开的 RSSI 黑箱映射；现改为显式信道方程 |
| $\rho_t(\boldsymbol{\theta})$ | $\rho_{i,t}(\boldsymbol{\theta})$ | 当前共享 actor 按无人机局部观测逐样本计算概率比，不以中心状态直接产生动作 |
| $\hat A_{i,t}$ | $\hat A_t$ | 旧版分别计算各无人机局部优势；当前规范由共享团队奖励和单一中心价值生成共同优势 |
| $a^{\mathrm{relay}}_{i,t}$ | 无 | 当前版本不设置独立中继动作；未来建立多跳路由和转发队列后再重新登记 |

### 2.15.2 符号查重和更新流程

1. 在新增公式、算法、代码字段或图表前，先在本表搜索符号的大小写、字体、上下标和同义含义。
2. 如果符号已登记且含义相同，直接复用原符号，不新建同义符号。
3. 如果符号已登记但含义不同，必须换用未登记符号；不得通过改变字体或大小写掩盖语义冲突。
4. 新符号登记必须同时填写 `Group`、`Symbol`、`Meaning`、`Status / first use` 四列，并在更新记录中写明日期和使用位置。
5. 删除或替换符号时，不得直接抹除旧记录；应移入“已弃用符号”表，并同步修改所有章节、代码注释和实验配置。
6. 每次论文章节或算法实现变更后，重新检查 `sections/`、`knowledge/`、`src/` 中的数学符号和字段名，确保没有重复含义。

7. 为兼容常用 Markdown 预览器，行内数学统一使用美元符号包围，独立公式统一使用双美元符号包围；不再使用反斜杠圆括号或反斜杠方括号作为正文数学分隔符。

### 2.15.3 更新记录

| Date | Version | Change |
| --- | --- | --- |
| 2026-07-25 | 1.0 | 建立系统、传感器、EKF、通信、MAPPO、目标函数和数据集符号表；登记弃用别名和查重流程 |
| 2026-07-26 | 2.0 | 补充坐标与时间同步、运动状态、传感器观测、Distributed EKF、动态图、通信链路、GAT、MAPPO 约束及模型—数据映射 |
| 2026-07-31 | 2.1 | 对齐文献符号：明确 $\boldsymbol{\Sigma}_{i,t}$ 与 EKF 文献 $\mathbf{P}_{i,t}$ 的协方差对应关系，并将通信质量统一放入 GAT 边特征 |
| 2026-07-31 | 2.2 | 统一 Markdown 数学分隔符，修复部分预览器无法显示的行内符号和独立公式 |
| 2026-07-31 | 2.3 | 补充 EKF 后验协方差、候选/有效通信边、链路质量阈值、数据包到达状态和受约束联合策略优化问题 |
| 2026-08-02 | 2.4 | 建立显式统计信道模型，补充路径损耗、接收功率、噪声、SNR、速率、丢包、排队时延和 AoI 方程，并登记信道参数符号 |
| 2026-08-02 | 2.5 | 将主环境校准为纯 Python 三维交互环境，区分输入/内部信道状态/局部观测/中心状态，补充潜在—候选—有效边、独立 UWB 边、逐边带宽、动作时序和 CI 邻居融合接口 |
| 2026-08-09 | 2.6 | 重构第 2.10—2.13 节职责：第 2 章保留估计信息、图状态、决策过程和联合优化定义，第 3 章统一承载 EKF、GAT 与 MAPPO 求解细节；同步校准局部策略概率比和优势符号 |
| 2026-08-26 | 2.7 | 补充仅由 baseline 训练划分确定的固定测量预算、剩余预算、预算动作掩码和预算约束分支，明确其不改变无预算分支与核心通信—定位目标 |
| 2026-08-28 | 2.8 | 登记动态障碍物局部运动观测的当前间隙、接近速率、一步预测间隙和有效性标记；明确安全 Smoke 的 41 维节点输入与历史 37 维兼容边界 |
| 2026-08-31 | 2.9 | 登记固定三步持久信息收益代理、当前/线性假设协方差传播和持久代理效用；明确其仅使用执行期可见估计量，并与固定 512 bytes 通信代价分开建模 |
| 2026-09-04 | 3.0 | 登记逐候选测量动作得分、候选执行期标记、训练专用不确定度统计量、学习势函数和学习奖励；明确 core-framework-v2 不改变报告奖励或物理目标 |
| 2026-09-04 | 3.1 | 登记采样动作与环境执行动作的区分；明确 core-framework-v3 只训练 Proposed Method、规则基线只作性能参照，并修正 PPO 旧对数概率的采样语义 |
| 2026-09-06 | 3.2 | 补充 core-framework-v8 的报告奖励直接训练、有限时域终止语义和 report_plus_feedback 定位进度臂；复用 $u_t$ 与 $s_{\Delta u}$，不新增同义符号 |
| 2026-09-07 | 3.3 | 登记 v9 候选级 actor 特征、`none` 质量上下文和 accepted-update 相对基准反馈；明确其只影响动作学习诊断/训练通道，不改变报告奖励、EKF 或通信物理模型 |
