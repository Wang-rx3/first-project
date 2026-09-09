# 3 研究方法

本章说明第 2 章系统模型的具体实现方法。第 2 章定义研究对象、状态、观测、通信图、优化变量和约束；本章说明数据如何进入系统、Distributed EKF 如何执行、动态图如何构造、GAT 如何编码，以及 MAPPO 如何根据联合代价进行训练和决策。

## 3.1 数据预处理、时间同步与坐标统一

**开发诊断边界（2026-09-04）。** seed=42 的训练趋势审计使用冻结的初始/最终策略和固定开发监测场景，区分回合长度造成的累计回报混杂与固定场景表现变化。梯度探针仅在预定 train 场景重新收集轨迹、计算原损失和 GAE，不执行优化器更新。探针 minibatch 之间不独立，不能作为独立实验重复，也不能复原未保存的中间训练过程。比较全局与假设 actor/critic 分组裁剪的范数只用于预注册筛选，不能视为 Adam 实际参数更新，更不能直接证明训练性能退化的原因；筛选失败时不运行裁剪对照。现行联合裁剪、奖励、滤波及教师机制保持不变。本段不引入新的论文数学符号。

### 3.1.1 统一处理流程

所有正式实验数据均由纯 Python 环境生成，并按照以下顺序处理：

~~~text
场景配置和随机种子初始化
    → 生成器版本和参数登记
    → 时间戳统一
    → 坐标系转换
    → 传感器质量标记
    → 通信事件和观测有效性判断
    → 统一 schema 输出
     → Python 多无人机环境 / Distributed EKF / GAT / MAPPO 接口
~~~

处理过程不依赖外部原始数据。Python 生成的 episode 保存到 `data/synthetic/`，生成配置和摘要保存在每个 episode 的 `scenario.json` 以及根目录 manifest 中。

### 3.1.2 时间和坐标处理

每个 episode 使用相对时间戳。当前纯 Python `episode-v1.2` 基线以决策频率推进，$\Delta t=\Delta t_{\mathrm{dec}}$，一次 `env.step()` 内完成运动、传感器、链路和 EKF 更新；`physics_hz`、传感器频率和 `link_hz` 只作为场景元数据记录，不代表当前已经运行独立的物理子步。所有观测先转换到 ENU 世界坐标系，再按照时间戳匹配到最近的决策窗口。若观测时间与当前窗口的差值超过容许阈值，则将该观测标记为缺失，不进行无依据插值。

PX4 或其他飞控接口使用 NED 坐标系时，在数据接口层完成 NED 到 ENU 的转换。转换后的数据在后续 EKF、图构造和评价过程中只能使用 ENU 坐标，避免同一实验混用两种坐标约定。

### 3.1.3 观测有效性和质量标记

每条观测在进入 EKF 前检查以下条件：

1. 时间戳存在且位于当前 episode；
2. 传感器在当前时刻处于可用状态；
3. 观测值满足字段、维度和数值范围检查；
4. 若观测来自邻居，则对应通信边在该时刻成功传输；
5. 观测没有超过 `stale_message_limit_s` 时间容许窗口；
6. NLOS、异常值和传感器退化状态被写入质量标记。

无效观测不被伪造为零值，而是以缺失标记传递给后续模块。

### 3.1.4 数据与模型接口

| 数据对象 | 主要来源 | 进入的模块 |
| --- | --- | --- |
| ground_truth.csv | Python 仿真环境 | 运动状态生成、离线误差评价 |
| imu.csv | Python 传感器模型 | EKF 预测 |
| uwb_ranges.csv | Python UWB 模型 | 协同测距和 EKF 更新 |
| visual_odometry.csv | Python 视觉里程计模型 | 视觉观测更新 |
| gnss.csv | Python GNSS 模型 | GNSS 正常、退化和失效实验 |
| links.csv | Python 统计信道模型 | 动态通信图和 GAT 边特征 |
| packet_events.csv | 通信模型 | 丢包、时延和 AoI 更新 |
| obstacles.csv | Python 场景生成器 | 碰撞和三维 LOS/NLOS 判定 |
| actions.csv | MAPPO 执行器 | 动作复现和轨迹评价 |
| rewards.csv | 奖励计算器 | 训练监控和消融分析 |

## 3.2 Distributed EKF 协同定位实现

### 3.2.1 输入和输出

每架无人机在决策时刻接收本机有效传感器观测和当前通信边成功传输的邻居信息。Distributed EKF 的输入包括：

- 上一时刻估计状态和协方差；
- 本机 IMU 观测；
- 本机视觉或 GNSS 观测；
- 当前有效的 UWB 相对测距；
- 邻居消息中的状态、测量和时间戳；
- 对应观测的噪声协方差和质量标记。

输出为：

$$
\hat{\mathbf{x}}_{i,t},
\qquad
\boldsymbol{\Sigma}_{i,t}.
$$

本文的 $\boldsymbol{\Sigma}_{i,t}$ 与相关 EKF 文献中常用的 $\mathbf{P}_{i,t}$ 表示同一类状态估计协方差；为保持与第 2 章符号表和实现代码一致，方法流程中统一使用 $\boldsymbol{\Sigma}_{i,t}$。

其中，真实位置只用于仿真环境和离线评价，不进入执行期 EKF。

### 3.2.1.1 传感器观测函数、雅可比与协方差

当前 15 维滤波状态的前三维为位置，接着三维为速度。各传感器采用与物理含义一致的观测函数；观测协方差由传感器配置或校准文件提供，并且必须为有限、对称半正定矩阵。

GNSS 正常或退化时提供位置观测：

$$
\mathbf z^{\mathrm{gnss}}_{i,t}
=
\mathbf p_{i,t}
+
\boldsymbol\epsilon^{\mathrm{meas}}_{i,t},
\qquad
\mathbf H^{\mathrm{gnss}}_{i,t}
=
\left[\mathbf I_3\ \ \mathbf 0_{3\times 12}\right],
\qquad
\mathbf R^{\mathrm{gnss}}_{i,t}\succcurlyeq 0.
$$

视觉里程计在当前纯 Python 基线中作为局部位置增量观测，而不是把视觉真值位置直接送入滤波器。传感器先生成当前决策窗口的位移：

$$
\Delta\mathbf p^{\mathrm{vo}}_{i,t}
=
\mathbf p_{i,t}-\mathbf p_{i,t-1}
+
\boldsymbol\epsilon^{\mathrm{meas}}_{i,t},
\qquad
\mathbf z^{\mathrm{vo}}_{i,t}
=
\Delta\mathbf p^{\mathrm{vo}}_{i,t},
\qquad
\mathbf H^{\mathrm{vo}}_{i,t}
=
\left[\mathbf I_3\ \ \mathbf 0_{3\times 12}\right],
\qquad
\mathbf R^{\mathrm{vo}}_{i,t}\succcurlyeq 0.
$$

这里的视觉观测是相对位移。创新等于测得位移减去当前预测位置与预测前估计位置之差。代码 `LocalEKF.update_visual_odometry` 保存预测前状态与协方差，以“上一时刻协方差乘状态转移矩阵转置”构造前后时刻交叉协方差；创新协方差包含两个时刻的位置协方差之和，减去交叉协方差及其转置的位置投影，再加观测噪声协方差。Kalman增益的分子同样减去“当前状态—上一位置”的交叉项，不能把过去估计位置看作独立的确定锚点。上式雅可比是单时刻位置选择矩阵；相对位移联合雅可比是当前选择矩阵与过去选择矩阵的负值的拼接。仿真真值仅用于传感器生成和离线评价，位移也不是未经除以时间间隔的速度观测。

seed42 corrected-v2对照保留上述相关相对位移EKF，不更改滤波公式。当前有效VO样本按固定高斯标准差生成，而 `tracking_quality` 独立随机生成，尚无证据表明其预测真实观测噪声。因此新对照关闭质量重加权，使用与传感器生成器相同的标称观测协方差；以下质量校准开关保留为历史实验模式，不把它表述为已校准的真实质量—噪声模型。

为处理视觉跟踪质量变化造成的后续滤波交互，实验可启用一个独立的 VO 质量校准开关。每个运行首先只用 train warm-up 中的有效 `tracking_quality` 与执行期 VO 创新统计拟合分段、单调保守的协方差倍率；该倍率随后固定复用于 train、validation 和 test。跟踪质量越低时，倍率不得更小；质量缺失采用有限的保守回退值。该校准只改变传给 VO EKF 更新的 $\mathbf R^{\mathrm{vo}}_{i,t}$，不改变位移采样、协同 UWB、奖励或策略动作，也不使用真实误差、LOS/NLOS 真值或未来数据。未启用开关时仍使用历史固定 `SensorConfig` 协方差，以保持旧检查点和旧实验可复现。
UWB 提供标量相对距离观测。Python 传感器模型生成观测时使用环境真值：

$$
z^{\mathrm{uwb}}_{ij,t}
=
\left\|\mathbf p_{i,t}-\mathbf p_{j,t}\right\|_2
+
\epsilon^{\mathrm{meas}}_{ij,t}.
$$

EKF 执行更新时不得读取上述真值距离，而是使用本机预测状态和已到达的邻居估计构造预测量：

$$
\hat z^{\mathrm{uwb}}_{ij,t}
=
\left\|\hat{\mathbf p}_{i,t}^{-}-\hat{\mathbf p}_{j,t}^{\mathrm{arr}}\right\|_2,
\qquad
\tilde z_{ij,t}
=
z^{\mathrm{uwb}}_{ij,t}-\hat z^{\mathrm{uwb}}_{ij,t}.
$$

其中 $\hat{\mathbf p}_{j,t}^{\mathrm{arr}}$ 是消息到达后经运动模型传播的邻居估计，$\mathbf p_{j,t}$ 只存在于仿真传感器生成和离线评价路径，不进入 actor、GAT 或在线 EKF。

其本机状态雅可比、邻居位置雅可比和邻居不确定度传播在 3.2.4 节统一计算；UWB 噪声方差、NLOS 偏差修正和异常观测拒绝也在同一节执行。GNSS 失效时跳过 GNSS 更新，视觉或 UWB 退化时通过观测质量标记增大对应协方差，而不是用真实位置补齐缺失观测。

### 3.2.2 单个决策窗口的执行顺序

每个决策窗口首先使用运动模型和 IMU 完成预测：

$$
\hat{\mathbf{x}}^{-}_{i,t}
=
f(\hat{\mathbf{x}}_{i,t-1},\mathbf{u}_{i,t-1},\Delta t),
$$

$$
\boldsymbol{\Sigma}^{-}_{i,t}
=
\mathbf{F}_{i,t-1}\boldsymbol{\Sigma}_{i,t-1}\mathbf{F}_{i,t-1}^{\mathsf T}
+
\mathbf{Q}_{i,t-1}.
$$

对每个通过有效性检查的本机或邻居观测，计算 Kalman 增益并顺序更新状态：

$$
\mathbf{K}_{i,t}
=
\boldsymbol{\Sigma}^{-}_{i,t}
\mathbf{H}_{i,t}^{\mathsf T}
\left(
\mathbf{H}_{i,t}\boldsymbol{\Sigma}^{-}_{i,t}\mathbf{H}_{i,t}^{\mathsf T}
+
\mathbf{R}_{i,t}
\right)^{-1},
$$

$$
\hat{\mathbf{x}}_{i,t}
=
\hat{\mathbf{x}}^{-}_{i,t}
+
\mathbf{K}_{i,t}
\left(
\mathbf{z}_{i,t}-h(\hat{\mathbf{x}}^{-}_{i,t})
\right).
$$

为保持数值稳定和后验协方差的半正定性，使用 Joseph 形式更新：

$$
\boldsymbol{\Sigma}_{i,t}
=
\left(
\mathbf I-\mathbf K_{i,t}\mathbf H_{i,t}
\right)
\boldsymbol{\Sigma}^{-}_{i,t}
\left(
\mathbf I-\mathbf K_{i,t}\mathbf H_{i,t}
\right)^{\mathsf T}
+
\mathbf K_{i,t}\mathbf R_{i,t}\mathbf K_{i,t}^{\mathsf T}.
$$

具体执行顺序为：

1. 使用 IMU 和上一时刻估计状态进行预测；
2. 更新预测协方差；
3. 检查本机视觉、GNSS 和 UWB 观测；
4. 根据 links.csv 或通信事件判断邻居消息是否成功到达；
5. 丢弃丢包、过期或质量标记不合格的消息；
6. 对有效本机观测和邻居相对观测进行更新；
7. 输出状态估计、位置协方差和质量指标；
8. 将输出传递给动态图构造和 GAT。

链路中断时不执行对应邻居更新；链路延迟但超过 `stale_message_limit_s` 时，丢弃该邻居消息。链路延迟但未过期时，使用消息时间戳进行有效性判断，不将其视为当前时刻同步观测。

### 3.2.3 延迟消息到达与状态传播

通信模型为每条成功数据包记录发送窗口、预计到达窗口和消息载荷。发送时保存邻居状态、协方差和时间戳；到达窗口按照第 2.10 节的离散规则计算。消息只有在已经到达、字段有效、未超过 `stale_message_limit_s` 且动作允许使用时，才进入本机协同定位队列。

如果消息在发送时刻 $\tau^{\mathrm{send}}_{j,t}$ 生成，而接收方在当前时刻 $t'$ 处理，则使用发送方状态模型将其传播到 $t'$：

$$
\hat{\mathbf{x}}^{\mathrm{arr}}_{j,t'}
:=
f
\left(
\hat{\mathbf{x}}^{\mathrm{send}}_{j,t},
\mathbf{u}_{j,t:t'},
t'-\tau^{\mathrm{send}}_{j,t}
\right),
$$

$$
\boldsymbol{\Sigma}^{\mathrm{arr}}_{j,t'}
:=
\mathbf F_{j,t:t'}
\boldsymbol{\Sigma}^{\mathrm{send}}_{j,t}
\mathbf F_{j,t:t'}^{\mathsf T}
+
\mathbf Q_{j,t:t'}.
$$

若发送方控制或传播所需信息不可获得，则该消息不能被伪造为当前时刻状态，应标记为不可用并丢弃。过期消息不进入 EKF、GAT 或动作观测，但仍保留在通信日志中用于时延和 AoI 统计。

### 3.2.4 多邻居观测处理

当前版本采用顺序更新：按照观测时间戳从早到晚处理有效观测，并在每次更新后刷新状态和协方差。若多个观测具有相同时间戳，则按照观测类型的固定顺序处理：

~~~text
本机 IMU 预测
    → 本机视觉/GNSS
    → UWB 相对测距
    → 邻居状态或协同观测
~~~

该顺序在同一组实验中保持不变。观测质量较低时，通过增大对应观测噪声协方差降低其更新权重，而不是直接删除所有退化观测；VO 质量校准只在启用且已有 warm-up 映射时执行这一协方差调整。

对于 UWB 相对测距，接收的邻居状态消息记为

$$
\mathfrak m_{j\rightarrow i,t}
=
\left(
\hat{\mathbf x}_{j,t},
\boldsymbol{\Sigma}_{j,t},
\tau^{\mathrm{msg}}_{j,t}
\right).
$$

只有当消息对应的数据包在有效通信边上成功到达且未超过时间窗口时，才构造局部伪观测：

$$
\mathbf H^{\mathrm{rel}}_{ij,t}
=
\left[
\frac{(\hat{\mathbf p}^{-}_{i,t}-\hat{\mathbf p}_{j,t})^{\mathsf T}}
{\|\hat{\mathbf p}^{-}_{i,t}-\hat{\mathbf p}_{j,t}\|_2}
\quad
\mathbf 0
\right].
$$

该雅可比表示本机位置沿 $i$ 与 $j$ 连线方向变化时，预测测距的变化率。邻居位置对相对距离的雅可比为：

$$
\mathbf H^p_{j,t}
:=
-\frac{
\left(
\hat{\mathbf p}^{-}_{i,t}-\hat{\mathbf p}_{j,t}
\right)^{\mathsf T}
}{
\left\|
\hat{\mathbf p}^{-}_{i,t}-\hat{\mathbf p}_{j,t}
\right\|_2
}.
$$

相对测距创新写为：

$$
\tilde z_{ij,t}
=
z^{\mathrm{uwb}}_{ij,t}
-
\left\|
\hat{\mathbf p}^{-}_{i,t}-\hat{\mathbf p}_{j,t}
\right\|_2.
$$

当 UWB 观测处于 NLOS 状态时，按照第 2.8 节的偏差模型进行偏差修正；如果没有可校准的偏差参数，则增大等效观测方差，不把 NLOS 测距当作无偏真值。对每个标量相对测距观测执行创新门限检查：

$$
\tilde z_{ij,t}^{\mathsf T}
\left(
\mathbf H^{\mathrm{rel}}_{ij,t}
\boldsymbol{\Sigma}^{-}_{i,t}
(\mathbf H^{\mathrm{rel}}_{ij,t})^{\mathsf T}
+\mathbf R^{\mathrm{rel}}_{ij,t}
\right)^{-1}
\tilde z_{ij,t}
\leq
\chi^2_{1,1-\alpha_{\mathrm{gate}}}.
$$

超过门限的观测只写入异常观测日志，不执行本次 EKF 更新。

邻居位置不确定度通过相对观测协方差传递。当前实现还使用Cauchy–Schwarz上界保护未知本机—邻居交叉相关；与实现对齐的等效标量观测方差为：

$$
\mathbf R^{\mathrm{rel}}_{ij,t}
=
R^{\mathrm{uwb}}_{ij,t}
+
\mathbf H^{p}_{j,t}
\boldsymbol{\Sigma}^{p}_{j,t}
(\mathbf H^{p}_{j,t})^{\mathsf T}
+2\sqrt{
\max\!\left(0,\mathbf H^{\mathrm{rel}}_{ij,t}\boldsymbol{\Sigma}^{-}_{i,t}(\mathbf H^{\mathrm{rel}}_{ij,t})^{\mathsf T}\right)
\max\!\left(0,\mathbf H^{p}_{j,t}\boldsymbol{\Sigma}^{p}_{j,t}(\mathbf H^{p}_{j,t})^{\mathsf T}\right)
}.
$$

实际传入标量更新的方差还加上两倍平方根项，其根号内为本机与邻居在视线方向上方差的乘积。本机方向方差由 $\mathbf H^{\mathrm{rel}}_{ij,t}\boldsymbol{\Sigma}^{-}_{i,t}(\mathbf H^{\mathrm{rel}}_{ij,t})^{\mathsf T}$ 给出，邻居方向方差由上式邻居投影项给出；负的数值残差按零截断，最终观测方差具有 $10^{-9}$ 的数值下限。测距被接受时允许修改状态均值，但在邻居协方差存在而未维护本机—邻居联合协方差时，最终本机协方差恢复为本次更新前的协方差。日志中的线性协方差收益是恢复前的假设值，不是最终不确定度收益。该保护不在corrected-v2中删除。

当前仿真还使用物理LOS/NLOS标签确定UWB噪声方差和偏差修正，这是滤波端的理想化信息假设，不是已证明可在真实执行端直接获得的信息。actor和代理教师不读取这些真值标签；二者的信息边界必须与滤波端假设分别说明。

当多个消息对同一邻居状态存在未知交叉相关时，当前基线只对邻居位置均值和位置协方差执行协方差交叉融合，再执行本机 EKF 更新。CI 位置协方差为：

$$
\boldsymbol{\Sigma}^{p,\mathrm{CI}}_{j,t}(\omega_{\mathrm{CI}})
=
\left[
\omega_{\mathrm{CI}}(\boldsymbol{\Sigma}^{p,(1)}_{j,t})^{-1}
+
(1-\omega_{\mathrm{CI}})(\boldsymbol{\Sigma}^{p,(2)}_{j,t})^{-1}
\right]^{-1},
\qquad 0\leq\omega_{\mathrm{CI}}\leq1.
$$

对应的保守位置均值为：

$$
\hat{\mathbf{p}}^{\mathrm{CI}}_{j,t}
=
\boldsymbol{\Sigma}^{p,\mathrm{CI}}_{j,t}
\left[
\omega_{\mathrm{CI}}
(\boldsymbol{\Sigma}^{p,(1)}_{j,t})^{-1}
\hat{\mathbf{p}}^{(1)}_{j,t}
+
(1-\omega_{\mathrm{CI}})
(\boldsymbol{\Sigma}^{p,(2)}_{j,t})^{-1}
\hat{\mathbf{p}}^{(2)}_{j,t}
\right].
$$

实现中在 $[0,1]$ 上离散搜索 $\omega_{\mathrm{CI}}$，选取使 $\operatorname{tr}(\boldsymbol{\Sigma}^{p,\mathrm{CI}}_{j,t})$ 最小的权重。不构造全局增广状态，也不把邻居估计当作无噪声真值。

### 3.2.5 EKF 方法输出

EKF 输出至少包括：

- 位置、速度和姿态估计；
- 位置协方差和完整状态协方差；
- 当前有效观测数量；
- 当前有效邻居数量；
- 最近一次成功接收的邻居信息时间；
- 供 GAT 使用的定位不确定度特征。

真实位置误差仅用于仿真评价；在线奖励使用滤波器内部的定位不确定度和通信状态。EKF 的完整状态、位置协方差、有效观测数和有效邻居数写入 `estimates.csv`。

每次预测或更新后都执行数值检查：将协方差替换为
$\frac{1}{2}(\boldsymbol{\Sigma}_{i,t}+\boldsymbol{\Sigma}_{i,t}^{\mathsf T})$，检查所有元素有限，并用最小特征值检查半正定性。若出现轻微负特征值，仅允许加入配置中的数值稳定项；若出现明显非半正定、奇异创新协方差或非有限状态，则拒绝该观测、记录异常并保留上一个有效滤波结果，不能继续把异常量传入 GAT 或奖励。

数值检查通过后，对估计位置执行一次与场景飞行区域一致的有界投影：

$$
\hat{\mathbf p}_{i,t}
\leftarrow
\operatorname{clip}_{[\mathbf p_{\min},\mathbf p_{\max}]}
\left(\hat{\mathbf p}_{i,t}\right).
$$

该投影不是对仿真真值的回填，而是对 EKF 后验均值施加的物理可行域约束；投影次数必须写入诊断信息，供检查 UWB 延迟融合或策略动作是否导致估计越界。投影后的位置估计和协方差才进入 GAT、MAPPO 观测及奖励计算。

滤波一致性只在离线评价阶段使用真实状态计算，不作为执行期观测。位置归一化估计误差平方（NEES）和相对测距归一化创新平方（NIS）分别定义为：

$$
\mathrm{NEES}^{p}_{i,t}
=
\frac{1}{3}
\left(\hat{\mathbf p}_{i,t}-\mathbf p_{i,t}\right)^{\mathsf T}
\left(\boldsymbol{\Sigma}^{p}_{i,t}\right)^{-1}
\left(\hat{\mathbf p}_{i,t}-\mathbf p_{i,t}\right),
$$

$$
\mathrm{NIS}^{\mathrm{rel}}_{ij,t}
=
\tilde z_{ij,t}^{\mathsf T}
\left(
\mathbf H^{\mathrm{rel}}_{ij,t}
\boldsymbol{\Sigma}^{-}_{i,t}
\left(\mathbf H^{\mathrm{rel}}_{ij,t}\right)^{\mathsf T}
+\mathbf R^{\mathrm{rel}}_{ij,t}
\right)^{-1}
\tilde z_{ij,t}.
$$

报告 NEES/NIS 的均值、分位数和超出理论置信区间的比例，用于判断协方差是否过度自信或过度保守；这些统计量只用于 EKF 一致性检查与消融分析，不进入 actor observation、centralized critic 输入或核心代价 $J_t$。

## 3.3 动态通信图构造与链路质量参数化检查

### 3.3.1 通信图构造

每个决策窗口使用上一个窗口末尾已经形成的局部候选图和动作掩码产生动作。动作执行后，环境内部根据更新后的真实位置、障碍物包络和通信配置重新生成潜在边，并将逐边带宽分配送入链路质量计算。对于每个有向无人机对，依次检查：

1. 几何距离是否进入潜在边集合；
2. 根据当前动作分配逐边带宽；
3. 三维视线是否被静态或动态障碍物遮挡；
4. 链路统计量是否满足候选边阈值；
5. 当前数据包是否成功传输；
6. 传输时延是否超过信息有效窗口。

通过距离、遮挡、链路质量和时延阈值检查的边加入下一窗口的 $G_{t+1}^{\mathrm{cand}}$；只有数据包成功传输、已经到达且未过期的边才加入 $G_{t+1}$。策略不能直接把任意节点加入有效邻居集合，只能在潜在出边上分配带宽，并从候选入边动作掩码中选择协同测量源。真实几何只在环境内部参与潜在边和 LOS/NLOS 判定，不能作为 actor 的输入。

### 3.3.2 链路统计模型

通信模型按照“距离与遮挡 → 路径损耗 → 接收功率与 RSSI → 噪声与 SNR → 速率 → 丢包和时延”的顺序执行。带宽动作先转换为链路带宽：

$$
b_{ij,t}=a^{\mathrm{bandwidth}}_{ij,t}B_i^{\mathrm{tot}},
\qquad
\sum_{j:(i,j)\in\mathcal E^{\mathrm{pot}}_t}a^{\mathrm{bandwidth}}_{ij,t}=1
\quad\text{when }\{j:(i,j)\in\mathcal E^{\mathrm{pot}}_t\}\neq\varnothing.
$$

若潜在出边为空，带宽分配返回空集合；不通过虚拟链路或隐式剩余带宽制造通信边。

随后调用统一信道函数，计算：

$$
L^{\mathrm{path}}_{ij,t},
\quad
P^{\mathrm{rx}}_{ij,t},
\quad
\mathrm{RSSI}_{ij,t},
\quad
\mathrm{SNR}_{ij,t},
\quad
R_{ij,t},
\quad
P_{\mathrm{loss},ij,t},
\quad
D_{\mathrm{com},ij,t}.
$$

所有功率使用 dBm，RSSI 和 SNR 使用 dB，$b_{ij,t}$ 使用 Hz，$R_{ij,t}$ 使用 bit/s。路径损耗采用 LOS/NLOS 条件化的对数距离模型；丢包率采用包含 SNR、LOS/NLOS 状态和队列长度的 Logistic 模型；时延只包含传播、序列化和排队三项，第一版不隐式执行自动重传。带宽为零时直接令 $R_{ij,t}=0$、丢包率为 1，并将链路标记为 no_bandwidth；当 $R_{ij,t}>0$ 时，候选阈值使用未截断的 $D^{\mathrm{raw}}_{ij,t}$，日志中的 $D_{\mathrm{com},ij,t}$ 只保存不超过 $D_{\mathrm{com},\max}$ 的有限值。

实现时，`generator.py` 和 `episode_generator.py` 使用同一 `ChannelConfig` 和 `compute_link_metrics` 接口。通信模式 `good/moderate/poor` 通过可用带宽和队列负载体现，不再对 RSSI 添加未解释的模式偏移。

通信参数全部来自版本化的 Python `ChannelConfig`、队列模型和场景配置；正式实验通过控制变量扫描、重复随机种子、敏感性分析和消融实验检查参数合理性，不使用公开数据集校准。Flightmare 仅保留为历史导入兼容接口，不是当前实验环境或数据源。

### 3.3.3 数据包和 AoI 更新

每条消息都记录发送时间、预计到达窗口、实际到达时间、源节点、目的节点和传输结果。成功发送的数据包只有在其离散到达窗口到达后才可进入接收方消息缓冲；到达且被接收时将 AoI 重置为该数据包的实际通信时延，未到达或丢包时仅按链路更新周期递增，不额外叠加失败数据包的时延，并将 AoI 截断在 $\mathrm{AoI}_{\max}$ 内。AoI 根据当前时刻与最近成功接收时间的差值更新，并在 `packet_events.csv` 中保存。队列使用数据包到达—服务递推：$Q_{ij,t}$ 是服务前队列，$Q_{ij,t+1}$ 是服务后的队列，不通过未记录的自动重传改变队列状态。消息到达后还必须经过字段有限性、协同测量动作许可和陈旧时间检查，才可进入 EKF/GAT；否则只保留在日志中。

为验证测量选择与通信资源之间的因果链，训练配置提供 `measurement_overhead_enabled` 开关。开关为真时，接收方 $i$ 选择非 `none` 的 `measurement_uav_id=j`，且 $j\to i$ 为潜在通信边，则在当前信道计算前向该边的 $A_{ji,t}$ 增加一个等效数据包。该包复用 $L_{\mathrm{pkt}}$，固定为 $512$ bytes（$4096$ bit），随后与普通队列负载共同进入速率服务、丢包、时延和 AoI 更新；开关为假时不增加负载。该实现只建模等效队列占用，不加入未记录的 MAC 握手或自动重传，也不改变 UWB、EKF、GAT、运动和奖励公式。因而，测量动作对 $J_{\mathrm{com}}$ 的影响必须通过日志中的队列和通信 KPI 体现，而不能通过隐藏的额外奖励项实现。

当邻居消息丢失或过期时：

- 不将其加入当前邻居观测；
- 不将其作为 GAT 的有效边消息；
- 不执行对应 EKF 协同更新；
- 在奖励中反映丢包、时延或信息年龄代价。

## 3.4 GAT 动态图表示方法

本节给出第 2.11 节图状态接口的具体 GAT 构造方法，并沿用该节“发送节点 $j$ → 接收节点 $i$”的有效入邻居约定。潜在边和候选边只进入资源分配特征与动作掩码，不进入有效邻居消息聚合。

### 3.4.1 节点和边特征

节点特征 $\mathbf h^{(0)}_{i,t}$、边特征 $\mathbf e_{ji,t}$、有效入邻居以及 actor/critic 可见性已经在第 2.11 节唯一规定，本节不重新定义这些向量。实现时首先根据当前有效通信图构造物理入边索引；潜在边和候选边仅构造资源分配特征及动作掩码，不能进入有效邻居消息聚合。

当前纯 Python 基线中，状态估计和协方差各为 15 维，目标特征为 3 维，局部障碍物特征为 4 维，因此节点特征为 37 维；边特征为 11 维。若实验配置改变状态或障碍物特征维度，必须同步保存新的输入维度和模型配置，不能通过截断真实字段维持旧维度。

安全运动特征配置将局部障碍物聚合扩展为八维：前四维保持上述历史顺序，后四维固定为 `dynamic_min_clearance`、`dynamic_closing_rate`、`dynamic_predicted_clearance` 和 `dynamic_motion_valid`。因此新配置的节点特征维度为 $15+15+3+8=41$。这些量由环境在当前决策窗口根据估计位置、估计速度和当前可见的动态障碍物运动信息聚合生成；actor 只接收聚合标量，不接收障碍物绝对坐标、原始速度、对象 ID、真实误差、碰撞标签或下一窗口真值。`dynamic_predicted_clearance` 只表示基于当前信息的一步模型预测，不改变实际环境仍采用的“运动—当前安全投影—动态障碍物更新—碰撞检查”顺序。

当当前窗口没有有效动态运动信息时，四个扩展字段仍保持有限值，并以 `dynamic_motion_valid=0` 显式表示缺失；该占位值不会被解释为安全状态。连续字段沿用训练划分的归一化统计量，二值有效性字段保持原值。默认配置继续关闭扩展并使用 37 维输入；安全 Smoke 显式开启扩展并从头训练 41 维检查点，旧检查点和旧实验结果不被迁移或覆盖。

正式安全消融使用三个固定分支：`baseline_37d` 关闭动态运动特征和质量/预算输入，`baseline_41d` 仅开启动态运动特征，`budgeted_quality_aware_41d` 在相同 41 维节点输入上进一步开启执行期质量特征和训练集测量预算。三者复用 EKF、GAT、centralized critic、信道、队列、奖励和动态障碍物物理时序；因此 `baseline_41d-baseline_37d` 表示观测特征效应，完整分支相对 `baseline_41d` 表示质量/预算机制效应。预算仍由 `baseline_37d` 的 train 回合拟合，完整分支的质量模型只由自身 warm-up 观测拟合，validation 和 test 不重新估计。正式评估在冻结的十个 hold-out 安全场景上进行，每个场景先平均五个策略种子，再以场景种子作为统计独立单位。这里的动态障碍物特征仍是仿真器依据当前动态障碍物运动字段构造的理想化聚合量，不代表真实传感器已经测得原始障碍物状态。

连续物理特征只使用训练划分统计量进行标准化，二值标记保持原值，缺失值和有效性掩码单独传递。验证、测试和部署阶段复用训练阶段保存的统计量，不重新估计均值或方差。`graph_interface.py` 负责构造 NumPy 节点特征、边特征、三类边索引和动作掩码；`ObservationNormalizer` 负责训练统计量拟合与复用；GAT 模块最后完成 NumPy 到 PyTorch 张量的转换。没有成功接收邻居状态的候选节点保留显式占位和无效掩码，不使用仿真真实位置补全。

### 3.4.2 注意力计算

第 $l$ 层对有效入邻居进行注意力聚合。为保留接收节点自身信息，在 GAT 计算图内部加入虚拟自环 $i\to i$；该自环不写入物理通信图、链路日志或通信统计。约定虚拟自环特征为 $\mathbf e_{ii,t}=\mathbf 0$，并令求和范围包含该自环：

$$
\alpha^{(l)}_{ij,t}
=
\frac{
\exp\left(
\operatorname{LeakyReLU}
\left(
\boldsymbol{\eta}^{(l)\mathsf T}
\left[
\mathbf W^{(l)}\mathbf h^{(l)}_{j,t}
\Vert
\mathbf W^{(l)}\mathbf h^{(l)}_{i,t}
\Vert
\mathbf W_e^{(l)}\mathbf e_{ji,t}
\right]
\right)
\right)
}{
\displaystyle
\sum_{k\in\mathcal N_{i,t}\cup\{i\}}
\exp\left(
\operatorname{LeakyReLU}
\left(
\boldsymbol{\eta}^{(l)\mathsf T}
\left[
\mathbf W^{(l)}\mathbf h^{(l)}_{k,t}
\Vert
\mathbf W^{(l)}\mathbf h^{(l)}_{i,t}
\Vert
\mathbf W_e^{(l)}\mathbf e_{ki,t}
\right]
\right)
\right)
}.
$$

节点更新为：

$$
\mathbf h^{(l+1)}_{i,t}
=
\sigma
\left(
\sum_{j\in\mathcal N_{i,t}\cup\{i\}}
\alpha^{(l)}_{ij,t}
\left(
\mathbf W^{(l)}\mathbf h^{(l)}_{j,t}
+
\mathbf W_e^{(l)}\mathbf e_{ji,t}
\right)
\right).
$$

这里 $\alpha^{(l)}_{ij,t}$ 的第一下标 $i$ 表示接收节点，第二下标 $j$ 表示发送邻居；对应物理边为 $j\to i$。上式给出单个注意力头的抽象形式；实际 `EdgeGATNetwork` 的中间层采用多头注意力并拼接各头输出，最后一层使用单头输出，同时加入残差映射、ELU 激活和可配置 dropout。注意力 softmax 分别在每个接收节点的入边集合上归一化。

若当前节点没有有效邻居，网络仍加入零边特征的虚拟自环以保留本机信息，不填充虚假的邻居状态。网络层数、隐藏维度、注意力头数和 dropout 由实验配置固定并写入运行清单，验证和测试阶段不得改变网络结构。

### 3.4.3 图表示输出与接口

完成预设层数的注意力更新后，最后一层节点输出定义为第 2.11 节的图表示 $\mathbf g_{i,t}$。执行期 actor 使用本机图表示、候选/潜在出边质量和动作掩码；centralized critic 使用全局节点和边图表示。图表示不是物理位置或定位真值，不能绕过通信成功条件直接获得邻居状态。

当前 `GraphBatch` 接口分别保存 `effective_edge_index`、`candidate_edge_index` 和 `potential_edge_index`，并分别提供对应边特征、动作掩码、带宽掩码以及 UWB 可用性字段。该接口只负责图输入组织和 NumPy/PyTorch 转换；`src/learning/gat.py` 的 `EdgeGATLayer`/`EdgeGATNetwork` 实现带边特征的多头注意力、虚拟自环、入边归一化和节点图表示输出。GAT 不把虚拟自环写入物理通信图，也不把潜在/候选边当作已成功接收的邻居消息。训练阶段拟合的归一化统计量随模型检查点保存，作为验证、测试和部署时不可重新估计的运行依赖。

## 3.5 MAPPO 联合决策方法

### 3.5.0 当前论文主方法与历史分支边界

本文当前主方法统一记为 **Local-EKF-GAT-MAPPO (Proposed Method)**。为保持已有数据、checkpoint 和脚本的可复现性，其内部兼容标识仍为 `baseline_mappo`；该标识不表示本文采用了一个弱控制器，也不表示它不使用邻居信息。该方法实际使用本地 Distributed EKF 输出、动态图 GAT 表示和 MAPPO 的集中训练—分散执行机制。

当前论文主线实验使用 `core-framework-v3` 协议，主方法的有效实现集合为 `local_only_fixed`、`random_cooperative`、`budgeted_greedy_ig` 和 `baseline_mappo`（图表中显示为 **Local-EKF-GAT-MAPPO (Proposed Method)**）。其中最后一个是本文的完整本地 EKF+GAT+MAPPO 框架，不是弱基线；前三者为不训练的规则参照。`mappo_mlp` 只属于历史架构对照，在 v3 中不训练、不读取、不绘图。

`communication_constrained_proxy_guided_mappo` 属于此前围绕代理教师和通信约束开展的历史探索分支，不属于当前论文主方法或当前主线对比集合。后续新实验和主线图表不再调用或展示该分支；历史代码和结果目录保留为 deprecated 归档，以避免破坏历史实验的可追溯性。论文中的主线比较应以 Local-EKF-GAT-MAPPO 与明确说明的弱/强基线为对象，并在表格中保留内部 ID 映射。

### 3.5.1 CTDE 信息流

执行期每架无人机只能使用：

- 本机 $\hat{\mathbf{x}}_{i,t}$ 和 $\boldsymbol{\Sigma}_{i,t}$；
- 当前有效邻居的 GAT 消息；
- 本机可见障碍物和目标信息；
- 本机通信、能量和动作约束。

训练期 centralized critic 可以使用中心状态 $s^{\mathrm c}_t$、完整通信图、障碍物真值摘要和全局奖励；执行期 actor 不读取真实位置、真实定位误差、未来障碍物状态或未成功传输的邻居消息。中心状态只通过环境的 `info["central_state"]` 提供，不混入 actor observation。

Actor 和 centralized critic 的连续输入均只使用训练集拟合的归一化统计量；验证、测试和部署阶段固定复用这些统计量，不能用当前评估数据重新计算。

当前规范采用参数共享 actor：每架无人机使用同一组策略参数处理自己的局部观测。Actor 将第 3.4 节的本机 GAT 图表示与有效、候选入边和潜在出边的汇总特征以及动作掩码拼接，再由多层感知机产生测量、带宽和速度动作头。系统只使用一个 centralized critic $V_{\boldsymbol{\phi}}(s^{\mathrm c}_t)$，将固定槽位编码后的中心状态、全局图摘要和允许的障碍物摘要映射为一个团队价值；critic 的中心输入不回流到 actor。

在 `core-framework-v2` 中，测量头不再只对候选边特征做全局均值聚合。对接收 UAV $i$ 的每个候选邻居 $j$，分别构造逐候选输入并计算动作得分：

$$
s_{ij,t}
=
f_{\boldsymbol{\theta}}\left(
\mathbf h^{\mathrm{GAT}}_{i,t},
\mathbf h^{\mathrm{GAT}}_{j,t},
\mathbf e_{ji,t},
\mathbf m_{ji,t},
b_{i,t}
\right).
$$

其中 $\mathbf h^{\mathrm{GAT}}_{j,t}$ 只在邻居估计已成功接收并且节点可见时使用；不可见候选保留显式缺失表示，并仍使用当前可见的候选入边特征和动作掩码。$\mathbf m_{ji,t}$ 是执行期标记向量，包含候选、有效、可行动作、节点可见和剩余预算信息。`none` 使用独立得分且始终保留；预算耗尽时动作掩码只允许 `none`。该设计使“候选邻居身份—测量动作 logit”一一对应，禁止通过真实位置、真实误差、LOS/NLOS 真值、NIS 或未来结果补全候选信息。

为形成公平的架构对照，`mappo_mlp` 使用相同的逐候选动作头，只将 GAT 换成固定 UAV 身份槽位、候选/接收角色和存在掩码的两层 MLP。它保留相同的可见输入和动作空间，因此 GAT 的增量应解释为图消息聚合能力，而不是额外的候选动作表达能力。

### 3.5.1.1 v9 候选级动作选择修正

针对早期实验中 validation 阶段合法候选存在但策略大量选择 `none` 的现象，
`core-framework-v9-action-selection` 保留 v2 逐候选动作头作为控制组，并增加
`action_conditioned_v3` 作为单一动作表示修正。对每个候选邻居 $j$，构造当前时刻可见的
候选描述向量：

$$
\boldsymbol{\psi}_{ij,t}
=
\operatorname{concat}\left(
\text{covariance-confidence},
\text{range-variance proxy},
\text{link quality},
\text{AoI},
\text{queue},
\text{feasibility flags}
\right).
$$

测量动作得分使用：

$$
s_{ij,t}
=
f_{\boldsymbol{\theta}}\left(
\mathbf h^{\mathrm{GAT}}_{i,t},
\mathbf h^{\mathrm{GAT}}_{j,t},
\mathbf e_{ji,t},
\mathbf m_{ji,t},
\boldsymbol{\psi}_{ij,t},
\mathbf b_{i,t}
\right).
$$

其中 $\boldsymbol{\psi}_{ij,t}$ 只使用本机 EKF 当前估计协方差、已接收邻居消息、当前
链路字段、UWB 可用性和动作掩码；不读取真实位置、真实 RMSE、真实 LOS/NLOS 标签、未来
NIS 或未来通信结果。`none` 动作保留独立 logit，并额外接收当前可行候选数量、候选质量
最大值/均值和剩余预算比例等摘要，从而允许策略根据当前候选集合判断是否值得执行邻居测量。
预算耗尽或没有合法候选时仍只允许 `none`，不设置人为的最低非 `none` 选择率。

v9 的 `action-linked-feedback-v2` 仅作为动作选择反馈校准组。对当前实际选择且被本地 EKF
接受的邻居更新，计算执行期可见的候选质量摘要 $q^{\mathrm{cand}}_{ij,t}$，并与 warm-up
冻结的 `none` 基准 $q^{\mathrm{none}}_{i,t}$ 做差分。令 $\mu_q$ 和 $s_q$ 分别为 warm-up
差分的稳健位置和尺度，则延迟反馈写为：

$$
r^{\mathrm{act}}_{i,t+\delta}
=
\alpha_{\mathrm{act}}\tanh\left(
\frac{q^{\mathrm{cand}}_{ij,t}-q^{\mathrm{none}}_{i,t}-\mu_q}
{s_q+\varepsilon}
\right).
$$

该反馈只进入 PPO 的训练通道，释放时记录所选邻居、接受状态、反馈时间、更新前后协方差
和反馈来源；报告奖励、$J_{\mathrm{loc}}$、$J_{\mathrm{com}}$、RMSE、EKF 公式和通信物理
模型保持不变。如果协方差恢复后该反馈没有对应的物理状态变化，只能将其解释为训练代理，
不能宣称已经带来最终定位改善。v9 同时输出训练/validation 的 `none` 率、候选 logit、
动作熵和候选质量—选择概率关系，用于区分“动作头缺少候选信息”和“反馈通道诱发塌缩”。

### 3.5.2 动作执行

策略输出严格使用第 2.12 节定义的混合动作，不在本节重新定义动作空间。测量选择使用带 `none` 选项的掩码 Categorical 分布；动作掩码在计算概率前将不可行节点排除，因此策略不会采样候选集合之外的对象。当前版本没有中继动作头。

带宽头仅对当前潜在出边生成参数，经 `softplus + 1` 转换为正的浓度参数后构造 Dirichlet 分布。采样结果天然非负且在有效潜在出边上和为 1；没有潜在出边时返回空分配。轨迹头使用三维对角高斯分布产生原始速度调整，经双曲正切和欧氏范数裁剪得到速度增量；环境随后再次执行速度、加速度、飞行区域和碰撞安全投影。

训练时从上述分布采样动作，验证和测试时可使用测量概率最大项、Dirichlet 均值和连续分布均值形成确定性动作。动作在当前时刻产生并作用于下一决策窗口，环境记录的是安全投影后的实际执行结果。

`core-framework-v3` 明确区分策略采样动作 $\mathbf a^{\mathrm{samp}}_{i,t}$ 与环境执行动作
$\mathbf a^{\mathrm{exec}}_{i,t}$。前者是混合动作分布真正采样出的测量、带宽和速度动作，
其旧对数概率用于 PPO 概率比；后者是在环境中经过有效性检查、裁剪、安全投影和预算约束后
真正施加的动作。v3 的 `sampled` 模式不把后者反向写回前者，二者的差异单独写入动作执行
诊断日志。这样既保留安全投影，又避免用不可逆的执行动作重算旧概率而破坏 PPO 语义。

### 3.5.3 预算约束分支

预算约束只作用于 `budgeted_quality_aware_mappo`，baseline 和原 `quality_aware_mappo` 保持无预算行为。每个随机种子先完成 baseline 的训练划分，再按第 2.12 节的定义统计每架无人机的非 `none` 测量次数并取整得到 $K^{\mathrm{meas}}_{\nu,i}$。该计算在训练阶段完成一次；validation 和 test 直接读取同一组预算，不重新估计，也不使用测试信息。

在每个 episode 的 `reset` 时将已用预算置零。生成 actor 动作前，环境依据当前剩余预算更新测量动作掩码；当 $B^{\mathrm{rem}}_{\nu,i,t}=0$ 时，仅保留 `none`。预算比例作为额外的标量输入只拼接到预算分支的测量动作头，不能进入带宽头、速度头或 centralized critic。采样到合法的非 `none` 测量后，环境在该决策窗口消耗一个预算单位；无效动作会先被归一化为 `none`，因此不会消耗预算。

每个窗口记录预算上限、使用前剩余量、选择、消耗量和使用后剩余量到 `measurement_budget.csv`，并在 `scenario.json`、`manifest.json` 和训练日志中保存预算来源。预算开关不改变 EKF、GAT、信道、奖励、通信开销模型或核心目标；开启通信开销时，非 `none` 测量仍按第 3.3.3 节产生等效 $L_{\mathrm{pkt}}$ 队列负载。预算耗尽、动作掩码和旧分支兼容性必须通过单元测试验证。

### 3.5.4 训练期测量率约束

`rate_controlled_quality_aware_mappo` 在 `budgeted_quality_aware_mappo` 的质量特征和训练集固定预算基础上，再加入训练期测量率约束；`baseline_mappo` 与旧 `budgeted_quality_aware_mappo` 的默认行为保持不变。每个随机种子先读取 baseline 的 `train` 动作日志，按第 2.12 节的 $\kappa^{\mathrm{target}}_{\nu,i}$ 定义拟合各无人机目标测量率，validation 和 test 只复用该目标。

在每个 PPO minibatch 中，使用当前 actor 的 `none` 动作概率计算期望非空测量率 $\bar\kappa_{\nu,i}$。只有实际出现在该 minibatch 的无人机参与约束，并按第 2.12 节的 $L^{\mathrm{rate}}$ 加入总损失。正式配置固定使用 `measurement_rate_constraint_coef=0.5` 和 `measurement_rate_tolerance=0.05`；约束项只影响反向传播，不改变环境中的动作掩码、预算消耗、通信队列、EKF、奖励或推理阶段的动作定义。训练日志逐更新保存目标率、期望率和约束损失，便于验证约束是否有限且确实生效。

### 3.5.5 自适应测量率约束分支

在固定系数审计判定约束能力不足后，新增 `adaptive_budgeted_quality_aware_mappo` 分支作为单变量稳定性验证。该分支与 `budgeted_quality_aware_mappo` 共享质量特征、训练集固定预算、512 字节测量开销、EKF、GAT、信道和奖励实现；唯一新增训练控制是第 2.12 节定义的每 UAV 自适应乘子。其附加损失使用当前 minibatch 的 $v^{\mathrm{rate}}_{\nu,i}$ 和已保存的、与策略梯度分离的 $\lambda^{\mathrm{dual}}_{\nu,i}$。

每次 train episode 完成一次 `trainer.update()` 后，按 $\eta^{\mathrm{dual}}_{\mathrm{rate}}=0.25$ 更新乘子并裁剪到 $[0,\lambda^{\mathrm{dual}}_{\max}]$，其中 $\lambda^{\mathrm{dual}}_{\max}=5$。validation 和 test 仅进行确定性推理，不更新乘子；乘子状态与 policy、optimizer 状态一同保存到 `final.pt`。baseline 和历史 budgeted 分支的乘子开关保持关闭，因此该分支不会改变既有实验输出的解释。

### 3.5.5 代理引导的自适应测量选择

在测量率稳定性得到验证后，新增 `proxy_guided_adaptive_budgeted_quality_aware_mappo` 分支，用于处理“测量数量稳定但测量质量仍可能不稳定”的问题。该分支保留质量特征、baseline 训练集固定预算、自适应测量率乘子、512 字节测量通信开销以及原有 EKF、GAT、信道、队列和奖励实现；唯一新增的是执行期可见的信息增益—通信代价代理和 measurement head 的辅助 teacher loss。

对当前可见的每个候选邻居，依据第 2 章定义的 $\widehat G^{\mathrm{IG}}_{ij,t}$、$\widehat C^{\mathrm{com}}_{ij,t}$ 构造 $U^{\mathrm{proxy}}_{ij,t}$，不使用真实位置、真实误差、LOS/NLOS 真值、NIS 或未来通信结果。代理尺度只由同一随机种子的 baseline `train` 图观测拟合，validation 和 test 仅复用冻结参数。当前动作掩码先排除不可行候选，`none` 永远保留；预算耗尽后仍只允许 `none`。

使用掩码后的代理效用以 $T^{\mathrm{proxy}}=0.25$ 形成 teacher 分布，并加入
$\alpha_{\mathrm{proxy}}=0.25$ 倍的 $L^{\mathrm{proxy}}$。teacher 目标和 actor body 上下文均与反向传播分离，辅助梯度只进入 measurement head，不改变带宽、速度、GAT、centralized critic 或环境奖励。该分支的 Smoke 先确认代理有限、预算和队列不变量、teacher loss 与梯度路径正确；只有 Smoke 通过后，才开展五个随机种子的正式稳定性验证。该实验属于机制与稳定性验证，不单独构成最终算法优越性的证明。

### 3.5.6 执行期通信代价约束（历史归档，不属于当前主线）

在此前探索中曾实现 `communication_constrained_proxy_guided_mappo`。该分支保留代理教师、质量特征、训练集硬预算和 512 字节测量负载，但关闭原双侧测量率约束。候选通信代价按第 2.12 节的 $\widehat c^{\mathrm{meas}}_{ij,t}$ 计算，使每次非 `none` 测量至少承担固定包代价，`none` 保持零代价。该分支现仅作为历史归档，不再属于本文当前主方法、基线或新实验协议。

每个 seed/UAV 的 $c^{\mathrm{target}}_{\nu,i}$ 只从 baseline train 动作与执行期可见链路字段拟合。训练 minibatch 按 measurement head 的完整动作概率计算 $\bar c_{\nu,i}$，再以单侧 $L^{\mathrm{com,dual}}$ 更新 measurement head；actor body 在该辅助路径中停止梯度，带宽、速度、GAT 和 centralized critic 不受该约束梯度影响。通信乘子只在 `trainer.update()` 后更新并写入 checkpoint，validation/test 只进行推理。该约束不作为新的环境奖励项，也不改变 $J_{\mathrm{loc}}$、$J_{\mathrm{com}}$、EKF 或信道公式。

为避免 teacher 与真实通信约束对同一个固定测量负载重复计罚，通信约束分支的 measurement teacher 使用目标相对代价：只对候选代价超过 $c^{\mathrm{target}}_{\nu,i}$ 的部分施加代理惩罚。该差异仅存在于 detached teacher 的 utility 计算，真实队列、丢包、时延、AoI、$J_{\mathrm{com}}$ 和硬测量预算仍使用完整测量负载；因此它不改变环境目标，也不使 baseline 或历史 proxy 分支的结果失去兼容性。

### 3.5.7 持久信息收益代理测量选择

为处理协同 EKF 单次更新收益在后续预测中衰减的问题，新增
`persistent_proxy_guided_quality_mappo` 分支。该分支与 `baseline_mappo` 使用相同的
EKF、GAT、信道、队列、奖励、带宽动作和速度动作；差异仅限于测量动作头接收一个
固定视界的执行期可见代理，并在训练时加入其 detached teacher 损失。持久代理不写回
滤波器状态，不替换保守协方差恢复，也不把代理值加入环境奖励。

对候选邻居 $j$，令 $H_{\mathrm{persist}}=3$。使用当前估计协方差
$\boldsymbol{\Sigma}^{\mathrm{cur}}_{i,t}$ 和候选相对测距产生的假设线性协方差
$\boldsymbol{\Sigma}^{\mathrm{lin}}_{i,t}(j)$，按当前模型的名义转移矩阵和过程噪声传播两种
假设：

$$
\widetilde{\boldsymbol{\Sigma}}^{(b)}_{i,t+H_{\mathrm{persist}}}
=
\mathbf F^{H_{\mathrm{persist}}}
\boldsymbol{\Sigma}^{(b)}_{i,t}
\left(\mathbf F^{H_{\mathrm{persist}}}\right)^{\mathsf T}
+
\sum_{\ell=0}^{H_{\mathrm{persist}}-1}
\mathbf F^\ell\mathbf Q(\mathbf F^\ell)^{\mathsf T},
\qquad b\in\{0,1\},
$$

其中 $\boldsymbol{\Sigma}^{(0)}_{i,t}=\boldsymbol{\Sigma}^{\mathrm{cur}}_{i,t}$，
$\boldsymbol{\Sigma}^{(1)}_{i,t}=\boldsymbol{\Sigma}^{\mathrm{lin}}_{i,t}(j)$。定义未归一化的
位置协方差迹减少量为：

$$
g^{\mathrm{persist}}_{ij,t}
=
\max\!\left(
0,
\operatorname{tr}\!\left(\widetilde{\boldsymbol{\Sigma}}^{(0),p}_{i,t+H_{\mathrm{persist}}}\right)
-
\operatorname{tr}\!\left(\widetilde{\boldsymbol{\Sigma}}^{(1),p}_{i,t+H_{\mathrm{persist}}}\right)
\right).
$$

这里上标 $p$ 表示位置协方差子块；该量仅由执行期估计状态、估计协方差、UWB 噪声、
消息年龄和当前可见链路字段计算。使用同一随机种子的 baseline `train` 样本拟合分位
尺度后得到 $\widehat G^{\mathrm{persist}}_{ij,t}\in[0,1]$。通信代价继续单独表示为
$\widehat C^{\mathrm{com}}_{ij,t}$，因此 measurement head 的 teacher 效用为：

$$
U^{\mathrm{persist}}_{ij,t}
=
\widehat G^{\mathrm{persist}}_{ij,t}
-
w^{\mathrm{proxy}}_{\mathrm{com}}\widehat C^{\mathrm{com}}_{ij,t},
\qquad
U^{\mathrm{persist}}_{i,\mathrm{none},t}=0.
$$

teacher 只在当前动作掩码允许的候选集合上计算，`none` 始终保留；固定测量预算耗尽后
只允许 `none`。teacher 分布温度固定为 $T^{\mathrm{proxy}}=0.25$，辅助损失系数为
$\alpha_{\mathrm{proxy}}=0.25$。teacher 目标和 actor body 上下文均停止梯度，所以该
辅助损失只更新 measurement head，不改变 GAT、带宽头、速度头或 centralized critic。
持久代理禁止读取真实位置、真实误差、真实测距、LOS/NLOS、NIS、未来通信结果和 test
数据；test 仅生成并封存，不参与代理拟合、调参或模型选择。

为验证持久代理是否能在通信代价受控的同时改善测量选择，`stabilized-v4` 将该机制作为
一个独立的版本化主线配置接入，而不改变旧分支的默认行为。该配置将
$\widehat C^{\mathrm{com}}_{ij,t}$ 实例化为包含固定 $512$ bytes 测量负载的执行期可见
候选代价，并在 teacher 中使用同一代价的 baseline-train 目标相对形式；持久信息收益仍
只使用当前估计协方差、名义预测模型、UWB 噪声、消息年龄和当前可见链路字段。通信约束
仍以单侧期望代价上限作用于 measurement head，乘子只在训练更新后更新，validation/test
不更新。`baseline_mappo`、规则基线和历史 proxy 分支不继承该改动，旧结果保持可追溯。
该版本是对“收益持久性—通信负载”假设的独立验证，不能在实验结果产生前预设其优于
baseline；正式结论仍须依据 validation 的 seed-level 配对统计。

新增分支先使用 seed=2026 的 `2/5/5/5` Smoke 检查代理计算、teacher 梯度、预算、
512 字节测量负载、EKF 数值稳定性和 `episode-v1.3`。Smoke 通过后才使用预注册的
`17181、18190、19209、20218、21227` 五个训练种子完成 `20/100/20/20` 正式验证。
最终统计以训练种子为独立单位，episode、时间窗口和单次 EKF 更新只用于机制诊断。

为保证规则强基线可复现，`stabilized-v4-corrected` 对 `budgeted_greedy_ig` 显式传入
同一 seed 的 `baseline_mappo/train` 执行期代理模型；代理模型不读取 validation 或 test，
并将来源写入运行报告。若代理模型缺失，贪心基线直接报错而不退化为始终选择 `none`。
该修正只补齐比较接口，不改变贪心规则、主方法、EKF、奖励或通信模型，也不覆盖历史
`stabilized-v4` 结果。seed=42 的 corrected development 结果显示主方法相对 GAT-MAPPO
基线的定位—负载折中有所改善，但与预算贪心基线接近，故该单种子结果只能作为开发期
描述性证据，不能推出跨初始化稳定优越性。

### 3.5.8 奖励实现

奖励严格按照第 2.13 节唯一规定的 $J_t$、$r_{i,t}$ 和 $r_t$ 计算，本节不重新定义目标函数或奖励权重。`RewardConfig` 保存各分量的归一化尺度和权重；默认尺度只允许用于 smoke/pilot，正式实验必须只用训练集估计尺度并写入实验配置，验证和测试阶段直接复用该尺度，禁止重新拟合。

每个决策窗口先计算 EKF 位置协方差迹、候选链路上的丢包、时延和 AoI 统计以及有效图的最大连通分量比例，再使用仅由训练集 95% 分位数确定并冻结的尺度归一化。候选边为空时采用第 2.13 节规定的最差有限值，不能把空集合记为零代价。能耗分量使用安全投影后的实际速度增量，碰撞分量使用三维安全检测结果。真实定位误差仅用于离线评价，不进入执行期 actor 的奖励输入。

### 3.5.9 MAPPO 更新

每个 rollout 收集状态、动作、奖励、下一个状态和动作可用性掩码。策略概率比为：

$$
\rho_{i,t}(\boldsymbol{\theta})
=
\frac{
\pi_{\boldsymbol{\theta}}
(\mathbf a_{i,t}\mid \mathbf o_{i,t})
}{
\pi_{\boldsymbol{\theta}_{\mathrm{old}}}
(\mathbf a_{i,t}\mid \mathbf o_{i,t})
}.
$$

使用 clipped surrogate objective 更新 actor。所有无人机共享同一个团队奖励和中心价值估计，但每架无人机仍按照自己的局部观测和实际动作计算策略概率比：

$$
L^{\mathrm{clip}}
=
\mathbb E_t
\left[
\min
\left(
\rho_{i,t}(\boldsymbol{\theta})\hat A_t,
\operatorname{clip}
\left(
\rho_{i,t}(\boldsymbol{\theta}),
1-\epsilon_{\mathrm{clip}},
1+\epsilon_{\mathrm{clip}}
\right)
\hat A_t
\right)
\right].
$$

每个 rollout 保存局部观测、混合动作样本、旧策略对数概率、共享团队奖励 $r_t$、中心状态价值、下一中心状态和终止标志。令 $d_t$ 表示当前 transition 是否终止，GAE 使用同一个团队 TD 残差：

$$
\delta_t
=
r_t
+
\gamma_{\mathrm{disc}}(1-d_t)V_{\boldsymbol{\phi}}(s^{\mathrm c}_{t+1})
-
V_{\boldsymbol{\phi}}(s^{\mathrm c}_t),
$$

$$
\hat A_t
=
\delta_t
+
\gamma_{\mathrm{disc}}\lambda_{\mathrm{GAE}}(1-d_t)\hat A_{t+1}.
$$

critic 的团队回报目标由同一共享优势和中心价值构造：

$$
\hat R_t
=
\hat A_t+V_{\boldsymbol{\phi}}(s^{\mathrm c}_t).
$$

该优势在当前训练批次内标准化后，分别与各无人机的局部策略概率比配对。历史基线将 `terminated` 和 `truncated` 都视为 `done`，二者的下一状态价值均置零；稳定化协议显式保存两类终止标志，并仅对时间上限截断的 TD 目标 bootstrap。GAE 递推仍在每个 episode 边界截断，不能把下一个 episode 的回报传回上一个 episode。

Actor 使用上述裁剪目标，critic 使用中心状态回报的均方价值损失；总损失同时加入熵正则以维持探索，并在反向传播后执行梯度范数裁剪。稳定化协议在不改变原学习率、PPO epoch、熵系数和裁剪阈值的前提下，可采用 Huber critic、value clipping 和目标 KL 早停，以限制异常回报造成的价值目标尖峰。实现按固定 `policy_epochs` 和 `minibatch_size` 对同一 rollout 更新，完成后清空 rollout。训练回合在环境返回 `terminated`、`truncated` 或达到预设时域时结束；当前基线按实验配置的固定训练回合数停止，不使用未记录的提前终止规则。验证和测试阶段不调用参数更新。

稳定化协议还区分环境报告奖励和优化奖励：环境返回的 `reward_total` 及其碰撞、能耗和联合目标分解保持原定义；当 `learning_reward_mode=bounded_potential` 时，PPO 使用有限幅度的 `learning_reward_total`，并将其势差项、碰撞学习惩罚上限和裁剪范围写入 manifest。该双通道只改变优化信号的数值尺度，不改变实际运动、通信、EKF 或奖励报告。动作经过环境的测量有效性检查和安全投影后，训练样本同时保存采样动作与实际执行动作。在历史 `executed` 模式中，旧对数概率按实际执行的测量动作、带宽和速度增量重算；在 `core-framework-v3` 的 `sampled` 模式中，旧对数概率严格对应采样动作，执行动作的变化只作为环境执行诊断和后续奖励/通信/定位结果的一部分，不污染 PPO 概率比。

当前可运行基线由 `src/learning/mappo.py` 提供：`MAPPOPolicy` 共享一个边感知 GAT actor，并以 `info["central_state"]` 经固定槽位编码后的联合向量作为 centralized critic 输入；`MAPPOTrainer.collect_rollout(...)` 调用环境 `reset/step` 收集 transition，`MAPPOTrainer.update()` 执行 GAE、clipped PPO 目标、价值损失、熵正则和梯度裁剪。环境本身仍只负责物理状态转移，不把训练器耦合进 `MultiUAVJointEnv`。该实现是可复现实验基线，后续可在不改变环境接口的前提下替换为更复杂的 MAPPO 训练配置。

`core-framework-v2` 另设训练专用的进度反馈，但不改变论文报告的环境奖励。令

$$
u_t=\frac{1}{N_u}\sum_{i\in\mathcal U}
\log\det\left(\boldsymbol{\Sigma}^{p}_{i,t}+\varepsilon\mathbf I\right),
$$

并只用 warm-up/train 数据拟合其中心和尺度，以及 $J_{\mathrm{com},t}$ 的中心和尺度。将两者裁剪归一化为 $\tilde u_t$ 和 $\tilde c_t$ 后，定义

$$
\Phi_t=-\tanh\left(0.35\tilde u_t+0.65\tilde c_t\right),
$$

$$
R_t^{\mathrm{learn}}
=
\operatorname{clip}\left[
R_t^{\mathrm{base}}+
0.25\left(\gamma_{\mathrm{disc}}\Phi_{t+1}-\Phi_t\right),-1,1
\right].
$$

环境仍报告原始 $R_t^{\mathrm{report}}=-(J_{\mathrm{loc},t}+J_{\mathrm{com},t}+C_{\mathrm{collision},t}+C_{\mathrm{energy},t})$ 及其分解；只有 PPO 使用 $R_t^{\mathrm{learn}}$。验证和测试复用 warm-up/train 冻结的尺度，不重新拟合。这样改善的是定位协方差和通信状态变化对策略更新的可学习反馈，不把训练奖励变化冒充为定位性能改善。

`core-framework-v2` 同时采用四回合 rollout 后更新、Huber critic、value clipping、时间上限 bootstrap 和 actor/critic 分组梯度裁剪。该组配置保持学习率、PPO epoch、熵系数和基础损失定义不变；分组裁剪仅限制数值异常对两个参数组的相互耦合，并记录裁剪前范数、裁剪因子和实际参数变化。GAE 在 episode 边界截断，不能把下一回合的回报传回前一回合。

### 3.5.10 `core-framework-v3` Proposed-only 稳定化协议

`core-framework-v3` 是当前论文主线的 Proposed-only 训练协议，而不是新增的算法分支。
它固定使用三架 UAV、Moderate、GNSS denied、50 步回合、CPU 和四个完整 episode 聚合后
更新的 MAPPO 训练配置。主方法使用本地 EKF 输出、边感知 GAT、逐候选测量动作头和
集中训练—分散执行的 MAPPO；三个规则方法只在共同 validation 场景上评估，不调用训练更新。
旧 MLP 曲线不会进入 v3 的前三张主图，但历史 MLP 目录不删除。

v3 的第一阶段只修正 `ppo_action_logprob_mode=sampled`，并保留 Huber critic、value clipping、
时间上限 bootstrap、目标 KL 早停、actor/critic 分组梯度裁剪、`rollout_episodes_per_update=4`
和现有报告奖励。若语义修正后 KL 仍异常，后续阶段才单独降低 actor 学习率；只有数值稳定而
定位反馈仍弱时，才启用预先登记的反馈对齐配置。每一阶段均使用独立输出目录，不覆盖历史
checkpoint 或结果。

v3 的五张主图严格区分训练诊断和最终性能：前三张只绘制 Proposed Method，分别报告累计
报告回报/学习回报、平均团队奖励以及 MAPPO 损失与 approximate KL；后两张才与
`local_only_fixed`、`random_cooperative` 和 `budgeted_greedy_ig` 比较 validation 的定位误差
与通信性能。训练回报允许受探索和碰撞影响而波动，不能用平滑曲线或删除 MLP 曲线制造收敛
结论；seed=42 结果只能作为开发验证，不能替代跨随机初始化的确认实验。

### 3.5.11 `core-framework-v4` 训练反馈对照

`core-framework-v4` 直接以 v3 的 Local-EKF-GAT-MAPPO 实现为起点，只增加训练期反馈
通道，不改变环境报告奖励、EKF、GAT、通信物理模型、动作空间或历史 v3 结果。v4 固定
使用三架 UAV、Moderate、GNSS denied、50 步回合、CPU、四回合 rollout 后更新，以及
`ppo_action_logprob_mode=sampled`、Huber critic、value clipping、时间上限 bootstrap 和
actor/critic 分组梯度裁剪。每个反馈臂均从同一 v3 初始参照重新训练，结果写入独立目录。

环境仍报告论文物理目标：

$$
R_t^{\mathrm{report}}
=
-\left(
J_{\mathrm{loc},t}+J_{\mathrm{com},t}
+C_{\mathrm{collision},t}+C_{\mathrm{energy},t}
\right).
$$

其中 $J_{\mathrm{loc}}$ 是协方差不确定度代价，位置向量范数 RMSE 只用于离线定位评价。
为增强该协方差变化对 PPO 的可学习反馈，环境在执行窗口更新后计算：

$$
\Delta u_t
=
\frac{u_t-u_{t+1}}{s_u+\varepsilon},
$$

其中 $u_t$ 为第 3.5 节定义的平均位置协方差对数行列式，$s_u$ 仅由 warm-up 数据确定；
因此协方差下降时 $\Delta u_t>0$。同时令 $B_t^{\mathrm{meas}}$ 为当前窗口实际执行的非
`none` 协同测量产生的等效负载字节数，令

$$
b_t=\frac{B_t^{\mathrm{meas}}}{512N_u},
\qquad
c_t^{\mathrm{load}}=\max(0,b_t-b^\ast),
$$

其中 $b^\ast$ 是从 v3 baseline 的 `train` 划分得到并冻结的平均测量负载目标。$b_t$
和 $c_t^{\mathrm{load}}$ 仅表示训练反馈所用的实际执行负载，不重定义历史 $J_{\mathrm{com}}$
或通信统计量。

v4 的三个反馈臂分别为 `loc_progress_v1`、`actual_load_v1` 和
`loc_progress_actual_load_v1`。对组合臂，PPO 训练奖励为：

$$
R_t^{\mathrm{learn}}
=
\operatorname{clip}\left[
R_t^{\mathrm{v3}}
+0.10\Delta u_t
-0.10c_t^{\mathrm{load}},-1,1
\right].
$$

单因素臂只保留对应的一项附加反馈。$R_t^{\mathrm{learn}}$ 与 $R_t^{\mathrm{report}}$
分离记录；validation/test 不重新拟合 $s_u$ 或 $b^\ast$，test 只生成并封存，不能参与
训练、模型选择或图表选择。v4 的前三张主图只展示 Proposed 的训练诊断，后两张在共同
validation 场景上与 `local_only_fixed`、`random_cooperative` 和 `budgeted_greedy_ig`
比较。由于 seed=42 只是开发实验，v4 不提供跨随机初始化稳定性结论；若反馈臂仍不能
改善 RMSE—通信权衡，下一步应检查有效通信边、协同 UWB 更新持久性和 GAT 候选表示，
而不是继续搜索反馈系数。

### 3.5.12 `core-framework-v4-delta-scale` 相邻窗口定位反馈尺度修正

`core-framework-v4-delta-scale` 是对 v4 `loc_progress_v1` 反馈尺度的单因素修正。
它不改变环境报告奖励、RMSE、$J_{\mathrm{loc}}$、$J_{\mathrm{com}}$、本地 EKF、GAT、
通信模型或动作空间，只修正训练专用定位进度信号的归一化分母。对同一 episode 的相邻
决策窗口，先计算原始不确定度下降量

$$
\Delta u^{\mathrm{raw}}_t=u_{t-1}-u_t,
$$

其中 $u_t$ 为前文定义的各 UAV 位置协方差对数行列式平均值。episode 边界不形成差分，
因此新回合的初始协方差不会被错误地与上一回合末状态比较。仅使用 warm-up 的这些相邻
窗口差分拟合稳健尺度

$$
s_{\Delta u}=\max\left(1.4826\,\operatorname{MAD}(\Delta u^{\mathrm{raw}}_t),10^{-3}\right),
$$

并在训练中使用

$$
\Delta\tilde u_t=
\operatorname{clip}\left(\frac{\Delta u^{\mathrm{raw}}_t}{s_{\Delta u}},-3,3\right).
$$

该尺度的来源、MAD、中位数、样本数和下限保护标志写入归一化统计文件；每个窗口同时
记录 $d_t$、裁剪前后的反馈和使用的尺度。validation/test 只读取 warm-up 冻结的
$s_{\Delta u}$，缺少该尺度时直接拒绝运行。训练奖励仍与报告奖励分离：只有 PPO 的
learning channel 使用该反馈，报告奖励及其物理分解保持历史定义。该修正用于检验“绝对
协方差尺度被误用为相邻变化尺度”这一已确认的反馈链路问题，不能单凭训练曲线改善宣称
定位性能已经改善；最终判断仍以 validation 的 RMSE、P95、有效协同更新和实际通信代价
为准。

### 3.5.13 `core-framework-v10-safety` 联合安全执行与终止归因

`core-framework-v10-safety` 只修正安全执行的时序语义和训练曲线中的终止归因，不改变
报告奖励、EKF、GAT、MAPPO、信道、队列、物理噪声或碰撞惩罚。新协议使用
`safety_execution_mode=joint_swept_v1`；历史结果继续使用 `legacy`。在一个决策窗口内，
环境先保存所有 UAV 与动态障碍物在时刻 $t$ 的状态，再由所有请求动作共同计算候选位置，
预测障碍物的时刻 $t+1$ 位置，并联合检查 UAV--UAV 端点距离、线段交叉、UAV--动态障碍物
扫掠路径和边界越界。冲突 UAV 进行对称安全投影后统一提交状态，随后更新障碍物并执行最终
碰撞检查。日志同时保留投影前风险、投影后物理碰撞、保护距离违反和不可避免动态碰撞，
避免把安全投影与真实碰撞混为同一事件。

训练日志和图表汇总使用固定的终止分类优先级：碰撞标志或
`termination_reason=collision` 归为 `collision`；非碰撞的 `terminated=true` 归为
`early_termination`；`truncated=true` 或 `termination_reason=time_limit` 归为
`time_limit`；短于最大步数但没有可靠原因的回合归为 `unknown_short`；其余为 `normal`。
奖励曲线中的 `X` 仅表示碰撞终止，空心三角形仅表示非碰撞提前终止，正常的时间上限回合
不标记，`unknown_short` 不作原因推断。每个 episode 最多归入一种类别，并将回合长度、
最小安全距离、碰撞对象和安全投影次数写入 `training_termination_attribution.csv`。

该协议仍保留单回合累计报告奖励、PPO 学习奖励和回合长度的原始曲线；标记用于解释尖峰，
不删除异常回合、不平滑替代原始证据，也不据此宣称 MAPPO 已经解决安全问题。性能结论
继续使用 validation 的定位误差、通信负载和更新质量，test 只生成、封存并进行独立 schema
检查。v10 的 3-UAV、seed=42 结果是开发性验证，不代表跨随机初始化的稳定性。

### 3.5.14 `core-framework-v11-comm-occluder` 纯通信遮挡区域

为避免通信机制与物理安全语义混淆，v11 将 `screening_moving_nlos_screen` 作为纯通信
遮挡区域处理。它保留在链路可见性计算中，因此仍能影响 LOS/NLOS、丢包、时延和 AoI；
但 `physical_collision_enabled=false`，不会进入 UAV--障碍物碰撞检测、保护距离违规、
安全投影或实体动态障碍物观测。真实建筑物和动态车辆仍按实体障碍物处理。

该配置与 v10 使用独立输出目录，v10 及更早结果仍保留原语义。v11 同时开启实体动态
障碍物运动特征和 `joint_swept_v1` 联合安全检查，因而策略可以看到实体障碍物的接近
趋势，而不会因通信遮挡几何被重复施加物理碰撞约束。该语义修正只改变模型定义和安全
归因，不改变通信遮挡本身对链路的影响，也不应单独被解释为定位性能已经改善。

## 3.6 纯 Python 仿真环境接口与执行流程

### 3.6.1 环境接口

环境由 NumPy 驱动的状态转移、障碍物、传感器、信道、拓扑和 EKF 模块组成，并向 MAPPO 提供统一交互接口：

```python
obs, info = env.reset(seed=seed, scenario_config=config)
obs, reward, terminated, truncated, info = env.step(action)
```

`env.step` 的完整多智能体输入是 `{uav_id: action}`；上面的动作字典表示单个无人机动作，环境也保留将其解释为首架无人机动作的简写形式。环境实现与论文模块的对应关系为：`src/simulation/motion_model.py` 负责三维速度增量和安全投影，`src/simulation/obstacle_model.py` 负责 AABB/球体、碰撞和 LOS，`src/simulation/sensor_model.py` 负责 IMU/UWB/视觉/GNSS，`src/synthetic/channel_model.py` 负责统一链路预算，`src/simulation/topology.py` 负责潜在—候选—有效边，`src/simulation/ekf.py` 负责本地 EKF 和 CI，`src/simulation/graph_interface.py` 负责 GAT 张量和动作掩码，`src/learning/normalization.py` 负责训练统计量归一化，`src/learning/gat.py` 负责 GAT 前向编码，`src/learning/mappo.py` 负责独立的 MAPPO rollout 和参数更新，`src/simulation/environment.py` 负责 `reset/step` 闭环。

动作结构固定为：

```python
{
    "measurement_uav_id": ..., 
    "bandwidth": {"uav_01": 0.5, "uav_02": 0.5},
    "delta_v": [dvx, dvy, dvz]
}
```

局部 observation 至少包含本机 EKF 状态和协方差、目标特征、局部障碍物特征、潜在/候选出边质量、候选入边质量、潜在出边特征、UWB 可用性、有效邻居消息、队列、AoI、当前带宽/速率和动作掩码。`node_features` 只包含本机及成功接收邻居的估计节点；未成功接收的候选节点使用显式占位特征，不能把真实邻居状态或真实距离传给 actor。测量掩码对应候选入边源节点，带宽掩码对应潜在出边。centralized critic 所需的联合状态和全局图通过 `info["central_state"]` 提供，但不进入执行期 actor observation。

动作处理顺序固定为：读取时刻 $t$ 的候选图和动作掩码 → 检查协同测量入边 → 在潜在出边上归一化逐边带宽 → 执行速度和安全投影 → 完成 $[t,t+\Delta t_{\mathrm{dec}})$ 的运动、传感器、信道和 EKF 更新 → 返回时刻 $t+1$ 的 observation、reward 和 info。环境不内置 MAPPO 训练器；`MAPPOTrainer` 作为独立学习模块调用该接口，并保持 observation、action、reward、central state 和 mask 的接口稳定。

在本规范中，接收方无人机 $i$ 的 `measurement_uav_id=j` 表示选择 $j$ 的已到达成功邻居消息/UWB 伪观测进入本地 EKF；选择 `none` 表示本窗口不使用邻居协同测量。当 `measurement_overhead_enabled=true` 时，非 `none` 的选择还会按第 3.3.3 节产生一个 $512$ bytes 的等效队列包；这只改变通信资源状态和派生 KPI，不改变测量本身的物理含义。当前没有 `relay_uav_id` 字段或中继动作头；多跳转发只有在未来建立明确的路由、转发队列和端到端 AoI 模型后才能扩展。逐边带宽和轨迹动作仍会直接改变下一窗口的通信和运动状态。

### 3.6.2 单个 episode 流程

~~~text
初始化无人机、障碍物、目标、传感器和通信参数（当前基线为决策频率单步更新）
    ↓
执行上一决策窗口动作并更新运动状态
    ↓
更新动态障碍物和三维安全状态
    ↓
生成真实运动状态和传感器观测
    ↓
时间同步、坐标转换和观测质量检查
    ↓
按配置写入测量等效队列负载（可选）
    ↓
计算三维 LOS/NLOS、信道、队列、丢包、时延和 AoI
    ↓
更新成功消息的离散到达缓冲并丢弃过期消息
    ↓
更新潜在图、候选图和有效通信图
    ↓
执行 Distributed EKF
    ↓
构造 GAT 节点/边特征并生成图表示
    ↓
读取训练集冻结的测量预算，按剩余预算更新测量动作掩码
    ↓
MAPPO actor 接口生成协同测量、逐边带宽和轨迹动作
    ↓
预算分支消耗合法非 `none` 测量单位并记录 `measurement_budget.csv`
    ↓
计算归一化联合代价、奖励和安全事件
    ↓
保存状态、观测、图特征、动作和评价数据并进入下一决策时刻
~~~

### 3.6.3 训练和验证隔离

训练、验证和测试使用不重叠的回合场景编号与随机流；场景类别和物理参数可相同，不能将编号隔离误写为所有物理参数不同。历史协议允许开发选择；corrected-v2 固定奖励权重、训练回合数与候选数，validation 仅按预先登记规则选择 R/P/T 开发候选，test 封存且只允许独立 schema 检查，不用于选型和绘图。

训练过程中不使用测试集真实位置、未来障碍物轨迹或未来通信状态。ground truth 只用于仿真环境转移、离线评价和训练期 critic 可访问的全局信息。

当前代码已按本章规范实现：消息到达缓冲在 `env.step` 中释放，动作接口删除独立中继动作，所有智能体共享团队奖励和单一 centralized critic，连通性按有效图无向化后的最大连通分量计算，actor 与 critic 均复用训练阶段拟合的归一化统计量；预算分支的预算只由 baseline 训练划分确定，并通过动作掩码和 `measurement_budget.csv` 记录。后续新增模块必须继续通过同一接口和测试，不得重新引入公开数据校准依赖。

## 3.7 复杂度、日志和可复现性

每个决策时刻记录：

- EKF 状态和协方差；
- 有效观测和有效邻居数量；
- GAT 节点、边和注意力输出；
- MAPPO 动作、逐边带宽和动作掩码；
- 每个奖励分量；
- 碰撞、越界、丢包、时延和 AoI；
- 随机种子、场景编号和模型版本。

实验结果由 ground_truth.csv、estimates.csv、links.csv、packet_events.csv、graph_features.csv、obstacles.csv、actions.csv 和 rewards.csv 共同复现；训练过程还保存 `training_metrics.jsonl`、`attention_maps.jsonl`、`normalization_stats.json`、`manifest.json` 和 checkpoint。算法网络层数、隐藏维度、学习率、batch size 和训练轮数属于实验配置，统一记录在 4_experimental_protocol.md，不在本章重复固定。

复杂度分析按模块统计：Distributed EKF 的计算量随状态维度和当前有效观测维度增长；GAT 的计算量随当前有效通信边数量、节点特征维度和图层数增长；MAPPO 的训练计算量随 rollout 样本量、智能体数量和 actor/critic 网络规模增长。执行期只处理本机及当前有效邻居信息，通信和图计算量不随不可见的全局真值直接增加。

### 3.7.1 seed=42 的 corrected-v2 开发协议（2026-09-03）

本协议保留本地 EKF、GAT、MAPPO 和未知交叉相关下的协方差保护，不以改权重或删除保护获得表面改善。$J_{\mathrm{loc}}$ 是经裁剪的协方差不确定度代价，不是位置 RMSE；裁剪饱和和协方差恢复都可能使状态均值改善不能即时改变该代价。邻居消息之间的 CI 也不意味着已维护本机与邻居的完整交叉协方差。

通信量按五种不同口径记录：链路风险、包含固定包负载的候选代理代价、策略每窗口期望代价、实际选中动作代理代价、实际字节数；$J_{\mathrm{com}}$ 则是原有归一化通信目标，不能与其中任一项直接等同。P 方案让教师、约束及日志共用候选代价函数，并在教师中只扣除超过 baseline-train 每窗口目标的正差。该减法是启发式引导，并非严格的预算推导，也不证明已消除重复成本。教师和通信辅助梯度只进入 measurement head。

R 使用与固定 VO 噪声生成器一致的标称观测协方差、历史教师和每回合一次更新；P 只改变教师代价口径；T 在 P 基础上每四个完整回合更新一次，学习基线也同步改为四回合，预算与目标从该组基线 train 重算。100 回合分别对应 100 或 25 次更新。GAE 在回合边界截断，余数回合单独更新。乘子仍按每次更新调整，因此更新频次变化不是纯粹的方差干预，必须同时报告优化步数和乘子调整次数。

外生随机数按场景、窗口、UAV/边及噪声类型索引；历史重放继续使用旧随机流。MLP 槽位对照保留候选与接收角色、身份、边及存在掩码，不进行图消息传递；旧均值池化编码器保持可加载，不与新版本结果混用。固定控制 B 组只作冻结策略干预诊断，承认训练与干预的状态分布差异。

所有性能结论限定为一个训练 seed 下的配对场景描述。训练 0/20/40/60/80/100 回合的固定场景监测不选择 checkpoint、不提前停止、不修改随机状态；规则方法没有训练 loss。无接受更新时质量比例为 NA，而更新次数为零；完整方法不必在协同负载上低于无协同测量的独立方法。

### 固定50步目标对齐（2026-09-06）

有限时域协议将50步定义为任务终点；到达时域末端和碰撞终止时，下一状态价值均为零。GAE在episode边界截断。历史稳定协议的时间上限bootstrap只保留用于历史复现，不适用于本协议。归一化剩余时间 $\tau^{\mathrm{remain}}_t$ 接入actor body和centralized critic，不改变GAT消息传递；采样观测随rollout保存，新输入维度要求从头训练。

`report_direct`逐窗口使用原始报告奖励，完整绕过势函数、附加反馈、碰撞限幅和最终裁剪。它与现有delta-scale奖励构成两组有限时域对照；报告指标和物理模型相同。单种子开发结果不能解释为跨初始化稳定性。固定前五个validation场景另报告共同时间前缀RMSE，避免提前终止的时长混杂。

### `core-framework-v5` 单种子稳定化开发协议（2026-09-06）

在不改变本地 EKF、GAT、通信信道、动作空间和报告奖励定义的前提下，
`core-framework-v5` 将训练反馈拆分为可归因的单因素阶段。`report-clean`
令 $R_t^{\mathrm{learn}}=R_t^{\mathrm{report}}$；`bounded-collision` 仅对 PPO
学习通道设置碰撞惩罚上限，环境报告仍保留真实碰撞代价；`loc-progress-v2`
在同一 episode 内使用已有 $u_t$ 计算相邻窗口的原始下降量，并用 warm-up
拟合的 $s_{\Delta u}$ 归一化；`actual-load` 增加实际非 `none` 测量负载相对
baseline-train 目标的训练反馈；组合臂同时启用后两项。所有附加反馈只进入
$R_t^{\mathrm{learn}}$，不改写 $R_t^{\mathrm{report}}$、$J_{\mathrm{loc}}$、
$J_{\mathrm{com}}$ 或 RMSE。

各阶段均固定为 seed=42、3 UAV、50 步、`20/100/20/20`，先通过
`2/5/5/5` Smoke；规则方法只在 validation 评估，不执行 MAPPO 更新。训练图
使用 train，性能图使用 validation，test 只封存并做 schema 检查。实际测量
负载按 $512$ bytes 等效包记录，且与 `none` 动作区分。由于单因素阶段用于
归因，组合阶段的定位收益和通信负载必须同时报告，不能只凭累计奖励或 loss
下降宣称方法有效。

本协议的最终开发结论仍限定为 seed=42 描述性证据；只有后续多个独立训练
seed 的配对统计才能支持跨初始化稳定性结论。

### `core-framework-v8`：报告奖励与有限时域的有序验证（2026-09-06）

为避免训练曲线同时混入历史碰撞限幅、势函数和实际负载反馈，v8 将奖励验证拆为两个有明确
顺序的训练臂。`reference` 臂令

$$R_t^{\mathrm{learn}}=R_t^{\mathrm{report}}.$$ 

其中报告奖励保持本章的物理定义，真实碰撞惩罚、通信目标和能耗项均不改变；学习环境不再对碰撞
分量单独截断，也不添加训练专用势函数或负载反馈。这样得到的训练曲线可直接解释为 MAPPO 对
论文报告目标的优化表现。该臂同时将 50 步任务建模为有限时域任务，令时间上限状态不 bootstrap，
并将归一化剩余时间提供给 actor 和 centralized critic；GAE 仍在每个 episode 边界截断。

若 `reference` 臂在固定场景上没有定位改善，才启用 `loc-progress-v2`。该臂使用
`report_plus_feedback`，其基础项仍是 $R_t^{\mathrm{report}}$，仅增加同一 episode 内相邻窗口
协方差对数行列式的下降量：

$$d_t=u_{t-1}-u_t,\qquad
\Delta\tilde u_t=\operatorname{clip}(d_t/s_{\Delta u},-3,3).$$

其中 $s_{\Delta u}$ 只由 warm-up 的同 episode 相邻差分按稳健 MAD 规则拟合；该臂不使用 RMSE、
真实位置或 validation/test 数据作为训练反馈。$R_t^{\mathrm{learn}}$ 中的附加定位项与报告奖励
分开记录，$J_{\mathrm{loc}}$ 仍解释为协方差不确定度，RMSE 仍是外部定位性能指标。v8 仅使用
seed=42 作为开发验证，不能据此声称跨随机初始化稳定。
