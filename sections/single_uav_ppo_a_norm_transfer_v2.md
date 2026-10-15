# A-only 全时域归一化与策略迁移对照 v2

日期：2026-10-11。保留上一轮所有数据；本轮目录名 `single_uav_ppo_a_norm_transfer_v2_seed2026_20261011`。无障碍任务，B 暂停，无 BC/DAgger/教师样本/辅助控制算法，不提交或推送。

## 证据与实验范围

旧统计把真实零目标增量的 X 分量编码为约 -1.627；源 Actor 在目标处、速度零时仍输出范数约 0.162 m/s 的动作。P 旧 online 边界步平均 reward -0.598，比普通步更负；normalized advantage 却平均 +0.430，普通步 -0.342，晚期步 +0.557。离线回放显示最终 P 对晚期旧动作的 log-prob 平均提高约 0.015，而早期降低约 0.153。此为分布、估值和时域信用问题的证据，不是边界总体正奖励或因果机制已经证明。

本轮只隔离统计与网络迁移，不修改原 reward 系数、reward potential、bootstrap_time_limits、时域、动作变换、基础投影或动力学。旧实现和上一轮完整结果做 SHA 保护。原严格停车条件为真实距离 ≤1 m、速度 ≤0.5 m/s、连续 5 个决策步、无碰撞，训练与评估继续使用原 FoundationEnv；有实体障碍的任务不运行。

## 登记的顺序与划分

1. 旧 CSV/权重纯离线回放与完整分布审计；不重跑历史轨迹。
2. 六个边界面向内/向外各一次，共 12 个单步探针，验证真实速度重置、加速度冲量及记账一致性。
3. 12 个独立校准训练场景，覆盖三个坐标轴的正负方向、4 m 与 8 m 两种距离；与原 6 个 PPO 训练、8 个新普通验证、4 个新挑战验证的几何、ID、噪声 seed 均分离。校准不使用任何旧或新验证数据。
4. 冻结源 P/G Actor，在旧短统计下分别执行 12×120 个校准步。只使用旧 Strict 环境的无障碍完整时域观察路径，成功由 Reporter 只读记录，不使用校准 reward 学习。120 步时限后停止，任何提前碰撞/真实终止均失败停止，不在 terminal 后继续推进。训练环境仍保留严格成功早终止。共 2880 transition、2904 observation（每轨迹含 reset 和 120 个 post state）。
5. 拟合全时域统计，冻结原始数据、特征布局和哈希；完成训练专用零目标/零估计速度探针，锁定 R 许可及 primary，随后才读取验证结果。
6. C/N/G/获许可的 R 各完成一次同预算 720 步 PPO；仅初末评估，不追加回合或选择最佳 checkpoint。

## 新统计与兼容性

Actor 42 维、中央输入 174 维。估计速度 `3:6`、世界目标增量 `30:33` 使用物理零中心，各向量组三轴共用同一正尺度，保留方向和符号；尺度为训练联合 RMS、固定 floor（速度 6、目标 2.4）、训练最大绝对值除以 8 三者最大。其他特征使用全时域 mean/std，尺度至少为训练最大中心化绝对值除以 8；常量尺度至少 1。clip=10 保留。中央输入的相应位置从真实 graph 节点核查；不凭注释推断。训练数据覆盖检查是校准内检验，不能写成验证分布保证。

新旧统计在相同 raw 状态上比较归一化、first-GAT 合成前激活、embedding、动作分布。first-GAT affine 重参数只作为离线机制探针；仅对新旧都没有裁剪的区域验证等价，不进入 PPO。裁剪不可逆，不宣称全域函数保持。

## 四个受控条件

| 条件 | 统计 | Actor 初始化 |
|---|---|---|
| C | 原短统计 | 完整源 P Actor，复现前轮 P |
| N | 新全时域统计 | 与 C 完全相同的 23 个继承 Actor 张量 |
| G | 新全时域统计 | 首层 GAT 使用原 fresh seed，其他 18 张量与 N 一致 |
| R | 新全时域统计 | 与 N 一致，只将 motion mean head 的 weight/bias 两张量置零 |

C–N 隔离统计变更，N–G 隔离首层迁移，N–R 隔离输出模块迁移；不称完整 factorial。R 保留源 encoder、Actor body、其余 heads 和 logstd，置零只发生在初始化，在线正常采样、PPO 更新，不按目标修改执行动作。

只有新统计下 6 个训练专用零目标/零估计速度探针的源均值动作范数全都 >0.05 m/s，才许可 R 训练并指定 primary=R。否则不训练 R，primary=N。选择记录在所有 validation 前冻结；不能根据验证结果换 primary。

每个条件使用相同 fresh Critic seed、Adam、空 rollout、训练顺序、采样/mini-batch RNG 和噪声流。每条件 720 online transition、6 update、4 epochs/minibatch64、48 实际 Adam step，学习率 1e-4，gamma .99、GAE .95、PPO clip .2、entropy .005、全局梯度限幅 .5、sampled log-prob。五个辅助全关；dt .2、速度12、加速度3、增量范数 .6、120步/24s、原边界及 legacy 基础安全投影不变。旧 optimizer/rollout/教师样本不导入。

## 评估、门槛和停止

每条件初末各评估原 6 train、8 fresh normal validation、4 challenge。Challenge 为 2 个静止起步及 2 个朝向目标的 1.2 m/s 起步，严格停车判据不变，用来排除靠初始低速度 coasting 被动过关。其数据不参与统计、许可、选择或调参。

Primary 的普通门槛：train ≥4/6、normal validation ≥5/8、无碰撞/接触、验证最终距离中位数 ≤2 m；train/validation 边界尝试与投影各自 ≤floor(实际执行步数×1%)。Challenge ≥3/4，且 stationary/fast 各至少1/2，并无碰撞/接触。任一门槛失败即停止；全部通过也仅作后续阶段候选，本轮不开障碍、多种子或多 UAV 性能实验。

最大研究交互为 23052：校准2880、PPO最多2880、评估最多17280、边界探针12。未获许可的 R 不用其他 arm 填补。工程测试单列，离线 forward 不计环境交互。每次 update 保存参数/optimizer/statistics、逐记录 GAE 与对固定训练状态的动作/采样 likelihood 漂移；保存实际/名义加速度、边界候选与轴向标志、reward组成及初末效果。

## 实际结果

实际校准24条完整轨迹、2880步、2904观测；四臂各720真实PPO步、6update、48Adam。新统计校准内姿态协方差及目标Y/Z裁剪为0，旧统计分别74.38%及59.81%/58.51%。6个训练专用N未重置探针均>0.05，验证前许可R并锁定primary。

最终普通验证C/N/G/R均0/8，距离中位数91.64/52.11/76.72/21.15 m，边界投影443/12/43/0（各960步）；train0/6、challenge0/4，均无碰撞、超时100%。R初始普通验证8/8但challenge0/4，属于低速恒速滑行；更新后没有可靠制动，门槛失败，不释放下一课程。研究步数22641，工程/离线步0；18项测试、81项独立检查、权重/Adam48步复核及符号审计通过，208项历史SHA不变。详见对应日志目录`norm_transfer_summary.md`与`independent_artifact_audit.json`，实际统一配置见对应data目录`effective_unified_configuration.json`。未新增论文数学符号，未提交或推送。
