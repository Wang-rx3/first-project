# v7：部分 GNSS 可用的单目标协同定位基线

## 冻结范围

本轮选择方案 B：UAV 0/1 获得 noisy GNSS，UAV 2 GNSS denied。只研究 UAV 2 在 0、1、none 中选择参考；不是完整多智能体 MAPPO。没有神经网络、优化器更新或奖励修改。

两种几何各使用种子 `20267101`、`20267102`：

| 场景 | UAV 0 起点 | UAV 1 起点 | UAV 2 起点 | 航点 |
|---|---|---|---|---|
| train_noncollinear | (40,40,22) | (40,90,42) | (65,65,32) | 各自起点向 x 正方向 35 m |
| train_collinear | (40,125,28) | (50,125,28) | (80,125,28) | 各自起点向 x 正方向 35 m |

空旷、无复杂遮挡；GNSS denial 是预先指定的可用性条件。noncollinear 为两个不同方向的三维参考，collinear 为退化几何对照。不同场景共用两个噪声种子，实际只有两个独立随机实现，不把 120 步或不同策略计作独立重复。

原固定导航参数、dt=0.2 s、horizon=120、动作轴/范数约束、速度/加速度上限、严格停车标准保持不变。v4 决策步为 2,6,...,118；每机保留 30 次预算，参考机选择 none 因此实际不消耗测量预算。TTL=0.6 s，UWB 每对更新间隔=0.8 s；请求扣预算，接受更新另计。

初始化独立抽取完整 15 维先验误差，位置每轴方差 4，其余每轴 1；跨机初始协方差为零。冻结 JSON 保存每个误差实现。详见 [数学与观测核查](../knowledge/partial_gnss_fusion_v7_basis.md)。不拟合训练归一化。

## 配对记录回放

先运行 4 条 None 物理轨迹，所有 UAV 只用自身 EKF 固定导航。保留实际 IMU/VO/GNSS、两路被动 UWB、通信消息及真值评估分区。选择策略为 None、Fixed0、Fixed1、Random-peer；Fixed 仅选指定合法参考，Random 只在合法参考中抽取，没有合法候选时选 none。

三个估计器分别是历史恢复 prior、独立标量测量 CI、维护交叉协方差的中央联合 EKF。每个估计器使用相同策略选择及预算；联合 EKF 额外拥有三机当前私有本地传感器和交叉协方差，仅为信息参考。分布式回放只用到达的旧消息；核对参考机滤波历史及消息传播结果与记录一致。

上限为 48 组×120 步=5760 组滤波步，4 条物理轨迹=480 物理步。回放使用 None 的固定广播/队列路径；请求按 512 byte-equivalent 计开销，**未重算请求导致的队列差异**，不能称为完整通信反事实。

## 预先定义的闭环门限

只有测量 CI 的 Random-peer 对 None 在两个几何的 pooled MSE 均有改善、至少 3/4 个场景种子块改善、最终平均位置协方差迹小于初始 12，且所有状态有限、PSD 合法、完整协方差迹低于 1e4 时，才进入候选闭环。

闭环最多 4 块×4 策略=16 条×120 步=1920 物理步；含记录产生总计上限 2400 物理步。只把候选接入隔离环境的 UAV 2；当前一步融合影响下一步导航。通信队列按实际请求在线运行；0/1 仍只使用自身本地滤波，不接受协同 UWB。同一块使用同一初态及 keyed 噪声实现，策略改变轨迹后的观测均值随物理状态变化，不强称后续轨迹相同。出现碰撞、非法状态、数值失败时停止，不替换种子。

这只是进入有限闭环检查的工程门限，不能替代统计一致性、论文实验或 MAPPO 科学准入门限。原 v3/v4 的邻居差异门槛未修改。

## 实际运行路径

解释器：`D:\research project\.venv\Scripts\python.exe`；工作目录：`D:\research project`。

已冻结设计，日后在 PyCharm 使用模块 `src.experiments.partial_gnss_baseline_v7`，参数 `--run-id USER_V7_REPLAY_001`；不要再传 `--freeze`。同名输出目录存在时拒绝运行。

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_partial_gnss_baseline_v7.py tests/test_partial_gnss_fusion_v7.py -q
.\.venv\Scripts\python.exe -m src.experiments.partial_gnss_baseline_v7 --run-id USER_V7_REPLAY_001
.\.venv\Scripts\python.exe -m src.analysis.partial_gnss_report_v7 --run-id USER_V7_REPLAY_001
```

这些命令用于复查隔离基线，不代表本轮已批准正式训练或已解决失败。当前实际结果为 20267101 两几何完成、20267102 两几何安全停止；候选闭环未执行。恢复运行 `bounded_resume_s20267101_102` 的 `manifest.json` 计入 341 累计物理步、2880 本次回放步；首次失败运行另有 1440 回放步，累计 4320。完整回放重做不增加物理轨迹。详细失败链条见 `knowledge/partial_gnss_v7_findings.md`。

配置在 `data/configs/three_uav_partial_gnss_baseline_v7_20261012.json`；JSON 记录和汇总在 `data/synthetic/three_uav_partial_gnss_baseline_v7_20261012/<run-id>/`；CSV 在对应 `logs/preliminary/`；图在对应 `plots/preliminary/`。本轮实际执行 run-id 及量化结果由结果报告提供。

## 下一阶段边界

相对/绝对 RMSE、NEES/NIS、位置谱、UWB 请求/接受、GNSS 有效/接受、开销、安全和停车都报告。短时序相关数据只能作描述性诊断，不能证明普遍一致性。标量 CI 的条件假设、退化几何、延迟加速度模型及先验宽度仍是限制。是否进入邻居选择基线比较根据实际结果判断；正式奖励设计、多目标选择和 MAPPO 训练仍需用户确认。
