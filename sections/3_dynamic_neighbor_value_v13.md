# V13：动态异构参考质量下的邻居价值验证

日期：2026-10-13。可学习性结论为**有条件通过**，正式 MAPPO 训练准入仍**未通过**，`training_ready=false`。已出现跨四个确认种子的局部可预测信号，但没有获得稳定的完整闭环定位优于 Random 的证据。本轮没有训练、部署、提交或推送。

## 1. 实现及冻结设计

仅新增隔离 V13 模块。原 Distributed EKF、测量 CI 的 0.05 网格与公式、UWB、固定导航、传感器融合公式、奖励、GAT 与 PPO 参数均未改变。

使用街谷启发的三维周期**虚拟质量场**，由实际传感器位置和链路中点驱动。GNSS 标准差为 0.5/1.5/3 m；采样使用同一标准正态噪声，原 EKF 实际使用每包声明的 `std**2`，不把真值当 GNSS。链路额外衰减 0/6/12 dB，接入等待 0.02/0.22/0.42 s，进入原 SNR、丢包和队列逻辑；总时延求和后按原规则量化，TTL 精确比较不变。场不是建筑射线模型，也未经真实城市标定。

共同场偏移 0/10 m 在种子间平衡，不按 UAV 身份设优劣。所有 64 个记录中两架参考机的声明质量均在轨迹中交换优劣；实际 GNSS 噪声与声明标准差的逐包索引重构误差为零。GNSS/RF 共同变化，不能据此分别归因。

开发种子：2026101301、2026101302、2026101303、2026101304；确认种子：2026101341、2026101342、2026101343、2026101344。每组两种原几何、四种策略，合计 64 条 ×120 步 =7,680 个实际主实验步。种子是独立重复，几何和窗口是重复区组。全部种子/参数在执行前冻结，确认未用于选择规则或调参。

三机前五步静止，启动不测量、不补发。原窗口 6、10、…、118，原每机预算 30。原真实请求、通信队列、时延、协同融合及估计反馈驱动导航持续闭环，没有使用 None 固定队列回放替代。

附加 OFF 六步配对证明传感器、真实运动、估计/P 和队列逐项完全一致。旧失败种子 2026101324 的 ON 前缀仍在第 3 步触发 `stationary_start_unsupported`，记录保留。附加物理尝试实际共 **15** 次，上限 18；原前缀汇总漏计位于 `frame.failure` 的失败尝试，原 14 次报告未覆盖，补充证据单独更正。

## 2. 可获得信息与 Greedy

策略只读 UAV2 当前本机估计/P、实际到达邻机估计/P、RSSI、原预决策链路时延、AoI、估计距离和方向。新增 GNSS 标准差及测量时间随实际发送消息携带，与发送前载荷对齐；接收端不读取邻机当前私有 EKF、未到达消息、场相位、偏移或真值。

GNSS 质量头为 16 字节概念开销；原 512 字节等效队列包保持，ON 按 496+16、OFF 按 512+0 记账。**没有真实编解码器，不声称完整 15 状态及 15×15 协方差能以 496 字节序列化。**等效开销不等于部署带宽证明。

V11 Greedy 完全复用原规则。V13 规则复用原 CI 零创新协方差探测，得分为：

```text
sqrt(position_trace_before) - sqrt(position_trace_after)
    - 0.01 m * (1 + native_prelink_delay_s / 0.2 s)
```

none=0，只选正分合法邻居，等分按原邻居顺序。单位为 m，不作数据拟合或归一化，不叠加任意几何/质量正加分。GNSS 质量通过真实邻机 P 影响探测，报告其额外状态解释力。根迹下降只是**不确定度代理**，不是已证明的真实误差期望下降。

## 3. 独立确认闭环结果

表中每行汇总四个独立种子；RMSE 按平方平均后开方。None 仍发送环境固定参考广播；请求开销只计协同测量请求，全部原生广播及质量头等效开销另保存在审计中。

|几何|策略|UAV2绝对RMSE m|集群相对RMSE m|请求|更新尝试|实际融合|严格停车/4|
|---|---|---:|---:|---:|---:|---:|---:|
|共线|None|2.500681|2.101268|0|0|0|0|
|共线|Random-peer|1.880723|1.622100|114|105|8|1|
|共线|V11 Greedy|1.867406|1.610901|112|109|8|1|
|共线|V13 Greedy|1.871627|1.614845|8|8|8|1|
|非共线|None|2.500681|2.159453|0|0|0|0|
|非共线|Random-peer|1.306372|1.270454|116|106|24|2|
|非共线|V11 Greedy|1.337533|1.285913|112|109|19|2|
|非共线|V13 Greedy|1.308777|1.264215|22|22|22|2|

精确全指标以 `report/grouped.csv`、`interpretation/per_seed_localization_communication.csv` 为准。

V13 相比 Random：共线绝对/相对 RMSE 改善 **0.484%/0.447%**；非共线绝对 RMSE **恶化 0.184%**、相对改善 **0.491%**。没有满足执行前冻结的两几何双指标各 5% 门槛。四种子中仅两个满足跨几何双指标平均正收益，未达原定三个；最大个例伤害仍低于 5%。

确认组请求总量从 **230 降至 30（减少86.96%）**，融合从 32 至30。V13 的主要实际优势是避免原 CI 大量拒绝或权重=1的无收益请求。请求窗口不同，**不能把这种优势归因于纯邻居身份优化**，也不能用融合率 100% 代替定位性能验收。

逐确认种子结果（单元为绝对 RMSE m；完整 NEES/NIS、相对 RMSE、预算、停车、最终误差和通信日志见 CSV）：

|几何|种子|None|Random|V11 Greedy|V13 Greedy|Random请求/融合|V13请求/融合|
|---|---:|---:|---:|---:|---:|---:|---:|
|共线|2026101341|2.0811|2.0521|2.0600|2.0742|29/2|2/2|
|共线|2026101342|2.1959|2.0941|2.0700|2.0794|29/2|2/2|
|共线|2026101343|3.7125|2.1888|2.1977|2.1888|27/2|2/2|
|共线|2026101344|1.4415|0.8726|0.7685|0.7715|29/2|2/2|
|非共线|2026101341|2.0811|1.8234|1.8327|1.8318|29/7|6/6|
|非共线|2026101342|2.1959|0.6453|0.6350|0.6311|29/5|5/5|
|非共线|2026101343|3.7125|1.6139|1.7219|1.6362|29/7|6/6|
|非共线|2026101344|1.4415|0.6931|0.6551|0.6487|29/5|5/5|

全 64 条轨迹安全完成，碰撞、边界投影、安全投影均为0；最小间隔 **9.681914 m**，最大加速度 **1.782747 m/s²**。这表示120步安全闭环完成，**不表示UAV2全样本达到真值严格停车**：V13确认仅3/8达到原停车判据。导航仍只使用自身 EKF；受限机的残余定位误差会影响真实停车。

全部开发/确认记录参考机定位 RMSE 最大值为 UAV0 **1.065360 m**、UAV1 **0.850644 m**；GNSS接受率最小均 **93.33%**。动态质量确实增加参考误差，不能把原无遮挡精度原样搬用。参考位置协方差仍各向同性（特征值比最大1）；区别主要表现为尺度、时效和相对几何，而非邻机 P 的方向差异。NEES/NIS只作描述统计，不宣称一般一致性保证。

## 4. 邻居价值与状态解释力

计划96个检查点全部完成，冻结状态分支与实际选中更新的状态/P及接受标志一致。共77个双合法可评分点；53个四状态净范数效用差超过0.01 m；确认分别为37个、25个。当前更新和未来三个**已记录UAV2本地包**在相同冻结测距/消息条件下配对，不重放队列、导航或未来参考机信息。最多1152个状态评价，原始数据保留SSE敏感性，旧30个线性代理门槛未降低或替换。

确认净效用排序准确率 **24/25=96%**，Pearson **0.6463**，Spearman **0.9031**，四种子净效用方向均正，冻结净效用计数/排序门槛通过。然而37个双候选点中 **20个两个邻居都没有定位均值收益**；25个净效用可分点有 **10个不足0.01 m的真实定位收益差**，差异主要来自相同的已知时延成本。不能据96%宣称稳定定位价值映射。

去掉成本后，确认真正四状态定位收益差≥0.01 m的点只有 **15** 个，其中 **14** 个在第6/10步，另1个在第62步。15个点排序正确 **14/15=93.33%**，Pearson **0.6430**、Spearman **0.7464**；去掉代理本身的成本项后结果相近。各确认种子分别4、4、3、4个点，准确率75%、100%、100%、100%。这支持**早期有限局部可预测性**，不足以证明持续120步内存在充足优化空间或泛化规律。

针对全部37个确认双合法点，状态差与**不含成本的真实四状态定位收益差**的相关性：

|可观测特征差（UAV0减UAV1）|Pearson|Spearman|
|---|---:|---:|
|邻机位置协方差迹|-0.5786|-0.5922|
|接收到的GNSS标准差|-0.4626|-0.5957|
|AoI|-0.2863|-0.4277|
|预决策链路时延|-0.2788|-0.3704|
|RSSI|0.2106|0.2684|
|估计距离|0.0321|-0.0362|
|自身沿两个方向的方差差|0.1600|0.0589|

这些是预定义状态的描述性检查，未拟合模型，也没有挑选最高相关字段重新训练或更换确认样本。邻机质量与P开始提供有效信号，距离单独解释力很弱；联合GNSS/RF场无法拆分各因素的因果作用。不能推断质量更好一定修正某次真实均值误差，创新方向仍包含不可预测噪声与本机未观测偏差。

## 5. 判断与最小下一步

1. 动态异构节点在冻结场中真实交换优劣，早期不同邻居具有可重复的局部定位价值差异：**有限实验支持**。
2. GNSS质量/P及时效信息比原同质参考情形更能解释收益；存在有限状态到局部邻居价值映射：**有条件通过**，不是全时域/城市场景泛化结论。
3. Greedy没有稳定提高完整闭环定位精度；新规则明显减少无效请求：**已验证于本设计**。
4. 原CI在少量早期融合后出现长期无均值收益，新增质量变化未形成持续测量竞争；需要区分“是否请求”与“请求哪个邻居”。不能为了通过而改CI、增大噪声或放大奖励。
5. **正式GAT+MAPPO训练准入未通过**。建议下一轮先冻结等请求窗口、等预算的 Random-peer 与可观测选择对照，分别报告原始定位收益和通信成本，确认是否仍有超越简单规则的累计收益空间。本轮不执行该追加实验、不降低5%验收标准。

若后续获准进入学习验证，动作仍为none/UAV0/UAV1，只对UAV2计算测量选择策略损失。输入可保留自身P/估计速度、任务信息、已到达邻机P/估计状态、GNSS质量及其时间戳、AoI、RSSI、延迟、估计方向、合法掩码、剩余预算。真值标签只用于离线评价；根迹收益仍需与累计误差验证，不能直接作为已被证明的定位奖励。只有多个受限机同时决策并存在资源竞争后，才能称为完整多智能体MAPPO任务。

## 6. 文件及复现

- `src/experiments/neighbor_selection_environment_v13.py`：隔离空间质量场、动态测量声明、原无线指标适配、到达消息质量头及审计。
- `src/experiments/neighbor_selection_execution_v13.py`：四策略接入原真实闭环，日志和冻结检查点。
- `src/experiments/neighbor_selection_validation_v13.py`：冻结种子、规则、历史哈希、运行上限、阶段预检、闭环和配对分支。
- `src/analysis/neighbor_selection_policy_v13.py`：非学习代理、可观测字段白名单和状态编码。
- `src/analysis/neighbor_selection_report_v13.py`：完整/截断分离、指标、旧/新门槛独立报告和图表。
- `src/analysis/neighbor_selection_evidence_v13.py`：纯读取逐包噪声、时间戳、身份与成本验证；补充计数更正，原报告保留。
- `src/analysis/neighbor_selection_conclusions_v13.py`：真实定位收益与共同成本拆分、原始收益相关和请求匹配分析。
- `knowledge/neighbor_selection_v13_protocol.md`：执行前冻结协议。

PyCharm解释器为 `D:\research project\.venv\Scripts\python.exe`，工作目录 `D:\research project`，模块 `src.experiments.neighbor_selection_validation_v13`。实际执行顺序如下（PowerShell）：

```powershell
& '.\.venv\Scripts\python.exe' -m src.experiments.neighbor_selection_validation_v13 --phase freeze
& '.\.venv\Scripts\python.exe' -m pytest tests/test_neighbor_selection_environment_v13.py tests/test_neighbor_selection_policy_v13.py tests/test_neighbor_selection_report_v13.py tests/test_neighbor_selection_counterfactual_v11.py tests/test_neighbor_selection_policy_v11.py tests/test_partial_gnss_ci_search_v10.py -q -p no:cacheprovider --junitxml=logs/preliminary/three_uav_dynamic_neighbor_value_v13_20261013/pure_units_release.xml
& '.\.venv\Scripts\python.exe' -m src.experiments.neighbor_selection_validation_v13 --phase release --junit logs/preliminary/three_uav_dynamic_neighbor_value_v13_20261013/pure_units_release.xml
& '.\.venv\Scripts\python.exe' -m src.experiments.neighbor_selection_validation_v13 --phase precheck --batch-id research_001
& '.\.venv\Scripts\python.exe' -m src.experiments.neighbor_selection_validation_v13 --phase develop --batch-id research_001
& '.\.venv\Scripts\python.exe' -m src.analysis.neighbor_selection_evidence_v13 --phase develop --batch-id research_001
& '.\.venv\Scripts\python.exe' -m src.experiments.neighbor_selection_validation_v13 --phase confirm --batch-id research_001
& '.\.venv\Scripts\python.exe' -m src.analysis.neighbor_selection_evidence_v13 --phase confirm --batch-id research_001
& '.\.venv\Scripts\python.exe' -m src.experiments.neighbor_selection_validation_v13 --phase counterfactual --batch-id research_001
& '.\.venv\Scripts\python.exe' -m src.experiments.neighbor_selection_validation_v13 --phase report --batch-id research_001
& '.\.venv\Scripts\python.exe' -m src.analysis.neighbor_selection_evidence_v13 --phase final --batch-id research_001
& '.\.venv\Scripts\python.exe' -m src.analysis.neighbor_selection_conclusions_v13
```

这些是本轮已执行的命令。设计和`research_001`已经存在，上限用完；脚本拒绝覆盖与追加重复物理运行。不要重跑这些写入阶段覆盖结果。只读核查可用：

```powershell
& '.\.venv\Scripts\python.exe' -c "from src.experiments.neighbor_selection_validation_v13 import verify; print(verify()['schema'])"
```

数据：`data/synthetic/three_uav_dynamic_neighbor_value_v13_20261013/batches/research_001/`；CSV：`logs/preliminary/three_uav_dynamic_neighbor_value_v13_20261013/research_001/`；图片：`plots/preliminary/three_uav_dynamic_neighbor_value_v13_20261013/research_001/`。配置：`data/configs/three_uav_dynamic_neighbor_value_v13_20261013.json`。

预数据纯测试74项通过，补充纯审计测试3项通过。最初一次测试收集命令引用了不存在的旧测试文件，0项执行；其XML保留，实际存在路径的74项成功结果用于release。补充读取审计最初误把原记录的nested case字段当顶层，原失败审计保留，纯测试确认后修复为nested case，v2对64条全部通过。两项修正没有改变物理结果、种子、策略或准入门槛。

完成后停止，等待用户确认。
