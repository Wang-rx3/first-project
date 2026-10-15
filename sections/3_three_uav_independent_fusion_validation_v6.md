# 三 UAV 独立融合方法验证 v6

日期 2026-10-12。独立验证，不部署，不训练；原 v3/v4/v5、固定导航、在线 EKF 和 MAPPO 奖励保留。数学来源核查与独立推导见 `knowledge/cooperative_fusion_v6_mathematical_basis.md`。未取得 2002 原论文可逐公式视觉核查的完整 PDF；作者上传原文文字可读，但部分公式缺失。严格表述限于独立推导的固定线性增益、有效输入协方差/二阶矩界、独立测距噪声条件。

## 预先固定的比较

配置 `data/configs/three_uav_fusion_validation_v6_seed2026_20261012.json`，对应 `data/synthetic/` 同名目录下 `design.lock.json` 与 `execution_sources.lock.json`，在首个物理步前保存。复用 TRAIN compact-X/bridge-X，分别用种子 2026116100、2026116102；初始位置/速度误差数组完全沿用 v5，边际位置 4I、跨机位置 2I、速度独立 I。跨机先验是已知诊断假设，不是在线真实相关性识别。姿态/bias 无误差注入；不声称全15维 NEES 校准。

四条原生物理记录，每条120步，共480联合步。独立生产器仅将派生配置的 `cooperative_ekf_enabled` 置false，三架固定导航仍只读各自RAW本机EKF和航点；控制参数、动力学、dt、限幅、安全投影、预算、消息TTL和UWB周期均保留。相对融合不反馈到导航，避免方法间轨迹变化混入估计比较。影子结果不能被当作这些候选控制器的闭环停车验证。

选择使用既有 v4 `WindowRoundRobin`，仅第2、6、…、118步开放，保留原生合法性和游标。消息和物理资格由生产器执行；每个方法使用相同有效 directed range ID、发送/到达时刻和30次预算上限。接收资格与创新门控分开，若不同方法拒绝某观测应单列。本轮339个资格满足测距在每个方法/条件中全部接受，356个实际预算/等价overhead对应17次更新前无效测量，不伪造免费测量。

公共记录保存真实状态仅供评估；估计器接口严格传入白名单IMU加速度/角速度、VO位移/协方差、带噪GNSS和测距，不传真实bias或true_range。分布式方法各自拥有发送前的本机belief历史，并按真实包发送时刻、到达时刻和缓存增量复现原生常速度/协方差传播；不把源belief悄悄替换成当前私有估计。每对消费间隔至少0.8s，超过0.6s TTL；若遇到应走fresh CI缓存的情况，验证失败而非静默省略CI。

两个信息条件独立报告：`no_anchor` 保持原生GNSS-denied；`nominal_gnss_uav00` 在同条物理轨迹上，使用现有 `sample_gnss` 的 nominal 模式给 uav_00 生成标准差0.5m的带噪绝对观测。独立 indexed RNG 键已锁定；不改传感器噪声参数。这是“该GNSS可用且模型可信”条件性probe，不是已证明现实系统拥有此参考，也不是精确真值锚点。观测只用于影子估计器，未供Actor或导航使用。

## 独立候选与接口

| 文件/函数 | 作用 | 部署边界 |
|---|---|---|
| `src/analysis/cooperative_fusion_estimators_v6.py:YoungBoundEKF.update_relative_range` | 调原生创新、门控和均值更新，然后只在独立对象中写入固定增益Young上界 | 未改 `src/simulation/ekf.py` |
| 同文件 `JointEKF.predict_and_local_updates` | 45维三机联合状态及完整交叉协方差；9维过去位置克隆处理原始VO位移，再边缘化 | 中央参考，非分布式替代 |
| 同文件 `JointEKF.update_range` | 单个标量测距同时更新两端及相关状态，消费唯一物理ID | 不重复已有样本 |
| 同文件 `measurement_information` | 同一标量方向作用于本机及三维相对状态，计算条件trace/logdet下降 | 不伪造三轴相对观测；联合C当前不可由本地接口直接获得 |
| `src/experiments/three_uav_fusion_validation_v6.py` | 原生记录生产、白名单、延迟发送belief回放、三方法两条件影子比较 | 无NN/PPO更新，最高480物理步 |
| `src/experiments/stage3_validate_fusion_v6.py` | PyCharm直接文件/模块入口 | 独占run-id，拒绝覆盖 |
| `src/analysis/cooperative_fusion_linear_v6.py` | 256个冻结线性高斯样本及90个上界网格 | 0物理步，不是非线性一致性证明 |
| `src/analysis/cooperative_fusion_report_v6.py` | 逐块/汇总误差、NEES/NIS、信息信用、代价和绘图 | 离线真值只评价 |
| `src/analysis/cooperative_fusion_verify_v6.py` | 原始记录、发送belief、导航、测量ID及NEES独立复算 | 0物理步 |
| `src/analysis/cooperative_fusion_geometry_v6.py` | 径向/横向、简化Jacobian及曲率诊断 | 真值只用于诊断，不构成完整可观测性证明 |

联合参考获得中央汇集的所有本地IMU/VO和当前联合估计、已知初始交叉先验；分布式方法仅拥有本地观测和实际延迟包。它们使用相同原始测距/噪声/预算，但可用信息不同。不能将联合参考的优势全部归因于融合公式或声称它在当前通信合同下可部署。

Young 同增益实验与递归实验分开：同增益只改变报告协方差，均值完全相同；递归写回后会改变后续VO、测距增益和发送协方差。上界输入需为测量时刻有效误差界，原生延迟常速度belief在加速条件下未被证明满足该条件。数值PSD不等于条件递归或一般非线性保证。

## 主要实际结果

四块 pooled 指标，每条120步：

| 条件 | 方法 | 绝对RMSE m | 相对向量RMSE m | 位置NEES均值 | UWB NIS均值 | 末步位置trace均值 m² |
|---|---|---:|---:|---:|---:|---:|
| 无锚 | 恢复prior | 3.14099 | 1.71556 | 2.45678 | 0.01543 | 12.0961 |
| 无锚 | Young递归 | 3.11315 | 1.75006 | 0.67169 | 0.01090 | 7220.0890 |
| 无锚 | 联合参考 | 3.11523 | 1.66467 | 3.32501 | 1.05742 | 8.4728 |
| 条件GNSS | 恢复prior | 2.02903 | 2.42051 | 1.98933 | 0.03731 | 8.0707 |
| 条件GNSS | Young递归 | 2.24113 | 2.67375 | 1.39134 | 0.06011 | 754.0735 |
| 条件GNSS | 联合参考 | 0.90285 | 1.27480 | 2.65606 | 1.03133 | 0.7993 |

四个配对块共享两个随机种子家族，时间点相关；无显著性/泛化结论。所有24组完整运行、无数值删失；各做一次保存记录的重复回放，科学数值精确相符。主影子计算2880联合filter步，重复2880；物理模拟仍480。

无锚联合相对NEES均值10.5846，相关样本中24.17%超过3维卡方95%单侧参考；条件GNSS相对NEES均值3.3556，8.82%越限。不能宣布联合参考一致性通过。尤其compact-X/6102无锚相对NEES约30.825。一般NEES统计检验需要独立重复，本轮只有两个误差/噪声家族；这些是过度自信风险指示而非正式总体检验。分布式相对NEES使用未知cross上界 `2*(own_pos_cov+peer_pos_cov)`，其低值不能与联合维护cross的NEES直接当作同一种校准评分。

高NEES不是共同平移本身：相对误差评价已消去平移。无锚联合协方差始终包含初始共同平移2I分量，减去该分量后的最小PSD slack约0.0008以上，未虚构消除共同坐标误差。曲率诊断中compact-X/6102方向平均误差约9.45°，真实范围范数误差RMSE0.128m，而投影径向向量误差RMSE0.311m；二者差中的几何曲率项RMSE0.260m，精确几何恒等式误差<4e-14。这确认曲率项不可忽略，但尚不能把全部不一致唯一归因于某一线性化或可观测性机制。完整动态可观测性、门控条件效应和误差模型仍待核查。

## 数学验证、代价与候选奖励

256个独立冻结线性样本中：恢复prior/Young同增益RMSE同为1.05666；恢复报告方差1，实际理论方差1.23447；Young报告1.23447，NEES均值由1.11653降到0.90446，但均值误差没变。已知交叉协方差的线性联合参考RMSE0.07678、NEES0.92713，使用不同且可为负的正确增益；它拥有已知相关性，不能用于未知相关性的上线承诺。90个标量/三维上界检查通过。

本机/邻机variance1/4、cross2的联合协方差合法，是对恢复prior普遍保证的反例。Young的代数依据见数学资料，但源协方差界、噪声独立性、固定Jacobians及增益都须成立。递归膨胀既改变均值也让NEES变低；不推荐把它作为本轮性能方案，更不能简单裁剪矩阵掩盖问题。

平均每联合filter步pipeline计时（含检查及指标计算，不是纯核微基准）无锚：恢复1.891ms、Young2.089ms、联合2.858ms；条件GNSS：2.400/3.055/3.813ms。当前协方差存储恢复/Young为三份15²×8=5400byte，联合45²×8=16200byte；额外54维克隆矩阵23328byte。实现还保留最后克隆，二者合计39528byte；这些是矩阵字节，不是全部工作区/RSS峰值。历史、临时矩阵及传感器集中传输额外代价不能遗漏，也未计作本轮现有通信开销。记录中356个overhead为512byte等价queue单位，不是精确RF能耗。

联合条件相对trace/logdet在一次标量测距后保留横向不确定度，能区分有历史信息后的重复方向与互补方向；纯测试验证正交例子。实际逐块logdet与即时相对SSE改善的描述性相关约0.049–0.743，剩余累计SSE关联约0.35–0.44，但受剩余时域/时间趋势混杂，不能当成因果预测或奖励验证通过。恢复prior最终trace信用仍为零；Young最终trace变化主要反映上界膨胀，不是可信信息收益。

**推荐定位**：恢复prior不再作为一致性方案；Young最适合当前接口的独立保守对照，现阶段不适合性能优化/奖励；联合EKF最适合继续作为交叉相关性和信息收益参考，当前相对一致性不足且有信息优势，不直接替换分布式执行层。本轮没有足够依据推荐三者之一直接部署。下一步保持本地EKF/消息骨架，先验证观测一致的相对误差与相关性建模，并解释/抑制有依据的重复保守膨胀，再选择可由实际分布式信息计算的奖励。SCI独立/相关分解是可研究方向，未实现也未凭二次引用宣布胜出。

可开始离线奖励候选设计，但不足以冻结正式MAPPO奖励或进入训练。候选联合相对logdet/trace减实际测量成本需首先有可信posterior，并解决执行期cross信息可得性；不能把中央参考状态交给Actor，也不能用真值误差替代执行观测。本轮未改原奖励、PPO参数或准入门槛。

### 下一独立版本建议，尚未实施

对当前本地EKF与延迟belief接口，优先研究“将已知独立的当前过程/观测噪声单列，仅对未知相关的旧belief部分施加上界”。发送payload在当前本地预测前冻结，当前IMU/VO/独立GNSS噪声具有时间因果上的独立性候选；可复用 `last_prediction_context` 和已有VO/GNSS线性化计算维护两个误差二阶矩贡献，再经测距固定增益变换，给未知相关部分使用加权上界、独立部分按普通协方差相加。

这比当前对整个15维prior反复施加同一上界更符合已有因果结构，不需要新导航、放大噪声或改变通信预算；但仍须验证模型噪声独立性、偏差、门控以及非线性条件，不能从“发送更早”单独宣布完整误差独立。它不保证解决旧belief未知相关性本身造成的膨胀，也不能自动带来定位信息增益。若需要进一步提高精度，应维护可验证的相关性结构/相对状态信息，不简单裁剪协方差。此为下一候选的明确研究建议，不是本轮已证明优越的SCI算法或部署建议；须用户确认后另建版本与当前三方法配对。

## PyCharm与实际脚本

Python解释器 `D:\research project\.venv\Scripts\python.exe`，工作目录 `D:\research project`。使用模块运行或直接运行stage3入口。以下run-id已存在，必须换独立名字才可重新运行；不要重跑覆盖本轮。

```powershell
.\.venv\Scripts\python.exe -m src.analysis.cooperative_fusion_linear_v6 --run-id linear_s2026
.\.venv\Scripts\python.exe -m src.experiments.stage3_validate_fusion_v6 --run-id paired_s2026
.\.venv\Scripts\python.exe -m src.analysis.cooperative_fusion_report_v6 --run-id paired_s2026
.\.venv\Scripts\python.exe -m src.analysis.cooperative_fusion_verify_v6 --run-id paired_s2026
.\.venv\Scripts\python.exe -m src.analysis.cooperative_fusion_geometry_v6 --run-id paired_s2026
.\.venv\Scripts\python.exe -m src.analysis.cooperative_fusion_geometry_v6 --run-id paired_s2026 --curvature-only
```

输出根目录各为 `data/synthetic/three_uav_fusion_validation_v6_seed2026_20261012/`、同名 `logs/preliminary/`、`plots/preliminary/`，分别有 `linear_s2026` 与 `paired_s2026`。所有原始record、配置、状态和指标保存；逐块全部数据见 `fusion_comparison.csv`，科学总结见 `diagnostic_summary.md`，检查见 `independent_verification.json`，图见 `paired_fusion_rmse.png` 与 `covariance_trajectory.png`。

本轮完成后停止，等待用户确认；不部署到主系统，不启动校准/训练。
