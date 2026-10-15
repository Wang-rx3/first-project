# v6 固定增益协方差上界：原文核查与适用边界

2026-10-12。本轮只验证独立模块，不改在线 EKF。这里使用程序变量解释独立推导，不引入新的论文正式符号。

## 原始资料实际可获取范围

- Chen、Arambel、Mehra，IEEE TAC 47(11), 1879–1882，2002，DOI 10.1109/TAC.2002.804475。[提供的原稿](https://citeseerx.ist.psu.edu/document?doi=f35a8803c33af3cc88393c273ecfb0f0ac0f6fbe&repid=rep1&type=pdf) 返回 403。[作者上传的原文网页](https://www.researchgate.net/publication/260661631_Estimation_under_unknown_correlation_covariance_intersection_revisited) 可读取正文各节、页码及 Theorem 1—Fixed Gains。原文说明不考虑动力学、输入协方差界有效、固定线性组合；上界族通常不紧。网页公式提取有缺失，两条 PDF 下载均失败。**未取得可逐公式视觉核验的完整 PDF**，不宣称已经完整核验式(11)–(15)或将定理直接推广到本项目。
- [Hao Li 的作者补充原稿](https://arxiv.org/pdf/2101.10159) 三页完整 PDF 可读；其 SCI 讨论明确使用独立与相关两部分协方差。当前接口没有该分解，不能直接应用并宣称满足理论条件。
- [Chang、Chen、Mehta 原始协同定位论文](https://arxiv.org/html/2108.08789v1) 全文可读；其 CI 融合及相关观测模型不是“更新均值后恢复 prior”。它支持继续研究有明确定义的分布式融合，不支持当前启发式的普遍保证。

## 独立推导，条件结论

固定测距方向与增益时，线性化后的本机误差为 `local_term + neighbor_term + noise_term`。其中：

```text
local_term = (identity - gain @ local_jacobian) @ local_error
neighbor_term = gain @ neighbor_jacobian @ neighbor_error
noise_term = gain @ range_noise
```

均值为零、输入协方差有效且测距噪声独立时，本机项和邻机项的二阶矩界分别为 `local_cov` 与 `neighbor_cov`。对任意零到一之间的固定 `weight`，外积矩阵的加权 Young 不等式给出两项和的二阶矩上界：

```text
upper = local_cov / (1 - weight) + neighbor_cov / weight + gain @ range_noise_cov @ gain.T
```

理由是加权两项外积之和减去和向量的外积，正好是一个缩放差向量的外积，必为半正定。输入上界再经固定线性变换保持半正定序。独立噪声使交叉项期望为零，因此噪声项可单独加入。若只知道 MSE 界而存在偏差，需使用误差二阶矩而非仅中心化协方差；遗漏偏差不能由该公式补救。

本机项 trace 和邻机项 trace 固定时，这一族上界 trace 的最小值由平方根比例得到：`weight = sqrt(trace(neighbor_cov)) / (sqrt(trace(local_cov)) + sqrt(trace(neighbor_cov)))`。这是**该上界族内的 trace 最优**，不是所有未知相关性上界中的最优，也不是自动最优均值增益。一个项为零时使用连续极限，不以人工 epsilon 或放大系数调整结果。

与原文可读的 Section IV 固定增益上界族讨论相符；上述代数是本轮可核查的独立依据，不依赖网页缺失公式或二次引用。

## 不能省略的限制

1. 本机和邻机的协方差须在测量时刻确为误差二阶矩界。延迟包的常速度传播在有加速度时可能产生偏差；仅按原生规则增加过程协方差，没有证明一定覆盖该偏差。
2. 增益和测距 Jacobian 在这里固定。实际 EKF 的方向依赖随机估计，线性化余项及条件偏差未被该上界覆盖。条件证明需要相应的条件误差界，不能直接把无条件 prior 代入后宣布严格递归一致性。
3. 创新门控改变接受样本的分布。线性无门控结论与门控后的 NIS/NEES、时间相关样本的统计检验应分开。
4. 测距噪声须与两端估计误差独立；同一物理测距样本不能重复计为新信息。保持原生唯一 directed sample ID、消息有效期、测量周期与预算。
5. 递归写回这个上界会改变下一次本机增益、VO 更新和传出的邻机协方差；它可能膨胀且改变均值。必须与“同 gain、只比较报告协方差”的冻结实验区分。更小的 NEES 或更大的协方差都不等于定位质量更好。

## 联合参考与奖励解释

45 维联合 EKF 显式维护三机交叉协方差；VO 用 9 维上一位置克隆处理当前减过去的位移，然后边缘化克隆。它使用中央汇集的本地传感器和当前联合估计，不使用真值，但拥有分布式延迟包之外的信息，是参考，不是部署候选的公平同信息最优证明。

无绝对观测时，内部测距与位移 VO 对全体相同坐标平移不敏感。仅在条件性已有 nominal GNSS 可用的独立 probe 中使用 uav_00 的带噪位置观测；这不是精确真值锚点，也未供导航或 Actor 使用。锚定与无锚结果分列。

联合相对 posterior 的 trace/logdet 条件下降可作评价基准，但需要交叉协方差，当前分布式接口不可直接计算。单机 Young bound 的变动虽然可计算，却是保守误差界的变化，不能直接叫信息收益。所有指标须对照累计实际误差，且随机期望不要求逐次预测实现后的误差符号。本轮不修改 MAPPO 奖励。
