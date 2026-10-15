# 协同融合可靠性：原始资料及提案边界（2026-10-12）

本轮只进行诊断与离线线性参考，不修改部署 EKF 或训练奖励。

- Chen、Arambel、Mehra，*Estimation under Unknown Correlation: Covariance Intersection Revisited*，IEEE TAC，2002，DOI 10.1109/TAC.2002.804475。[原始稿](https://citeseerx.ist.psu.edu/document?doi=f35a8803c33af3cc88393c273ecfb0f0ac0f6fbe&repid=rep1&type=pdf)。研究给定线性组合增益时的未知相关性协方差上界，以及优化组合增益时与 CI 的关系。它不支持“任何均值更新后恢复 prior 即保守”的推断。
- Li、Nashashibi、Yang，*Split Covariance Intersection Filter: Theory and Its Application to Vehicle Localization*，IEEE TITS，2013，DOI 10.1109/TITS.2013.2267800。[作者代码与论文资料](https://github.com/LI-Hao-SJTU/SplitCIF)。SCI 的独立与相关部分需要明确维护；当前只传递总协方差的接口，不能未经改造就宣称拥有这种分解。[作者补充理论](https://arxiv.org/pdf/2101.10159)。
- Chang、Chen、Mehta，*Resilient and consistent multirobot cooperative localization with covariance intersection*，[作者论文](https://arxiv.org/abs/2108.08789)。通信与相对观测需要分别建模；存在已研究的分布式一致性路线，不应因当前启发式出现局限而否定整个协同定位框架。
- Chen 等，*Kalman Filter Auto-tuning through Enforcing Chi-Squared Normalized Error Distributions with Bayesian Optimization*，[作者论文](https://arxiv.org/abs/2306.07225)。这里只引用对 NEES/NIS 分布检验适用条件的讨论，不采用该文调参算法。相关时间点、门控后的样本及保守创新方差不能直接当成独立、准确校准的卡方样本。

## 仅作为提案的固定增益协方差修复

以下使用程序变量名说明已有估计误差变换，不新增论文正式符号。将相对测距在冻结状态处线性化，`direction` 从邻居指向本机；本机观测行 `jacobian` 的位置部分为该方向。`gain` 使用现有 EKF 增益。若输入协方差界有效、测距噪声与两端误差独立，并把增益/方向视为给定，则更新误差分解为本机误差线性变换、邻居投影误差增益项以及独立噪声项。

```text
transform = identity - gain @ jacobian
local_part = transform @ prior_covariance @ transform.T
neighbor_part = gain @ projected_neighbor_variance @ gain.T
independent_noise_part = gain @ range_variance @ gain.T
upper = local_part / (1 - mixing_weight) + neighbor_part / mixing_weight + independent_noise_part
```

`mixing_weight` 在零到一的开区间内。由矩阵形式的加权 Young 不等式，未知交叉项被这个上界覆盖；最小化该族上界的 trace，可取邻居项 trace 平方根除以两项 trace 平方根之和。退化的零项需要单独处理。离线脚本应检验合法联合协方差下 `upper - actual_posterior` 的半正定性，而非只检验 `upper` 自身半正定。

这能修复“固定增益下无证明恢复 prior”的协方差语义，但可能很保守；它不自动带来信息增益，也不保证非线性测距、延迟消息、条件门控或模型失配下的严格一致性。不得直接将该上界写回在线 EKF。本轮等待用户确认后，才可建立独立融合版本比较。

## 奖励提案的前置条件

最终 posterior 的 trace/logdet 增益减实际测量开销，需要 posterior 有可信的一致性与可辨识信息语义。当前恢复 prior 时，该增益为零；直接换奖励会奖励 none，不能据此推断协同均值没有价值。

滚动测距方向互补性可以作为相对定位任务的候选指标，但几何 Gram 秩只说明约束方向，不等于定位精度、概率信息量或绝对坐标可观测性。若累计信息矩阵，必须处理共享状态、重复消息和测量相关性，不能把所有邻居当独立信息相加。仿真真值只能评价这些候选指标，不得成为执行期 Actor 输入或规则选择依据。
