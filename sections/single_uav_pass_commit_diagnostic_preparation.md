# Single-UAV PASS_COMMIT Diagnostic 准备说明

## 状态与边界

本诊断在已有 lateral-exit braking assistance ON 基线之后，检验一个局部的 pass-progress non-regression
干预。准备入口仅读取冻结日志、场景清单、模型检查点和源文件；只有显式 `--human-run` 才会延迟导入
Evaluator。原 attempt 已在 ON episode 0001 因 action-decomposition assertion 中止；OFF 20/20 与冻结
baseline 完全匹配。当前版本只修复 decomposition bookkeeping，并等待从 OFF 20 + ON 20 重新开始的人工
paired run。两臂都开启 lateral braking，radial intent assistance 关闭；仅 ON 启用 PASS_COMMIT。
Goal Recovery、终止判定、safety/boundary 投影、动力学与既有 action transform 不变。

## 运行门控

PASS_COMMIT 使用当前 EKF 估计位置指向目标的单位向量，并以现有障碍几何计算 pass margin。连续三步
估计 lateral speed 的绝对值不超过 `0.5 m/s` 后开始记录 margin；连续两步 margin delta 小于
`-1e-8 m` 才触发并 latch。当前步还必须满足既有 past-closest 与 clearance-increasing gate。精确 pass、
release、episode 结束、clearance 小于 `2.5 m`、目标恢复、障碍身份不匹配/不可用或几何失败都会永久
终止；lateral assistance active 仅暂时互斥，不清除 latch。

校正量仅抵消当前 Actor 加既有 baseline candidate delta-v 在目标单位向量上的负 pass-progress 分量，
请求不额外限幅，实际应用受冻结 cap 限制，故应用后分量仍可能为负。分解校验明确加总 applied/capped
PASS_COMMIT vector；requested vector 只用于 telemetry。该分量在现有 action transform 之前加入，
逐步检查 transform 调用数与既有 `transform_applied` 标志一致。helper 只接收 estimated state，不接触
actual-state。

## 离线 cap 与 OFF 基线

Cap 从 prior authoritative lateral-exit assistance ON trace 的六个固定 main-failure episode
`0002, 0004, 0005, 0006, 0011, 0020` 重放既有 helper gates 后派生。逐行 cap CSV 保留所有来源记录和
纳入/排除原因；单个 episode 可为零行，但总体合格 cancellation 样本数必须大于零。统计量使用 Type-7
Q1/median/Q3/maximum，冻结规则为 `min(Q3, 0.075 m/s)`。

PASS_COMMIT OFF 基线 fingerprint 独立于父 radial-intent preregistration，锁定 20 个 episode 的顺序、
seeds、scenario manifest、checkpoint、policy/source/config、逐 episode trace/outcome 与 aggregate hash。
只规范化 condition，并只排除新增 `pass_commit_*` 字段。不匹配时必须以
`PASS_COMMIT_OFF_BASELINE_MISMATCH` 停止，且在 ON 输出布局/reset 前完成校验。新 preregistration 与
父 radial-intent 身份哈希分域；所有 preparation artifacts 不得覆盖非同内容的既有目录。

## 准备与人工入口

显式准备命令：

```text
python src/experiments/single_uav_pass_commit_diagnostic.py --prepare-only
```

本修复版本准备文件写入
`logs/preliminary/single_uav_pass_commit_diagnostic_seed2026/preparation_action_decomposition_fix_v1/`，
固定七个 canonical 文件：
`preregistered_pass_commit_endpoints.json`、`pass_commit_preregistration_lock.json`、
`pass_commit_runtime_source_lock.json`、`pass_commit_cap_derivation.json`、
`pass_commit_cap_derivation_rows.csv`、`pass_commit_off_baseline_contract.json`、
`pass_commit_preparation_summary.json`。

经人工审核并明确授权后，才可使用：

```text
python src/experiments/single_uav_pass_commit_diagnostic.py --human-run
```

本说明不新增论文符号；若后续正文引入新符号，须先查验并登记于 `sections/2_system_model.md`。

本次准备已完成：54 条合格 cancellation row，episode 分布为 `0002:5, 0004:1, 0005:15, 0006:15,
0011:15, 0020:3`；Type-7 Q1/median/Q3/maximum 分别为 `0.04689/0.11859/0.22227/3.22722 m/s`，
冻结 cap 为 `0.075 m/s`。父 active preregistration SHA-256 为
`74b999edb1158fc960374025f418d33c7812a57653c07f6d0e73fab8b69e02ac`；新 active hash 为
`ddd08b7fc729c759b51eff091bfc25b6eb273aef2368e40dea4d57c317dee73a`。原 OFF baseline contract 文件保持
逐字节不变，fingerprint 仍为 `de53e383264db1c46ea793519f7b627f3ef83fbb447ca33d9d44cf72a603c97e`。
新的日志输出根为
`logs/preliminary/single_uav_pass_commit_diagnostic_seed2026/run_action_decomposition_fix_v1/`；runtime
source-lock、cap 重放、OFF contract 离线验证、69 项纯测试、`py_compile` 与 `git diff --check` 均通过。
尚未执行新的 paired human-run。
