# Single-UAV Pre-pass Radial-Intent-Persistence Diagnostic 准备说明

## 状态

诊断 helper、Evaluator opt-in 集成、配对身份检查与离线准备入口已经实现。canonical preparation
文件已由只读历史日志离线物化并通过 runtime-lock 重算校验：active preregistration SHA-256 为
`bf70fb95188ca73c1ba87997a090d22331a3ac76a27daaaea1d86da09f59e40b`，OFF baseline
fingerprint 为 `9d10d1529c4a0126899d72428c227a710cb00a2293205d32a113308e4819787f`，冻结 cap 为
`0.075 m/s`。本轮执行了 `py_compile` 与 40 项纯 `unittest`，没有执行 rollout、仿真、formal、
smoke、训练或 `environment.step()`。

最终 consistency fix 将 cap derivation 的歧义字段拆为 `fail_closed_policy=true` 与
`fail_closed_triggered=false`；preregistration 保留 policy 兼容字段 `fail_closed=true`，summary、cap、
runtime lock 和 preregistration lock 均显式记录相同语义。因为 canonical payload 锁定 diagnostic 源码
hash，先前 `5a2b175b...` 在任何 rollout 前被上述新 hash 取代，而不是复用。

## 冻结问题与干预

问题限定为：在 Actor 连续历史显示 radial intent persistence deficient、估计 lateral exit 已就绪、
且所有排除条件满足时，提供小幅 radial-only assistance 是否改善 pre-pass reorientation。人群固定为
20 个有序 episode，以全体 paired ITT 为主；七个既有 failure episode 仅作描述性机制 subgroup，
episode 0008 只作 downstream terminal adverse case。OFF/ON 采用同一 formal20、120-step 配置；
Goal Recovery、signed observability 与旧 lateral exit assistance 在两臂都保持启用，radial intent
assistance 是唯一处理差异。

Encounter frame 固定为 `encounter_time_ekf_estimated_goal_ray_projected_to_xy`。运行 helper 仅接受
估计状态与 Actor 输出，不接收或读取 actual-state 参数。固定 target floor 为
`0.031192317367950993 m/s`，deadband 为 `1e-8 m/s`；估计 lateral speed 的绝对值不超过
`0.5 m/s` 连续三步才认为 exit ready；Actor 需要完整连续五步历史，goalward radial command
超过 deadband 的步数少于四步才属于 deficient。实际 assistance 只沿冻结 radial axis，幅度为
不低于 floor 所需的请求与离线 cap 中较小者。放行 gate 还要求 frozen frame/obstacle 匹配、无
release latch、未 passed、无 post-bypass recovery、Goal Recovery 与 lateral assistance 均 inactive、
episode 未结束；冲突时 fail closed。Active assistance 与既有 transform 恰好融合/调用一次，之后沿用
原 safety、boundary 与 dynamics 路径。

Cap 仅从历史 lateral-assistance ON `steps.csv` 中 episode 0002、0004、0005、0006、0011、0012、0020
推导。准备器按 episode 与步序检查连续性，重建估计 frame gate 和 radial projection，并与冻结 radial
dot 字段交叉核对；输出排除原因计数及 Type-7 median、Q1、Q3、maximum，最终规则为 Q3 与
`0.075 m/s` 的较小者。无合格样本、缺失字段或投影不匹配时拒绝准备，不做 cap sweep。

## OFF 基线与身份保护

OFF contract 继承 prior lateral-exit ON steps 的明确有序字段及逐行表示，覆盖 lateral assistance、
Goal Recovery/terminal、transform/projection、pass/recovery/outcome 和 episode/step/seeds 身份；唯一
规范化是 `condition` 标签。新 radial telemetry 不属于该继承比较。运行中 OFF 与冻结 prior-ON trace
不逐行完全一致时，在创建或运行 ON 前以 `RADIAL_INTENT_OFF_BASELINE_MISMATCH` 停止。配对比较同时
锁定 episode 顺序、seeds、checkpoint、policy/controller identity、配置及源文件 hashes。

## 准备文件与人工入口

显式离线准备入口：

```text
python -m src.experiments.single_uav_radial_intent_persistence_diagnostic --prepare-only
```

准备输出限定于
`logs/preliminary/single_uav_radial_intent_persistence_diagnostic_seed2026/preparation/`，仅含：

- `preregistered_radial_intent_persistence_endpoints.json`
- `radial_intent_preregistration_lock.json`
- `radial_intent_runtime_source_lock.json`
- `radial_intent_cap_derivation.json`
- `radial_intent_cap_derivation_rows.csv`
- `radial_intent_off_baseline_contract.json`
- `radial_intent_persistence_preparation_summary.json`

writer 不复制数据；同内容可复用，异内容拒绝覆盖。runtime loader 会从冻结源日志重算 cap 与 OFF
contract，并连同完整 source hash（含 signed-observability gate dependency）逐项核对。完成 artifact review
并取得人工授权后，paired runtime
入口为：

```text
python -m src.experiments.single_uav_radial_intent_persistence_diagnostic --human-run --condition paired
```

本说明只描述准备契约，不报告实验结果或因果结论；没有新增论文符号登记。
