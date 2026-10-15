LATERAL_EXIT_BRAKING_DIAGNOSTIC_VALIDITY_FIXED

# Single-UAV Minimal Lateral-Exit Braking Assistance OFF/ON 准备报告

## 状态

本轮只完成 paired causal diagnostic 的代码、身份锁、日志 schema、预注册 endpoint 与纯测试准备，
没有运行环境、rollout、formal、smoke、simulation 或训练，也没有创建新的诊断输出目录。

## 唯一处理变量

- OFF：当前 authority-repaired v2.3、`max_steps=120`、Goal Recovery ON 的冻结 signed-observability 行为；新 assistance 关闭。
- ON：完全相同配置，仅启用 minimal lateral-exit braking assistance。
- 固定 cap：`0.075 m/s per step`，来源是成功参考 Actor counter-steer magnitude 的 Q3 约 `0.0751 m/s`；这是诊断参数，不是最优参数声明，不支持 sweep。

## Activation gates

每步只使用 pre-step EKF estimated position/velocity、goal、static obstacle map、raw Actor action 及 encounter-frozen lateral frame。以下条件必须同时成立：

1. estimated-state encounter 已建立，并将 encounter-time EKF estimated goal ray 显式投影到 XY 平面后，冻结 static obstacle ID 与 horizontal goal-frame lateral unit；
2. 相对于同一个 frozen obstacle 已越过 estimated closest approach；
3. clearance 连续三个 transition 严格增加；
4. 最近五个 raw Actor lateral command 中至少四个与 encounter 初始 estimated lateral motion 方向相反；
5. 当前 estimated lateral velocity 仍保持 encounter 初始方向；
6. Goal Recovery 尚未进入 post-bypass/active 状态；
7. episode 尚未结束。

frozen obstacle 缺失或 tracker identity 不一致、速度过零、counter-steer/clearance gate 失效或 Goal Recovery 接管时，assistance 立即归零。若 estimated goal ray 的 XY 投影退化，则不构造替代轴，记录 `horizontal_goal_frame_degenerate` 并保持 assistance unavailable。helper API 不接受 simulator actual position/velocity；actual state 仅在动作提交后用于 outcome logging。`+ lateral` 仍表示 goal-frame-left，`- lateral` 仍表示 goal-frame-right。

## Assist 与 pipeline

激活时使用固定实现：

```text
requested lateral delta-v
  = -sign(estimated lateral velocity)
    * min(0.075, abs(estimated lateral velocity))
    * encounter lateral unit
```

该向量在冻结 frame 中只有 lateral component。每步先用未修改 raw Actor action 计算一次既有 Goal Recovery：

- Goal Recovery active：完全保留既有 fusion，assistance 强制为零；
- Goal Recovery inactive 且 ON gates 全部通过：`raw Actor + assistance` 仅调用一次既有 action transform；
- OFF 或 gate 未通过：保持既有 raw-Actor/no-transform passthrough。

随后继续通过既有 safety、boundary 与 dynamics。既有 physical/shared cap `0.6 m/s` 不变，assistance 不绕过 safety。

## OFF compatibility 与 paired identity

canonical paired 入口必须先完整执行 OFF，并在第一次 ON reset 前与冻结 signed-observability artifact 做 `lateral_exit_off_baseline_trace_v1` 精确比较。比较覆盖 ordered identity、seeds、raw Actor、Goal Recovery、environment command、executed action、estimated trace、reward 与 termination；仅排除新增 assistance-only columns。任何差异立即抛出：

```text
LATERAL_EXIT_OFF_BASELINE_MISMATCH
```

冻结基线自身的 canonical trace fingerprint 已离线验证为
`d0d6ec4891d314923c01be839cc3def1d071462cbccff1971a0f0e22ebcfd07b`（20 episodes、2,182 steps）；
五个输入 artifact 的 SHA-256 与已完成 authority audit 中的 before/after 记录一致。

检查通过后才允许创建 ON artifact。Actor、Critic、GAT、normalizer 在 first condition 前、OFF 后和 ON 后分别哈希；20 个 episode 的 scenario/noise seed、顺序、checkpoint、policy、observation dimension、controller config、Goal Recovery config 与 `max_steps=120` 均由两臂实际 provenance 比较。reward、dynamics、action transform、safety projection 与 boundary projection 没有独立 component digest，分别报告 `NOT_INDEPENDENTLY_VERIFIED`，不会作为非空字符串进入布尔 guard；OFF exact trajectory fingerprint 继续提供行为等价保护。`only_lateral_exit_braking_assistance_changed` 只由 paired identity guards、OFF fingerprint 和精确 OFF/ON treatment-config diff 派生，不再硬编码。

## Telemetry 与 endpoint

step telemetry 包含全部 gate、inactive reason、encounter/frozen obstacle identity、estimated lateral velocity、Actor lateral command、requested/applied assistance、combined pre-transform、final applied lateral command，以及原有 transform/safety/boundary/Goal Recovery 与 outcome 字段。

预注册采用 all-20 intention-to-treat paired endpoints，并保留 0013/0014 特殊案例：`passed_obstacle`、collision、沿 encounter 初始横向运动方向的最大 excursion、encounter 后到 `lateral_exit_threshold_0p5_mps` 的时间及 censoring、minimum frozen-obstacle clearance。primary 的方向性 excursion 仅更正名称为 `maximum_initial_direction_lateral_excursion_from_encounter_m`，计算行为未改变。

OFF-triggered secondary cohort 只由 OFF 首个 fully eligible step 选择，ON 使用同一个绝对 anchor，禁止重新选择。轨迹型 secondary endpoint 统一使用 `[anchor, min(OFF last step, ON last step)]`：integrated absolute lateral velocity、mean decay、peak velocity、activation 后 displacement、extra drift、50% speed reduction time，以及新增的 `steps_to_lateral_exit_threshold_0p5_mps_after_common_anchor`。后者明确报告 `THRESHOLD_REACHED_WITHIN_COMMON_WINDOW` 或 `RIGHT_CENSORED_AT_COMMON_WINDOW_END`；另行区分 `OFF_NEVER_ELIGIBLE` 与 `ON_TERMINATED_BEFORE_OFF_ANCHOR`。`passed_obstacle`、recovery、entered-1m、stable、collision、timeout 及 projection counts 始终使用各自完整 episode，不受 common window 截断。

本次 schema 修复发生在任何 OFF/ON rollout 之前。正式 active preregistration canonical SHA-256 为
`a93837621a04c9fa7895881155a90878b664eeac318fbcb0cf7dded5afedd2c2`；旧值
`6804dc2e8207da57bbcfb202a7fb936379ea5a815adae771b74ceefd060a1679`
仅保留为 `superseded_pre_run_preregistration_hash`。

## CLI 与未来输出

人工运行入口：

```text
python -m src.experiments.single_uav_lateral_exit_braking_assistance --condition paired
```

也支持锁定的 `off` 或 `on` 单条件准备入口；正式因果结论以 `paired` 为准。未来输出根为：

```text
logs/preliminary/single_uav_lateral_exit_braking_assistance_diagnostic_seed2026/
```

采用 fresh-output 拒绝覆盖；本轮未创建该目录。

## 修改与验证

新增 diagnostic assistance helper、独立 diagnostic module、两组纯测试，并在现有 frozen evaluator 添加默认关闭的窄 hook；没有修改 Goal Recovery arithmetic、Actor/GAT/Critic、reward、dynamics、action-transform implementation、safety、boundary 或 Terminal parameters。

人工首次启动曾在 rollout 前暴露 normalizer identity hash 的 JSON serialization 问题：`ObservationNormalizer.normalizers` 内含嵌套 `FeatureNormalizer` 对象。现已仅在 runtime identity 层加入只读递归 snapshot，支持 dataclass、带 `__dict__` 的纯状态对象、NumPy scalar/array、mapping、list/tuple、非有限浮点和 cycle fail-closed；不调用 normalizer 方法，不改变 normalization arithmetic。Actor、Critic、GAT 与 combined policy hash 保持原实现，active preregistration hash 与 OFF baseline fingerprint 均未改变。

最终准备状态：

```text
lateral_exit_braking_diagnostic_prepared=true
lateral_exit_assist_cap_mps=0.075
actor_modified=false
reward_modified=false
dynamics_modified=false
goal_recovery_modified=false
terminal_controller_modified=false
safety_modified=false
boundary_modified=false
training_performed=false
rollout_executed=false
formal_executed_by_codex=false
smoke_executed_by_codex=false
environment_step_called=false
```
