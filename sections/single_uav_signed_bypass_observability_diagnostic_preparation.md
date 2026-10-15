SIGNED_BYPASS_OBSERVABILITY_READY

# Single-UAV Signed Bypass Observability Diagnostic Preparation

本次变更只建立旁路日志、纯数学 helper、严格输出与身份锁；没有运行环境、`environment.step`、控制器、策略推理或新 CLI。冻结对象仍为 Terminal Capture v2.3 authority-repair、`max_steps=120`、20 个静态 holdout episode。

## A. 坐标系与符号

底层 contract 是 3D position/velocity/action/obstacle geometry，ENU 约定为 `+x/+y` 水平、`+z` 向上；姿态导航只使用 yaw，body `+x` forward、`+y` left。冻结 static pool 的起点、目标、障碍中心全部保持 `z=20.0`，因此绕障诊断采用唯一的水平目标框架：

- `radial_unit`：目标方向的 XY 投影单位向量；
- `lateral_unit=[-radial_y, radial_x, 0]`；正号是 goal-frame-left，负号是 goal-frame-right；
- `vertical_unit=[0,0,1]`，单独记录，不混入左右判断。

若 XY 投影退化或 goal ray 非水平，helper 失败关闭并返回 `SIGNED_LATERAL_AXIS_NOT_WELL_DEFINED`；不伪造左右。

## B. 实际/估计几何

估计 UAV position/velocity/goal delta 来自现有 actor-visible node features；实际 position/velocity 只读 `environment.states["uav_00"]`，并分别记录 post-step estimated 与 post-step actual。旧的 radial/tangential magnitude 字段保留。

## C. 障碍字段来源

复用 `_EstimatedRecoveryTracker` 当前选择的 obstacle、`center_m`、`_obstacle_extent`、`along_obstacle_m`、clearance 与 extent；不重新选择障碍、不调用 Teacher/Oracle。新增 obstacle relative radial/lateral/vertical、relative vector、signed side、clearance、extent 和沿障碍通过门字段。

## D. Actor 字段

保留 raw actor vector；旁路投影 `actor_raw_radial_delta_v_mps`、`actor_raw_lateral_delta_v_mps`、`actor_raw_vertical_delta_v_mps`，并记录 pre-transform 与现有 correction/fusion 后流水线分量。

## E. 实际运动字段

记录 estimated/actual radial、lateral、vertical velocity；每步 displacement 及从 encounter anchor 起的 signed lateral displacement；actual truth 不回写 observation。

## F. Encounter 冻结框架

首次达到 tracker 已有 `extent + SOFT_RECOVERY_BUFFER_M` 几何边界时冻结 frame、estimated/actual encounter position 与 step。后续同时保留冻结轴字段；未进入 encounter 时保持 `AMBIGUOUS/NA`，不强行贴标签。

## G. 日志字段与诊断 helper

新增字段覆盖：

- obstacle signed geometry 与 `signed_obstacle_side`；
- `pre_transform`、`post_transform`、`pre_safety`、`post_boundary`、`post_safety`、`final_applied` 的 radial/lateral/vertical；现有 boundary/safety delta 作为可追溯来源；
- passed threshold/margin、`passed_condition_now`、`soft_recovery_condition_now`；
- sign、累计 sign changes、same-side run length、actor/actual side agreement、clearance increasing、past closest approach、actor/actual still same-side；
- 推荐 bypass side 仅在确定几何规则可证明时使用；当前日志明确写 `AMBIGUOUS` 且标明未调用 Oracle。

未来 focus 标签支持 `WRONG_INITIAL_BYPASS_SIDE`、`CORRECT_INITIAL_SIDE_BUT_NO_EXIT`、`LATERAL_SIGN_OSCILLATION`、`ACTOR_TO_MOTION_DIRECTION_MISMATCH`、`RADIAL_COLLAPSE_WITH_VALID_LATERAL`、`NOT_ASSESSABLE`。安全/边界投影诊断字段继续保留。

## H. 修改文件

- `src/experiments/single_uav_signed_bypass_observability.py`：纯 frame、分解、sign/run、passed-margin、fingerprint、标签 helper，以及不把聚合 `same_*` 检查冒充独立 component identity 的 fail-closed 比较 helper；
- `src/experiments/single_uav_goal_recovery_correction_evaluation.py`：可选旁路 logging、独立 CLI flag、身份/轨迹锁和五个输出文件 writer；默认旧 CLI 路径不启用 signed logging；
- `tests/test_single_uav_signed_bypass_observability.py`：32 个纯离线测试；
- 本报告与根目录 `README.md`。

## I. Identity lock

未来运行固定 checkpoint/SHA、policy hash、observation dimension、ordered episode identity、scenario/noise seeds、controller、stable criterion、Goal Recovery `post_bypass_only`、Terminal Capture v2.3 authority repair 与 horizon 120。现有 baseline summary/current evaluation config 没有独立持久化的 reward、dynamics、action-transform、safety-projection 或 boundary-projection identity digest；因此五项均输出 `NOT_INDEPENDENTLY_VERIFIED`，现有聚合 `same_reward` 等只保留为 supporting context。signed fields 不进入 observation、reward、controller input、RNG 或 reset order。step-level trajectory fingerprint 继续提供行为等价保护；不一致时抛出 `LOGGING_PERTURBED_TRAJECTORY`。

输出根为 `logs/preliminary/single_uav_signed_bypass_observability_diagnostic_seed2026/`，文件名为 `signed_bypass_summary.json`、`signed_bypass_episodes.csv`、`signed_bypass_steps.csv`、`signed_bypass_focus_failures.csv`、`signed_bypass_direction_transitions.csv`。

## J. Tests

`D:\app-install\python\python.exe -m unittest tests/test_single_uav_signed_bypass_observability.py -v`：51/51 通过。测试覆盖轴退化、非水平 goal ray、正交/重构、左右符号、障碍/Actor/流水线分解、冻结通过门、sign changes/run length/agreement、五类失败标签、fingerprint、五项 component identity 的显式/缺失状态、CLI、Goal Recovery activation、旧 CLI 与独立输出根。

## K. py_compile

通过：

`D:\app-install\python\python.exe -m py_compile src/experiments/single_uav_signed_bypass_observability.py src/experiments/single_uav_goal_recovery_correction_evaluation.py tests/test_single_uav_signed_bypass_observability.py`

## L. git diff check

已通过 `git diff --check`（仅有既存的 Windows LF/CRLF 警告，无 whitespace error）；不自动提交或推送。

## M. Future PyCharm CLI

在人为确认后运行：

`D:\app-install\python\python.exe -m src.experiments.single_uav_goal_recovery_correction_evaluation --single-uav-signed-bypass-observability-diagnostic --quiet`

该命令仅在未来人工运行；本次准备阶段未调用。

## 最终状态旗标

```text
signed_bypass_observability_prepared=true
behavior_algorithm_modified=false
controller_modified=false
actor_modified=false
critic_modified=false
gat_modified=false
reward_modified=false
dynamics_modified=false
observation_modified=false
safety_projection_modified=false
boundary_projection_modified=false
action_transform_modified=false
terminal_controller_modified=false
goal_recovery_modified=false
training_performed=false
rollout_executed=false
formal_executed=false
smoke_executed=false
environment_step_called=false
historical_logs_modified=false
```

等待人工运行未来诊断；不在本次准备阶段生成新的 rollout 数据。
