HORIZON120_DIAGNOSTIC_READY

# Single-UAV Near-Goal Capture Horizon-Only Diagnostic Preparation

- CLI: `--terminal-capture-v23-authority-repair-horizon120-diagnostic`
- Baseline: `logs/preliminary/single_uav_terminal_capture_v23_authority_repair_validation_seed2026/`
- Baseline summary SHA-256: `4048b73c42423d56eb95f006d8177b205752b4d5d4580c28d65d699e551ef2e2`.
- Checkpoint SHA-256: `b7ed3ba79cf3055906bd0425cf4a7c4e613b206a8680ebf17f3b586fbc31b3c6`.
- Policy hash: `c423ad25f7832ad74fc1782ac56aabfaeacb2f004e217cb5a02b225cd04a78b5`.
- Baseline manifest hash: `becf1bba9fb868787356889f3825c71645788ff766e6ead1506886ec0fbf2114`.
- Candidate: `max_steps=120`; frozen baseline: `max_steps=100`; `formal=true`; 20 episodes; ON-only.
- Independent roots:
  - `data/synthetic/single_uav_near_goal_capture_horizon120_diagnostic_seed2026/`
  - `logs/preliminary/single_uav_near_goal_capture_horizon120_diagnostic_seed2026/`
  - `plots/preliminary/single_uav_near_goal_capture_horizon120_diagnostic_seed2026/`
- First100 indices: `0–99`; extension indices: `100–119` (natural-language steps 101–120).
- Focus episodes: `test_r99_episode_0001`, `test_r99_episode_0008`, `test_r99_episode_0017`.
- Required identity result: `only_horizon_changed=true`.

## Frozen identity

The preparation locks checkpoint path/SHA-256, loaded policy hash, observation dimension,
scenario manifest and ordered episode identity, Actor/Critic/GAT, reward, dynamics, obstacle,
safety/boundary projection, action transform, stable criterion, max correction, `a_eff`, stopping
margin, v2 governor, v2.1 arbitration, v2.2 latch, v2.3 target and authority-repair behavior.

## Prepared status

```text
horizon120_diagnostic_prepared=true
max_steps_baseline=100
max_steps_candidate=120
only_horizon_changed=true
controller_code_modified=false
terminal_capture_v23_modified=false
authority_repair_modified=false
actor_modified=false
gat_modified=false
critic_modified=false
reward_modified=false
dynamics_modified=false
action_transform_modified=false
stable_criterion_modified=false
training_performed=false
rollout_executed=false
formal_executed_by_codex=false
smoke_executed_by_codex=false
source_logs_modified=false
```

No new horizon120 CLI, `environment.step`, formal/smoke evaluation, rollout, training, or UAV
trajectory was executed by Codex.  This report records preparation only and awaits a manual run.
