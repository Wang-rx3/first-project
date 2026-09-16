# Current single-UAV visualization diagnostic report

## Scope and provenance

This is a diagnostic-only reproduction and visualization run. It reads the new
external-recovery rerun at D:\research project\logs\preliminary\single_uav_external_recovery_method_study_seed2026_visualization_rerun_20260914_01 and its evaluation data at
D:\research project\logs\preliminary\single_uav_external_recovery_method_study_seed2026_visualization_rerun_20260914_01\dagger_bc. It does not modify reward, environment, network,
checkpoint, PPO/MAPPO, DAgger, BC, or Teacher logic.

- git commit: 78b7bce015fb54eaf4508f371f05dc0e7926829c
- branch: main
- current rerun PPO updates: 0
- current rerun training_performed: False
- baseline MAPPO/PPO source: D:\research project\logs\preliminary\single_uav_braking_progress025_seed2026\training_metrics.csv
- evaluation seed values: [2026, 4021000, 4021043, 4021086, 4021129, 4021172, 4021215, 4021258, 4021301, 4021344, 4021387, 4021430, 4021473, 4021516, 4021559, 4021602, 4021645, 4021688, 4021731, 4021774, 4021817]
- historical directories overwritten: no

The project did not contain the optional docs/current_state.md,
docs/next_task.md, or docs/claim_gate.md files at audit time.

## Reproduction integrity

- episode rows: 240
- step rows: 22593
- step groups: 240
- step-index continuity issues: 0
- static same-scenario pairing: True
- static pairing intersection: 20 scenarios
- reward-return checks: 100 traces, missing 0, max absolute difference 0.000000000
- requested plots unavailable because of missing fields: none

## Unified normal far-distance test comparison

| Policy | Episodes | 8m | 5m | 3m | 1m | Stable | Min distance (m) | Mean max hold | Hold >=1 | Hold >=3 | Hold >=5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline RL | 20 | 20/20 | 17/20 | 11/20 | 1/20 | 0/20 | 3.135 | 0.000 | 0/20 | 0/20 | 0/20 |
| DAgger Actor | 20 | 20/20 | 20/20 | 20/20 | 20/20 | 20/20 | 0.464 | 5.000 | 20/20 | 20/20 | 20/20 |
| Fine-tuned Actor | 20 | 20/20 | 20/20 | 0/20 | 0/20 | 0/20 | 3.584 | 0.000 | 0/20 | 0/20 | 0/20 |
| BC Actor | 20 | 15/20 | 5/20 | 1/20 | 0/20 | 0/20 | 6.945 | 0.000 | 0/20 | 0/20 | 0/20 |
| Oracle / Teacher | 20 | 20/20 | 17/20 | 11/20 | 1/20 | 0/20 | 3.135 | 0.000 | 0/20 | 0/20 | 0/20 |

## Unified static holdout comparison

| Policy | Episodes | 8m | 5m | 3m | 1m | Stable | Min distance (m) | Mean max hold | Hold >=1 | Hold >=3 | Hold >=5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline RL | 20 | 0/20 | 0/20 | 0/20 | 0/20 | 0/20 | 26.109 | 0.000 | 0/20 | 0/20 | 0/20 |
| DAgger Actor | 20 | 0/20 | 0/20 | 0/20 | 0/20 | 0/20 | 24.647 | 0.000 | 0/20 | 0/20 | 0/20 |
| Fine-tuned Actor | 20 | 0/20 | 0/20 | 0/20 | 0/20 | 0/20 | 24.738 | 0.000 | 0/20 | 0/20 | 0/20 |
| BC Actor | 20 | 8/20 | 3/20 | 1/20 | 0/20 | 0/20 | 9.451 | 0.000 | 0/20 | 0/20 | 0/20 |
| Oracle / Teacher | 20 | 20/20 | 20/20 | 20/20 | 20/20 | 19/20 | 0.081 | 4.850 | 20/20 | 19/20 | 19/20 |

The funnel is a sequential intersection, so its stage counts are non-increasing;
it is not a set of unrelated marginal event counts.

## Figure/data mapping

1. 01_cumulative_reward_curve.png uses raw baseline train episode returns and cumulative mean.
2. 02_mappo_training_loss.png uses real baseline trainer metrics, one row per PPO update; it is not BC MSE.
3. 03_xy_trajectory_overview.png uses three policies on one paired static scenario with obstacle, goal, boundary, and event markers.
4. 04_rollout_diagnostics.png uses one BC episode that passed/recovered but did not enter 1 m where such an episode exists.
5. 05_teacher_vs_actor_action_scatter.png uses finite Teacher/Actor values attached to the same BC evaluation observation record.
6. 06_raw_vs_applied_action.png shows raw, supervisor/boundary/applied action paths and correction norm.
7. 07_dataset_distance_phase_heatmap*.png uses actual demonstration split, collection mode, phase and distance bins.
8. 08_task_completion_funnel.png shows BC Actor and Oracle/Teacher sequential static-holdout stages.

## Current behavior diagnosis

- obstacle bypass: BC passed obstacle 18/20 and post-bypass recovery 18/20.
- post-bypass radial return: mean Actor radial action -0.021 m/s versus Teacher 0.222 m/s; negative radial-action fraction=0.488.
- post-bypass lateral-persistence proxy: mean tangential velocity 0.930 m/s.
- execution intervention: BC static safety steps=5, boundary steps=34, pre-risk unsafe steps=5.
- largest funnel drop: {'policy': 'bc_actor', 'from': 'post_bypass_recovery', 'to': 'entered_8m', 'drop': 10}

Teacher/Actor radial-action comparison in terminal bins:

- 5–3m: n=43, MAE=0.431, actor/teacher magnitude ratio=0.406
- 3–1m: n=19, MAE=0.417, actor/teacher magnitude ratio=0.321
- ≤1m: n=0, MAE=NA, actor/teacher magnitude ratio=NA

Dataset distance totals: {'>8m': 7681, '3–1m': 241, '5–3m': 265, '8–5m': 143, '≤1m': 212}.
Largest distance-bin share: >8m at 0.9.
Phase totals: {'boundary_onset': 21, 'bypass': 2133, 'decision': 44, 'normal': 4400, 'post_bypass': 1339, 'recovery': 605}.

## Answers to requested questions

1. Episode return first/last: -69.656 to -69.629; mean -69.032. Raw points are retained.
2. MAPPO loss: 100 trainer updates, fields ['actor_loss', 'critic_loss', 'entropy', 'approximate_kl', 'clip_fraction', 'loss']; source is baseline history, not the current BC/DAgger rerun.
3. BC failure location: static entered 1 m 0/20; stable 0/20.
4. Failure category: the direct measurable candidates are bypass, post-bypass radial recovery, lateral persistence and execution intervention; see the values above rather than treating any proxy as causal.
5. Teacher/Actor radial differences are in the scatter CSV by distance bin.
6. Action shrinkage is indicated by Actor/Teacher magnitude ratios below 1 in the scatter CSV; no statistical threshold is implied.
7. Static success attribution requires reading raw/applied correction and projection counts together; nonzero correction prevents Actor-only attribution.
8. Dataset dominance is quantified by the distance and phase totals above.
9. Funnel maximum drop: {'policy': 'bc_actor', 'from': 'post_bypass_recovery', 'to': 'entered_8m', 'drop': 10}.
10. The next evidence-led issues are to preserve far-distance navigation, reduce measured Actor/Teacher mismatch in the failing distance bins, and cover under-represented transition/recovery states before changing reward or safety semantics.

## Evidence boundary

This single-seed fixed-holdout output is a local diagnostic dashboard. It does
not establish multi-seed generalization, causal superiority, or manuscript-level
evidence. Missing optional fields and pairing caveats are recorded in the
manifest and output CSV files.
