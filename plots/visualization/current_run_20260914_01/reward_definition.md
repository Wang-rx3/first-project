# Reward definition

The episode-return curve uses the baseline training episode_metrics.csv field episode_return for split=train. It is the sum of step rewards and is not cumulatively summed across episodes. The cumulative mean is only a display statistic.

- episode source: D:\research project\logs\preliminary\single_uav_braking_progress025_seed2026\episode_metrics.csv
- baseline trace root: D:\research project\data\synthetic\single_uav_braking_progress025_seed2026
- traces checked: 100
- missing reward traces: 0
- maximum absolute difference between episode_return and step-reward sum: 0.000000000
- MAPPO loss is read from trainer training_metrics.csv and is never replaced by BC MSE.
