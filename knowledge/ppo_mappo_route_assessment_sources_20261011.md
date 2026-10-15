# PPO / MAPPO 路线评估的一手来源

用途：2026-10-11 A-only 基础导航关键可行性评估。不是系统文献综述，不据本文献推断本项目单UAV性能。

1. Schulman, J., Wolski, F., Dhariwal, P., Radford, A., Klimov, O. *Proximal Policy Optimization Algorithms*, 2017, arXiv:1707.06347v2. https://arxiv.org/abs/1707.06347
   - 官方摘要核验：方法交替执行在线环境采样和多轮小批优化；论文在仿真机器人运动与Atari任务报告经验表现。
   - 可支持：PPO确实是可用于连续控制的在线策略梯度方法。
   - 不可支持：720步迁移smoke必能学习本项目严格停车，或本项目失败证明PPO普遍不可行。

2. Yu, C., Velu, A., Vinitsky, E., Gao, J., Wang, Y., Bayen, A., Wu, Y. *The Surprising Effectiveness of PPO in Cooperative, Multi-Agent Games*, NeurIPS 2022 Datasets and Benchmarks, arXiv:2103.01955v4. https://arxiv.org/abs/2103.01955
   - 官方摘要核验：PPO-based多智能体方法在particle-world、SMAC、football、Hanabi四类协作环境得到较强经验结果，并研究实现及超参数因素。
   - 可支持：保留MAPPO作为协作决策学习方法有文献基础。
   - 不可支持：本项目的EKF/GAT/UAV通信任务或固定导航拆分已被验证。

读取范围为官方arXiv页面与摘要。论文的经验结论与本项目的新证据分别登记，固定基础导航的路线建议是结合本项目目标与机会成本的工程研究判断，不是上述论文直接证明的结论。
