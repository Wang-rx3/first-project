# 4 实验协议

## 4.1 数据集分工

本课题的训练、验证、测试、消融和参数检查全部使用纯 Python 环境生成的数据；不同实验只使用不同的场景配置、随机种子和数据划分，不拼接外部数据集。

| 实验 | Python 数据配置 | 目标 |
| --- | --- | --- |
| E1 | `python_sim_benchmark`：GNSS denied/退化、可控传感器噪声 | Distributed EKF 协同定位和协方差一致性 |
| E2 | `python_sim_benchmark`：3/5/7 架无人机、可控测距和动态图 | GAT 拓扑特征、注意力和连通性评估 |
| E3 | `python_sim_benchmark`：异构传感器配置和时间偏移 | 时间同步、延迟消息和多源融合鲁棒性 |
| E4 | `python_sim_benchmark`：信道参数扫描和重复种子 | RSSI/SNR、丢包/时延/AoI 模型敏感性检查 |
| E5 | `python_sim_benchmark`：联合动作和障碍场景 | MAPPO 通信—定位联合优化主实验 |
| E6 | `python_sim_preliminary` / `python_sim_pilot` | 工程联通性、规模扩展和绘图流程 |

## 4.2 自建 benchmark 场景

当前主实验环境为基于 NumPy 的纯 Python 仿真环境，并使用 PyTorch 预留 GAT/MAPPO 张量接口；Flightmare 仅作为历史兼容说明，不作为本研究主实验的运行依赖。

自建仿真数据覆盖 3、5、7 架无人机，建筑物和动态车辆，LOS/NLOS 切换，GPS 正常/退化/失效，以及不同通信半径、带宽、丢包率和时延。训练、验证和测试场景使用不同的布局和随机种子；测试集不复用训练场景。

每个 benchmark 目录按 `train/`、`validation/`、`test/` 划分，每个 episode 独立保存场景和数据流：

- `scenario.json`：Python 仿真环境版本与配置、坐标系、采样率、场景参数和随机种子；
- `ground_truth.csv`：100 Hz 物理真值；
- `imu.csv`：100 Hz 带零偏和噪声 IMU；
- `uwb_ranges.csv`：10 Hz LOS/NLOS 测距观测；
- `visual_odometry.csv`：10 Hz 视觉里程计观测；
- `gnss.csv`：5 Hz GNSS 正常、退化和失效状态；
- `links.csv`：10 Hz 有向通信边；
- `packet_events.csv`：逐包发送、接收和丢包事件；
- `actions.csv`、`rewards.csv`：5 Hz MAPPO 交互记录；
- `obstacles.csv`：建筑物、车辆和行人状态；
- `trajectory.csv`：面向快速分析的汇总表。

正式数据集的 episode 数量配置为：调试集 30×30 秒、论文主数据集 300/60/60 个训练/验证/测试 episode，每个 60 秒；测试集固定加入 7 架无人机和未见过的场景布局。开发期可先使用 `generate-episodes --preset smoke`。

## 4.3 评价指标

- 定位：位置 RMSE、速度 RMSE、协方差一致性和 GNSS 失效期间的误差增长；
- 通信：丢包率 `P_loss`、通信时延 `D_com`、AoI 和网络连通度 `C_net`；
- 轨迹：任务完成率、轨迹长度、能耗、碰撞率和轨迹平滑性；
- 联合目标：

  \[
  J_t =
  w_{\mathrm{unc}}\bar E_{\mathrm{uncert},t}
  +w_{\mathrm{loss}}\bar P_{\mathrm{loss},t}
  +w_{\mathrm{AoI}}\overline{\mathrm{AoI}}_t
  +w_{\mathrm{delay}}\bar D_{\mathrm{com},t}
  -w_{\mathrm{net}}\bar C_{\mathrm{net},t}.
  \]

  其中各权重使用 `2_system_model.md` 符号表中的规范名称，不再新增未登记的希腊字母权重。

## 4.4 消融和泛化

比较以下设置：仅轨迹优化、仅通信优化、仅定位优化、去除 GAT、去除 Distributed EKF、去除通信模型，以及完整方法。另在未参与训练的障碍布局、通信条件和无人机数量上进行泛化测试。

## 4.5 数据可追溯性

每个 Python 数据集必须记录生成器版本、配置哈希、随机种子、坐标系、时间基准、场景编号、训练/验证/测试 split 和文件哈希。模型生成字段必须在 `scenario.json` 和 manifest 中标记为 `generated_by_model`，不表述为实测数据。论文文件只作为知识来源，不进入数据目录或参数校准流程。

## 4.6 三架无人机最小闭环 pilot

正式实验前，先使用固定随机种子运行独立的 `run-pilot`，验证纯 Python 环境、Distributed EKF、GAT、MAPPO 和 episode 日志能够完整联通。pilot 默认使用 3 架无人机、GNSS-denied、20 个决策步、CPU PyTorch，并依次执行归一化 warm-up、物理通信/EKF probe、GAT/MAPPO smoke、短训练和训练后验证。

pilot 输出位于 `data/synthetic/python_sim_pilot/`，与 `python_sim_benchmark` 分离，包含 `episode-v1.2` 的训练/验证文件、`normalization_stats.json`、GAT 注意力、PPO 训练指标、参数变化、探测结果和确定性校验。pilot 参数来自 Python 配置，pilot 结果只能证明软件接口和数据链路可执行，不能替代正式多随机种子实验或性能结论。

运行命令：

```powershell
& "C:\Users\lenovo\anaconda3\python.exe" main.py run-pilot --output data/synthetic/python_sim_pilot --seed 2026 --device cpu
& "C:\Users\lenovo\anaconda3\python.exe" main.py validate-episodes data/synthetic/python_sim_pilot
& "C:\Users\lenovo\anaconda3\python.exe" main.py audit-symbols
```

## 4.7 3/5/7 架无人机工程预实验

Pilot 通过后，先运行 3、5、7 架无人机的短周期工程预实验。每个规模独立初始化 MAPPO/GAT，默认使用 30 个训练 episode、10 个验证 episode、10 个测试 episode，每个 episode 包含 50 个决策步。训练阶段执行 MAPPO 更新，验证和测试阶段使用确定性策略且不执行参数更新。

各规模结果分别写入 `data/synthetic/python_sim_preliminary/scale_3/`、`scale_5/` 和 `scale_7/`，并生成 checkpoint、episode-v1.2 数据、训练指标、注意力摘要、评估 CSV 和 PNG 图表。该阶段使用已登记的 Python 通信/定位参数，只用于检查规模扩展、训练稳定性、日志完整性和绘图流程；正式论文结论仍需增加多随机种子、对照方法和参数敏感性实验。

运行命令：

```powershell
& "C:\Users\lenovo\anaconda3\python.exe" main.py run-preliminary --output data/synthetic/python_sim_preliminary --uav-counts 3 5 7 --seed 2026 --steps 50 --train-episodes 30 --validation-episodes 10 --test-episodes 10 --device cpu
& "C:\Users\lenovo\anaconda3\python.exe" main.py validate-preliminary data/synthetic/python_sim_preliminary
& "C:\Users\lenovo\anaconda3\python.exe" main.py summarize-preliminary data/synthetic/python_sim_preliminary
```
