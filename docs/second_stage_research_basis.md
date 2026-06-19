# 第二阶段研究基础筛选

日期：2026-06-14

结论先放前面：第二阶段不应把重点放在通用 PDE foundation model 上。你当前系统的数据和问题形态是 6-DOF AUV 状态、推进器动作、Fossen 风格名义动力学、轨迹跟踪误差，以及可能的实机/高保真仿真残差。真正能作为研究基础的主线是：

```text
名义 Fossen/Isaac 动力学
  + 可解释参数估计
  + 学习型残差水动力
  + 不确定性/置信区间
  + 用残差增强仿真再训练 policy
```

因此保留 6 类工作，其中 4 类是主线，2 类是辅助模型选择依据。

## 当前项目锚点

当前仓库已经有一个合适的名义模型底座：

- `warpauv_env.py` 中默认水动力参数包括 `linear_damping`、`quadratic_damping`、`added_mass_diag`、`volume`、`com_to_cob_offset`、`water_current_w`。
- `rigid_body_hydrodynamics.py` 施加的流体力为浮力、相对速度阻尼和可选附加质量 Coriolis 项。
- PhysX 负责刚体惯性、重力和积分，因此第二阶段不应再完整重写刚体动力学，而应学习/估计流体和执行器相关的不确定项。

第二阶段最自然的研究问题可以表述为：

```text
给定观测状态 x_t、动作 a_t、名义模型 f_nom、以及实际/高保真转移 x_{t+1},
学习或估计一个 physically constrained 的动力学修正项，
使增强仿真 f_aug = f_nom + Delta_f 能更准确预测 AUV 运动并提升轨迹跟踪 policy 的泛化。
```

## 保留的工作

| 优先级 | 工作 | 保留原因 | 在本项目中的用法 |
| --- | --- | --- | --- |
| 主线 1 | Neural Lander | 名义动力学 + 神经残差 + 稳定性约束，逻辑最贴近“在已有物理模型上补未知力” | 学习 AUV 未建模水动力残差 `Delta tau`，并用 spectral normalization 限制残差模型 |
| 主线 2 | PINC underwater vehicle | 明确面向水下车辆，输入包含初始状态、控制和时间，用物理约束改善长时域预测 | 做 WarpAUV 的 physics-informed controlled dynamics model 或 rollout loss |
| 主线 3 | Uncertainty-aware adaptive dynamics | 在线估计物理参数并给出不确定性，和你的 alpha/参数估计路线最匹配 | 对阻尼、浮力、附加质量、推进器系数做带约束的在线/离线估计 |
| 主线 4 | Swift / Nature drone racing | 给出 sim-to-real 的完整闭环：真实数据识别残差，增强仿真，fine-tune policy | 用少量真实/高保真轨迹拟合残差，再在增强 Isaac 中重训或微调策略 |
| 辅助 1 | FED-LSTM | 证明序列模型能从实验数据学习非定常水动力 | 当静态 MLP 残差不足时，用状态-动作历史窗口预测残差 |
| 辅助 2 | Neural ODE + attention hydrodynamic model | 连续时间自适应水动力模型，适合处理速度/水流条件变化 | 作为 residual ODE 或参数随时间变化模型的备选结构 |

## 1. Neural Lander

来源：Guanya Shi 等，`Neural Lander: Stable Drone Landing Control using Learned Dynamics`, ICRA 2019 / arXiv 1811.08027。

这篇不是水下机器人论文，但方法结构非常适合你的第二阶段。它把系统写成“名义动力学 + 未知扰动力”，用 DNN 学习高阶空气动力相互作用，并通过 spectral normalization 约束网络 Lipschitz 常数。论文的关键不是“无人机降落”，而是这套思路：

```text
真实动力学 = 名义模型 + 未知残差力
残差力由神经网络从状态/控制输入中学习
网络输出进入控制律或仿真器
用 Lipschitz 约束降低闭环不稳定风险
```

迁移到 WarpAUV 时，可以把无人机 ground effect residual 换成水下未建模项：

```text
M nu_dot = tau_thr + tau_fluid_nom + Delta_tau_hydro
Delta_tau_hydro = NN(z_t)
```

其中 `z_t` 可包括：

- `nu_r`：相对水流速度。
- `eta` 或姿态相关量：影响浮力矩和耦合项。
- 推进器命令或实际推力：捕捉推进器-流体耦合。
- 历史窗口：如果存在明显非定常效应。

可直接借鉴的点：

- 用名义物理模型先解释大部分动力学，只让网络学残差。
- 残差目标可以由加速度反推，例如 `Delta_tau = M * nu_dot_meas - tau_nom`，注意要滤波和同步动作延迟。
- 对残差网络加 spectral normalization 或其他 Lipschitz/权重约束，避免残差模型在闭环中放大错误。

不要照搬的点：

- 它主要关注位置动力学和近地空气动力，不直接给出 AUV 的 Fossen 参数。
- 它的控制律可以作为理论参考，但你当前更可能把残差模型用于增强 Isaac 仿真和 PPO 微调，而不是直接替换控制器。

## 2. PINC for underwater vehicle

来源：Abdelhakim Amer 等，`Modelling of Underwater Vehicles using Physics-Informed Neural Networks with Control`, arXiv 2504.20019。

这是最直接相关的一篇。它把 PINN 扩展成带控制输入的 PINC，用初始状态、控制动作和时间来学习水下车辆动力学转移。论文中采用 BlueROV2 的简化 4-DOF Fossen 模型，任务是开环系统辨识和长时域预测。

可直接借鉴的点：

- 输入不只是时间/坐标，还包括控制输入和初始状态，这适合 AUV 受动作驱动的动力学。
- loss 不只看一步预测，也可以加物理残差和 rollout loss。
- 名义物理模型可以作为 regularizer，而不是要求完全准确。
- 适合回答“为什么学习模型能外推到训练域外”：因为训练时引入了动力学方程约束。

迁移到 WarpAUV 时建议不要照搬 4-DOF 简化版本，而是用你当前 6-DOF 接口：

```text
state: eta, nu, optional thruster_state
control: 6 thruster commands or mapped wrench
physics residual: Fossen/Isaac nominal dynamics residual
prediction target: x_{t+k} or nu_dot / Delta_tau
```

推荐实现方式：

1. 先训练一个 supervised residual model：`Delta_tau = NN(x, a)`。
2. 再加入 PINC 风格 loss：要求增强模型 rollout 后的 `x_{t+1:t+H}` 接近日志轨迹。
3. 将物理残差项限制在水动力/推进器不确定部分，不重复学习 PhysX 已经处理的刚体重力和惯性积分。

局限：

- 原论文排除了推进器动态，而你的当前模型已经包含推进器一阶响应。
- 原论文是 4-DOF BlueROV2，不能直接作为 WarpAUV 的最终模型，只能借训练框架和 loss 设计。

## 3. Uncertainty-aware adaptive dynamics

来源：Edward Morgan 等，`Uncertainty-Aware Adaptive Dynamics For Underwater Vehicle-Manipulator Robots`, arXiv 2603.06548。

这篇适合作为你的 alpha/参数估计路线的主参考。它不是把所有东西交给黑箱网络，而是保持模型对 lumped physical parameters 线性，在线估计参数，并通过凸物理一致性约束保证估计结果可实现。它还用 moving horizon estimation 累积一段时间窗内的回归信息，并输出参数不确定性。

可直接借鉴的点：

- 参数估计对象应保持物理意义，例如阻尼、摩擦、浮力、惯性/附加质量、推进器系数。
- 估计时加入物理一致性约束，而不是允许参数任意漂移。
- 不只输出一个参数点估计，还输出置信区间或覆盖率，用来判断当前模型是否可信。
- moving horizon 比单步最小二乘更适合噪声和耦合强的水下系统。

迁移到 WarpAUV 的参数集合可以是：

```text
theta = {
  D_l, D_q,
  added_mass_diag,
  volume, com_to_cob_offset,
  rotor_constant, dyn_time_constant,
  optional water_current_w
}
```

建议作为第二阶段的第一条实验线：

1. 固定 policy，采集多种轨迹跟踪日志。
2. 用日志中的 `x_t, a_t, x_{t+1}` 估计 `theta`。
3. 比较 nominal、fixed identified、adaptive identified 三种模型的 rollout RMSE。
4. 再比较用这些模型训练/微调后的 policy OOD 跟踪误差。

这条线最容易写成有说服力的贡献，因为它能保持物理解释性，并能和你现有 `linear_damping`、`quadratic_damping`、`volume` 等代码参数自然对应。

## 4. Swift / champion-level drone racing

来源：Elia Kaufmann 等，`Champion-level drone racing using deep reinforcement learning`, Nature 2023。

这篇也不是水下机器人，但它非常适合支撑你的 sim-to-real 或 sim-to-sim residual augmentation 设计。它的核心做法是：先在仿真中用 model-free RL 训练策略，再用少量真实飞行数据识别感知和动力学残差，把残差模型加入仿真，最后在增强仿真中 fine-tune policy。

对你来说，最有用的不是无人机比赛任务，而是这个实验闭环：

```text
base simulator -> train policy
real/high-fidelity logs -> identify residual model
augmented simulator -> fine-tune policy
OOD evaluation -> compare robustness
```

迁移到 WarpAUV：

- 如果没有实机数据，可以先用“高保真动力学/扰动版本 Isaac”作为 pseudo-real。
- 残差可分为动力学残差和观测残差。你当前阶段优先做动力学残差。
- 评估指标直接沿用你已经在轨迹 notebook 中使用的位置 RMSE、速度 RMSE、动作范数和 OOD 轨迹集合。

这篇适合写在方法论部分，说明为什么“学习残差后回到仿真中训练 policy”是合理路线。

## 5. FED-LSTM hydrodynamic model

来源：Fei Han 等，`Learn to Swim: Data-Driven LSTM Hydrodynamic Model for Quadruped Robot Gait Optimization`, ICRA 2025 / arXiv 2505.03146。

这篇证明了 LSTM 可以从水槽/拖曳实验数据中学习非定常、非线性水动力。它的输入是多个时间步的流速、关节角和角速度，输出作用在机器人腿部的力和力矩。

对 WarpAUV 的价值是模型结构层面的：

- 如果 `Delta_tau = MLP(x_t, a_t)` 无法解释滞后、涡脱落、推进器响应或水流历史影响，可以改成序列模型。
- 输入窗口可以从论文中的 16 步思路迁移为：

```text
[nu_r, action, thruster_state, eta_error]_{t-15:t}
  -> Delta_tau_t 或 x_{t+1}
```

不建议作为第一主线，因为它研究的是腿式水下机器人，形态和 WarpAUV 差别较大；但它可以作为“非定常水动力需要历史窗口”的依据。

## 6. Neural ODE + attention hydrodynamic model

来源：Cong Wang 等，`Learning Adaptive Hydrodynamic Models Using Neural ODEs in Complex Conditions`, arXiv 2410.00490。

这篇用于复杂水下条件下的自适应水动力建模，使用 Neural ODE 和 attention 处理实时传感数据，并预测 force states。它适合作为连续时间学习动力学的参考。

对 WarpAUV 的可用点：

- 如果你希望残差模型不依赖固定离散步长，可以把残差写成 Neural ODE：

```text
d h / dt = f_theta(h, x, a)
Delta_tau = g_theta(h, x, a)
```

- attention 可以用于在不同速度、水流、姿态或推进器状态之间自适应加权。

但它不如 PINC 和 uncertainty-aware adaptive dynamics 直接，因为平台是 amphibious quadruped，不是推进器式 AUV。因此建议只作为模型备选，不作为第二阶段理论主支柱。

## 不纳入第二阶段主线的工作

下面这些工作很强，但当前不适合作为你第二阶段的主基础。原因不是质量不够，而是它们解决的问题形态和你现在的数据形态不同。

| 工作 | 判断 | 原因 |
| --- | --- | --- |
| Poseidon | 暂不纳入主线 | 预测 PDE solution operator，输入输出是流体场网格，例如密度、速度、压力。你现在没有围绕 AUV 的 Eulerian flow-field 数据。 |
| PROSE-FD | 暂不纳入主线 | 多模态 PDE foundation model，适合 2D 流体系统预测，不直接处理 AUV 状态-动作-力矩闭环。 |
| BCAT | 暂不纳入主线 | 做 2D fluid field next-frame prediction，适合有连续流场帧的数据，而不是当前的 6-DOF 日志。 |
| NeuralOperator / FNO | 暂不纳入主线 | 是成熟工具库，可自己训练 operator，但仍需要 PDE/CFD 场数据；当前阶段更应先做低维动力学残差和参数估计。 |

这些工作可作为远期方向：如果之后你生成 AUV 周围流场 CFD 数据，例如局部速度场、压力场、涡量场，再考虑 Poseidon/FNO/BCAT。否则现在引入它们会把问题从“水下机器人动力学辨识”变成“流体场预测”，研究范围会发散。

## 建议的第二阶段实验路线

第一步：建立日志数据集。

```text
log = {
  eta_t, nu_t, action_t, thruster_state_t,
  target_t, target_velocity_t,
  eta_{t+1}, nu_{t+1}
}
```

数据来源先用 Isaac nominal + 扰动/参数偏差版本，之后再替换成实机或更高保真仿真。

第二步：做可解释参数估计 baseline。

- 估计 `D_l, D_q, volume, com_to_cob_offset, rotor_constant, dyn_time_constant`。
- 对比 nominal 参数和 identified 参数。
- 指标：one-step error、multi-step rollout RMSE、轨迹跟踪 RMSE。

第三步：做神经残差模型。

```text
Delta_tau = NN(nu_r, eta, action, optional history)
```

- 先用 MLP + spectral normalization。
- 如果误差表现出滞后，再换 LSTM/GRU。
- 加物理约束：阻尼残差不要长期注入非物理能量，浮力/恢复力残差要有合理范围。

第四步：做 PINC 风格 rollout training。

- 把名义动力学和残差模型放进 rollout。
- 训练目标从一步 `Delta_tau` 扩展到 `H` 步状态预测。
- 比较是否改善 OOD 轨迹预测。

第五步：做增强仿真中的 policy 微调。

- `f_aug = f_nom(theta_hat) + Delta_tau_theta`
- 在增强 Isaac 中 fine-tune trajectory policy。
- 用 lissajous、helix、spiral、chirp、racetrack 评估泛化。

## 可写入论文的精简表述

第二阶段可以这样表述：

> 本阶段不直接采用通用 PDE foundation model 预测流体场，而是在已有 Fossen/Isaac 名义动力学基础上学习和估计低维 AUV 动力学误差。具体而言，参考 Neural Lander 的名义模型加神经残差思想、PINC 的控制输入物理约束建模、以及 uncertainty-aware adaptive dynamics 的物理一致参数估计框架，本文将推进器驱动 AUV 的未建模水动力表示为可解释参数偏差与有界神经残差的组合。随后借鉴 Swift 的 residual simulator augmentation 思路，将识别出的残差模型注入仿真并微调轨迹跟踪策略，以提升分布外轨迹和动力学扰动下的跟踪鲁棒性。

## 参考来源

- Neural Lander: Stable Drone Landing Control using Learned Dynamics, arXiv 1811.08027: https://arxiv.org/abs/1811.08027
- Champion-level drone racing using deep reinforcement learning, Nature 2023: https://www.nature.com/articles/s41586-023-06419-4
- Modelling of Underwater Vehicles using Physics-Informed Neural Networks with Control, arXiv 2504.20019: https://arxiv.org/abs/2504.20019
- Uncertainty-Aware Adaptive Dynamics For Underwater Vehicle-Manipulator Robots, arXiv 2603.06548: https://arxiv.org/abs/2603.06548
- Learn to Swim: Data-Driven LSTM Hydrodynamic Model for Quadruped Robot Gait Optimization, arXiv 2505.03146: https://arxiv.org/abs/2505.03146
- Learning Adaptive Hydrodynamic Models Using Neural ODEs in Complex Conditions, arXiv 2410.00490: https://arxiv.org/abs/2410.00490
- Poseidon: Efficient Foundation Models for PDEs, arXiv 2405.19101: https://arxiv.org/abs/2405.19101
- PROSE-FD: A Multimodal PDE Foundation Model for Forecasting Fluid Dynamics, arXiv 2409.09811: https://arxiv.org/abs/2409.09811
- BCAT: A Block Causal Transformer for PDE Foundation Models for Fluid Dynamics, arXiv 2501.18972: https://arxiv.org/abs/2501.18972
- NeuralOperator library: https://github.com/neuraloperator/neuraloperator
