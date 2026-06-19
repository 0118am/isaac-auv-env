# 第二阶段研究基础：AUV 动力学修正与增强仿真

**核心判断**  
当前阶段不做通用 PDE/流体场预测，而是在已有 **Fossen/Isaac 名义动力学** 上学习和估计低维 AUV 动力学误差：

```text
f_aug = f_nom(theta_hat) + Delta_tau_NN
```

**需要保留的文章基础**

- **Neural Lander: Stable Drone Landing Control using Learned Dynamics**  
  主要方法/结论：在名义动力学上学习未知扰动力，用 spectral normalization 约束神经网络，并证明带学习残差的控制器具有稳定性和扰动抑制能力。  
  我可以借鉴的点：把 WarpAUV 的未建模水动力写成 `Delta_tau`，用“名义 Fossen/Isaac 模型 + 有界神经残差”的形式建模。

- **Modelling of Underwater Vehicles using Physics-Informed Neural Networks with Control**  
  主要方法/结论：将 PINN 扩展为带控制输入的 PINC，用初始状态、控制动作和时间学习 BlueROV2 动力学，长时域预测优于纯数据驱动模型。  
  我可以借鉴的点：把控制输入、6-DOF 状态和名义动力学残差一起放进 loss，训练 physics-informed dynamics / rollout model。

- **Uncertainty-Aware Adaptive Dynamics For Underwater Vehicle-Manipulator Robots**  
  主要方法/结论：用 moving horizon estimation 在线估计物理参数，并加入物理一致性约束和不确定性评估，使估计参数可解释、可实现。  
  我可以借鉴的点：作为我的 alpha/参数估计主线，估计 `D_l, D_q, volume, COB, rotor_constant, tau` 等参数并给出置信度。

- **Champion-level drone racing using deep reinforcement learning / Swift**  
  主要方法/结论：先在仿真中训练 RL policy，再用少量真实数据识别感知和动力学残差，把残差注入仿真后继续 fine-tune。  
  我可以借鉴的点：建立 residual-augmented Isaac，用高保真/实机日志拟合动力学残差，再微调轨迹跟踪 policy。

- **Learn to Swim: Data-Driven LSTM Hydrodynamic Model for Quadruped Robot Gait Optimization / FED-LSTM**  
  主要方法/结论：用实验水动力数据训练 LSTM，证明历史窗口能更好预测非定常、非线性水动力，并优于经验公式模型。  
  我可以借鉴的点：如果静态 MLP 残差不够，用 `[nu_r, action, thruster_state]_{t-H:t}` 的历史序列预测 `Delta_tau`。

- **Learning Adaptive Hydrodynamic Models Using Neural ODEs in Complex Conditions**  
  主要方法/结论：用 Neural ODE 和 attention 从传感数据中学习复杂水下条件下的自适应水动力模型，适合不同速度和流体条件。  
  我可以借鉴的点：作为连续时间残差模型备选，用于处理水流、速度变化或参数随时间变化的情况。

**不作为当前主线**  
Poseidon / PROSE-FD / BCAT / NeuralOperator-FNO 主要面向 **Eulerian 流体场或 PDE 网格预测**；除非后续有 AUV 周围 CFD/流场数据，否则当前只作为远期方向。

**建议实验链路**

```text
轨迹日志 -> 参数估计 baseline -> 神经残差 -> PINC rollout loss
        -> 增强 Isaac -> policy fine-tune -> OOD 轨迹 RMSE 对比
```

Sources: Neural Lander, Swift, PINC, uncertainty-aware adaptive dynamics, FED-LSTM, Neural ODE hydrodynamics.
