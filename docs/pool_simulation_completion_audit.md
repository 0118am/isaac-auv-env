# 高保真水池动力学实施完成度审计

审计日期：2026-06-21

范围：原始目标要求尽可能准确复现水池中的 6-DOF 运动、推进器响应、边界/自由液面/缆绳扰动及闭环量测误差，并把实测数据可靠地转换成仿真参数。

## 结论

当前工作树已经完成 P0-P2 主干模型、环境配置接入、数学级测试和主要 CSV 标定 pipeline，但还不能证明“最大程度准确的真实水池仿真”已经实现。决定性缺口不是基础方程，而是：

1. 尚无目标水池、目标 BlueROV2 配置的实测日志和 measured profile，默认附加质量仍可为零、推进器仍可使用厂家/解析先验。
2. 尚无 Isaac/PhysX 运行级的真实轨迹回放、阶跃响应、自由衰减和闭环跟踪误差对比证据；当前自动测试主要是纯数学/配置测试。
3. 物理传感器函数尚未接入真实 EKF 状态传播，安装外参/时钟偏移也未形成完整运行链。
4. 字面意义的最高保真 P3 项仍未实现：advance-ratio/桨叶完整模型、CFD/势流边界查表、波浪/晃荡、动态 cable 接触与放线机构。

因此本审计不支持把总目标标记为完成。

## 要求与证据

| 要求 | 当前证据 | 状态 |
|---|---|---|
| 刚体质量、COM、完整 3x3 惯量可配置并写入 PhysX | `RigidBodyProfile`、`rigid_body_properties.py`、`warpauv_env.py` runtime mass/inertia/COM 写入；静态 CSV pipeline 可拟合质量和惯量 | 已实现并有数学/配置测试 |
| 浮力、排水体积和 COB 恢复力矩 | `calculate_buoyancy_forces(...)`；静浮/姿态恢复力矩拟合与静态 pipeline | 已实现并有直接数学测试 |
| 相对水流速度 `nu_r` | `HydrodynamicForceModels.calculate_relative_velocity(...)`；环境中阻尼、附加质量和推进器入流均使用相对流速 | 已实现并有直接数学测试 |
| 常值、随机时间水流与空间流场 | 环境随机过程、`calculate_trilinear_current_field(...)`、环境 CSV pipeline | 已实现并有数学/插值测试 |
| 对角/full 6x6 附加质量 | `calculate_added_mass_inertia_wrench(...)`、`calculate_added_mass_coriolis_wrench(...)`、full-matrix pipeline | 已实现并有功率/符号/合成辨识测试 |
| 对角/full 6x6 线性与二次阻尼 | full matrix 运算、速度相关倍率、耗散投影和 passivity gate | 已实现并有耗散与合成辨识测试 |
| 推进器位置/方向、静态曲线、入流曲面、一阶动态、延迟/量化/丢包 | `thruster_dynamics.py` 与推进器 CSV pipeline | 已实现并有直接数学/合成辨识测试 |
| 电压缩放、电池下垂、尾流干扰和 reaction torque | 环境运行路径已接入；电池下垂/电压指数函数和推进器 pipeline 有合成测试 | 模型与标定链已实现；缺目标硬件实测数据 |
| 近壁/池底和自由液面经验修正 | `pool_effects.py`、环境接入、环境 CSV pipeline | 已实现简化模型并有正反向测试；非 CFD 高保真 |
| tether 单段和准静态多段动力学 | `tether_dynamics.py`、环境接入、tether CSV pipeline | 已实现中等保真并有直接测试；缺动态接触/放线 |
| 观测延迟、噪声、bias、丢帧、滤波和漂移 | `sensor_models.py`、环境 policy observation 接入 | 已实现并有直接测试 |
| IMU、深度、DVL、外部定位物理量测 | 独立传感器函数和 `SensorProfile` | 函数级实现；尚未形成真实 EKF/外参运行链 |
| 参数 profile、JSON、合并、审计和 domain randomization | `pool_dynamics_profile.py` 与 builder/audit CLI | 已实现并有 round-trip/audit 测试 |
| 实验任务、updates 模板、CSV schema 和日志校验 | calibration task/template/schema API 与 audit CLI | 已实现并有文件级测试 |
| 静态、推进器、环境、6-DOF 水动力、tether 自动拟合 | `fit_pool_*_logs.py` workflows | 已实现并有合成端到端测试 |
| 真实水池轨迹复现精度 | 无目标水池数据、无 measured profile、无 replay RMSE 报告 | 缺失外部证据 |

## 验证覆盖边界

当前 `tests/test_dynamics_math.py` 覆盖 78 个与 added mass、damping、thruster、boundary、surface、tether、current、sensor、profile 或 calibration pipeline 相关的测试函数。它能证明：

- 公式符号、张量形状、插值、耗散性和配置映射符合当前实现约定。
- 合成数据可恢复预设参数，并能生成可验证的 `PoolDynamicsProfile` updates。
- 主要 CLI 的核心函数和 JSON/CSV 文件路径可工作。

它不能证明：

- Isaac/PhysX 中高频积分、接触、求解器参数和实际 USD 刚体属性组合后的轨迹精度。
- 实测传感器时钟、坐标外参、滤波器和控制链的端到端延迟。
- 目标水池墙面、泵流、自由液面、缆绳及推进器相互作用的经验倍率是否正确。

## 剩余工作优先级

### 阻止“真实高保真完成”的必需项

1. 采集并校验目标车辆/水池的静态、推进器、水流、6-DOF 水动力和 tether 日志。
2. 运行各 `fit_pool_*_logs.py`，合并生成 measured profile，并通过 `audit_pool_profile.py --fail-on-warning`。
3. 增加 Isaac 运行级验证：相同输入下对比实测和仿真的自由衰减、轴向阶跃、yaw/heave 响应及轨迹 RMSE。
4. 用留出实验验证参数，而不是只在用于拟合的日志上报告残差。

### P1/P2 可继续补强

- 增加传感器 reference CSV 到 `SensorProfile` scale/bias/noise/dropout 的自动拟合。
- 把物理传感器和真实 EKF 接入训练/评估运行链，加入安装外参与时钟偏移。
- 为 wake、reaction torque 和 speed-dependent damping 增加专用实验 pipeline。

### P3 可选项

- 动态离散 cable、池壁/机器人接触和放线机构。
- CFD/势流/边界元近壁与自由液面查表。
- 完整 advance-ratio 桨叶模型、瞬态尾流和自由液面波浪/晃荡。

## 当前验证命令

```bash
/home/jining_yang/miniconda3/envs/env_isaaclab/bin/python tests/test_dynamics_math.py
/home/jining_yang/miniconda3/envs/env_isaaclab/bin/python -m py_compile \
  calibration_tools.py pool_dynamics_profile.py rigid_body_hydrodynamics.py \
  thruster_dynamics.py pool_effects.py tether_dynamics.py sensor_models.py \
  water_current_fields.py custom_workflows/fit_pool_static_logs.py \
  custom_workflows/fit_pool_thruster_logs.py custom_workflows/fit_pool_environment_logs.py \
  custom_workflows/fit_pool_hydrodynamics_logs.py custom_workflows/fit_pool_tether_logs.py
```
