# 高保真水池仿真的动力学参数与模型清单

本文面向当前 `isaac-auv-env` 中的 BlueROV2 Heavy/WarpAUV 仿真，目标是尽可能准确复现水池实验中的 6-DOF 运动、推进器响应和主要扰动。结论先说在前面：最高性价比路线不是直接上 CFD，而是以 Fossen 6-DOF 刚体水动力模型为主干，使用水池实验辨识参数，并逐步补上附加质量、耦合阻尼、推进器真实曲线、近壁/自由液面和传感器/执行器误差。

## 1. 当前仓库已有能力

当前代码已经有一个合理的水下动力学边界：

- Isaac/PhysX 负责刚体质量、惯量、刚体陀螺项和重力积分。
- `rigid_body_hydrodynamics.py` 额外施加流体外力：浮力、相对水流阻尼、可选附加质量科氏项。
- `thruster_dynamics.py` 实现 BlueROV2 Heavy 的 8 个 T200 推进器几何、一阶推进器动态、正反向不对称二次推力映射。
- `warpauv_env.py` 已经接入水流扰动、阻尼参数、推进器死区、推进器强度/时间常数随机化和质量/体积随机化。

因此后续增强应避免重复实现刚体动力学和重力，重点补“水体相对运动产生的外力/等效惯性”和“真实水池实验误差源”。

## 2. 必须纳入的核心模型

### 2.1 刚体质量属性

参数：

- `m`：整机湿前称量质量，含电池、配重、传感器、外设。
- `Ixx, Iyy, Izz, Ixy, Ixz, Iyz`：惯性张量，优先来自 CAD + 实测质量修正；高精度时用摆振/扭摆实验校准。
- `r_G`：质心相对机体系原点偏移。

当前状态：

- `bluerov2_heavy_model.py` 用 11.5 kg 和盒体估算对角惯量。
- `warpauv_env.py` 可运行时写入 PhysX mass/inertia/COM，惯量字段兼容 3 维对角、完整 3x3 矩阵或 PhysX 9 值扁平矩阵。

建议：

- 短期保留对角惯量，但用 CAD 或实验更新数值。
- 若外设、浮力块、机械臂不对称，可直接把 `inertia_diag` 配置替换为完整对称 3x3 惯量张量。

### 2.2 静水力：浮力与恢复力矩

参数：

- `rho`：水池水密度。淡水随温度约在 997-1000 kg/m^3 范围变化，精确仿真时记录水温。
- `V`：排水体积。
- `r_B`：浮心位置。
- `W = m g`，`B = rho g V`：重力和浮力。
- `r_BG = r_B - r_G`：浮心相对质心偏移，决定 roll/pitch 被动稳定性。

模型：

```text
F_b^w = -rho V g^w
tau_b^b = r_BG^b x R(q)^T F_b^w
```

当前状态：

- 已实现中性浮力体积和 `com_to_cob_offset`。

建议：

- 把质量、体积、浮心作为可辨识/可随机化参数，不要只随机质量或只随机体积。
- 加入“略正浮/略负浮”配置，因为水池实物很少严格中性。
- 通过倾斜静态实验辨识 `r_BG`：让机器人在水中自由静止，测 roll/pitch 平衡和恢复振荡。

### 2.3 相对流速与水池水流

参数：

- `v_current^w`：世界系水流速度。
- `tau_current`：水流一阶变化时间常数。
- `sigma_current`：随机扰动强度。
- 可选空间场：`v_current(x, y, z, t)`。

模型：

```text
nu_r = nu - nu_current
```

当前状态：

- 已有常值水流和一阶平滑随机水流。
- 已支持可选的三线性插值水流场：把水池局部坐标中的规则网格流速作为世界系水流增量，按机器人当前位置叠加到当前水流上。
- 已支持把多位置散点水流采样拟合成规则网格，再导出为 `water_current_field_*` 配置。

建议：

- 水池高保真优先用空间非均匀水流：泵、回流、墙面附近速度剖面会造成位置相关偏差；如果有 ADV/流速仪或视觉漂标数据，可用 `fit_water_current_field_grid(...)` 生成 `water_current_field_*` 网格参数。
- 若没有 ADV/流速仪，可用漂浮小球/中性浮标视频跟踪估计水流均值、方差和相关时间。

### 2.4 附加质量

参数：

- 最低配置：`M_A = diag(X_du, Y_dv, Z_dw, K_dp, M_dq, N_dr)`。
- 高保真配置：完整 6x6 对称附加质量矩阵，包含 surge-sway-yaw、heave-pitch 等耦合项。

模型：

```text
M = M_RB + M_A
C = C_RB + C_A
tau_A = -M_A dot(nu_r) - C_A(nu_r) nu_r
```

当前状态：

- 代码支持对角 6 维 `added_mass_diag`，也支持 full 6x6 附加质量矩阵。
- 已实现 `C_A(nu_r) nu_r` 和显式 `-M_A dot(nu_r)` 惯性项。
- 标定工具支持从阶跃/多轴激励日志联合拟合对角或 full 6x6 有效惯性、附加质量和阻尼；默认参数仍为 0，因此需要实测、辨识或文献参数后才会产生附加质量效果。

建议：

- 第一阶段：用 `fit_diagonal_added_mass_linear_quadratic_damping(...)` 从轴向阶跃/自由运动日志拟合非零对角附加质量，并利用现有差分 + 低通的 `dot(nu_r)` 估计接入 `-M_A dot(nu_r)`。
- 第二阶段：用 `fit_full_matrix_added_mass_linear_quadratic_damping(...)` 或文献数据把 `added_mass_diag` 替换为完整 6x6 矩阵。
- 继续谨慎调节 `added_mass_accel_filter_alpha`，避免差分加速度放大速度噪声。

### 2.5 水动力阻尼

参数：

- 线性阻尼：`D_l`，单位分别为 N/(m/s)、N m/(rad/s)。
- 二次阻尼：`D_q`，单位分别为 N/(m/s)^2、N m/(rad/s)^2。
- 高保真：完整 6x6 阻尼矩阵、速度相关阻尼、交叉项、升力/侧滑项。

模型：

```text
tau_D = -(D_l + D_q |nu_r|) nu_r
```

当前状态：

- 已有 6 个自由度的对角线性 + 二次阻尼。
- 代码支持 6x6 线性/二次阻尼矩阵，并支持按 `|nu_r|` 查表的速度相关阻尼倍率；倍率可是一条共用曲线，也可以是每个自由度一条曲线，线性阻尼和二次阻尼可分别配置。
- 标定工具支持从多轴激励日志直接拟合 full 6x6 线性/二次阻尼矩阵，可用于落地 `Y_r, N_v, Z_q, M_w` 等耦合项。

建议：

- 水池仿真最优先辨识 `D_q`，因为低速 ROV 的停止距离、转弯半径、轨迹跟踪误差通常对二次阻尼很敏感。
- 对 BlueROV2 这类非流线开放框架，sway/heave/yaw 阻尼不应简单等同于 surge。
- 高保真时引入耦合阻尼，例如 `Y_r, N_v, Z_q, M_w`，否则横移和偏航耦合会偏理想；这些项可用 `fit_full_matrix_linear_quadratic_damping(...)` 从多轴激励日志拟合。
- 如果自由衰减或恒推实验显示某自由度在不同速度段的阻尼拟合残差有系统偏差，可启用 `speed_dependent_damping_enabled` 并填入 `damping_speed_points`、`linear_damping_speed_scales` 和/或 `quadratic_damping_speed_scales`。这些曲线只是标定入口，数值仍应由实测拟合得到。

### 2.6 推进器与推进分配

参数：

- 每个推进器位置 `r_i`、推力方向 `e_i`。
- `T_i(cmd, V, v_inflow)`：PWM/归一化命令到推力的曲线。
- 正反向最大推力、死区、饱和、速率限制。
- 电机/ESC 时间常数、命令延迟、PWM 分辨率。
- 电池电压、电压下垂、ESC 限流。
- 可选反扭矩：推进器旋转产生的 reaction torque。

当前状态：

- 已有 8 个 T200 位置/方向、一阶响应、死区、正反向不对称二次推力。
- 已支持推进器命令步进延迟、归一化命令速率限制、归一化命令量化、通信丢包保持上一条命令、电池电压推力缩放和 episode 内电压下垂。
- 已支持推力台分段线性查表接口，可使用全推进器共用曲线或每个推进器独立曲线。
- 已支持二维实测推进器曲面：`command x axial_inflow_speed -> thrust`，可直接填入 advance-ratio/入流条件下的推力台或水洞标定结果；启用该二维表时，不再额外叠加简化二次入流损失，避免重复计算入流影响。
- 已支持简化轴向入流 `thrust-loss` 模型：车辆/水流沿当前推力方向冲刷推进器时按二次规律降低有效推力。
- 已支持简化推进器尾流干扰模型：上游推进器射流会按距离、尾流半径、扩散率和源推进器推力强度降低下游推进器有效推力。
- 已支持推进器反扭矩基础模型：按 signed thrust、推进器轴向、旋向和反扭矩系数施加 reaction torque。
- 仍未接入基于实测 advance-ratio 的完整桨叶模型。

建议：

- 用 Blue Robotics T200 曲线作为先验，但最好在本水池、本电池、本桨叶方向下做推力台标定。
- 电压依赖已支持基础形式；后续应用实测电池曲线替换默认二次缩放。
- 简化 `thrust-loss` 和二维 `command x axial_inflow_speed` 实测表均已支持；若有实测 advance-ratio 曲线，优先使用二维表替换经验二次损失。
- 简化尾流干扰已支持；后续用整机重复推力台或固定姿态阶跃实验标定尾流长度、半径、扩散率和损失系数。
- 反扭矩基础模型已支持；后续用电机/桨叶实验标定 `thruster_reaction_torque_coeff` 和旋向。
- 对垂直推进器和水平推进器分别标定，安装遮挡和框架干扰会导致同型号推进器有效推力不同。

## 3. 水池特有的高保真模型

### 3.1 边界效应：墙、底、自由液面

水池不是无限大水域，靠近池壁、池底和水面时，附加质量、阻尼和推进效率都会变化。

参数：

- 水池尺寸：`Lx, Ly, depth`。
- 机器人到墙/底/水面的距离。
- 近壁修正系数：`k_D(d)`, `k_MA(d)`, `k_T(d)`。

建议模型：

- 简化版：当距离小于 1-2 个车体长度时，对阻尼、附加质量、推进器推力乘以经验修正。
- 高保真版：用 CFD 或势流/边界元预计算查表。

当前代码已支持简化版 box-boundary 模型：在配置的水池边界附近按距离平滑放大阻尼/附加质量并降低推进效率。默认关闭；打开后可设置边界范围、影响距离和三个边界倍率。尚未接入 CFD、势流或实验预计算查表。

优先级：

- 如果实验轨迹远离墙、底、水面，先不加。
- 如果水池小、轨迹靠边、深度浅，近壁修正优先级很高。

### 3.2 水面波浪、晃荡和气泡

水池中通常没有海浪，但会有推进器激起的自由液面扰动、反射波和气泡。

当前代码已支持默认关闭的自由液面近场修正：当机器人接近配置的平面 `free_surface_z` 时，对 heave、roll、pitch 阻尼和附加质量施加经验倍率，并可降低等效浮力和推进器推力，用于近水面部分出水、自由液面附加惯性和推进器通气的一阶近似。该模型不是波浪/晃荡求解器，后续可用实验或 CFD 查表替换经验倍率。

建议：

- 机器人深度超过 1 个车体高度且速度低时，可忽略。
- 靠近水面或做 heave/yaw 大动作时，启用自由液面对 heave/roll/pitch 阻尼、附加质量、浮力和推力的修正。
- 视觉任务需要加入气泡、浑浊、焦散和反射；这影响感知，不一定直接影响动力学。

### 3.3 缆绳/安全绳/线束

如果水池实验存在 tether，动力学必须建模。

参数：

- 缆绳长度、直径、线密度、浮力/负浮力。
- 轴向刚度、阻尼、弯曲近似。
- 水阻系数、接触摩擦。
- 顶端锚点位置和放线策略。

当前代码已支持两级 tether 模型：默认一段式 slack cable，在超过松弛长度后施加非线性弹簧/阻尼拉力，并可叠加相对水流速度二次拖曳；也支持准静态多段 tether 近似，把线径、线材密度、水体浮力密度和分布拖曳折算到机体连接点。锚点、机体系连接点、松弛长度、刚度、阻尼、拖曳系数、分段数、线径和密度均可配置。尚未实现池壁/机器人接触、放线机构和完整动态 cable solver。

拉力计/弹簧秤实验可用 `fit_tether_spring_damper(...)` 从缆绳长度、张力和沿缆绳速度拟合 `tether_slack_length`、`tether_stiffness` 和 `tether_damping`。速度符号与仿真一致：`velocity_along_tether = body_velocity dot direction_to_anchor`，机器人远离锚点时该值为负，阻尼张力增加。若单独测得横向拖曳力，可用 `fit_tether_drag_coefficient(...)` 拟合 `tether_drag_coeff`。

CSV 日志可直接交给 tether pipeline；锚点、机体连接点、分段数、线径与密度由实测几何参数显式传入：

```bash
/home/jining_yang/miniconda3/envs/env_isaaclab/bin/python \
  custom_workflows/fit_pool_tether_logs.py \
  calibration_logs/ \
  --anchor-pos-w 0 0 8 \
  --attach-offset-b -0.2 0 0 \
  --num-segments 4 \
  --segment-diameter 0.006 \
  --segment-density 1200 \
  --output tether_updates.json \
  --report tether_fit_report.json
```

当自动 slack 搜索分辨率不足或已有几何测量先验时，可用 `--slack-candidates` 显式给出候选长度。pipeline 不会从张力曲线“猜”线径、密度或锚点，这些不可辨识量必须来自直接测量。

建议模型：

- 简化版：在机器人尾部施加一根非线性弹簧阻尼 + 二次拖曳。
- 中等保真版：准静态多段 cable model，折算浮力/负浮力、重力和分布拖曳。
- 高保真版：动态离散多段 cable solver，每段受浮力、重力、阻尼和接触。

## 4. 传感器与控制链误差

严格说这不是“水动力学”，但要复现水池实验闭环表现必须加入：

- IMU：角速度/加速度噪声、bias、温漂、安装外参误差。
- 深度计：压力噪声、零点偏置、温度漂移、低通滤波。
- DVL/视觉定位/动捕：延迟、丢帧、量测噪声、坐标系外参、低频漂移。
- 控制链：指令延迟、控制频率、零阶保持、通信抖动、PWM 量化。
- 状态估计器：EKF/滤波延迟和模型误差。

当前代码已支持通用 policy observation 层的高斯噪声、episode 常值 bias、固定步数延迟、低频更新保持、观测通道丢帧保持、一阶低通和 bias 随机游走漂移，可用于训练时做状态估计误差/传感器误差 domain randomization。噪声、bias、丢帧、低通和漂移既可以是标量/完整观测向量，也可以按语义通道配置，例如 `position_error_b`、`linear_velocity_b`、`angular_velocity_b`、`actions` 等。`sensor_models.py` 还提供了独立的物理传感器读数工具：IMU specific force/gyro、深度计、DVL body-frame velocity 和外部视觉/声学定位的量测、量程有效性、丢测保持、scale/bias/noise/限幅。推进器控制链已支持命令死区、步进延迟、丢包保持、速率限制、命令量化和一阶电机动态。尚未接入真实 EKF 状态传播。

对于强化学习策略，若训练观测是理想状态，实物只提供传感器估计状态，则传感器模型的收益可能和水动力参数同等重要。

## 5. 校准参数落地方式

当前代码新增了 `pool_dynamics_profile.py`，用于把水池实测参数收束成一个统一 profile，然后一次性写入 `WarpAUVEnvCfg` 或 `WarpAUVTrajEnvCfg`：

- `RigidBodyProfile`：质量、排水体积、惯量、COM、COB、水密度和黏度。
- `HydrodynamicsProfile`：线性/二次阻尼、速度相关阻尼倍率曲线、附加质量、附加质量加速度滤波、平均水流和可选三线性插值水流场。
- `ThrusterProfile`：推进器一阶时间常数、死区、延迟、速率限制、命令量化、丢包概率、正反向最大推力、一维实测 lookup 推力表、二维 `command x axial_inflow_speed` 实测推力表、入流损失、尾流干扰和反扭矩。
- `BatteryProfile`：标称电压、初始电压、最低电压、电压下垂和电压-推力指数。
- `PoolBoundaryProfile`：水池边界、近壁影响距离、阻尼/附加质量/推力修正系数。
- `FreeSurfaceProfile`：自由液面位置、影响距离、heave/roll/pitch 阻尼、附加质量、浮力和推力近场修正。
- `TetherProfile`：缆绳锚点、机体系连接点、松弛长度、刚度、阻尼、水阻、分段数、线径、线材密度和浮力密度。
- `ObservationProfile`：观测噪声、bias、固定步数延迟、更新周期、丢帧概率、低通系数和 bias 漂移；多数观测参数可用标量、完整观测向量或语义通道 dict 表达。
- `SensorProfile`：IMU、深度计、DVL 和外部定位的物理传感器参数，包括 scale、bias、noise、量程、丢测概率和 reference frame/range 设置；这些字段会输出为 cfg 参数，供物理传感器链或 EKF 接入。
- `DomainRandomizationProfile`：把实测不确定性转成 reset-time 随机化范围。

使用方式有两种：

```python
from .pool_dynamics_profile import PoolDynamicsProfile, apply_pool_dynamics_profile
from .warpauv_env import WarpAUVTrajEnvCfg

profile = PoolDynamicsProfile(...)
cfg = apply_pool_dynamics_profile(WarpAUVTrajEnvCfg(), profile)
```

或者在 cfg 上显式设置：

```python
cfg = WarpAUVTrajEnvCfg()
cfg.pool_dynamics_profile = profile
```

也可以把实测 profile 保存为 JSON 文件后加载：

```python
from .pool_dynamics_profile import load_pool_dynamics_profile_json

cfg = WarpAUVTrajEnvCfg()
cfg.pool_dynamics_profile = load_pool_dynamics_profile_json("measured_pool_profile.json")
```

各个标定函数的 `to_cfg_updates()` 可直接用 `merge_pool_dynamics_cfg_updates(...)` 反向合并到 nested profile；命令行入口 `custom_workflows/build_pool_profile_from_calibration.py` 支持把多个 JSON updates 文件按顺序叠加，后面的文件覆盖前面的字段：

```python
from .pool_dynamics_profile import merge_pool_dynamics_cfg_updates

profile = merge_pool_dynamics_cfg_updates(
    cfg_updates=[
        mass_fit.to_cfg_updates() | inertia_fit.to_cfg_updates(),
        volume_fit.to_cfg_updates() | cob_fit.to_cfg_updates(),
        damping_fit.to_cfg_updates(),
        thruster_fit.to_cfg_updates(include_deadband=True),
    ],
    domain_randomization_updates=current_fit.to_domain_randomization_updates(stage_count=3),
    name="measured-pool-2026-06-21",
)
```

```bash
/home/jining_yang/miniconda3/envs/env_isaaclab/bin/python \
  custom_workflows/build_pool_profile_from_calibration.py \
  --updates rigid_body_updates.json \
  --updates hydrodynamics_updates.json \
  --updates thruster_updates.json \
  --domain-randomization-updates randomization_updates.json \
  --name measured-pool-2026-06-21 \
  --output measured_pool_profile.json
```

`--updates` 文件既可以是 flat cfg updates，例如 `{"mass": 11.6, "added_mass_diag": [...]}`，也可以是 `{"cfg_updates": {...}, "domain_randomization_updates": {...}}` 包装格式。默认遇到未知字段会失败，避免把单位写错或把环境无关字段悄悄丢掉。

环境初始化时会自动 apply。默认 profile 与现有默认参数一致，且边界效应、tether、lookup 表、入流损失、反扭矩、观测误差和物理传感器误差默认关闭，因此不会改变旧训练脚本行为。

训练或实物对比前，可以用 `audit_pool_dynamics_profile(...)` 生成高保真缺口清单：

```python
from .pool_dynamics_profile import PoolProfileAuditOptions, audit_pool_dynamics_profile

report = audit_pool_dynamics_profile(
    profile,
    PoolProfileAuditOptions(
        near_boundaries_expected=True,
        near_surface_expected=True,
        tether_expected=True,
        spatial_current_expected=True,
        physical_sensors_expected=True,
    ),
)

for finding in report.findings:
    print(finding.severity, finding.section, finding.recommendation)
```

该审计不会修改 profile；它只指出哪些高影响模型仍在默认/关闭状态，例如零附加质量、未启用推进器实测表、无水流场、无近壁/自由液面修正、缺少 tether 或物理传感器参数等。

如果需要把缺口直接变成实验任务，可用 `pool_profile_calibration_tasks(...)` 或审计 CLI 的 `--checklist`。任务会包含 priority、section、触发原因、建议实验、对应标定函数和最终要写入的 cfg update keys；若要生成可填写的 JSON 骨架，用 `pool_profile_calibration_update_template(...)` 或 CLI 的 `--template`；若要给实验日志生成 CSV 列规范，用 `pool_profile_calibration_log_schemas(...)` 或 CLI 的 `--log-schemas` / `--write-log-templates`：

```python
from .pool_dynamics_profile import (
    pool_profile_calibration_log_schemas,
    pool_profile_calibration_tasks,
    pool_profile_calibration_update_template,
)

for task in pool_profile_calibration_tasks(profile, PoolProfileAuditOptions(physical_sensors_expected=True)):
    print(task.priority, task.section, task.calibration_functions, task.update_keys)

template = pool_profile_calibration_update_template(profile, PoolProfileAuditOptions(physical_sensors_expected=True))
schemas = pool_profile_calibration_log_schemas(profile, PoolProfileAuditOptions(physical_sensors_expected=True))
```

同一检查也可直接用于 JSON profile：

```bash
/home/jining_yang/miniconda3/envs/env_isaaclab/bin/python custom_workflows/audit_pool_profile.py \
  measured_pool_profile.json \
  --near-boundaries \
  --near-surface \
  --tether \
  --spatial-current \
  --physical-sensors \
  --fail-on-warning
```

加 `--json` 可输出机器可读报告；加 `--checklist` 可输出实验任务清单，`--checklist --json` 则输出机器可读任务列表；加 `--template` 会输出含 `update_payload.cfg_updates` 和 `update_payload.domain_randomization_updates` 的待填写模板；加 `--log-schemas` 会输出实验日志 schema；加 `--write-log-templates calibration_logs/` 会写出空 CSV 表头和 `schemas.json`；采集后用 `--validate-log-dir calibration_logs/` 检查缺文件、必需列、最少样本数、空值、非法布尔值和非有限数值，校验失败时退出码为 2。模板外层带有说明和任务元数据，不会被 profile builder 直接当成可合并 updates；填写完成后，把 `update_payload` 单独保存为 updates JSON，再交给 `custom_workflows/build_pool_profile_from_calibration.py`。`--fail-on-warning` 适合在训练前脚本或 CI 中阻止使用明显缺参的 measured profile。

## 6. 参数辨识实验建议

### 6.1 静态实验

- 干重称量：得到 `m`。
- 排水/中性浮力实验：得到 `V` 和净浮力。
- 倾斜恢复实验：估计 `r_BG`。
- CAD + 摆振实验：估计惯量。

称重和摆振实验可先更新刚体属性。`fit_mass_from_scale_readings(...)` 用多次称重读数给出 `mass`；`fit_inertia_tensor_from_axis_moments(...)` 用多个机体系转轴上的实测转动惯量拟合完整对称 3x3 惯量张量；复摆实验可先用 `compound_pendulum_moments_from_periods(...)` 或直接用 `fit_inertia_tensor_from_compound_pendulum(...)` 把小角度周期、质量和转轴到质心距离转换成 `inertia_diag`：

```python
from .calibration_tools import (
    fit_inertia_tensor_from_compound_pendulum,
    fit_mass_from_scale_readings,
)

mass_fit = fit_mass_from_scale_readings(scale_readings_kg)
inertia_fit = fit_inertia_tensor_from_compound_pendulum(
    axis_b_samples=pendulum_axes_b,
    period_s_samples=measured_period_s,
    mass=mass_fit.mass,
    pivot_to_com_distance_samples=pivot_to_com_distance_m,
)

rigid_body_updates = mass_fit.to_cfg_updates() | inertia_fit.to_cfg_updates()
```

惯量拟合至少需要 6 个独立轴向约束才能完全辨识 `Ixx/Iyy/Izz/Ixy/Ixz/Iyz`；`design_rank < 6` 时只能作为部分约束或 CAD 先验的校验。复摆公式假设小角度、刚性悬挂和已知质心位置，周期应取多次振荡平均值，避免手动秒表误差直接进入惯量。

静水力日志可用 `calibration_tools.py` 直接转成刚体 profile 参数。`fit_buoyancy_volume_from_forces(...)` 从实测浮力向量反推排水体积；`fit_com_to_cob_offset_from_buoyancy_wrenches(...)` 从体坐标浮力/恢复力矩拟合 `com_to_cob_offset`；若实验只记录静态姿态和恢复力矩，则用 `fit_com_to_cob_offset_from_static_torques(...)` 由姿态、体积和水密度自动生成体坐标浮力方向：

```python
from .calibration_tools import (
    fit_buoyancy_volume_from_forces,
    fit_com_to_cob_offset_from_static_torques,
)

volume_fit = fit_buoyancy_volume_from_forces(
    buoyancy_force_w_samples=measured_buoyancy_force_w,
    water_density=997.0,
    gravity_w=[0.0, 0.0, -9.81],
)

cob_fit = fit_com_to_cob_offset_from_static_torques(
    root_quats_w=static_orientation_quats_wxyz,
    buoyancy_torque_b_samples=measured_restoring_torque_b,
    volume=volume_fit.volume,
    water_density=997.0,
)

rigid_body_updates = volume_fit.to_cfg_updates() | cob_fit.to_cfg_updates()
```

体积拟合输入应是水对机器人施加的浮力，不是重力与浮力相加后的净力；净浮力实验若只得到 `B - W`，需先加回 `m g` 得到 `B`。浮心偏移最好用多个非共面姿态，否则某些方向不可观，`design_rank < 3` 时只能把结果当作部分约束。

按前述 CSV schema 采集后，可直接运行静态标定 pipeline。它会自动发现并校验 `rigid_body_mass_readings.csv`、`rigid_body_buoyancy_forces.csv`、`rigid_body_static_buoyancy_torques.csv`、`rigid_body_axis_moments.csv` 和 `rigid_body_compound_pendulum_periods.csv`，把存在的数据联合转换成 builder-compatible updates JSON：

```bash
/home/jining_yang/miniconda3/envs/env_isaaclab/bin/python \
  custom_workflows/fit_pool_static_logs.py \
  calibration_logs/ \
  --output static_updates.json \
  --report static_fit_report.json

/home/jining_yang/miniconda3/envs/env_isaaclab/bin/python \
  custom_workflows/build_pool_profile_from_calibration.py \
  --updates static_updates.json \
  --output measured_pool_profile.json
```

当轴向惯量和复摆周期两份日志同时存在时，pipeline 会先把复摆周期换算为 COM 转动惯量，再与直接轴向惯量样本联合拟合完整 3x3 张量；输出报告包含残差、设计矩阵秩和 PSD 投影前后最小特征值。

### 6.2 推进器实验

- 单推进器推力台：PWM/归一化命令、电压、电流到推力曲线。
- 正反向分别标定。
- 整机 bollard pull：验证每个轴向最大合力和力矩。
- 阶跃命令：估计推进器时间常数、延迟、饱和和死区。

推进器/电池 CSV 可用自动 pipeline 转成一维静态 lookup、二维 `command x inflow` lookup、`dyn_time_constant`、命令延迟步数、电池线性下垂和电压推力指数：

```bash
/home/jining_yang/miniconda3/envs/env_isaaclab/bin/python \
  custom_workflows/fit_pool_thruster_logs.py \
  calibration_logs/ \
  --physics-dt 0.02 \
  --output thruster_updates.json \
  --report thruster_fit_report.json
```

`thruster_index` 可以全部使用同一个标签，表示所有推进器共享一条曲线；若要保存独立曲线，必须提供严格的 `0` 到 `7` 八组数据，避免生成 `ThrusterProfile` 无法解释的两组或四组曲线。命令点和入流速度点由 CSV 中的离散值自动推断；每个 bin/cell 都必须有样本。`--physics-dt` 用于把拟合的秒级响应延迟换算成 `thruster_command_delay_steps`。若存在 `battery_voltage_thrust_samples.csv`，pipeline 会拟合当前 runtime 使用的线性 `V(t)=V0-drop*t` 模型；`thrust_scale` 列有值时还会按 `--nominal-voltage` 拟合电压推力指数。

### 6.3 自由运动实验

- 无推进自由衰减：估计线性阻尼、部分二次阻尼。
- 轴向恒推/阶跃：估计 `M_A + D` 的组合效应。
- yaw/roll/pitch 阶跃和衰减：估计旋转阻尼和附加惯量。
- 多轴激励轨迹：联合辨识耦合项。

当前代码新增了 `calibration_tools.py`，可把水池日志中的 `time_s`、相对速度 `nu_r`、已知外部激励 `applied_wrench` 和等效质量/惯量 `effective_mass` 拟合成可直接写入 profile 的阻尼参数：

```python
from .calibration_tools import fit_diagonal_linear_quadratic_damping

fit = fit_diagonal_linear_quadratic_damping(
    time_s=time_s,
    nu_r=relative_velocity_6d,
    applied_wrench=known_thruster_or_test_wrench_6d,
    effective_mass=mass_plus_added_mass_diag,
)
updates = fit.to_cfg_updates()
```

若常量阻尼在不同速度段残差明显不同，可进一步用 `fit_speed_dependent_damping_scales(...)` 拟合 `linear_damping_speed_scales` 和 `quadratic_damping_speed_scales`，并把 `fit.to_cfg_updates(speed_points)` 合并进 `HydrodynamicsProfile` 或环境 cfg。这个工具默认按对角 6-DOF 独立拟合；完整 6x6 耦合项仍建议用多轴激励数据做离线系统辨识。

对 full 6x6 拟合结果，建议先做物理一致性检查再写入 profile：`project_added_mass_to_physical(...)` 会把附加质量对称化并投影到正半定矩阵，`project_linear_damping_to_dissipative(...)` 会保证线性阻尼满足 `nu_r.T @ D @ nu_r >= 0`，`calculate_damping_dissipated_power(...)` / `damping_is_dissipative_for_samples(...)` 可用实测或设计速度包络检查线性+二次阻尼是否在该速度范围内耗能而不是产能。

`hydrodynamics_motion_wrench_log.csv` 可直接交给 6-DOF 自动 pipeline。默认 `full` 模式联合拟合完整附加质量、线性阻尼和二次阻尼矩阵，并要求 `[dot(nu_r), nu_r, |nu_r|nu_r]` 设计矩阵秩达到 18：

```bash
/home/jining_yang/miniconda3/envs/env_isaaclab/bin/python \
  custom_workflows/fit_pool_hydrodynamics_logs.py \
  calibration_logs/ \
  --base-profile static_measured_profile.json \
  --fit-mode full \
  --output hydrodynamics_updates.json \
  --report hydrodynamics_fit_report.json
```

pipeline 从 base profile 的 `mass` 和 3x3 惯量构造与环境 `root_lin_vel_b/root_ang_vel_b` 一致的 COM/root 块对角刚体质量矩阵；若 CSV 完整提供六个加速度列则直接使用，否则由 `time_s` 和 `nu_r` 差分。full 模式会把附加质量投影到 PSD、把线性阻尼投影到耗散域，并在实测速度样本上检查线性+二次阻尼总耗散功率；秩不足或仍出现产能样本时默认拒绝输出。数据不足时可先用 `--fit-mode diagonal`，但高保真最终仍应通过多轴激励补齐 full 6x6 可辨识性。

### 6.4 水流与边界实验

- 漂浮/中性标记物跟踪：估计水池水流均值、方差和相关时间。
- 靠墙/靠底重复同一动作：拟合近壁修正系数。
- 不同深度 heave 实验：判断自由液面是否需要建模。

水流日志可用 `fit_water_current_process(...)` 转成静态平均水流和一阶平滑随机水流参数：

```python
from .calibration_tools import fit_water_current_process

fit = fit_water_current_process(time_s, measured_current_w)
cfg_updates = fit.to_cfg_updates()
randomization_updates = fit.to_domain_randomization_updates(stage_count=3)
```

`cfg_updates` 会写入 `water_current_w`，`randomization_updates` 会给出 `water_current_smooth`、`water_current_tau_range`、`water_current_max_by_stage`、`water_current_vertical_max_by_stage` 和 `water_current_variation_std_by_stage`。如果水流位置相关性明显，仍应优先用 `water_current_field_*` 填空间网格；时间随机过程只描述同一位置或全局平均水流的缓慢变化。

多位置散点采样可用 `fit_water_current_field_grid(...)` 转成规则网格：

```python
from .calibration_tools import fit_water_current_field_grid

field_fit = fit_water_current_field_grid(
    sample_positions=pool_local_positions,
    sample_currents_w=measured_current_w,
    grid_shape=[5, 5, 3],
    bounds=[-7.0, 7.0, -7.0, 7.0, 1.0, 15.0],
)
field_updates = field_fit.to_cfg_updates()
```

该工具使用逆距离加权插值；若采样点正好落在网格节点，会直接使用实测值。它适合把漂标/ADV 的稀疏空间采样变成仿真可用的规则场，后续若要更高精度可替换为 CFD 或专门的流场重建方法。

靠墙/靠底重复实验可用 `fit_pool_boundary_effect_scales(...)` 估计简化边界模型的三个倍率。输入样本应是相对开阔水域基线的比值，例如 `D_wall / D_open`、`M_A_wall / M_A_open` 或 `thrust_wall / thrust_open`：

```python
from .calibration_tools import fit_pool_boundary_effect_scales

boundary_fit = fit_pool_boundary_effect_scales(
    sample_positions=pool_local_positions,
    bounds=[-7.0, 7.0, -7.0, 7.0, 1.0, 15.0],
    effect_distance=0.75,
    damping_scale_samples=measured_damping_ratio,
    added_mass_scale_samples=measured_added_mass_ratio,
    thrust_scale_samples=measured_thrust_ratio,
)
boundary_updates = boundary_fit.to_cfg_updates()
```

不同深度的 heave/roll/pitch、静浮和推进实验可用 `fit_free_surface_effect_scales(...)` 标定自由液面近场修正。它支持把 roll/pitch 两列或 heave/roll/pitch 三列样本共同拟合成一个模型倍率：

```python
from .calibration_tools import fit_free_surface_effect_scales

surface_fit = fit_free_surface_effect_scales(
    sample_positions=pool_local_positions,
    surface_z=water_surface_z,
    effect_distance=0.5,
    heave_damping_scale_samples=heave_damping_ratio,
    roll_pitch_damping_scale_samples=roll_pitch_damping_ratio,
    added_mass_scale_samples=heave_roll_pitch_added_mass_ratio,
    buoyancy_scale_samples=buoyancy_ratio,
    thrust_scale_samples=surface_thrust_ratio,
)
surface_updates = surface_fit.to_cfg_updates()
```

上述四类环境 CSV 也可统一交给自动 pipeline：

```bash
/home/jining_yang/miniconda3/envs/env_isaaclab/bin/python \
  custom_workflows/fit_pool_environment_logs.py \
  calibration_logs/ \
  --current-stages 3 \
  --current-grid-shape 5 5 3 \
  --pool-bounds -7 7 -7 7 1 15 \
  --boundary-effect-distance 0.75 \
  --surface-z 1.0 \
  --surface-effect-distance 0.5 \
  --output environment_updates.json \
  --report environment_fit_report.json
```

若存在 `pool_boundary_effect_samples.csv`，必须显式提供 `--pool-bounds`；若存在 `free_surface_effect_samples.csv`，必须提供 `--surface-z`。空间流场 bounds 可用 `--current-bounds` 指定，省略时由采样位置自动扩展。近壁/自由液面 CSV 的倍率列可以部分填写，pipeline 会对每个非空倍率列独立筛选样本和拟合。水流时间序列同时输出 `water_current_w` 和 domain-randomization 的相关时间、最大水平/垂直流速与变化强度。

推进器台架数据可直接转成静态查表、二维入流查表、一阶响应和电压衰减参数。稳态推力台样本建议记录 normalized command、实测推力、供电电压；若有循环水槽或拖曳台，再记录轴向入流速度：

```python
from .calibration_tools import (
    fit_thruster_static_lookup_table,
    fit_thruster_inflow_lookup_table,
    fit_thruster_first_order_response,
    fit_thruster_voltage_exponent,
)

static_fit = fit_thruster_static_lookup_table(
    command_samples=normalized_command,
    thrust_samples=measured_thrust_n,
    command_points=[-1.0, -0.5, 0.0, 0.5, 1.0],
    deadband_thrust_threshold=0.1,
)
static_updates = static_fit.to_cfg_updates(include_deadband=True)

inflow_fit = fit_thruster_inflow_lookup_table(
    command_samples=normalized_command,
    inflow_speed_samples=axial_inflow_speed_mps,
    thrust_samples=measured_thrust_n,
    command_points=[-1.0, 0.0, 1.0],
    inflow_speed_points=[-0.5, 0.0, 0.5],
)
inflow_updates = inflow_fit.to_cfg_updates()

step_fit = fit_thruster_first_order_response(
    time_s=time_s,
    measured_thrust=step_response_thrust_n,
    command_step_time_s=command_step_time_s,
)
dynamic_updates = step_fit.to_cfg_updates(physics_dt_s=sim_dt)

voltage_fit = fit_thruster_voltage_exponent(
    voltage_samples=voltage_v,
    thrust_scale_samples=measured_thrust_ratio_to_nominal,
    nominal_voltage=16.0,
)
voltage_updates = voltage_fit.to_cfg_updates()
```

如果只有普通水池而没有循环水槽，先用 `fit_thruster_static_lookup_table(...)` 和 `fit_thruster_first_order_response(...)`；二维入流表可以后续用拖曳台、循环水槽或 CFD/厂家数据补齐。

## 6. 本项目落地优先级

### P0：保持并校准当前基础模型

- 校准 `mass`, `volume`, `com_to_cob_offset`, `inertia_diag`；其中 `inertia_diag` 兼容旧 3 维对角值，也可填完整 3x3 对称惯量张量。
- 用 `calibration_tools.py` 从自由衰减/恒推日志拟合 `linear_damping`, `quadratic_damping`。
- 如恒推/自由衰减实验显示常量阻尼无法同时拟合低速和高速段，用 `fit_speed_dependent_damping_scales(...)` 生成速度相关阻尼倍率曲线并对每个关键自由度单独拟合。
- 校准 `t200_max_forward_thrust`, `t200_max_reverse_thrust`, `dyn_time_constant`, `thruster_deadband`。
- 保持相对水流速度 `nu_r` 的阻尼计算。

### P1：补齐最影响真实水池的模型

- 给 `added_mass_diag` 填非零值，并实现/近似 `-M_A dot(nu_r)`。
- 已增加推进器电压依赖、命令延迟、速率限制、命令量化、通信丢包保持、推力台曲线查表接口、二维入流实测推力表和简化轴向入流推力损失；下一步填入实测表和实测 advance-ratio 曲线。
- 已增加通用观测噪声、bias、延迟、低频更新保持、通道丢帧、低通和 bias 漂移，并提供 IMU/深度计/DVL/视觉定位的独立物理量测工具；下一步接入真实 EKF 状态传播。
- 已增加水池水流随机过程参数的实验估计工具；下一步可把多位置漂标/ADV 数据拟合为空间水流场网格。

### P2：从对角模型升级到耦合模型

- `M_A` 从 6 维对角扩展为 6x6 对称矩阵；可用 `fit_full_matrix_added_mass_linear_quadratic_damping(...)` 从多轴加速度激励日志得到初值。
- `D_l`, `D_q` 从 6 维对角扩展为矩阵/交叉项；可用 `fit_full_matrix_linear_quadratic_damping(...)` 从多轴激励日志得到初值。
- 加入 `Y_r, N_v, Z_q, M_w` 等横移-转动耦合，并通过耗散性测试检查矩阵是否物理合理。
- 加入推进器-船体、推进器-推进器干扰。
- 已加入 tether 单段模型和准静态多段浮力/负浮力/拖曳模型；下一步可做 cable 接触、放线机构和完整动态求解。

### P3：极高保真水池

- 已加入简化池壁/池底边界倍率模型和自由液面近场修正；下一步用实验或 CFD 查表替换经验倍率。
- 用 CFD/势流/边界元预计算水动力查表。
- 对局部流场、推进器尾流、气泡、自由液面波动做流固耦合或离线校正。

P3 能提高物理真实性，但代价很高。若目标是 RL 控制策略在水池实物上可迁移，P0-P2 + 参数随机化通常比在线 CFD 更实用。

## 7. 建议的代码改造点

- `rigid_body_hydrodynamics.py`
  - 已支持 full 6x6 `added_mass_matrix`。
  - 已支持 full 6x6 `linear_damping_matrix`，二次阻尼也可传入 6x6 系数。
  - 已增加 `calculate_added_mass_inertia_wrench(relative_acceleration_b, added_mass)`。
  - 已增加 `calculate_speed_dependent_damping_scale(nu_r, speed_points, scale_points)`，用于把速度段标定曲线转换成每个自由度的阻尼倍率。

- `calibration_tools.py`
  - 已增加 `fit_diagonal_added_mass_linear_quadratic_damping(...)`，从阶跃/多轴日志联合拟合对角附加质量、线性阻尼和二次阻尼。
  - 已增加 `fit_full_matrix_added_mass_linear_quadratic_damping(...)`，从多轴加速度激励日志拟合 full 6x6 附加质量、线性阻尼和二次阻尼。
  - 已增加 `project_added_mass_to_physical(...)`、`project_linear_damping_to_dissipative(...)`、`calculate_damping_dissipated_power(...)` 和 `damping_is_dissipative_for_samples(...)`，用于把含噪声的耦合矩阵标定结果投影/检查到物理可行域。
  - 已增加 `fit_diagonal_linear_quadratic_damping(...)`，从自由衰减/恒推日志拟合对角线性与二次阻尼。
  - 已增加 `fit_full_matrix_linear_quadratic_damping(...)`，从多轴激励日志拟合 full 6x6 线性/二次阻尼矩阵。
  - 已增加 `fit_speed_dependent_damping_scales(...)`，把不同速度段的阻尼残差转换成 profile 可用的倍率曲线。
  - 已增加 `fit_mass_from_scale_readings(...)`、`fit_inertia_tensor_from_axis_moments(...)`、`compound_pendulum_moments_from_periods(...)` 和 `fit_inertia_tensor_from_compound_pendulum(...)`，把称重/CAD/复摆实验转换成 `mass` 和完整 3x3 `inertia_diag`。
  - 已增加 `fit_buoyancy_volume_from_forces(...)`、`fit_com_to_cob_offset_from_buoyancy_wrenches(...)` 和 `fit_com_to_cob_offset_from_static_torques(...)`，把静浮/倾斜恢复实验转换成 `volume`、`water_rho` 和 `com_to_cob_offset`。
  - 已增加 `fit_water_current_process(...)`，从漂标/ADV/DVL 水流日志估计平均水流、扰动强度、水平/垂直最大流速和相关时间常数。
  - 已增加 `fit_water_current_field_grid(...)`，把散点水流采样拟合成三线性插值可用的规则网格。
  - 已增加 `fit_pool_boundary_effect_scales(...)` 和 `fit_free_surface_effect_scales(...)`，把近壁/近自由液面重复实验中的倍率样本转换成环境 cfg 可直接使用的经验模型参数。
  - 已增加 `fit_thruster_static_lookup_table(...)`、`fit_thruster_inflow_lookup_table(...)`、`fit_thruster_first_order_response(...)` 和 `fit_thruster_voltage_exponent(...)`，把推进器台架/阶跃/电压实验转换成查表推力曲线、一阶时间常数、命令延迟步数和电压推力指数。
  - 已增加 `fit_battery_voltage_sag(...)`，把时间-电压日志拟合成环境 runtime 使用的初始电压、最低观测电压和线性每秒下垂。
  - 已增加 `fit_tether_spring_damper(...)` 和 `fit_tether_drag_coefficient(...)`，把安全绳/缆绳的拉力-伸长、速度阻尼和二次拖曳实验转换成 tether cfg 参数。
  - 已提供 `to_cfg_updates()`，便于把拟合结果直接合并进 `PoolDynamicsProfile` 或环境 cfg。

- `warpauv_env.py`
  - 已保存上一帧 `nu_r`，并用物理步长估计/低通 `dot(nu_r)`，用于附加质量惯性项。
  - 已增加 pool 配置：尺寸、深度、近壁启用开关、近壁阻尼/附加质量/推力修正系数。
  - 已增加通用观测噪声、bias、延迟、更新周期、丢帧、低通和 bias 漂移配置，训练时可作为 domain randomization。
  - 已增加电池电压状态、episode 内电压下降和推进器延迟/速率限制状态。
  - 已接入默认关闭的速度相关线性/二次阻尼倍率曲线，可与近壁、自由液面阻尼倍率叠加。

- `thruster_dynamics.py`
  - 已支持查表/分段曲线，而不只二次映射。
  - 已支持 command delay、rate limit、voltage scaling 的基础模型。
  - 可选加入推进器反扭矩。

- `sensor_models.py`
  - 已支持通用 observation delay/noise/bias/update-hold/dropout/low-pass/bias-drift。
  - 已增加 `apply_sensor_channel_model(...)`，统一处理 scale、bias、噪声、限幅、丢测和上一帧保持。
  - 已增加 `calculate_imu_measurement(...)`，用 wxyz 姿态把世界系加速度转换成机体系 specific force，并输出 gyro 量测。
  - 已增加 `calculate_depth_sensor_measurement(...)`、`calculate_dvl_velocity_measurement(...)` 和 `calculate_position_sensor_measurement(...)`，用于深度计、DVL 和外部视觉/声学定位的物理量测与有效性标志。

- `pool_dynamics_profile.py`
  - 已增加 `SensorProfile` 及其 IMU/深度计/DVL/外部定位子 profile，用于把物理传感器参数和水动力、推进器、tether 参数保存在同一个 measured-pool profile 中。
  - 已增加 `merge_pool_dynamics_cfg_updates(...)`，把各标定函数输出的 flat cfg updates 反向合并成 nested `PoolDynamicsProfile`。
  - 已增加 `audit_pool_dynamics_profile(...)` 和 `PoolProfileAuditOptions`，用于在训练/实验前生成 profile 高保真缺口清单和 readiness score。
  - 已增加 `pool_profile_calibration_tasks(...)` 和 `PoolCalibrationTask`，把审计缺口转换成带实验建议、标定函数和 update keys 的任务清单。
  - 已增加 `pool_profile_calibration_update_template(...)`，把任务清单转换成待填写的 `cfg_updates` / `domain_randomization_updates` JSON 骨架。
  - 已增加 `pool_profile_calibration_log_schemas(...)`、`PoolCalibrationLogSchema` 和 `PoolCalibrationLogColumn`，把任务清单转换成实验日志 CSV 列规范。
  - 已增加 `validate_pool_calibration_log_directory(...)` 和结构化 validation report，用 schema 校验实际 CSV 日志。

- `custom_workflows/build_pool_profile_from_calibration.py`
  - 已增加命令行 profile 构建入口，可按顺序合并多个标定 JSON updates 文件和 domain randomization updates，输出可审计的 measured-pool JSON。

- `custom_workflows/fit_pool_static_logs.py`
  - 已增加静态 CSV 标定 pipeline，把称重、浮力、静态恢复力矩、轴向惯量和复摆周期日志转换成 `mass`、`volume`、`water_rho`、`com_to_cob_offset` 与完整 3x3 `inertia_diag` updates，并输出拟合诊断报告。

- `custom_workflows/fit_pool_thruster_logs.py`
  - 已增加推进器/电池 CSV 标定 pipeline，把静态推力台、轴向入流、阶跃响应和时间-电压/推力倍率日志转换成一维/二维 lookup、死区、一阶时间常数、命令延迟、电池下垂和电压推力指数 updates，并输出拟合诊断报告。

- `custom_workflows/fit_pool_environment_logs.py`
  - 已增加环境 CSV 标定 pipeline，把时间水流、空间水流场、近壁倍率和自由液面倍率日志转换成 cfg/domain-randomization updates，并输出拟合诊断报告。

- `custom_workflows/fit_pool_hydrodynamics_logs.py`
  - 已增加 6-DOF 水动力 CSV 标定 pipeline，支持 diagonal/full 附加质量与阻尼联合拟合、设计秩检查、附加质量 PSD 投影、线性阻尼耗散投影和实测速度包络 passivity gate。

- `custom_workflows/fit_pool_tether_logs.py`
  - 已增加 tether CSV 标定 pipeline，把张力-伸长-速度和相对流速-拖曳力日志转换成松弛长度、刚度、阻尼、拖曳系数及实测 cable geometry updates。

- `custom_workflows/audit_pool_profile.py`
  - 已增加命令行 profile 审计入口，可直接读取 measured-pool JSON，输出文本或 JSON 报告，并用 `--fail-on-warning` / `--fail-on-critical` 作为训练前 gate。
  - 已支持 `--checklist` / `--checklist --json`，把当前 profile 缺口输出成实验标定任务清单。
  - 已支持 `--template`，生成待填写的标定 updates 模板。
  - 已支持 `--log-schemas` 和 `--write-log-templates DIR`，生成实验日志 schema JSON 或空 CSV 表头模板。
  - 已支持 `--validate-log-dir DIR`，在拟合前校验日志完整性并用退出码作为 gate。

- `tests/`
  - 保留阻尼耗散性测试。
  - 已增加 full-matrix `C_A` power-preserving 测试。
  - 已增加 `-M_A dot(nu_r)` 符号和总 fluid wrench 接入测试。

## 8. 参考资料

- Fossen marine craft model: https://www.fossen.biz/html/marineCraftModel.html
- UUV Simulator README/features: https://github.com/uuvsimulator/uuv_simulator
- Blue Robotics BlueROV2 product specifications: https://bluerobotics.com/store/rov/bluerov2/
- Blue Robotics T200 product page and technical details: https://bluerobotics.com/store/thrusters/t100-t200-thrusters/t200-thruster-r2-rp/
- 6-DOF experimental hydrodynamic characterization: https://arxiv.org/abs/2501.17018
- BlueROV2 pool-based mapping platform: https://arxiv.org/abs/2407.10901
- AUV hydrodynamics over complex beds: https://arxiv.org/abs/1904.13305
- Oscillating body near wall unsteady drag: https://arxiv.org/abs/2307.05991
