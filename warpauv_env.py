"""
WarpAUV environment for IsaacLabs

Author: Kevin Chang and Levi "Veevee" Cai (cail@mit.edu)
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import Tuple

from .assets.warpauv import WARPAUV_CFG

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.envs.ui import BaseEnvWindow
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass
from isaaclab.markers import CUBOID_MARKER_CFG, VisualizationMarkers, RED_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG, BLUE_ARROW_X_MARKER_CFG
from isaaclab.utils.math import quat_apply, quat_conjugate
import isaaclab.utils.math as math_utils
import gymnasium as gym
import numpy as np

from .bluerov2_heavy_model import BLUEROV2_HEAVY, KGF_TO_NEWTON
from .rigid_body_hydrodynamics import HydrodynamicForceModels, calculate_speed_dependent_damping_scale
from .thruster_dynamics import (
    T200_REVERSE_TO_FORWARD_RATIO,
    DynamicsFirstOrder,
    ThrusterCommandProcessor,
    ConversionFunctionT200,
    ConversionFunctionLookupTable,
    ConversionFunctionInflowLookupTable,
    calculate_axial_inflow_thrust_scale,
    calculate_reaction_torques,
    calculate_thruster_wake_interaction_scale,
    calculate_voltage_thrust_scale,
    get_thruster_com_and_orientations,
)
from .sensor_models import (
    ObservationDelayBuffer,
    ObservationFilterState,
    apply_observation_sensor_model,
    build_observation_group_parameter,
)
from .pool_effects import calculate_free_surface_scales, calculate_pool_boundary_scales
from .tether_dynamics import calculate_multisegment_tether_wrench
from .pool_dynamics_profile import apply_pool_dynamics_profile
from .water_current_fields import calculate_trilinear_current_field
from .rigid_body_properties import inertia_matrix_tensor


def _nominal_hydro_coeff_tensor(values, device: torch.device, name: str) -> torch.Tensor:
    """Normalize 6-DOF hydrodynamic coefficients to a single-env tensor."""

    tensor = torch.as_tensor(values, dtype=torch.float32, device=device)
    if tensor.ndim == 1 and tensor.shape[0] == 6:
        return tensor.reshape(1, 6)
    if tensor.ndim == 2:
        if tensor.shape == (6, 6):
            return tensor.reshape(1, 6, 6)
        if tensor.shape == (1, 6):
            return tensor
    if tensor.ndim == 3 and tensor.shape == (1, 6, 6):
        return tensor
    raise ValueError(f"{name} must be a 6-vector or 6x6 matrix, got shape {tuple(tensor.shape)}.")


def _repeat_hydro_coeff_for_envs(nominal: torch.Tensor, num_envs: int) -> torch.Tensor:
    repeats = (num_envs,) + tuple(1 for _ in range(nominal.ndim - 1))
    return nominal.repeat(repeats)


def _scale_hydro_coefficients(coefficients: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    if coefficients.ndim == 2:
        return coefficients * scale
    if coefficients.ndim == 3:
        if scale.ndim == 2 and scale.shape[1] == 6:
            matrix_scale = torch.sqrt(torch.clamp(scale.unsqueeze(1) * scale.unsqueeze(2), min=0.0))
            return coefficients * matrix_scale
        return coefficients * scale.reshape(scale.shape[0], 1, 1)
    raise ValueError(f"Expected batched 6-vector or 6x6 coefficients, got {tuple(coefficients.shape)}.")


class WarpAUVEnvWindow(BaseEnvWindow):
    """Window manager for the warpauvenv environment."""

    def __init__(self, env: WarpAUVEnv, window_name: str = "IsaacLab"):
        """Initialize the window.

        Args:
            env: The environment object.
            window_name: The name of the window. Defaults to "IsaacLab".
        """
        # initialize base window
        super().__init__(env, window_name)
        # add custom UI elements
        with self.ui_window_elements["main_vstack"]:
            with self.ui_window_elements["debug_frame"]:
                with self.ui_window_elements["debug_vstack"]:
                    # add command manager visualization
                    self._create_debug_vis_ui_element("targets", self.env)

@configclass
class WarpAUVEnvCfg(DirectRLEnvCfg):
    ui_window_class_type = WarpAUVEnvWindow

    sim: SimulationCfg = SimulationCfg(dt=1 / 120)

    # robot
    robot_cfg: RigidObjectCfg = WARPAUV_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4, env_spacing=4.0, replicate_physics=True)
    debug_vis = True

    observation_space: gym.spaces.Space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(17,), dtype=np.float64)
    action_space: gym.spaces.Space = gym.spaces.Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float64)
    state_space: gym.spaces.Space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(17,), dtype=np.float64)
    # env
    decimation = 2
    cap_episode_length = True
    episode_length_s = 5.0
    episode_length_before_reset = None
    num_actions = 8
    num_observations = 17
    num_states = 0
    use_boundaries = True
    max_auv_x = 7
    max_auv_y = 7
    max_auv_z = 7
    starting_depth = 8
    min_goal_steps = 100
    goal_completion_radius = 0.01
    goal_dims = 4
    eval_mode = False
    pool_dynamics_profile = None

    goal_spawn_radius = 2.0
    init_guidance_rate = 0.1
    init_vel_max = 0.2

    # Trajectory tracking is disabled for the original pos-hold task.  The
    # trajectory task flips this flag in WarpAUVTrajEnvCfg so old 17-D policies
    # keep their original observation and reward contract.
    trajectory_tracking = False
    trajectory_eval_mode = False
    # Training command distribution: random smooth trajectories around each
    # environment origin.  These bounds define the "train set" for RL.
    trajectory_amp_x_range = [0.4, 1.6]
    trajectory_amp_y_range = [0.25, 1.0]
    trajectory_amp_z_range = [0.08, 0.35]
    trajectory_period_range = [16.0, 28.0]
    trajectory_curriculum = True
    trajectory_curriculum_stage_steps = [15_000, 40_000, 70_000]
    trajectory_curriculum_stage_0_types = [0, 1, 2, 3]
    trajectory_curriculum_stage_1_types = [0, 1, 2, 3, 4]
    trajectory_curriculum_stage_2_types = [0, 1, 2, 3, 4, 5]
    trajectory_curriculum_stage_3_types = [0, 1, 2, 3, 4, 5, 6]
    trajectory_curriculum_amp_scales = [0.45, 0.65, 0.85, 1.0]
    trajectory_curriculum_z_amp_scales = [0.35, 0.65, 0.85, 1.0]
    trajectory_curriculum_period_min = [24.0, 22.0, 20.0, 16.0]
    trajectory_curriculum_period_max = [30.0, 30.0, 28.0, 28.0]
    # Deterministic eval commands.  trajectory_eval_type maps to:
    # 1=lissajous, 3=helix, 4=spiral, 5=chirp, 6=racetrack,
    # 7=random_smooth.
    # With curriculum enabled, each stage below chooses the active train set.
    # Codes not listed in the current stage remain held-out eval tests.
    trajectory_eval_type = 1
    trajectory_eval_amp_x = 1.2
    trajectory_eval_amp_y = 0.6
    trajectory_eval_amp_z = 0.25
    trajectory_eval_period = 24.0
    trajectory_eval_duration_s = 32.0
    trajectory_eval_radius_min = 0.3
    trajectory_eval_radius_max = 1.2
    trajectory_eval_chirp_rate = 1.6
    trajectory_eval_align_initial_target = True
    trajectory_train_types = [0, 1, 2, 3, 4, 5]

    # rewards
    rew_scale_terminated = 0.0
    rew_scale_alive = 0.0
    rew_scale_completion = 1000

    rew_scale_pos = 0.2
    rew_scale_ang = 0.5
    rew_scale_vel = 0.0
    rew_scale_ang_vel = 0.05
    rew_scale_lin_vel = 0.05
    rew_scale_track_vel = 0.2
    rew_scale_actions = 0.02
    rew_scale_action_rate = 0.0
    rew_pos_sigma = 1.0
    rew_ang_sigma = 1.0
    rew_track_vel_sigma = 0.5
    rew_ang_vel_sigma = 0.5
    observation_noise_std = 0.0
    observation_bias_range = 0.0
    observation_delay_steps = 0
    observation_update_period_steps = 1
    observation_dropout_probability = 0.0
    observation_lowpass_alpha = 1.0
    observation_bias_drift_std = 0.0
    pool_boundary_effects_enabled = False
    pool_bounds = [-7.0, 7.0, -7.0, 7.0, 1.0, 15.0]
    pool_boundary_effect_distance = 0.75
    pool_boundary_damping_scale = 1.5
    pool_boundary_added_mass_scale = 1.2
    pool_boundary_thrust_scale = 0.85
    free_surface_effects_enabled = False
    free_surface_z = 1.0
    free_surface_effect_distance = 0.5
    free_surface_heave_damping_scale = 1.4
    free_surface_roll_pitch_damping_scale = 1.2
    free_surface_added_mass_scale = 1.15
    free_surface_buoyancy_scale = 0.95
    free_surface_thrust_scale = 0.90
    tether_enabled = False
    tether_anchor_pos_w = [0.0, 0.0, 8.0]
    tether_attach_offset_b = [-0.2, 0.0, 0.0]
    tether_slack_length = 2.0
    tether_stiffness = 20.0
    tether_damping = 5.0
    tether_drag_coeff = 0.0
    tether_num_segments = 1
    tether_segment_diameter = 0.004
    tether_segment_density = 1100.0
    tether_segment_buoyancy_density = BLUEROV2_HEAVY.water_density_kg_m3
    thruster_inflow_loss_enabled = False
    thruster_inflow_loss_coefficient = 0.25
    thruster_inflow_reference_speed = 1.0
    thruster_inflow_min_scale = 0.5
    thruster_wake_interaction_enabled = False
    thruster_wake_loss_coefficient = 0.10
    thruster_wake_length = 0.6
    thruster_wake_radius = 0.08
    thruster_wake_expansion_rate = 0.15
    thruster_wake_min_scale = 0.7
    thruster_reaction_torque_coeff = 0.0
    thruster_spin_directions = [1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0]

    # dynamics
    center_of_mass_offset = list(BLUEROV2_HEAVY.center_of_mass_offset_m)
    inertia_diag = list(BLUEROV2_HEAVY.inertia_diag_kg_m2)
    com_to_cob_offset = list(BLUEROV2_HEAVY.center_of_buoyancy_from_com_m)
    water_rho = BLUEROV2_HEAVY.water_density_kg_m3 # kg/m^3
    water_beta = 0.001306 # Pa s, dynamic viscosity of water @ 50 deg F
    dyn_time_constant = 0.05 # time constant for linear dynamics for each rotor
    thruster_deadband = 0.08
    thruster_command_delay_steps = 0
    thruster_max_command_rate = 0.0
    thruster_command_resolution = 0.0
    thruster_command_dropout_probability = 0.0
    battery_voltage_nominal = 16.0
    battery_voltage = 16.0
    battery_min_voltage = 12.0
    battery_voltage_drop_per_s = 0.0
    battery_voltage_thrust_exponent = 2.0
    mass = BLUEROV2_HEAVY.mass_kg # kg, BlueROV2 Heavy with ballast
    volume = BLUEROV2_HEAVY.neutral_buoyancy_volume_m3 # m^3, neutral buoyancy at water_rho

    # Calibrated to BlueROV2 Heavy vehicle bollard thrust: 9 kgf forward/lateral
    # through four vectored T200s and 14 kgf vertical through four T200s.
    t200_horizontal_forward_thrust = BLUEROV2_HEAVY.forward_bollard_thrust_kgf * KGF_TO_NEWTON / (
        2.0 * (0.7431448255 + 0.6691306064)
    )
    t200_vertical_forward_thrust = BLUEROV2_HEAVY.vertical_bollard_thrust_kgf * KGF_TO_NEWTON / 4.0
    t200_max_forward_thrust = [
        t200_horizontal_forward_thrust,
        t200_horizontal_forward_thrust,
        t200_horizontal_forward_thrust,
        t200_horizontal_forward_thrust,
        t200_vertical_forward_thrust,
        t200_vertical_forward_thrust,
        t200_vertical_forward_thrust,
        t200_vertical_forward_thrust,
    ]
    t200_max_reverse_thrust = [thrust * T200_REVERSE_TO_FORWARD_RATIO for thrust in t200_max_forward_thrust]
    use_thruster_lookup_table = False
    thruster_lookup_commands = [-1.0, 0.0, 1.0]
    thruster_lookup_thrusts = []
    use_thruster_inflow_lookup_table = False
    thruster_inflow_lookup_commands = [-1.0, 0.0, 1.0]
    thruster_inflow_lookup_speeds = [-1.0, 0.0, 1.0]
    thruster_inflow_lookup_thrusts = []

    # Fossen-style hydrodynamic parameters.  Damping is applied to relative
    # velocity nu_r = nu - nu_current, not absolute vehicle velocity.
    water_current_w = [0.0, 0.0, 0.0]
    water_current_field_enabled = False
    water_current_field_bounds = [-7.0, 7.0, -7.0, 7.0, 1.0, 15.0]
    water_current_field_shape = [1, 1, 1]
    water_current_field_values = []
    linear_damping = [0.00526, 0.00526, 0.00526, 0.00032, 0.00032, 0.00032]
    quadratic_damping = [39.196, 68.272, 135.402, 0.277, 1.387, 0.770]
    speed_dependent_damping_enabled = False
    damping_speed_points = [0.0, 1.0]
    linear_damping_speed_scales = []
    quadratic_damping_speed_scales = []
    added_mass_diag = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    added_mass_inertia_scale = 1.0
    added_mass_accel_filter_alpha = 0.35

    # domain randomization
    # todo: isaaclabs has a built-in method somehow
    class domain_randomization:
        use_custom_randomization = False
        # com_to_cob_offset_radius = 0 # uniform from sphere around predicted com_to_cob_offset
        com_to_cob_offset_radius = 0.0 # uniform from sphere around predicted com_to_cob_offset
        volume_range = [BLUEROV2_HEAVY.neutral_buoyancy_volume_m3, BLUEROV2_HEAVY.neutral_buoyancy_volume_m3]
        mass_range = [BLUEROV2_HEAVY.mass_kg, BLUEROV2_HEAVY.mass_kg]
        thruster_command_delay_steps_range = [0, 0]
        thruster_max_command_rate_range = [0.0, 0.0]
        thruster_command_resolution_range = [0.0, 0.0]
        thruster_command_dropout_probability_range = [0.0, 0.0]
        battery_voltage_range = [16.0, 16.0]
        battery_voltage_drop_per_s_range = [0.0, 0.0]
        observation_noise_std_range = [0.0, 0.0]
        observation_bias_range = [0.0, 0.0]
        observation_delay_steps_range = [0, 0]
        observation_update_period_steps_range = [1, 1]
        observation_dropout_probability_range = [0.0, 0.0]
        observation_lowpass_alpha_range = [1.0, 1.0]
        observation_bias_drift_std_range = [0.0, 0.0]


@configclass
class WarpAUVTrajEnvCfg(WarpAUVEnvCfg):
    """Configuration for the moving-target trajectory-tracking task."""

    # Observation layout:
    # [target_quat(4), target_pos_error_b(3), target_lin_vel_b(3),
    #  root_quat_w(4), root_lin_vel_b(3), root_ang_vel_b(3), last_action(8)].
    observation_space: gym.spaces.Space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(28,), dtype=np.float64)
    state_space: gym.spaces.Space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(28,), dtype=np.float64)
    num_observations = 28
    episode_length_s = 24.0
    trajectory_tracking = True
    init_guidance_rate = 0.2
    rew_scale_pos = 2.0
    rew_scale_ang = 0.25
    rew_scale_track_vel = 0.6
    rew_scale_ang_vel = 0.02
    rew_scale_lin_vel = 0.0
    rew_scale_actions = 0.003
    rew_scale_action_rate = 0.0012
    rew_pos_sigma = 0.35
    rew_ang_sigma = 0.75
    rew_track_vel_sigma = 0.35
    rew_ang_vel_sigma = 0.5

    class domain_randomization(WarpAUVEnvCfg.domain_randomization):
        use_custom_randomization = True
        disturbance_curriculum = True
        # Boundaries are in policy steps. With num_steps_per_env=256, these are
        # roughly iterations 80, 180, 320, and 450 for a 600-iteration run.
        disturbance_curriculum_stage_steps = [20_000, 46_000, 82_000, 115_000]
        water_current_max_by_stage = [0.0, 0.05, 0.10, 0.15, 0.20]
        water_current_vertical_max_by_stage = [0.0, 0.01, 0.02, 0.025, 0.03]
        water_current_smooth = True
        water_current_tau_range = [8.0, 24.0]
        water_current_variation_std_by_stage = [0.0, 0.004, 0.008, 0.012, 0.016]
        damping_scale_by_stage = [0.0, 0.0, 0.15, 0.25, 0.30]
        thruster_scale_by_stage = [0.0, 0.0, 0.0, 0.10, 0.15]
        thruster_tau_scale_by_stage = [0.0, 0.0, 0.0, 0.25, 0.50]
        thruster_deadband_scale_by_stage = [0.0, 0.0, 0.0, 0.10, 0.20]


class WarpAUVEnv(DirectRLEnv):
    cfg: WarpAUVEnvCfg

    def __init__(self, cfg: WarpAUVEnvCfg, render_mode: str | None = None, **kwargs):
        if getattr(cfg, "pool_dynamics_profile", None) is not None:
            apply_pool_dynamics_profile(cfg, cfg.pool_dynamics_profile)
        super().__init__(cfg, render_mode, **kwargs)

        # Debug mode?
        self._debug = False

        # Initialize buffers
        self._actions = torch.zeros(self.num_envs, self.cfg.num_actions, device=self.device)
        self._previous_actions = torch.zeros(self.num_envs, self.cfg.num_actions, device=self.device)
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._goal = torch.zeros(self.num_envs, self.cfg.goal_dims, device=self.device)

        # Moving-target command buffers.  These are always allocated so the
        # debug visualizer and eval scripts can use one API, but only the
        # trajectory task updates them every policy step.
        self._target_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._target_lin_vel_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._target_quat_w = torch.zeros(self.num_envs, 4, device=self.device)

        # Per-environment trajectory parameters sampled at reset.  traj_type is
        # 0=circle, 1=Lissajous, 2=single-axis sine, 3=helix, 4=spiral, 5=chirp,
        # 6=racetrack, 7=random smooth Fourier curve.
        self._traj_center_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._traj_type = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self._traj_axis = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self._traj_amp_x = torch.zeros(self.num_envs, device=self.device)
        self._traj_amp_y = torch.zeros(self.num_envs, device=self.device)
        self._traj_amp_z = torch.zeros(self.num_envs, device=self.device)
        self._traj_period = torch.ones(self.num_envs, device=self.device)
        self._traj_phase_x = torch.zeros(self.num_envs, device=self.device)
        self._traj_phase_y = torch.zeros(self.num_envs, device=self.device)
        self._default_root_state = torch.zeros(self.num_envs, 13, device=self.device)
        self._completion_buffer = torch.zeros(self.num_envs, device=self.device)
        self._completed_envs = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._default_env_origins = torch.zeros(self.num_envs, 3, device=self.device)
        self._goal_pos_w = self._default_env_origins # just for visualizations at the moment
        self._step_count = 0
        
        # Get thruster configurations
        self.thruster_com_offsets, self.thruster_quats = get_thruster_com_and_orientations(self.device)
        self.num_thrusters = self.thruster_com_offsets.shape[0]
        if self.cfg.num_actions != self.num_thrusters:
            raise ValueError(
                f"Expected {self.num_thrusters} actions for the thruster model, got {self.cfg.num_actions}."
            )
        self.thruster_com_offsets = self.thruster_com_offsets.unsqueeze(0).repeat(self.num_envs, 1, 1)
        self.thruster_quats = self.thruster_quats.unsqueeze(0).repeat(self.num_envs, 1, 1)

        torch.manual_seed(0)

        if self.cfg.eval_mode:
            print("Setting manual seed")
            torch.manual_seed(0)

        # Debug visualization
        self.set_debug_vis(self.cfg.debug_vis)

        if self._debug: print("mass: ", list(self._robot.root_physx_view._masses))

        # Get specific information about the AUV
        self._gravity_w = torch.tensor(self.sim.cfg.gravity, device=self.device, dtype=torch.float32)
        self._gravity_magnitude = self._gravity_w.norm()

        self.inertia_matrices = inertia_matrix_tensor(
            self.cfg.inertia_diag,
            self.device,
        ).reshape(1, 3, 3).repeat(self.num_envs, 1, 1)
        self.inertia_tensors = torch.diagonal(self.inertia_matrices, dim1=-2, dim2=-1)
        self.masses = torch.full((self.num_envs, 1), self.cfg.mass, device=self.device)
        self._apply_nominal_rigid_body_properties()

        # todo: cleaner way to handle this
        if type(self.cfg.com_to_cob_offset) != torch.Tensor:
            self.com_to_cob_offsets = torch.tensor(self.cfg.com_to_cob_offset).repeat(self.num_envs, 1).to(self.device)
        else:
            self.com_to_cob_offsets = self.cfg.com_to_cob_offset.copy()

        if type(self.cfg.volume) != torch.Tensor:
            self.volumes = torch.full((self.num_envs, 1), self.cfg.volume, device=self.device)
        else:
            self.volumes = self.cfg.volume.copy()

        self.inertia_tensors_mean = self.inertia_tensors.mean(dim=1, keepdim=True) 

        # Initialize dynamics calculators
        self._init_thruster_dynamics()
        self._init_observation_sensor_model()
        
        # Set initial goals
        self._reset_idx(self._robot._ALL_INDICES)


    def _init_thruster_dynamics(self):
        if type(self.cfg.com_to_cob_offset) != torch.Tensor:
            self.cfg.com_to_cob_offset = torch.tensor(
                self.cfg.com_to_cob_offset,
                device=self.device,
                dtype=torch.float32,
                requires_grad=False,
            ).reshape(1, 3).repeat(self.num_envs, 1)

        # Fluid and motor models are vectorized over environments.  The damping
        # coefficients are body-frame Fossen diagonal entries for nu_r.
        self.force_calculation_functions = HydrodynamicForceModels(self.num_envs, self.device, False)
        self.thruster_dynamics = DynamicsFirstOrder(
            self.num_envs,
            self.num_thrusters,
            self.cfg.dyn_time_constant,
            self.device,
        )
        delay_range = getattr(self.cfg.domain_randomization, "thruster_command_delay_steps_range", [0, 0])
        max_delay_steps = max(int(self.cfg.thruster_command_delay_steps), int(delay_range[1]))
        self.thruster_command_processor = ThrusterCommandProcessor(
            self.num_envs,
            self.num_thrusters,
            max_delay_steps,
            self.device,
        )
        if self.cfg.use_thruster_inflow_lookup_table:
            if len(self.cfg.thruster_inflow_lookup_thrusts) == 0:
                raise ValueError(
                    "thruster_inflow_lookup_thrusts must be provided when use_thruster_inflow_lookup_table=True."
                )
            self.thruster_conversion = ConversionFunctionInflowLookupTable(
                self.cfg.thruster_inflow_lookup_commands,
                self.cfg.thruster_inflow_lookup_speeds,
                self.cfg.thruster_inflow_lookup_thrusts,
            )
        elif self.cfg.use_thruster_lookup_table:
            if len(self.cfg.thruster_lookup_thrusts) == 0:
                raise ValueError("thruster_lookup_thrusts must be provided when use_thruster_lookup_table=True.")
            self.thruster_conversion = ConversionFunctionLookupTable(
                self.cfg.thruster_lookup_commands,
                self.cfg.thruster_lookup_thrusts,
            )
        else:
            self.thruster_conversion = ConversionFunctionT200(
                self.cfg.t200_max_forward_thrust,
                self.cfg.t200_max_reverse_thrust,
            )
        self._nominal_linear_damping = _nominal_hydro_coeff_tensor(
            self.cfg.linear_damping, self.device, "linear_damping"
        )
        self._nominal_quadratic_damping = _nominal_hydro_coeff_tensor(
            self.cfg.quadratic_damping, self.device, "quadratic_damping"
        )
        self._nominal_added_mass_diag = _nominal_hydro_coeff_tensor(
            self.cfg.added_mass_diag, self.device, "added_mass_diag"
        )
        self._nominal_water_current_w = torch.tensor(
            self.cfg.water_current_w, dtype=torch.float32, device=self.device
        ).reshape(1, 3)
        self.linear_damping = _repeat_hydro_coeff_for_envs(self._nominal_linear_damping, self.num_envs)
        self.quadratic_damping = _repeat_hydro_coeff_for_envs(self._nominal_quadratic_damping, self.num_envs)
        self.added_mass_diag = _repeat_hydro_coeff_for_envs(self._nominal_added_mass_diag, self.num_envs)
        self.water_current_w = self._nominal_water_current_w.repeat(self.num_envs, 1)
        self.water_current_mean_w = self._nominal_water_current_w.repeat(self.num_envs, 1)
        self.water_current_horizontal_max = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.water_current_vertical_max = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.water_current_tau = torch.full((self.num_envs,), 12.0, dtype=torch.float32, device=self.device)
        self.thruster_force_scale = torch.ones(self.num_envs, self.num_thrusters, device=self.device)
        self.thruster_time_constant = torch.full(
            (self.num_envs,), self.cfg.dyn_time_constant, dtype=torch.float32, device=self.device
        )
        self.thruster_deadband = torch.full(
            (self.num_envs, self.num_thrusters), self.cfg.thruster_deadband, dtype=torch.float32, device=self.device
        )
        self.thruster_delay_steps = torch.full(
            (self.num_envs,), int(self.cfg.thruster_command_delay_steps), dtype=torch.long, device=self.device
        )
        self.thruster_max_command_rate = torch.full(
            (self.num_envs, 1), self.cfg.thruster_max_command_rate, dtype=torch.float32, device=self.device
        )
        self.thruster_command_resolution = torch.full(
            (self.num_envs, 1), self.cfg.thruster_command_resolution, dtype=torch.float32, device=self.device
        )
        self.thruster_command_dropout_probability = torch.full(
            (self.num_envs, 1), self.cfg.thruster_command_dropout_probability, dtype=torch.float32, device=self.device
        )
        self.battery_initial_voltage = torch.full(
            (self.num_envs, 1), self.cfg.battery_voltage, dtype=torch.float32, device=self.device
        )
        self.battery_voltage = torch.full(
            (self.num_envs, 1), self.cfg.battery_voltage, dtype=torch.float32, device=self.device
        )
        self.battery_voltage_drop_per_s = torch.full(
            (self.num_envs, 1), self.cfg.battery_voltage_drop_per_s, dtype=torch.float32, device=self.device
        )
        self.thruster_dynamics.tau = self.thruster_time_constant
        self._previous_nu_r = torch.zeros((self.num_envs, 6), dtype=torch.float32, device=self.device)
        self._filtered_nu_r_dot = torch.zeros_like(self._previous_nu_r)
        self._has_previous_nu_r = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def _init_observation_sensor_model(self) -> None:
        delay_range = getattr(self.cfg.domain_randomization, "observation_delay_steps_range", [0, 0])
        max_delay_steps = max(int(self.cfg.observation_delay_steps), int(delay_range[1]))
        self.observation_delay_buffer = ObservationDelayBuffer(
            self.num_envs,
            self.cfg.num_observations,
            max_delay_steps,
            self.device,
        )
        self.observation_filter_state = ObservationFilterState(
            self.num_envs,
            self.cfg.num_observations,
            self.device,
        )
        self.observation_delay_steps = torch.full(
            (self.num_envs,),
            int(self.cfg.observation_delay_steps),
            dtype=torch.long,
            device=self.device,
        )
        self.observation_update_period_steps = torch.full(
            (self.num_envs,),
            int(self.cfg.observation_update_period_steps),
            dtype=torch.long,
            device=self.device,
        )
        self.observation_noise_std = torch.zeros(
            (self.num_envs, self.cfg.num_observations),
            dtype=torch.float32,
            device=self.device,
        )
        self.observation_bias = torch.zeros_like(self.observation_noise_std)
        self.observation_dropout_probability = torch.zeros_like(self.observation_noise_std)
        self.observation_lowpass_alpha = torch.ones_like(self.observation_noise_std)
        self.observation_bias_drift_std = torch.zeros_like(self.observation_noise_std)
        self._set_fixed_observation_noise(self._robot._ALL_INDICES)

    def _observation_cfg_tensor(self, value, name: str) -> torch.Tensor:
        reference = torch.zeros(
            (self.num_envs, self.cfg.num_observations),
            dtype=torch.float32,
            device=self.device,
        )
        if isinstance(value, dict):
            return build_observation_group_parameter(value, self._observation_group_slices(), reference)

        tensor = torch.as_tensor(value, dtype=torch.float32, device=self.device)
        if tensor.ndim == 0:
            return tensor.reshape(1, 1).repeat(self.num_envs, self.cfg.num_observations)
        if tensor.ndim == 1 and tensor.shape[0] == self.cfg.num_observations:
            return tensor.reshape(1, self.cfg.num_observations).repeat(self.num_envs, 1)
        raise ValueError(f"{name} must be a scalar or length-{self.cfg.num_observations} sequence.")

    def _observation_group_slices(self) -> dict[str, slice]:
        if self.cfg.trajectory_tracking:
            return {
                "target_quat": slice(0, 4),
                "position_error_b": slice(4, 7),
                "target_linear_velocity_b": slice(7, 10),
                "attitude_quat": slice(10, 14),
                "linear_velocity_b": slice(14, 17),
                "angular_velocity_b": slice(17, 20),
                "actions": slice(20, 28),
            }
        return {
            "goal_quat": slice(0, 4),
            "position_error_b": slice(4, 7),
            "attitude_quat": slice(7, 11),
            "linear_velocity_b": slice(11, 14),
            "angular_velocity_b": slice(14, 17),
        }

    def _set_fixed_observation_noise(self, env_ids: Sequence[int] | torch.Tensor) -> None:
        if not isinstance(env_ids, torch.Tensor):
            env_ids_device = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        else:
            env_ids_device = env_ids.to(device=self.device, dtype=torch.long)

        fixed_noise = self._observation_cfg_tensor(self.cfg.observation_noise_std, "observation_noise_std")
        bias_range = self._observation_cfg_tensor(self.cfg.observation_bias_range, "observation_bias_range")
        dropout_probability = self._observation_cfg_tensor(
            self.cfg.observation_dropout_probability,
            "observation_dropout_probability",
        )
        lowpass_alpha = self._observation_cfg_tensor(self.cfg.observation_lowpass_alpha, "observation_lowpass_alpha")
        bias_drift_std = self._observation_cfg_tensor(
            self.cfg.observation_bias_drift_std,
            "observation_bias_drift_std",
        )
        self.observation_noise_std[env_ids_device] = fixed_noise[env_ids_device]
        self.observation_bias[env_ids_device] = (
            2.0 * torch.rand(len(env_ids_device), self.cfg.num_observations, device=self.device) - 1.0
        ) * bias_range[env_ids_device]
        self.observation_dropout_probability[env_ids_device] = torch.clamp(
            dropout_probability[env_ids_device],
            min=0.0,
            max=1.0,
        )
        self.observation_lowpass_alpha[env_ids_device] = torch.clamp(
            lowpass_alpha[env_ids_device],
            min=0.0,
            max=1.0,
        )
        self.observation_bias_drift_std[env_ids_device] = torch.clamp(
            bias_drift_std[env_ids_device],
            min=0.0,
        )

    def _apply_observation_sensor_model(self, obs: torch.Tensor) -> torch.Tensor:
        if (
            torch.all(self.observation_delay_steps <= 0)
            and torch.all(self.observation_noise_std <= 0.0)
            and torch.all(self.observation_bias == 0.0)
            and torch.all(self.observation_update_period_steps <= 1)
            and torch.all(self.observation_dropout_probability <= 0.0)
            and torch.all(self.observation_lowpass_alpha >= 1.0)
            and torch.all(self.observation_bias_drift_std <= 0.0)
        ):
            return obs
        return apply_observation_sensor_model(
            obs,
            self.observation_delay_buffer,
            self.observation_delay_steps,
            self.observation_noise_std,
            self.observation_bias,
            self.observation_filter_state,
            self.observation_update_period_steps,
            self.observation_dropout_probability,
            self.observation_lowpass_alpha,
            self.observation_bias_drift_std,
            self.physics_dt * self.cfg.decimation,
        )

    def _apply_nominal_rigid_body_properties(self) -> None:
        """Apply the Heavy mass, inertia, and COM to the live PhysX body."""

        all_env_ids = self._robot._ALL_INDICES
        self._apply_runtime_mass_properties(all_env_ids)
        self._apply_runtime_center_of_mass(all_env_ids)
        self._robot.data.default_mass = self._robot.root_physx_view.get_masses().clone()
        self._robot.data.default_inertia = self._robot.root_physx_view.get_inertias().clone()

    def _apply_runtime_mass_properties(self, env_ids: Sequence[int] | torch.Tensor) -> None:
        """Write per-env mass and matching inertia tensor into PhysX."""

        if not isinstance(env_ids, torch.Tensor):
            env_ids_device = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        else:
            env_ids_device = env_ids.to(device=self.device, dtype=torch.long)
        env_ids_cpu = env_ids_device.detach().cpu()

        physx_masses = self._robot.root_physx_view.get_masses().clone()
        selected_masses = self.masses[env_ids_device].to(device=physx_masses.device, dtype=physx_masses.dtype)
        if physx_masses.ndim == 1:
            physx_masses[env_ids_cpu] = selected_masses.reshape(-1)
        else:
            physx_masses[env_ids_cpu] = selected_masses.reshape(len(env_ids_cpu), -1)
        self._robot.root_physx_view.set_masses(physx_masses, env_ids_cpu)

        physx_inertias = self._robot.root_physx_view.get_inertias().clone()
        nominal_inertia = inertia_matrix_tensor(
            self.cfg.inertia_diag,
            physx_inertias.device,
            physx_inertias.dtype,
        )
        mass_ratio = selected_masses.reshape(-1, 1) / float(self.cfg.mass)
        flat_inertia = nominal_inertia.reshape(1, 9) * mass_ratio
        if physx_inertias.ndim == 3:
            physx_inertias[env_ids_cpu, :, :] = flat_inertia.reshape(len(env_ids_cpu), 1, 9)
        else:
            physx_inertias[env_ids_cpu, :] = flat_inertia
        self._robot.root_physx_view.set_inertias(physx_inertias, env_ids_cpu)

    def _apply_runtime_center_of_mass(self, env_ids: Sequence[int] | torch.Tensor) -> None:
        """Write the body-frame COM offset into PhysX."""

        if not isinstance(env_ids, torch.Tensor):
            env_ids_device = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        else:
            env_ids_device = env_ids.to(device=self.device, dtype=torch.long)
        env_ids_cpu = env_ids_device.detach().cpu()

        physx_coms = self._robot.root_physx_view.get_coms().clone()
        com_pos = torch.tensor(self.cfg.center_of_mass_offset, device=physx_coms.device, dtype=physx_coms.dtype)
        identity_xyzw = torch.tensor([0.0, 0.0, 0.0, 1.0], device=physx_coms.device, dtype=physx_coms.dtype)
        if physx_coms.ndim == 3:
            physx_coms[env_ids_cpu, :, :3] = com_pos.reshape(1, 1, 3)
            physx_coms[env_ids_cpu, :, 3:7] = identity_xyzw.reshape(1, 1, 4)
        else:
            physx_coms[env_ids_cpu, :3] = com_pos.reshape(1, 3)
            physx_coms[env_ids_cpu, 3:7] = identity_xyzw.reshape(1, 4)
        self._robot.root_physx_view.set_coms(physx_coms, env_ids_cpu)

    def _setup_scene(self):
        self.cfg.robot_cfg.init_state = RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, self.cfg.starting_depth))
        self._robot = RigidObject(self.cfg.robot_cfg)

        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[])

        self.scene.articulations["robot"] = self._robot

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))

        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        if self._debug: print("original actions vec: ", actions)
        if self._debug: print("concatenated actions shape: ", self._actions)

        self._update_smooth_water_current()
        self._previous_actions[:] = self._actions
        self._actions[:] = actions
        self._actions[:] = torch.clip(self._actions, -1, 1).to(self.device)

    def _apply_action(self) -> None:
        self._thrust[:,0,:], self._moment[:,0,:] = self._compute_dynamics(self._actions)
        self._robot.permanent_wrench_composer.set_forces_and_torques(forces=self._thrust, torques=self._moment)

    def _get_observations(self) -> dict:
        if self.cfg.trajectory_tracking:
            # Keep the target synchronized with the current episode time before
            # constructing the policy observation.
            self._update_tracking_targets()
            target_pos_error_b = quat_apply(
                quat_conjugate(self._robot.data.root_quat_w),
                self._target_pos_w - self._robot.data.root_pos_w,
            )
            target_lin_vel_b = quat_apply(
                quat_conjugate(self._robot.data.root_quat_w),
                self._target_lin_vel_w,
            )
            obs = torch.cat(
                [
                    self._target_quat_w,
                    target_pos_error_b,
                    target_lin_vel_b,
                    self._robot.data.root_quat_w,
                    self._robot.data.root_lin_vel_b,
                    self._robot.data.root_ang_vel_b,
                    self._actions,
                ],
                dim=-1,
            )
            observations = {"policy": self._apply_observation_sensor_model(obs)}
            return observations

        #desired_pos_b = quat_apply(quat_conjugate(self._robot.data.root_quat_w), self._goal - self._robot.data.root_pos_w)
        offset_from_origin_b = quat_apply(quat_conjugate(self._robot.data.root_quat_w), self._default_env_origins - self._robot.data.root_pos_w)

        # Uniquefy and normalize all quaternions
        # goal = self._goal
        # root_quat_w = self._robot.data.root_quat_w
        # goal = math_utils.normalize(math_utils.quat_unique(self._goal))
        # root_quat_w = math_utils.normalize(math_utils.quat_unique(self._robot.data.root_quat_w))

        obs = torch.cat(
            [
                self._goal,
                offset_from_origin_b,
                self._robot.data.root_quat_w,
                self._robot.data.root_lin_vel_b,
                self._robot.data.root_ang_vel_b,
            ],
            dim=-1
        )
        observations = {"policy": self._apply_observation_sensor_model(obs)}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        if self.cfg.trajectory_tracking:
            # Reward position and velocity tracking against the moving target.
            # The old pos-hold reward stays unchanged in the branch below.
            self._update_tracking_targets()
            target_lin_vel_b = quat_apply(
                quat_conjugate(self._robot.data.root_quat_w),
                self._target_lin_vel_w,
            )
            total_reward = _compute_tracking_rewards(
                self.cfg.rew_scale_pos,
                self.cfg.rew_scale_ang,
                self.cfg.rew_scale_track_vel,
                self.cfg.rew_scale_ang_vel,
                self.cfg.rew_scale_actions,
                self.cfg.rew_scale_action_rate,
                self.cfg.rew_pos_sigma,
                self.cfg.rew_ang_sigma,
                self.cfg.rew_track_vel_sigma,
                self.cfg.rew_ang_vel_sigma,
                self._robot.data.root_pos_w,
                self._robot.data.root_quat_w,
                self._robot.data.root_lin_vel_b,
                self._robot.data.root_ang_vel_b,
                self._target_pos_w,
                self._target_quat_w,
                target_lin_vel_b,
                self._actions,
                self._previous_actions,
            )
            return total_reward

        offsets_from_origin = quat_apply(quat_conjugate(self._robot.data.root_quat_w), self._default_env_origins - self._robot.data.root_pos_w)

        total_reward = _compute_rewards(
            self.cfg.rew_scale_pos,
            self.cfg.rew_scale_ang,
            self.cfg.rew_scale_lin_vel,
            self.cfg.rew_scale_ang_vel,
            self.cfg.rew_scale_actions,
            self._robot.data.root_lin_vel_b,
            self._robot.data.root_ang_vel_b,
            self.reset_terminated,
            self._robot.data.root_pos_w,
            self._robot.data.root_quat_w,
            self._goal,
            offsets_from_origin,
            self._completed_envs,
            self._actions
        )

        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.cfg.cap_episode_length:
            time_out = self.episode_length_buf >= self.max_episode_length - 1
        else:
            time_out = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self._step_count = self._step_count + 1

        if self.cfg.episode_length_before_reset is not None:
            time_out = time_out | (self.episode_length_buf >= int(self.cfg.episode_length_before_reset))

        if self.cfg.use_boundaries:
            out_of_bounds = (
                (torch.abs(self._robot.data.root_pos_w[:, 0] - self.scene.env_origins[:, 0]) > self.cfg.max_auv_x) | 
                (torch.abs(self._robot.data.root_pos_w[:, 1] - self.scene.env_origins[:, 1]) > self.cfg.max_auv_y) | 
                (torch.abs(self._robot.data.root_pos_w[:, 2] - self.cfg.starting_depth) > self.cfg.max_auv_z)
            )
        else:
            out_of_bounds = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        return out_of_bounds, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        super()._reset_idx(env_ids)
        self.thruster_dynamics.reset(env_ids)
        self.thruster_command_processor.reset(env_ids)
        self.observation_delay_buffer.reset(env_ids)
        self.observation_filter_state.reset(env_ids)
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0

        self._default_root_state[env_ids, :] = self._robot.data.default_root_state[env_ids]
        self._default_root_state[env_ids, :3] += self.scene.env_origins[env_ids]

        self._default_env_origins[env_ids, :] = self._default_root_state[env_ids, :3]

        if self.cfg.trajectory_tracking:
            # Sample the command before initial-state guidance so guided envs
            # can start exactly on the first target pose.
            self._reset_trajectory(env_ids)

        if not self.cfg.eval_mode:
            # Randomize initial position relative to the origin
            self._default_root_state[env_ids, :3] += self._sample_from_sphere(len(env_ids), self.cfg.goal_spawn_radius)

            # Randomize initial orientation relative to the origin
            # self._default_root_state[env_ids, 3:7] = math_utils.random_orientation(len(env_ids), device=self.device)
            
            # Randomize initial linear and rotational velocities
            self._default_root_state[env_ids, 7:13] = math_utils.sample_uniform(
                -self.cfg.init_vel_max,
                self.cfg.init_vel_max,
                (len(env_ids), 6),
                device=self.device,
            )

        self._step_count = 0
        
        # Apply domain randomization
        self._reset_domain(env_ids)
        self._previous_nu_r[env_ids] = 0.0
        self._filtered_nu_r_dot[env_ids] = 0.0
        self._has_previous_nu_r[env_ids] = False

        # Reset goals
        if not self.cfg.trajectory_tracking:
            self._reset_goal(env_ids)

        if not self.cfg.eval_mode:
            # Apply guidance (set to goal position and orientation)
            envs_to_guide = math_utils.sample_uniform(0, 1, len(env_ids), self.device) < self.cfg.init_guidance_rate
            env_ids_to_guide = env_ids[envs_to_guide]
            if self.cfg.trajectory_tracking:
                # Guidance is a curriculum trick: a small fraction of resets
                # start at the desired pose, which exposes the policy to
                # near-target stabilization as well as catch-up behavior.
                self._default_root_state[env_ids_to_guide, :3] = self._target_pos_w[env_ids_to_guide, :3]
                self._default_root_state[env_ids_to_guide, 3:7] = self._target_quat_w[env_ids_to_guide, 0:4]
            else:
                self._default_root_state[env_ids_to_guide, :3] = self._default_env_origins[env_ids_to_guide, :3]
                self._default_root_state[env_ids_to_guide, 3:7] = self._goal[env_ids_to_guide, 0:4]
        elif self.cfg.trajectory_tracking and self.cfg.trajectory_eval_align_initial_target:
            self._default_root_state[env_ids, :3] = self._target_pos_w[env_ids, :3]
            self._default_root_state[env_ids, 3:7] = self._target_quat_w[env_ids, 0:4]

        self._robot.write_root_pose_to_sim(self._default_root_state[env_ids, :7], env_ids)
        self._robot.write_root_velocity_to_sim(self._default_root_state[env_ids, 7:], env_ids)


    # OVERRIDE THIS FUNC TO CHANGE GOAL
    def _reset_goal(self, env_ids: Sequence[int]):
        # Get random orientation
        self._goal[env_ids, 0:4] = math_utils.random_orientation(len(env_ids), device=self.device)

        # Get random yaw orientation with 0 pitch and roll
        # self._goal[env_ids,0:4] = math_utils.random_yaw_orientation(len(env_ids), device=self.device)

        # Get fix RPY
        # rs = torch.zeros(len(env_ids), device=self.device) + 0.0
        # ps = torch.zeros(len(env_ids), device=self.device) + 0.0
        # ys = torch.zeros(len(env_ids), device=self.device) + 0.0
        # self._goal[env_ids,0:4] = math_utils.quat_from_euler_xyz(rs, ps, ys)

    def _get_trajectory_curriculum_stage(self) -> int:
        """Return the active trajectory curriculum stage from global policy steps."""

        if not self.cfg.trajectory_curriculum:
            return -1

        stage = 0
        for step_boundary in self.cfg.trajectory_curriculum_stage_steps:
            if self.common_step_counter >= step_boundary:
                stage += 1

        return min(stage, len(self.cfg.trajectory_curriculum_amp_scales) - 1)

    def _get_trajectory_training_profile(self):
        """Return trajectory type/range settings for the current curriculum stage."""

        if not self.cfg.trajectory_curriculum:
            return (
                self.cfg.trajectory_train_types,
                self.cfg.trajectory_amp_x_range,
                self.cfg.trajectory_amp_y_range,
                self.cfg.trajectory_amp_z_range,
                self.cfg.trajectory_period_range,
            )

        stage = self._get_trajectory_curriculum_stage()
        stage_types = (
            self.cfg.trajectory_curriculum_stage_0_types,
            self.cfg.trajectory_curriculum_stage_1_types,
            self.cfg.trajectory_curriculum_stage_2_types,
            self.cfg.trajectory_curriculum_stage_3_types,
        )[stage]
        amp_scale = self.cfg.trajectory_curriculum_amp_scales[stage]
        z_amp_scale = self.cfg.trajectory_curriculum_z_amp_scales[stage]
        amp_x_range = [self.cfg.trajectory_amp_x_range[0] * amp_scale, self.cfg.trajectory_amp_x_range[1] * amp_scale]
        amp_y_range = [self.cfg.trajectory_amp_y_range[0] * amp_scale, self.cfg.trajectory_amp_y_range[1] * amp_scale]
        amp_z_range = [
            self.cfg.trajectory_amp_z_range[0] * z_amp_scale,
            self.cfg.trajectory_amp_z_range[1] * z_amp_scale,
        ]
        period_range = [
            self.cfg.trajectory_curriculum_period_min[stage],
            self.cfg.trajectory_curriculum_period_max[stage],
        ]

        return stage_types, amp_x_range, amp_y_range, amp_z_range, period_range

    def _reset_trajectory(self, env_ids: Sequence[int]):
        """Sample trajectory parameters for reset environments."""

        num_env_ids = len(env_ids)
        zeros = torch.zeros(num_env_ids, device=self.device)

        # The trajectory center is the environment origin at the nominal
        # starting depth, so the command stays local to each cloned env.
        self._traj_center_w[env_ids, :] = self._default_env_origins[env_ids, :]
        self._target_quat_w[env_ids, :] = math_utils.quat_from_euler_xyz(zeros, zeros, zeros)

        if self.cfg.trajectory_eval_mode:
            # Fixed eval trajectories use deterministic parameters so repeated
            # evaluations are comparable across checkpoints.
            self._traj_type[env_ids] = self.cfg.trajectory_eval_type
            self._traj_axis[env_ids] = 0
            if self.cfg.trajectory_eval_type == 7:
                amp_x_lower, amp_x_upper = self.cfg.trajectory_amp_x_range
                amp_y_lower, amp_y_upper = self.cfg.trajectory_amp_y_range
                amp_z_lower, amp_z_upper = self.cfg.trajectory_amp_z_range
                period_lower, period_upper = self.cfg.trajectory_period_range
                self._traj_amp_x[env_ids] = math_utils.sample_uniform(
                    amp_x_lower, amp_x_upper, (num_env_ids,), device=self.device
                )
                self._traj_amp_y[env_ids] = math_utils.sample_uniform(
                    amp_y_lower, amp_y_upper, (num_env_ids,), device=self.device
                )
                self._traj_amp_z[env_ids] = math_utils.sample_uniform(
                    amp_z_lower, amp_z_upper, (num_env_ids,), device=self.device
                )
                self._traj_period[env_ids] = math_utils.sample_uniform(
                    period_lower, period_upper, (num_env_ids,), device=self.device
                )
                self._traj_phase_x[env_ids] = math_utils.sample_uniform(
                    0.0, 2.0 * torch.pi, (num_env_ids,), device=self.device
                )
                self._traj_phase_y[env_ids] = math_utils.sample_uniform(
                    0.0, 2.0 * torch.pi, (num_env_ids,), device=self.device
                )
            else:
                self._traj_amp_x[env_ids] = self.cfg.trajectory_eval_amp_x
                self._traj_amp_y[env_ids] = self.cfg.trajectory_eval_amp_y
                self._traj_amp_z[env_ids] = self.cfg.trajectory_eval_amp_z
                self._traj_period[env_ids] = self.cfg.trajectory_eval_period
                self._traj_phase_x[env_ids] = 0.0
                self._traj_phase_y[env_ids] = 0.0
        else:
            # Random smooth trajectories form the RL training command
            # distribution.  The shapes share one compact parameterization so
            # the observation interface remains identical across all samples.
            train_types, amp_x_range, amp_y_range, amp_z_range, period_range = self._get_trajectory_training_profile()
            amp_x_lower, amp_x_upper = amp_x_range
            amp_y_lower, amp_y_upper = amp_y_range
            amp_z_lower, amp_z_upper = amp_z_range
            period_lower, period_upper = period_range
            train_types = torch.as_tensor(train_types, device=self.device, dtype=torch.long)
            train_type_indices = torch.randint(0, len(train_types), (num_env_ids,), device=self.device)
            self._traj_type[env_ids] = train_types[train_type_indices]
            self._traj_axis[env_ids] = torch.randint(0, 3, (num_env_ids,), device=self.device)
            self._traj_amp_x[env_ids] = math_utils.sample_uniform(
                amp_x_lower, amp_x_upper, (num_env_ids,), device=self.device
            )
            self._traj_amp_y[env_ids] = math_utils.sample_uniform(
                amp_y_lower, amp_y_upper, (num_env_ids,), device=self.device
            )
            self._traj_amp_z[env_ids] = math_utils.sample_uniform(
                amp_z_lower, amp_z_upper, (num_env_ids,), device=self.device
            )
            self._traj_period[env_ids] = math_utils.sample_uniform(
                period_lower, period_upper, (num_env_ids,), device=self.device
            )
            self._traj_phase_x[env_ids] = math_utils.sample_uniform(
                0.0, 2.0 * torch.pi, (num_env_ids,), device=self.device
            )
            self._traj_phase_y[env_ids] = math_utils.sample_uniform(
                0.0, 2.0 * torch.pi, (num_env_ids,), device=self.device
            )

        self._update_tracking_targets(env_ids)

    def _update_tracking_targets(self, env_ids: Sequence[int] | None = None):
        """Update target pose/velocity from the stored trajectory parameters."""

        if env_ids is None:
            env_ids = self._robot._ALL_INDICES

        # Use policy-step time, not physics substep time, because observations
        # and rewards are produced once per decimated control step.
        t = self.episode_length_buf[env_ids].to(dtype=torch.float32) * self.cfg.sim.dt * self.cfg.decimation
        omega = (2.0 * torch.pi) / self._traj_period[env_ids]
        phase_x = omega * t + self._traj_phase_x[env_ids]
        phase_y = 2.0 * omega * t + self._traj_phase_y[env_ids]
        num_env_ids = len(env_ids)
        offsets_w = torch.zeros(num_env_ids, 3, device=self.device)
        target_lin_vel_w = torch.zeros(num_env_ids, 3, device=self.device)
        traj_type = self._traj_type[env_ids]

        # Ellipse/circle: x = Ax*cos(wt), y = Ay*sin(wt).
        circle_mask = traj_type == 0
        offsets_w[circle_mask, 0] = self._traj_amp_x[env_ids][circle_mask] * torch.cos(phase_x[circle_mask])
        offsets_w[circle_mask, 1] = self._traj_amp_y[env_ids][circle_mask] * torch.sin(phase_x[circle_mask])
        target_lin_vel_w[circle_mask, 0] = -self._traj_amp_x[env_ids][circle_mask] * omega[circle_mask] * torch.sin(phase_x[circle_mask])
        target_lin_vel_w[circle_mask, 1] = self._traj_amp_y[env_ids][circle_mask] * omega[circle_mask] * torch.cos(phase_x[circle_mask])

        # Figure-eight/Lissajous: x = Ax*sin(wt), y = Ay*sin(2wt).
        lissajous_mask = traj_type == 1
        offsets_w[lissajous_mask, 0] = self._traj_amp_x[env_ids][lissajous_mask] * torch.sin(phase_x[lissajous_mask])
        offsets_w[lissajous_mask, 1] = self._traj_amp_y[env_ids][lissajous_mask] * torch.sin(phase_y[lissajous_mask])
        target_lin_vel_w[lissajous_mask, 0] = self._traj_amp_x[env_ids][lissajous_mask] * omega[lissajous_mask] * torch.cos(phase_x[lissajous_mask])
        target_lin_vel_w[lissajous_mask, 1] = 2.0 * self._traj_amp_y[env_ids][lissajous_mask] * omega[lissajous_mask] * torch.cos(phase_y[lissajous_mask])

        # Single-axis sine randomly chooses x/y/z, which forces the policy to
        # learn each translational axis without needing a separate task.
        sine_mask = traj_type == 2
        sine_indices = torch.nonzero(sine_mask, as_tuple=False).squeeze(-1)
        sine_axes = self._traj_axis[env_ids][sine_indices]
        sine_amps = self._traj_amp_x[env_ids][sine_indices]
        sine_amps = torch.where(sine_axes == 1, self._traj_amp_y[env_ids][sine_indices], sine_amps)
        sine_amps = torch.where(sine_axes == 2, self._traj_amp_z[env_ids][sine_indices], sine_amps)
        sine_offsets = sine_amps * torch.sin(phase_x[sine_indices])
        sine_vels = sine_amps * omega[sine_indices] * torch.cos(phase_x[sine_indices])
        for axis in range(3):
            axis_mask = sine_axes == axis
            local_indices = sine_indices[axis_mask]
            offsets_w[local_indices, axis] = sine_offsets[axis_mask]
            target_lin_vel_w[local_indices, axis] = sine_vels[axis_mask]

        # 3D helix-like loop: circle in xy plus vertical sinusoid.
        helix_mask = traj_type == 3
        offsets_w[helix_mask, 0] = self._traj_amp_x[env_ids][helix_mask] * torch.cos(phase_x[helix_mask])
        offsets_w[helix_mask, 1] = self._traj_amp_y[env_ids][helix_mask] * torch.sin(phase_x[helix_mask])
        offsets_w[helix_mask, 2] = self._traj_amp_z[env_ids][helix_mask] * torch.sin(phase_y[helix_mask])
        target_lin_vel_w[helix_mask, 0] = -self._traj_amp_x[env_ids][helix_mask] * omega[helix_mask] * torch.sin(phase_x[helix_mask])
        target_lin_vel_w[helix_mask, 1] = self._traj_amp_y[env_ids][helix_mask] * omega[helix_mask] * torch.cos(phase_x[helix_mask])
        target_lin_vel_w[helix_mask, 2] = 2.0 * self._traj_amp_z[env_ids][helix_mask] * omega[helix_mask] * torch.cos(phase_y[helix_mask])

        # Spiral: radius grows and shrinks while the target rotates.
        spiral_mask = traj_type == 4
        radius_min = self.cfg.trajectory_eval_radius_min
        radius_max = self.cfg.trajectory_eval_radius_max
        radius_range = radius_max - radius_min
        radial_phase = 0.5 * omega[spiral_mask] * t[spiral_mask]
        radius = radius_min + 0.5 * radius_range * (1.0 - torch.cos(radial_phase))
        radius_dot = 0.25 * radius_range * omega[spiral_mask] * torch.sin(radial_phase)
        offsets_w[spiral_mask, 0] = radius * torch.cos(phase_x[spiral_mask])
        offsets_w[spiral_mask, 1] = radius * torch.sin(phase_x[spiral_mask])
        target_lin_vel_w[spiral_mask, 0] = (
            radius_dot * torch.cos(phase_x[spiral_mask])
            - radius * omega[spiral_mask] * torch.sin(phase_x[spiral_mask])
        )
        target_lin_vel_w[spiral_mask, 1] = (
            radius_dot * torch.sin(phase_x[spiral_mask])
            + radius * omega[spiral_mask] * torch.cos(phase_x[spiral_mask])
        )

        # Chirp: Lissajous shape with frequency increasing during the episode.
        # This probes speed/acceleration generalization.
        chirp_mask = traj_type == 5
        w0 = omega[chirp_mask]
        w1 = self.cfg.trajectory_eval_chirp_rate * w0
        chirp_duration = self.cfg.trajectory_eval_duration_s
        chirp_k = (w1 - w0) / chirp_duration
        chirp_t = t[chirp_mask]
        chirp_phase = w0 * chirp_t + 0.5 * chirp_k * chirp_t**2
        chirp_omega = w0 + chirp_k * chirp_t
        chirp_phase_x = chirp_phase + self._traj_phase_x[env_ids][chirp_mask]
        chirp_phase_y = 2.0 * chirp_phase + self._traj_phase_y[env_ids][chirp_mask]
        offsets_w[chirp_mask, 0] = self._traj_amp_x[env_ids][chirp_mask] * torch.sin(chirp_phase_x)
        offsets_w[chirp_mask, 1] = self._traj_amp_y[env_ids][chirp_mask] * torch.sin(chirp_phase_y)
        target_lin_vel_w[chirp_mask, 0] = self._traj_amp_x[env_ids][chirp_mask] * chirp_omega * torch.cos(chirp_phase_x)
        target_lin_vel_w[chirp_mask, 1] = 2.0 * self._traj_amp_y[env_ids][chirp_mask] * chirp_omega * torch.cos(chirp_phase_y)

        # Held-out random smooth 3D curve: a low-frequency Fourier blend with
        # continuous position and velocity.  This is meant for generalization
        # tests, not for the staged training curriculum.
        random_smooth_mask = traj_type == 7
        random_indices = torch.nonzero(random_smooth_mask, as_tuple=False).squeeze(-1)
        if len(random_indices) > 0:
            random_t = t[random_indices]
            random_omega = omega[random_indices]
            random_phase_x = self._traj_phase_x[env_ids][random_indices]
            random_phase_y = self._traj_phase_y[env_ids][random_indices]
            ax = self._traj_amp_x[env_ids][random_indices]
            ay = self._traj_amp_y[env_ids][random_indices]
            az = self._traj_amp_z[env_ids][random_indices]

            x_phase_1 = random_omega * random_t + random_phase_x
            x_phase_2 = 2.0 * random_omega * random_t + random_phase_y
            y_phase_1 = 1.5 * random_omega * random_t + random_phase_y
            y_phase_2 = 0.5 * random_omega * random_t + random_phase_x
            z_phase_1 = 0.75 * random_omega * random_t + random_phase_x + random_phase_y
            z_phase_2 = 2.0 * random_omega * random_t + random_phase_x

            offsets_w[random_indices, 0] = ax * (0.60 * torch.sin(x_phase_1) + 0.30 * torch.sin(x_phase_2))
            offsets_w[random_indices, 1] = ay * (0.55 * torch.sin(y_phase_1) + 0.35 * torch.sin(y_phase_2))
            offsets_w[random_indices, 2] = az * (0.70 * torch.sin(z_phase_1) + 0.25 * torch.sin(z_phase_2))
            target_lin_vel_w[random_indices, 0] = ax * random_omega * (
                0.60 * torch.cos(x_phase_1) + 0.60 * torch.cos(x_phase_2)
            )
            target_lin_vel_w[random_indices, 1] = ay * random_omega * (
                0.825 * torch.cos(y_phase_1) + 0.175 * torch.cos(y_phase_2)
            )
            target_lin_vel_w[random_indices, 2] = az * random_omega * (
                0.525 * torch.cos(z_phase_1) + 0.50 * torch.cos(z_phase_2)
            )

        # Held-out racetrack: straight segments joined by semicircles with
        # continuous velocity.  It stresses cornering behavior without hard
        # discontinuities in the command.
        racetrack_mask = traj_type == 6
        racetrack_indices = torch.nonzero(racetrack_mask, as_tuple=False).squeeze(-1)
        if len(racetrack_indices) > 0:
            radius = self._traj_amp_y[env_ids][racetrack_indices]
            half_straight = torch.clamp(self._traj_amp_x[env_ids][racetrack_indices] - radius, min=0.1)
            path_length = 4.0 * half_straight + 2.0 * torch.pi * radius
            speed = path_length / self._traj_period[env_ids][racetrack_indices]
            s = torch.remainder(speed * t[racetrack_indices], path_length)
            local_offsets = torch.zeros(len(racetrack_indices), 3, device=self.device)
            local_vels = torch.zeros(len(racetrack_indices), 3, device=self.device)

            top_mask = s < 2.0 * half_straight
            top_ix = torch.nonzero(top_mask, as_tuple=False).squeeze(-1)
            local_offsets[top_ix, 0] = -half_straight[top_ix] + s[top_ix]
            local_offsets[top_ix, 1] = radius[top_ix]
            local_vels[top_ix, 0] = speed[top_ix]

            right_mask = (s >= 2.0 * half_straight) & (s < 2.0 * half_straight + torch.pi * radius)
            right_ix = torch.nonzero(right_mask, as_tuple=False).squeeze(-1)
            right_s = s[right_ix] - 2.0 * half_straight[right_ix]
            right_theta = torch.pi / 2.0 - right_s / radius[right_ix]
            local_offsets[right_ix, 0] = half_straight[right_ix] + radius[right_ix] * torch.cos(right_theta)
            local_offsets[right_ix, 1] = radius[right_ix] * torch.sin(right_theta)
            local_vels[right_ix, 0] = speed[right_ix] * torch.sin(right_theta)
            local_vels[right_ix, 1] = -speed[right_ix] * torch.cos(right_theta)

            bottom_mask = (
                (s >= 2.0 * half_straight + torch.pi * radius)
                & (s < 4.0 * half_straight + torch.pi * radius)
            )
            bottom_ix = torch.nonzero(bottom_mask, as_tuple=False).squeeze(-1)
            bottom_s = s[bottom_ix] - (2.0 * half_straight[bottom_ix] + torch.pi * radius[bottom_ix])
            local_offsets[bottom_ix, 0] = half_straight[bottom_ix] - bottom_s
            local_offsets[bottom_ix, 1] = -radius[bottom_ix]
            local_vels[bottom_ix, 0] = -speed[bottom_ix]

            left_mask = s >= 4.0 * half_straight + torch.pi * radius
            left_ix = torch.nonzero(left_mask, as_tuple=False).squeeze(-1)
            left_s = s[left_ix] - (4.0 * half_straight[left_ix] + torch.pi * radius[left_ix])
            left_theta = -torch.pi / 2.0 - left_s / radius[left_ix]
            local_offsets[left_ix, 0] = -half_straight[left_ix] + radius[left_ix] * torch.cos(left_theta)
            local_offsets[left_ix, 1] = radius[left_ix] * torch.sin(left_theta)
            local_vels[left_ix, 0] = speed[left_ix] * torch.sin(left_theta)
            local_vels[left_ix, 1] = -speed[left_ix] * torch.cos(left_theta)

            offsets_w[racetrack_indices, :] = local_offsets
            target_lin_vel_w[racetrack_indices, :] = local_vels

        self._target_pos_w[env_ids, :] = self._traj_center_w[env_ids, :] + offsets_w
        self._target_lin_vel_w[env_ids, :] = target_lin_vel_w
        self._goal[env_ids, 0:4] = self._target_quat_w[env_ids, 0:4]
        self._goal_pos_w = self._target_pos_w

    def get_tracking_targets(self):
        """Return synchronized trajectory targets for eval/logging code."""

        if self.cfg.trajectory_tracking:
            self._update_tracking_targets()
        return self._target_pos_w, self._target_lin_vel_w, self._target_quat_w

    def _reset_domain(self, env_ids: Sequence[int]):
        if self.cfg.domain_randomization.use_custom_randomization:
            mass_lower, mass_upper = self.cfg.domain_randomization.mass_range
            self.masses[env_ids] = math_utils.sample_uniform(
                mass_lower,
                mass_upper,
                self.masses[env_ids].shape,
                self.device,
            )
            self._apply_runtime_mass_properties(env_ids)

        # Randomize COM to COB offset
        if self.cfg.domain_randomization.use_custom_randomization:
            self.com_to_cob_offsets[env_ids] = self.cfg.com_to_cob_offset[env_ids] + self._sample_from_sphere(len(env_ids), self.cfg.domain_randomization.com_to_cob_offset_radius)

        # Randomize volume
        if self.cfg.domain_randomization.use_custom_randomization:
            vol_lower, vol_upper = self.cfg.domain_randomization.volume_range
            self.volumes[env_ids] = math_utils.sample_uniform(vol_lower, vol_upper, self.volumes[env_ids].shape, self.device)

        self._reset_disturbance_domain(env_ids)
        self._reset_observation_domain(env_ids)

    def _get_disturbance_curriculum_stage(self) -> int:
        if not getattr(self.cfg.domain_randomization, "disturbance_curriculum", False):
            return len(self.cfg.domain_randomization.water_current_max_by_stage) - 1

        stage = 0
        for step_boundary in self.cfg.domain_randomization.disturbance_curriculum_stage_steps:
            if self.common_step_counter >= step_boundary:
                stage += 1
        return min(stage, len(self.cfg.domain_randomization.water_current_max_by_stage) - 1)

    def _reset_disturbance_domain(self, env_ids: Sequence[int]) -> None:
        if not isinstance(env_ids, torch.Tensor):
            env_ids_device = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        else:
            env_ids_device = env_ids.to(device=self.device, dtype=torch.long)
        num_resets = len(env_ids_device)

        self.water_current_w[env_ids_device] = self._nominal_water_current_w
        self.water_current_mean_w[env_ids_device] = self._nominal_water_current_w
        self.water_current_horizontal_max[env_ids_device] = 0.0
        self.water_current_vertical_max[env_ids_device] = 0.0
        self.water_current_tau[env_ids_device] = 12.0
        self.linear_damping[env_ids_device] = self._nominal_linear_damping
        self.quadratic_damping[env_ids_device] = self._nominal_quadratic_damping
        self.added_mass_diag[env_ids_device] = self._nominal_added_mass_diag
        self.thruster_force_scale[env_ids_device] = 1.0
        self.thruster_time_constant[env_ids_device] = self.cfg.dyn_time_constant
        self.thruster_deadband[env_ids_device] = self.cfg.thruster_deadband
        self.thruster_delay_steps[env_ids_device] = int(self.cfg.thruster_command_delay_steps)
        self.thruster_max_command_rate[env_ids_device] = self.cfg.thruster_max_command_rate
        self.thruster_command_resolution[env_ids_device] = self.cfg.thruster_command_resolution
        self.thruster_command_dropout_probability[env_ids_device] = self.cfg.thruster_command_dropout_probability
        self.battery_initial_voltage[env_ids_device] = self.cfg.battery_voltage
        self.battery_voltage[env_ids_device] = self.cfg.battery_voltage
        self.battery_voltage_drop_per_s[env_ids_device] = self.cfg.battery_voltage_drop_per_s

        if self.cfg.eval_mode or not self.cfg.domain_randomization.use_custom_randomization:
            self.thruster_dynamics.tau = self.thruster_time_constant
            return

        stage = self._get_disturbance_curriculum_stage()
        current_max = self.cfg.domain_randomization.water_current_max_by_stage[stage]
        vertical_current_max = self.cfg.domain_randomization.water_current_vertical_max_by_stage[stage]
        damping_scale = self.cfg.domain_randomization.damping_scale_by_stage[stage]
        thruster_scale = self.cfg.domain_randomization.thruster_scale_by_stage[stage]
        tau_scale = self.cfg.domain_randomization.thruster_tau_scale_by_stage[stage]
        deadband_scale = self.cfg.domain_randomization.thruster_deadband_scale_by_stage[stage]
        delay_min, delay_max = self.cfg.domain_randomization.thruster_command_delay_steps_range
        rate_min, rate_max = self.cfg.domain_randomization.thruster_max_command_rate_range
        resolution_min, resolution_max = self.cfg.domain_randomization.thruster_command_resolution_range
        dropout_min, dropout_max = self.cfg.domain_randomization.thruster_command_dropout_probability_range
        voltage_min, voltage_max = self.cfg.domain_randomization.battery_voltage_range
        voltage_drop_min, voltage_drop_max = self.cfg.domain_randomization.battery_voltage_drop_per_s_range

        if current_max > 0.0:
            theta = 2.0 * torch.pi * torch.rand(num_resets, device=self.device)
            mag = current_max * torch.sqrt(torch.rand(num_resets, device=self.device))
            self.water_current_mean_w[env_ids_device, 0] = mag * torch.cos(theta)
            self.water_current_mean_w[env_ids_device, 1] = mag * torch.sin(theta)
            self.water_current_mean_w[env_ids_device, 2] = (
                2.0 * torch.rand(num_resets, device=self.device) - 1.0
            ) * vertical_current_max
            self.water_current_w[env_ids_device] = self.water_current_mean_w[env_ids_device]
            self.water_current_horizontal_max[env_ids_device] = current_max
            self.water_current_vertical_max[env_ids_device] = vertical_current_max
            tau_min, tau_max = self.cfg.domain_randomization.water_current_tau_range
            self.water_current_tau[env_ids_device] = math_utils.sample_uniform(
                tau_min,
                tau_max,
                (num_resets,),
                self.device,
            )

        if damping_scale > 0.0:
            if self.linear_damping.ndim == 2:
                damping_shape = (num_resets, 6)
            else:
                damping_shape = (num_resets, 1, 1)
            damping_mult = 1.0 + damping_scale * (2.0 * torch.rand(damping_shape, device=self.device) - 1.0)
            self.linear_damping[env_ids_device] = self._nominal_linear_damping * damping_mult
            self.quadratic_damping[env_ids_device] = self._nominal_quadratic_damping * damping_mult

        if thruster_scale > 0.0:
            self.thruster_force_scale[env_ids_device] = 1.0 + thruster_scale * (
                2.0 * torch.rand(num_resets, self.num_thrusters, device=self.device) - 1.0
            )

        if tau_scale > 0.0:
            tau_mult = 1.0 + tau_scale * (2.0 * torch.rand(num_resets, device=self.device) - 1.0)
            self.thruster_time_constant[env_ids_device] = torch.clamp(
                self.cfg.dyn_time_constant * tau_mult,
                min=0.01,
            )

        if deadband_scale > 0.0:
            deadband_mult = 1.0 + deadband_scale * (
                2.0 * torch.rand(num_resets, self.num_thrusters, device=self.device) - 1.0
            )
            self.thruster_deadband[env_ids_device] = torch.clamp(
                self.cfg.thruster_deadband * deadband_mult,
                min=0.04,
                max=0.14,
            )

        if delay_max > delay_min:
            self.thruster_delay_steps[env_ids_device] = torch.randint(
                int(delay_min),
                int(delay_max) + 1,
                (num_resets,),
                device=self.device,
            )

        if rate_max > rate_min:
            self.thruster_max_command_rate[env_ids_device] = math_utils.sample_uniform(
                rate_min,
                rate_max,
                (num_resets, 1),
                self.device,
            )

        if resolution_max > resolution_min:
            self.thruster_command_resolution[env_ids_device] = math_utils.sample_uniform(
                resolution_min,
                resolution_max,
                (num_resets, 1),
                self.device,
            )

        if dropout_max > dropout_min:
            self.thruster_command_dropout_probability[env_ids_device] = math_utils.sample_uniform(
                dropout_min,
                dropout_max,
                (num_resets, 1),
                self.device,
            )

        if voltage_max > voltage_min:
            sampled_voltage = math_utils.sample_uniform(
                voltage_min,
                voltage_max,
                (num_resets, 1),
                self.device,
            )
            self.battery_initial_voltage[env_ids_device] = sampled_voltage
            self.battery_voltage[env_ids_device] = sampled_voltage

        if voltage_drop_max > voltage_drop_min:
            self.battery_voltage_drop_per_s[env_ids_device] = math_utils.sample_uniform(
                voltage_drop_min,
                voltage_drop_max,
                (num_resets, 1),
                self.device,
            )

        self.thruster_dynamics.tau = self.thruster_time_constant

    def _reset_observation_domain(self, env_ids: Sequence[int]) -> None:
        if not isinstance(env_ids, torch.Tensor):
            env_ids_device = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        else:
            env_ids_device = env_ids.to(device=self.device, dtype=torch.long)
        num_resets = len(env_ids_device)

        self.observation_delay_steps[env_ids_device] = int(self.cfg.observation_delay_steps)
        self.observation_update_period_steps[env_ids_device] = int(self.cfg.observation_update_period_steps)
        self._set_fixed_observation_noise(env_ids_device)

        if self.cfg.eval_mode or not self.cfg.domain_randomization.use_custom_randomization:
            return

        noise_min, noise_max = self.cfg.domain_randomization.observation_noise_std_range
        bias_min, bias_max = self.cfg.domain_randomization.observation_bias_range
        delay_min, delay_max = self.cfg.domain_randomization.observation_delay_steps_range
        update_period_min, update_period_max = self.cfg.domain_randomization.observation_update_period_steps_range
        dropout_min, dropout_max = self.cfg.domain_randomization.observation_dropout_probability_range
        lowpass_min, lowpass_max = self.cfg.domain_randomization.observation_lowpass_alpha_range
        bias_drift_min, bias_drift_max = self.cfg.domain_randomization.observation_bias_drift_std_range

        if noise_max > noise_min:
            sampled_noise = math_utils.sample_uniform(
                noise_min,
                noise_max,
                (num_resets, 1),
                self.device,
            )
            self.observation_noise_std[env_ids_device] = sampled_noise

        if bias_max > bias_min:
            sampled_bias_range = math_utils.sample_uniform(
                bias_min,
                bias_max,
                (num_resets, 1),
                self.device,
            )
            self.observation_bias[env_ids_device] = (
                2.0 * torch.rand(num_resets, self.cfg.num_observations, device=self.device) - 1.0
            ) * sampled_bias_range

        if delay_max > delay_min:
            self.observation_delay_steps[env_ids_device] = torch.randint(
                int(delay_min),
                int(delay_max) + 1,
                (num_resets,),
                device=self.device,
            )

        if update_period_max > update_period_min:
            self.observation_update_period_steps[env_ids_device] = torch.randint(
                int(update_period_min),
                int(update_period_max) + 1,
                (num_resets,),
                device=self.device,
            )

        if dropout_max > dropout_min:
            sampled_dropout = math_utils.sample_uniform(
                dropout_min,
                dropout_max,
                (num_resets, 1),
                self.device,
            )
            self.observation_dropout_probability[env_ids_device] = sampled_dropout

        if lowpass_max > lowpass_min:
            sampled_lowpass = math_utils.sample_uniform(
                lowpass_min,
                lowpass_max,
                (num_resets, 1),
                self.device,
            )
            self.observation_lowpass_alpha[env_ids_device] = torch.clamp(sampled_lowpass, min=0.0, max=1.0)

        if bias_drift_max > bias_drift_min:
            sampled_bias_drift = math_utils.sample_uniform(
                bias_drift_min,
                bias_drift_max,
                (num_resets, 1),
                self.device,
            )
            self.observation_bias_drift_std[env_ids_device] = sampled_bias_drift

    def _update_smooth_water_current(self) -> None:
        if (
            not self.cfg.domain_randomization.use_custom_randomization
            or not getattr(self.cfg.domain_randomization, "water_current_smooth", False)
        ):
            return

        stage = self._get_disturbance_curriculum_stage()
        variation_std = self.cfg.domain_randomization.water_current_variation_std_by_stage[stage]
        if variation_std <= 0.0 and torch.all(self.water_current_horizontal_max <= 0.0):
            return

        policy_dt = self.physics_dt * self.cfg.decimation
        tau = torch.clamp(self.water_current_tau, min=policy_dt)
        alpha = torch.exp(-policy_dt / tau).unsqueeze(-1)
        noise_scale = torch.sqrt(torch.clamp(1.0 - alpha * alpha, min=0.0))
        noise = torch.randn_like(self.water_current_w) * variation_std * noise_scale
        noise[:, 2] *= 0.5
        self.water_current_w[:] = (
            alpha * self.water_current_w
            + (1.0 - alpha) * self.water_current_mean_w
            + noise
        )

        xy = self.water_current_w[:, 0:2]
        xy_norm = torch.linalg.norm(xy, dim=1, keepdim=True)
        xy_limit = self.water_current_horizontal_max.unsqueeze(-1)
        xy_scale = torch.clamp(xy_limit / torch.clamp(xy_norm, min=1.0e-6), max=1.0)
        self.water_current_w[:, 0:2] = xy * xy_scale
        self.water_current_w[:, 2] = torch.clamp(
            self.water_current_w[:, 2],
            -self.water_current_vertical_max,
            self.water_current_vertical_max,
        )

    def _calculate_water_current_w(self) -> torch.Tensor:
        if not self.cfg.water_current_field_enabled:
            return self.water_current_w
        local_positions = self._robot.data.root_pos_w - self.scene.env_origins
        field_current_w = calculate_trilinear_current_field(
            local_positions,
            self.cfg.water_current_field_bounds,
            self.cfg.water_current_field_shape,
            self.cfg.water_current_field_values,
        )
        return self.water_current_w + field_current_w

    def _update_relative_acceleration_b(self, water_current_w: torch.Tensor) -> torch.Tensor:
        """Estimate filtered body-frame ``dot(nu_r)`` for added-mass inertia."""

        nu_r = self.force_calculation_functions.calculate_relative_velocity(
            self._robot.data.root_quat_w,
            self._robot.data.root_lin_vel_b,
            self._robot.data.root_ang_vel_b,
            water_current_w,
        )
        has_previous = self._has_previous_nu_r.unsqueeze(-1)
        previous_nu_r = torch.where(has_previous, self._previous_nu_r, nu_r)
        raw_nu_r_dot = (nu_r - previous_nu_r) / max(float(self.physics_dt), 1.0e-6)

        alpha = torch.clamp(
            torch.as_tensor(
                self.cfg.added_mass_accel_filter_alpha,
                dtype=torch.float32,
                device=self.device,
            ),
            min=0.0,
            max=1.0,
        )
        self._filtered_nu_r_dot[:] = alpha * raw_nu_r_dot + (1.0 - alpha) * self._filtered_nu_r_dot
        self._filtered_nu_r_dot[:] = torch.where(
            has_previous,
            self._filtered_nu_r_dot,
            torch.zeros_like(self._filtered_nu_r_dot),
        )
        self._previous_nu_r[:] = nu_r
        self._has_previous_nu_r[:] = True
        return self._filtered_nu_r_dot

    def _update_battery_voltage_scale(self) -> torch.Tensor:
        episode_time = (
            self.episode_length_buf.to(dtype=torch.float32).reshape(self.num_envs, 1)
            * self.physics_dt
            * self.cfg.decimation
        )
        self.battery_voltage[:] = torch.clamp(
            self.battery_initial_voltage - self.battery_voltage_drop_per_s * episode_time,
            min=self.cfg.battery_min_voltage,
        )
        scale = calculate_voltage_thrust_scale(
            self.battery_voltage,
            self.cfg.battery_voltage_nominal,
            self.cfg.battery_voltage_thrust_exponent,
            self.cfg.battery_min_voltage,
        )
        return scale.to(device=self.device, dtype=torch.float32)

    def _calculate_pool_boundary_scales(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ones = torch.ones((self.num_envs, 1), dtype=torch.float32, device=self.device)
        if not self.cfg.pool_boundary_effects_enabled:
            return ones, ones, ones

        local_positions = self._robot.data.root_pos_w - self.scene.env_origins
        return calculate_pool_boundary_scales(
            local_positions,
            self.cfg.pool_bounds,
            self.cfg.pool_boundary_effect_distance,
            self.cfg.pool_boundary_damping_scale,
            self.cfg.pool_boundary_added_mass_scale,
            self.cfg.pool_boundary_thrust_scale,
        )

    def _calculate_free_surface_scales(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        ones_6 = torch.ones((self.num_envs, 6), dtype=torch.float32, device=self.device)
        ones_1 = torch.ones((self.num_envs, 1), dtype=torch.float32, device=self.device)
        if not self.cfg.free_surface_effects_enabled:
            return ones_6, ones_6, ones_1, ones_1

        local_positions = self._robot.data.root_pos_w - self.scene.env_origins
        return calculate_free_surface_scales(
            local_positions,
            self.cfg.free_surface_z,
            self.cfg.free_surface_effect_distance,
            self.cfg.free_surface_heave_damping_scale,
            self.cfg.free_surface_roll_pitch_damping_scale,
            self.cfg.free_surface_added_mass_scale,
            self.cfg.free_surface_buoyancy_scale,
            self.cfg.free_surface_thrust_scale,
        )

    def _calculate_speed_dependent_damping_scales(
        self,
        water_current_w: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ones = torch.ones((self.num_envs, 6), dtype=torch.float32, device=self.device)
        if not self.cfg.speed_dependent_damping_enabled:
            return ones, ones

        has_linear_curve = len(self.cfg.linear_damping_speed_scales) > 0
        has_quadratic_curve = len(self.cfg.quadratic_damping_speed_scales) > 0
        if not has_linear_curve and not has_quadratic_curve:
            raise ValueError(
                "At least one damping speed scale curve must be provided when "
                "speed_dependent_damping_enabled=True."
            )

        nu_r = self.force_calculation_functions.calculate_relative_velocity(
            self._robot.data.root_quat_w,
            self._robot.data.root_lin_vel_b,
            self._robot.data.root_ang_vel_b,
            water_current_w,
        )
        linear_scale = ones
        quadratic_scale = ones
        if has_linear_curve:
            linear_scale = calculate_speed_dependent_damping_scale(
                nu_r,
                self.cfg.damping_speed_points,
                self.cfg.linear_damping_speed_scales,
            )
        if has_quadratic_curve:
            quadratic_scale = calculate_speed_dependent_damping_scale(
                nu_r,
                self.cfg.damping_speed_points,
                self.cfg.quadratic_damping_speed_scales,
            )
        return linear_scale, quadratic_scale

    def _calculate_tether_wrench(self, water_current_w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        zeros = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        if not self.cfg.tether_enabled:
            return zeros, zeros
        if water_current_w.ndim == 1:
            water_current_w = water_current_w.reshape(1, 3).repeat(self.num_envs, 1)
        return calculate_multisegment_tether_wrench(
            self._robot.data.root_pos_w,
            self._robot.data.root_quat_w,
            self._robot.data.root_lin_vel_w,
            water_current_w,
            self.cfg.tether_anchor_pos_w,
            self.cfg.tether_attach_offset_b,
            self.cfg.tether_slack_length,
            self.cfg.tether_stiffness,
            self.cfg.tether_damping,
            self.cfg.tether_drag_coeff,
            self.cfg.tether_num_segments,
            self.cfg.tether_segment_diameter,
            self.cfg.tether_segment_density,
            self.cfg.tether_segment_buoyancy_density,
            self._gravity_w,
            quat_conjugate,
            quat_apply,
        )

    def _calculate_thruster_axes_b(self) -> torch.Tensor:
        unit_x = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32, device=self.device).repeat(
            self.num_envs * self.num_thrusters,
            1,
        )
        return quat_apply(self.thruster_quats.reshape(-1, 4), unit_x).reshape(
            self.num_envs,
            self.num_thrusters,
            3,
        )

    def _calculate_thruster_axial_inflow(
        self,
        water_current_w: torch.Tensor,
        thruster_axes_b: torch.Tensor,
    ) -> torch.Tensor:
        nu_r = self.force_calculation_functions.calculate_relative_velocity(
            self._robot.data.root_quat_w,
            self._robot.data.root_lin_vel_b,
            self._robot.data.root_ang_vel_b,
            water_current_w,
        )
        relative_linvel_b = nu_r[:, 0:3]
        return torch.sum(relative_linvel_b.unsqueeze(1) * thruster_axes_b, dim=-1)

    def _calculate_thruster_inflow_scale(
        self,
        thruster_magnitudes: torch.Tensor,
        water_current_w: torch.Tensor,
        thruster_axes_b: torch.Tensor,
    ) -> torch.Tensor:
        if not self.cfg.thruster_inflow_loss_enabled:
            return torch.ones((self.num_envs, self.num_thrusters), dtype=torch.float32, device=self.device)

        axial_inflow_along_axis = self._calculate_thruster_axial_inflow(water_current_w, thruster_axes_b)
        thrust_direction = torch.sign(thruster_magnitudes).unsqueeze(-1)
        axial_inflow_speed = axial_inflow_along_axis * thrust_direction.squeeze(-1)
        return calculate_axial_inflow_thrust_scale(
            axial_inflow_speed,
            self.cfg.thruster_inflow_loss_coefficient,
            self.cfg.thruster_inflow_reference_speed,
            self.cfg.thruster_inflow_min_scale,
        )

    def _calculate_thruster_wake_scale(
        self,
        thruster_magnitudes: torch.Tensor,
        thruster_axes_b: torch.Tensor,
    ) -> torch.Tensor:
        if not self.cfg.thruster_wake_interaction_enabled:
            return torch.ones((self.num_envs, self.num_thrusters), dtype=torch.float32, device=self.device)

        if self.cfg.use_thruster_lookup_table and len(self.cfg.thruster_lookup_thrusts) > 0:
            reference = float(torch.as_tensor(self.cfg.thruster_lookup_thrusts, dtype=torch.float32).abs().max())
        else:
            max_forward = torch.as_tensor(self.cfg.t200_max_forward_thrust, dtype=torch.float32)
            max_reverse = torch.as_tensor(self.cfg.t200_max_reverse_thrust, dtype=torch.float32)
            reference = float(torch.max(torch.cat((max_forward.reshape(-1), max_reverse.reshape(-1)))))

        return calculate_thruster_wake_interaction_scale(
            self.thruster_com_offsets,
            thruster_axes_b,
            thruster_magnitudes,
            self.cfg.thruster_wake_length,
            self.cfg.thruster_wake_radius,
            self.cfg.thruster_wake_loss_coefficient,
            self.cfg.thruster_wake_expansion_rate,
            self.cfg.thruster_wake_min_scale,
            reference,
        )

    def _sample_from_circle(self, num_env_ids, r):
        sampled_radius = r * torch.sqrt(torch.rand((num_env_ids), device=self.device))
        sampled_theta = torch.rand((num_env_ids), device=self.device) * 2 * 3.14159
        sampled_x = sampled_radius * torch.cos(sampled_theta)
        sampled_y = sampled_radius * torch.sin(sampled_theta)
        return (sampled_x, sampled_y)

    def _sample_from_sphere(self, num_env_ids, r):
        coords = torch.randn((num_env_ids, 3), device=self.device)
        norms = torch.norm(coords, dim=1).unsqueeze(1)
        coords /= norms

        radii = r * torch.pow(torch.rand((num_env_ids, 1), device=self.device), 1/3)

        return radii * coords

    def _compute_dynamics(self, actions) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute dynamics from normalized T200 throttle commands.

        Actions are -1 for full reverse thrust and 1 for full forward thrust.
        Args:
            actions (torch.Tensor): Actions shape (num_envs, num_actions)

        Returns:
            [torch.Tensor]: Forces sent to the simulation
            [torch.Tensor]: Torques sent to the simulation
        """

        if self._debug: print("actions: ", actions)

        thruster_forces = torch.zeros((self.num_envs, self.num_thrusters, 3), device=self.device, dtype=torch.float)
        thruster_torques = torch.zeros((self.num_envs, self.num_thrusters, 3), device=self.device, dtype=torch.float)
        thruster_commands = torch.clone(actions)

        if self._debug: print("thruster commands: ", thruster_commands)

        thruster_commands = torch.where(
            torch.abs(thruster_commands) < self.thruster_deadband,
            torch.zeros_like(thruster_commands),
            thruster_commands,
        )
        thruster_commands = self.thruster_command_processor.update(
            thruster_commands,
            self.thruster_delay_steps,
            self.thruster_max_command_rate,
            self.physics_dt,
            self.thruster_command_resolution,
            self.thruster_command_dropout_probability,
        )

        # Update first-order thruster dynamics at the actual physics clock.
        # DirectRLEnv may run several physics steps per policy step (decimation),
        # so episode_length_buf * dt would under-count time and freeze dynamics
        # inside the decimation loop.
        physics_time = torch.full(
            (self.num_envs,),
            self._sim_step_counter * self.physics_dt,
            dtype=torch.float32,
            device=self.device,
        )
        thruster_commands = self.thruster_dynamics.update(thruster_commands, physics_time)

        voltage_scale = self._update_battery_voltage_scale()
        water_current_w = self._calculate_water_current_w()
        pool_damping_scale, pool_added_mass_scale, pool_thruster_scale = self._calculate_pool_boundary_scales()
        (
            surface_damping_scale,
            surface_added_mass_scale,
            surface_buoyancy_scale,
            surface_thruster_scale,
        ) = self._calculate_free_surface_scales()
        thruster_axes_b = self._calculate_thruster_axes_b()
        if self.cfg.use_thruster_inflow_lookup_table:
            axial_inflow_speed = self._calculate_thruster_axial_inflow(water_current_w, thruster_axes_b)
            converted_thrust = self.thruster_conversion.convert(thruster_commands, axial_inflow_speed)
        else:
            converted_thrust = self.thruster_conversion.convert(thruster_commands)
        thruster_magnitudes = (
            converted_thrust
            * self.thruster_force_scale
            * voltage_scale
            * pool_thruster_scale
            * surface_thruster_scale
        )
        if not self.cfg.use_thruster_inflow_lookup_table:
            inflow_scale = self._calculate_thruster_inflow_scale(
                thruster_magnitudes,
                water_current_w,
                thruster_axes_b,
            )
            thruster_magnitudes = thruster_magnitudes * inflow_scale

        # TODO: this could be taken out of the physics step
        thruster_forces = thruster_axes_b
        wake_scale = self._calculate_thruster_wake_scale(thruster_magnitudes, thruster_axes_b)
        thruster_magnitudes = thruster_magnitudes * wake_scale

        # apply the force magnitudes to the thruster forces
        thruster_forces = thruster_forces * thruster_magnitudes.unsqueeze(-1)

        # calculate the thruster torques
        # T = r x F
        # T (num_envs, num_thrusters_per_env, 3)
        # r (num_thrusters_per_env, 3)
        # F (num_envs, num_thrusters_per_env, 3)
        # it should broadcast r to be (num_envs, num_thrusters_per_env, 3)
        thruster_torques = torch.cross(self.thruster_com_offsets, thruster_forces, dim=-1)
        thruster_torques = thruster_torques + calculate_reaction_torques(
            thruster_magnitudes,
            thruster_axes_b,
            self.cfg.thruster_reaction_torque_coeff,
            self.cfg.thruster_spin_directions,
        )

        # now sum together all the forces/torques on each robot
        thruster_forces = torch.sum(thruster_forces, dim=-2) # sum over the thruster indices
        thruster_torques = torch.sum(thruster_torques, dim=-2) # sum over the thruster indices

        added_mass_inertia_scale = float(getattr(self.cfg, "added_mass_inertia_scale", 1.0))
        relative_acceleration_b = self._update_relative_acceleration_b(water_current_w)
        if added_mass_inertia_scale <= 0.0:
            relative_acceleration_b = None
        else:
            relative_acceleration_b = relative_acceleration_b * added_mass_inertia_scale

        ## Calculate hydrodynamics
        if self._debug: print("gravity magnitude: ", self._gravity_magnitude)
        damping_scale = pool_damping_scale * surface_damping_scale
        added_mass_scale = pool_added_mass_scale * surface_added_mass_scale
        linear_damping_speed_scale, quadratic_damping_speed_scale = self._calculate_speed_dependent_damping_scales(
            water_current_w
        )
        linear_damping = _scale_hydro_coefficients(self.linear_damping, damping_scale * linear_damping_speed_scale)
        quadratic_damping = _scale_hydro_coefficients(
            self.quadratic_damping,
            damping_scale * quadratic_damping_speed_scale,
        )
        added_mass_diag = _scale_hydro_coefficients(self.added_mass_diag, added_mass_scale)
        volumes = self.volumes * surface_buoyancy_scale
        fluid_forces, fluid_torques = self.force_calculation_functions.calculate_fossen_fluid_forces(
          self._robot.data.root_quat_w,
          self._robot.data.root_lin_vel_b,
          self._robot.data.root_ang_vel_b,
          self._gravity_w,
          self.cfg.water_rho,
          volumes,
          self.com_to_cob_offsets,
          linear_damping,
          quadratic_damping,
          water_current_w,
          added_mass_diag,
          relative_acceleration_b,
        )

        if self._debug: print("fluid forces: ", fluid_forces)
        if self._debug: print("fluid torques: ", fluid_torques)

        if self._debug: print("thruster forces: ", thruster_forces)
        if self._debug: print("thruster torques: ", thruster_torques)

        tether_forces, tether_torques = self._calculate_tether_wrench(water_current_w)

        forces = fluid_forces + thruster_forces + tether_forces
        torques = fluid_torques + thruster_torques + tether_torques

        if self._debug: print("final forces", forces)
        if self._debug: print("final torques", torques)

        return forces, torques

    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the first tome
        if debug_vis:
            if not hasattr(self, "goal_pos_visualizer"):
                marker_cfg = CUBOID_MARKER_CFG.copy()
                marker_cfg.markers["cuboid"].size = (0.05, 0.05, 0.05)
                # -- goal pose
                marker_cfg.prim_path = "/Visuals/Command/goal_position"
                self.goal_pos_visualizer = VisualizationMarkers(marker_cfg)

            if not hasattr(self, "goal_ang_visualizer"):
                marker_cfg = RED_ARROW_X_MARKER_CFG.copy()
                marker_cfg.prim_path = "/Visuals/Command/goal_ang"
                marker_cfg.markers["arrow"].scale = (0.125, 0.125, 1)
                self.goal_ang_visualizer = VisualizationMarkers(marker_cfg)

            if not hasattr(self, "goal_z_ang_visualizer"):
                marker_cfg = BLUE_ARROW_X_MARKER_CFG.copy()
                marker_cfg.prim_path = "/Visuals/Command/goal_z_ang"
                marker_cfg.markers["arrow"].scale = (0.125, 0.125, 1)
                self.goal_z_ang_visualizer = VisualizationMarkers(marker_cfg)

            if not hasattr(self, "x_b_visualizer"):
                marker_cfg = GREEN_ARROW_X_MARKER_CFG.copy()
                marker_cfg.markers["arrow"].scale = (0.125, 0.125, 1)
                marker_cfg.prim_path = "/Visuals/Command/x_b"
                self.x_b_visualizer = VisualizationMarkers(marker_cfg)

            if not hasattr(self, "z_b_visualizer"):
                marker_cfg = GREEN_ARROW_X_MARKER_CFG.copy()
                marker_cfg.markers["arrow"].scale = (0.125, 0.125, 1)
                marker_cfg.prim_path = "/Visuals/Command/z_b"
                self.z_b_visualizer = VisualizationMarkers(marker_cfg)
            
            # set their visibility to true
            self.goal_pos_visualizer.set_visibility(True)
            self.goal_ang_visualizer.set_visibility(True)
            self.goal_z_ang_visualizer.set_visibility(True)
            self.x_b_visualizer.set_visibility(True)
            self.z_b_visualizer.set_visibility(True)

        else:
            if hasattr(self, "goal_pos_visualizer"):
                self.goal_pos_visualizer.set_visibility(False)

            if hasattr(self, "goal_ang_visualizer"):
                self.goal_ang_visualizer.set_visibility(False)

            if hasattr(self, "goal_z_ang_visualizer"):
                self.goal_z_ang_visualizer.set_visibility(False)

            if hasattr(self, "x_b_visualizer"):
                self.x_b_visualizer.set_visibility(False)
            
            if hasattr(self, "z_b_visualizer"):
                self.z_b_visualizer.set_visibility(False)

    def _rotate_quat_by_euler_xyz(self, q: torch.tensor, x: float|torch.tensor, y: float|torch.tensor, z: float|torch.tensor, device=None):
        # Assumes q has shape [num_envs, 4]
        num_envs = q.shape[0]
        if device == None:
            device = self.device

        if type(x) == float:
            x = torch.zeros(num_envs, device=device) + x

        if type(y) == float:
            y = torch.zeros(num_envs, device=device) + y
        
        if type(z) == float:
            z = torch.zeros(num_envs, device=device) + z

        iq = math_utils.quat_from_euler_xyz(x, y, z)
        return math_utils.quat_mul(q, iq)


    def _debug_vis_callback(self, event):
        # Visualize the goal positions
        # self.goal_pos_visualizer.visualize(translations = self._default_env_origins)
        self.goal_pos_visualizer.visualize(translations = self._goal_pos_w)

        # Visualize goal orientations
        goal_quats_w = self._goal
        ang_marker_scales = torch.tensor([1, 1, 1]).repeat(self.num_envs, 1)
        ang_marker_scales[:, 0] = 1
        self.goal_ang_visualizer.visualize(translations=self._robot.data.root_pos_w, orientations=goal_quats_w, scales=ang_marker_scales)

        # Visualize goal orientations via another axis
        goal_z_quat = self._rotate_quat_by_euler_xyz(goal_quats_w, 0.0, -torch.pi/2, 0.0)
        ang_marker_scales = torch.tensor([1, 1, 1]).repeat(self.num_envs, 1)
        ang_marker_scales[:, 0] = 1
        self.goal_z_ang_visualizer.visualize(translations=self._robot.data.root_pos_w, orientations=goal_z_quat, scales=ang_marker_scales)

        # Visualize current X-direction
        x_w = self._robot.data.root_quat_w
        x_w_marker_scales = torch.tensor([1, 1, 1]).repeat(self.num_envs, 1)
        x_w_marker_scales[:, 0] = 1
        self.x_b_visualizer.visualize(translations=self._robot.data.root_pos_w, orientations=x_w, scales=x_w_marker_scales)

        # Visualize current Z-direction
        z_w_quat = self._rotate_quat_by_euler_xyz(self._robot.data.root_quat_w, 0.0, -torch.pi/2, 0.0)
        z_w_marker_scales = torch.tensor([1, 1, 1]).repeat(self.num_envs, 1)
        z_w_marker_scales[:, 0] = 1
        self.z_b_visualizer.visualize(translations=self._robot.data.root_pos_w, orientations=z_w_quat, scales=z_w_marker_scales)


class WarpAUVTrajEnv(WarpAUVEnv):
    """Separate Gym entry point for trajectory tracking.

    The implementation lives in WarpAUVEnv behind cfg.trajectory_tracking so
    the original pos-hold task and old checkpoints keep working unchanged.
    """

    cfg: WarpAUVTrajEnvCfg


@torch.jit.script
def quat_dist(q1, q2):
    return 1 - torch.sum(q1*q2, dim=-1)**2

@torch.jit.script
def _compute_rewards(
    rew_scale_pos: float,
    rew_scale_ang: float,
    rew_scale_lin_vel: float,
    rew_scale_ang_vel: float,
    rew_scale_actions: float,
    lin_vel: torch.Tensor,
    ang_vel: torch.Tensor,
    reset_terminated: torch.Tensor,
    root_pos: torch.Tensor,
    root_quat: torch.Tensor,
    goal: torch.Tensor,
    offsets_from_origin: torch.Tensor,
    completed_envs: torch.Tensor,
    actions: torch.Tensor,
):

    # Reward position accuracy, todo: scale the gaussian std appropriately
    rew_pos = rew_scale_pos * torch.exp(-1 * torch.norm(offsets_from_origin, dim=1)**2)

    # Reward angular accuracy, todo: scale the gaussian std appropriately
    # Uniquefy and normalize all quaternions
    rew_ang = rew_scale_ang * torch.exp(-1 * math_utils.quat_error_magnitude(goal[:,:], root_quat[:,:]))

    # Reward low linear and angular velocities for disturbance rejection.
    rew_lin_vel = rew_scale_lin_vel * torch.exp(-1 * torch.norm(lin_vel, dim=1)**2)
    rew_ang_vel = rew_scale_ang_vel * torch.exp(-1 * torch.norm(ang_vel, dim=1)**2)

    # Penalize action energy.
    rew_action = -rew_scale_actions * torch.norm(actions, dim=1)**2

    total_rew = rew_ang + rew_action + rew_pos + rew_lin_vel + rew_ang_vel
    return total_rew


@torch.jit.script
def _compute_tracking_rewards(
    rew_scale_pos: float,
    rew_scale_ang: float,
    rew_scale_track_vel: float,
    rew_scale_ang_vel: float,
    rew_scale_actions: float,
    rew_scale_action_rate: float,
    rew_pos_sigma: float,
    rew_ang_sigma: float,
    rew_track_vel_sigma: float,
    rew_ang_vel_sigma: float,
    root_pos: torch.Tensor,
    root_quat: torch.Tensor,
    root_lin_vel_b: torch.Tensor,
    root_ang_vel_b: torch.Tensor,
    target_pos_w: torch.Tensor,
    target_quat_w: torch.Tensor,
    target_lin_vel_b: torch.Tensor,
    actions: torch.Tensor,
    previous_actions: torch.Tensor,
):
    # Position and velocity are tracked against the moving command.  Unlike the
    # pos-hold reward, linear velocity is not pushed toward zero.  The Cauchy
    # kernel keeps useful reward slope at larger heavy2 tracking errors.
    pos_error = torch.norm(target_pos_w - root_pos, dim=1)
    ang_error = math_utils.quat_error_magnitude(target_quat_w[:, :], root_quat[:, :])
    track_vel_error = torch.norm(target_lin_vel_b - root_lin_vel_b, dim=1)
    ang_vel_error = torch.norm(root_ang_vel_b, dim=1)

    rew_pos = rew_scale_pos / (1.0 + (pos_error / rew_pos_sigma) ** 2)
    rew_ang = rew_scale_ang / (1.0 + (ang_error / rew_ang_sigma) ** 2)
    rew_track_vel = rew_scale_track_vel / (1.0 + (track_vel_error / rew_track_vel_sigma) ** 2)
    rew_ang_vel = rew_scale_ang_vel / (1.0 + (ang_vel_error / rew_ang_vel_sigma) ** 2)
    rew_action = -rew_scale_actions * torch.norm(actions, dim=1) ** 2
    rew_action_rate = -rew_scale_action_rate * torch.norm(actions - previous_actions, dim=1) ** 2

    total_rew = rew_pos + rew_ang + rew_track_vel + rew_ang_vel + rew_action + rew_action_rate
    return total_rew
