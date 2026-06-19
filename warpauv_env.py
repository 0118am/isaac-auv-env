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
from .rigid_body_hydrodynamics import HydrodynamicForceModels
from .thruster_dynamics import (
    T200_REVERSE_TO_FORWARD_RATIO,
    DynamicsFirstOrder,
    ConversionFunctionT200,
    get_thruster_com_and_orientations,
)

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

    # dynamics
    center_of_mass_offset = list(BLUEROV2_HEAVY.center_of_mass_offset_m)
    inertia_diag = list(BLUEROV2_HEAVY.inertia_diag_kg_m2)
    com_to_cob_offset = list(BLUEROV2_HEAVY.center_of_buoyancy_from_com_m)
    water_rho = BLUEROV2_HEAVY.water_density_kg_m3 # kg/m^3
    water_beta = 0.001306 # Pa s, dynamic viscosity of water @ 50 deg F
    dyn_time_constant = 0.05 # time constant for linear dynamics for each rotor
    thruster_deadband = 0.08
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

    # Fossen-style hydrodynamic parameters.  Damping is applied to relative
    # velocity nu_r = nu - nu_current, not absolute vehicle velocity.
    water_current_w = [0.0, 0.0, 0.0]
    linear_damping = [0.00526, 0.00526, 0.00526, 0.00032, 0.00032, 0.00032]
    quadratic_damping = [39.196, 68.272, 135.402, 0.277, 1.387, 0.770]
    added_mass_diag = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    # domain randomization
    # todo: isaaclabs has a built-in method somehow
    class domain_randomization:
        use_custom_randomization = False
        # com_to_cob_offset_radius = 0 # uniform from sphere around predicted com_to_cob_offset
        com_to_cob_offset_radius = 0.0 # uniform from sphere around predicted com_to_cob_offset
        volume_range = [BLUEROV2_HEAVY.neutral_buoyancy_volume_m3, BLUEROV2_HEAVY.neutral_buoyancy_volume_m3]
        mass_range = [BLUEROV2_HEAVY.mass_kg, BLUEROV2_HEAVY.mass_kg]


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


class WarpAUVEnv(DirectRLEnv):
    cfg: WarpAUVEnvCfg

    def __init__(self, cfg: WarpAUVEnvCfg, render_mode: str | None = None, **kwargs):
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

        self.inertia_tensors = torch.tensor(
            self.cfg.inertia_diag,
            device=self.device,
            dtype=torch.float32,
            requires_grad=False,
        ).reshape(1, 3).repeat(self.num_envs, 1)
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
        self.thruster_conversion = ConversionFunctionT200(
            self.cfg.t200_max_forward_thrust,
            self.cfg.t200_max_reverse_thrust,
        )
        self.linear_damping = torch.tensor(self.cfg.linear_damping, dtype=torch.float32, device=self.device)
        self.quadratic_damping = torch.tensor(self.cfg.quadratic_damping, dtype=torch.float32, device=self.device)
        self.added_mass_diag = torch.tensor(self.cfg.added_mass_diag, dtype=torch.float32, device=self.device)
        self.water_current_w = torch.tensor(self.cfg.water_current_w, dtype=torch.float32, device=self.device)

    def _apply_nominal_rigid_body_properties(self) -> None:
        """Apply the Heavy mass, inertia, and COM to the live PhysX body."""

        all_env_ids = self._robot._ALL_INDICES
        self._apply_runtime_mass_properties(all_env_ids)
        self._apply_runtime_center_of_mass(all_env_ids)
        self._robot.data.default_mass = self._robot.root_physx_view.get_masses().clone()
        self._robot.data.default_inertia = self._robot.root_physx_view.get_inertias().clone()

    def _apply_runtime_mass_properties(self, env_ids: Sequence[int] | torch.Tensor) -> None:
        """Write per-env mass and matching diagonal inertia into PhysX."""

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
        nominal_diag = torch.tensor(self.cfg.inertia_diag, device=physx_inertias.device, dtype=physx_inertias.dtype)
        mass_ratio = selected_masses.reshape(-1, 1) / float(self.cfg.mass)
        inertia_diag = nominal_diag.reshape(1, 3) * mass_ratio
        flat_inertia = torch.zeros((len(env_ids_cpu), 9), device=physx_inertias.device, dtype=physx_inertias.dtype)
        flat_inertia[:, 0] = inertia_diag[:, 0]
        flat_inertia[:, 4] = inertia_diag[:, 1]
        flat_inertia[:, 8] = inertia_diag[:, 2]
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
            observations = {"policy": obs}
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
        observations = {"policy": obs}
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

        thruster_commands[torch.abs(thruster_commands) < self.cfg.thruster_deadband] = 0.0

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

        thruster_magnitudes = self.thruster_conversion.convert(thruster_commands)

        # TODO: this could be taken out of the physics step
        thruster_forces[..., 0] = 1.0 # start with forces in the x direction
        thruster_forces = quat_apply(self.thruster_quats, thruster_forces) # rotate forces into body-frame axes

        # apply the force magnitudes to the thruster forces
        thruster_forces = thruster_forces * thruster_magnitudes.unsqueeze(-1)

        # calculate the thruster torques
        # T = r x F
        # T (num_envs, num_thrusters_per_env, 3)
        # r (num_thrusters_per_env, 3)
        # F (num_envs, num_thrusters_per_env, 3)
        # it should broadcast r to be (num_envs, num_thrusters_per_env, 3)
        thruster_torques = torch.cross(self.thruster_com_offsets, thruster_forces, dim=-1)

        # now sum together all the forces/torques on each robot
        thruster_forces = torch.sum(thruster_forces, dim=-2) # sum over the thruster indices
        thruster_torques = torch.sum(thruster_torques, dim=-2) # sum over the thruster indices

        ## Calculate hydrodynamics
        if self._debug: print("gravity magnitude: ", self._gravity_magnitude)
        fluid_forces, fluid_torques = self.force_calculation_functions.calculate_fossen_fluid_forces(
          self._robot.data.root_quat_w,
          self._robot.data.root_lin_vel_b,
          self._robot.data.root_ang_vel_b,
          self._gravity_w,
          self.cfg.water_rho,
          self.volumes,
          self.com_to_cob_offsets,
          self.linear_damping,
          self.quadratic_damping,
          self.water_current_w,
          self.added_mass_diag,
        )

        if self._debug: print("fluid forces: ", fluid_forces)
        if self._debug: print("fluid torques: ", fluid_torques)

        if self._debug: print("thruster forces: ", thruster_forces)
        if self._debug: print("thruster torques: ", thruster_torques)

        forces = fluid_forces + thruster_forces
        torques = fluid_torques + thruster_torques

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
