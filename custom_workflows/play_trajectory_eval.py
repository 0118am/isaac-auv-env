import argparse
import math
import os
from collections import deque

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip


TRAJECTORY_TYPE_IDS = {
    "lissajous": 1,
    "helix": 3,
    "spiral": 4,
    "chirp": 5,
    "racetrack": 6,
    "random_smooth": 7,
}

# This script evaluates the trajectory task without manually editing
# obs["policy"].  The desired command comes from WarpAUVTrajEnv itself, matching
# the training-time observation/reward path.
parser = argparse.ArgumentParser(description="Evaluate a WarpAUV trajectory-tracking policy.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument("--duration", type=float, default=32.0, help="Trajectory evaluation duration in seconds.")
parser.add_argument("--custom_weights", type=str, default=None, help="Path to a checkpoint outside the log directory.")
parser.add_argument(
    "--trajectory",
    type=str,
    default="lissajous",
    choices=TRAJECTORY_TYPE_IDS.keys(),
    help="Fixed eval trajectory to run. Only lissajous appears in the original eval; the others are OOD tests.",
)
parser.add_argument("--trajectory_amp_x", type=float, default=None, help="Override trajectory x amplitude.")
parser.add_argument("--trajectory_amp_y", type=float, default=None, help="Override trajectory y amplitude/radius.")
parser.add_argument("--trajectory_amp_z", type=float, default=None, help="Override trajectory z amplitude.")
parser.add_argument("--trajectory_period", type=float, default=None, help="Override trajectory period.")
parser.add_argument("--trajectory_radius_min", type=float, default=None, help="Override spiral minimum radius.")
parser.add_argument("--trajectory_radius_max", type=float, default=None, help="Override spiral maximum radius.")
parser.add_argument(
    "--random_curve_count",
    type=int,
    default=5,
    help="Number of random smooth curves to evaluate in parallel when --trajectory=random_smooth.",
)
parser.add_argument(
    "--align_initial_target",
    action="store_true",
    default=False,
    help="Start the vehicle on the first eval target instead of at the trajectory center.",
)
parser.add_argument(
    "--disable_trajectory_vis",
    action="store_true",
    default=False,
    help="Disable live desired/actual trajectory drawing in GUI eval.",
)
parser.add_argument(
    "--trail_stride",
    type=int,
    default=4,
    help="Draw live trajectory trails every N policy steps in GUI eval.",
)
parser.add_argument(
    "--trail_max_points",
    type=int,
    default=2500,
    help="Maximum desired/actual points kept in the live GUI trail.",
)
parser.add_argument(
    "--hold_open",
    action="store_true",
    default=False,
    help="Keep Isaac Sim open after eval so the live trails can be inspected.",
)
parser.add_argument(
    "--eval_current",
    type=float,
    nargs=3,
    default=None,
    metavar=("VX", "VY", "VZ"),
    help="Fixed world-frame water current in m/s for disturbance eval.",
)
parser.add_argument(
    "--eval_smooth_current",
    action="store_true",
    default=False,
    help="Let eval_current drift smoothly around its mean with a low-frequency current model.",
)
parser.add_argument(
    "--eval_current_variation_std",
    type=float,
    default=0.0,
    help="Std of smooth current variation in m/s. Used with --eval_smooth_current.",
)
parser.add_argument(
    "--eval_current_tau",
    type=float,
    default=12.0,
    help="Time constant in seconds for smooth current disturbance eval.",
)
parser.add_argument("--eval_damping_scale", type=float, default=1.0, help="Multiply linear/quadratic damping.")
parser.add_argument("--eval_thruster_scale", type=float, default=1.0, help="Multiply all thruster force outputs.")
parser.add_argument(
    "--eval_thruster_tau_scale",
    type=float,
    default=1.0,
    help="Multiply the first-order thruster response time constant.",
)
parser.add_argument("--eval_deadband_scale", type=float, default=1.0, help="Multiply thruster deadband.")
parser.add_argument(
    "--disturbance_name",
    type=str,
    default=None,
    help="Optional label used in the output directory name for disturbance eval.",
)

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import numpy as np
import pandas as pd
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab.utils.math import quat_apply, quat_error_magnitude
from rsl_rl.runners import OnPolicyRunner


def _resolve_checkpoint(log_root_path: str, agent_cfg: RslRlOnPolicyRunnerCfg) -> str:
    if args_cli.custom_weights:
        return args_cli.custom_weights
    return get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)


def _format_token(value: float) -> str:
    token = f"{value:.3g}".replace("-", "m").replace(".", "p")
    return token


def _sanitize_label(label: str) -> str:
    return "".join(char if char.isalnum() or char in ("_", "-") else "_" for char in label).strip("_")


def _disturbance_enabled() -> bool:
    current = args_cli.eval_current or [0.0, 0.0, 0.0]
    return (
        any(abs(value) > 1.0e-9 for value in current)
        or args_cli.eval_smooth_current
        or abs(args_cli.eval_current_variation_std) > 1.0e-9
        or abs(args_cli.eval_damping_scale - 1.0) > 1.0e-9
        or abs(args_cli.eval_thruster_scale - 1.0) > 1.0e-9
        or abs(args_cli.eval_thruster_tau_scale - 1.0) > 1.0e-9
        or abs(args_cli.eval_deadband_scale - 1.0) > 1.0e-9
    )


def _disturbance_label() -> str:
    if not _disturbance_enabled():
        return ""
    if args_cli.disturbance_name:
        return _sanitize_label(args_cli.disturbance_name)

    current = args_cli.eval_current or [0.0, 0.0, 0.0]
    parts = []
    if any(abs(value) > 1.0e-9 for value in current):
        parts.append("cur_" + "_".join(_format_token(value) for value in current))
    if args_cli.eval_smooth_current or args_cli.eval_current_variation_std > 0.0:
        parts.append(f"smooth{_format_token(args_cli.eval_current_variation_std)}")
    if abs(args_cli.eval_damping_scale - 1.0) > 1.0e-9:
        parts.append(f"damp{_format_token(args_cli.eval_damping_scale)}")
    if abs(args_cli.eval_thruster_scale - 1.0) > 1.0e-9:
        parts.append(f"thr{_format_token(args_cli.eval_thruster_scale)}")
    if abs(args_cli.eval_thruster_tau_scale - 1.0) > 1.0e-9:
        parts.append(f"tau{_format_token(args_cli.eval_thruster_tau_scale)}")
    if abs(args_cli.eval_deadband_scale - 1.0) > 1.0e-9:
        parts.append(f"dead{_format_token(args_cli.eval_deadband_scale)}")
    return _sanitize_label("_".join(parts) or "disturbance")


def _apply_eval_cfg_disturbance(env_cfg) -> None:
    current = args_cli.eval_current or [0.0, 0.0, 0.0]
    env_cfg.water_current_w = [float(value) for value in current]

    if abs(args_cli.eval_damping_scale - 1.0) > 1.0e-9:
        env_cfg.linear_damping = [float(value) * args_cli.eval_damping_scale for value in env_cfg.linear_damping]
        env_cfg.quadratic_damping = [float(value) * args_cli.eval_damping_scale for value in env_cfg.quadratic_damping]

    if abs(args_cli.eval_thruster_tau_scale - 1.0) > 1.0e-9:
        env_cfg.dyn_time_constant = float(env_cfg.dyn_time_constant) * args_cli.eval_thruster_tau_scale

    if abs(args_cli.eval_deadband_scale - 1.0) > 1.0e-9:
        env_cfg.thruster_deadband = float(env_cfg.thruster_deadband) * args_cli.eval_deadband_scale

    smooth_current = args_cli.eval_smooth_current or args_cli.eval_current_variation_std > 0.0
    env_cfg.domain_randomization.use_custom_randomization = smooth_current
    if smooth_current:
        env_cfg.domain_randomization.water_current_smooth = True
        env_cfg.domain_randomization.water_current_variation_std_by_stage = [args_cli.eval_current_variation_std] * 5
        env_cfg.domain_randomization.water_current_tau_range = [
            args_cli.eval_current_tau,
            args_cli.eval_current_tau,
        ]


def _apply_eval_runtime_disturbance(env) -> None:
    device = env.device
    current = torch.tensor(args_cli.eval_current or [0.0, 0.0, 0.0], dtype=torch.float32, device=device)
    env.water_current_w[:] = current.reshape(1, 3)

    if hasattr(env, "water_current_mean_w"):
        env.water_current_mean_w[:] = current.reshape(1, 3)
        horizontal_limit = torch.linalg.norm(current[0:2]).item() + 3.0 * max(args_cli.eval_current_variation_std, 0.0)
        vertical_limit = abs(current[2].item()) + 1.5 * max(args_cli.eval_current_variation_std, 0.0)
        env.water_current_horizontal_max[:] = horizontal_limit
        env.water_current_vertical_max[:] = vertical_limit
        env.water_current_tau[:] = args_cli.eval_current_tau

    if hasattr(env, "thruster_force_scale") and abs(args_cli.eval_thruster_scale - 1.0) > 1.0e-9:
        env.thruster_force_scale[:] = args_cli.eval_thruster_scale


class TrajectoryEvalVisualizer:
    """Small GUI-only helper for drawing desired and actual eval trails."""

    def __init__(
        self,
        enabled: bool,
        trajectory: str,
        checkpoint_name: str,
        max_points: int,
        stride: int,
    ):
        self.enabled = enabled
        self.trajectory = trajectory
        self.checkpoint_name = checkpoint_name
        self.stride = max(1, stride)
        self.desired_points = deque(maxlen=max(2, max_points))
        self.actual_points = deque(maxlen=max(2, max_points))
        self._draw = None
        self._labels = {}

        if not self.enabled:
            return

        self._init_debug_draw()
        self._init_status_window()

    def _init_debug_draw(self):
        try:
            try:
                from isaacsim.util.debug_draw import _debug_draw
            except Exception:
                from omni.isaac.debug_draw import _debug_draw

            self._draw = _debug_draw.acquire_debug_draw_interface()
            self._clear_draw()
        except Exception as exc:
            self.enabled = False
            print(f"[WARN]: Live trajectory drawing is unavailable: {exc}")

    def _init_status_window(self):
        try:
            import omni.ui as ui

            self._window = ui.Window("Trajectory Eval", width=360, height=150)
            with self._window.frame:
                with ui.VStack(spacing=4):
                    ui.Label(f"trajectory: {self.trajectory}")
                    ui.Label(f"checkpoint: {self.checkpoint_name}")
                    ui.Label("blue: desired target trail")
                    ui.Label("orange: actual AUV trail")
                    self._labels["time"] = ui.Label("time: 0.00 s")
                    self._labels["error"] = ui.Label("pos err: -- m | vel err: -- m/s")
        except Exception as exc:
            print(f"[WARN]: Trajectory status window is unavailable: {exc}")

    @staticmethod
    def _point(tensor: torch.Tensor) -> tuple[float, float, float]:
        values = tensor.detach().cpu().tolist()
        return float(values[0]), float(values[1]), float(values[2])

    def update(
        self,
        step: int,
        time_s: float,
        desired_pos_w: torch.Tensor,
        actual_pos_w: torch.Tensor,
        position_error: float,
        velocity_error: float,
    ):
        if not self.enabled:
            return

        self.desired_points.append(self._point(desired_pos_w))
        self.actual_points.append(self._point(actual_pos_w))

        if step % self.stride == 0:
            self._draw_trails()
            self._update_labels(time_s, position_error, velocity_error)

    def _draw_trails(self):
        if self._draw is None:
            return

        desired = list(self.desired_points)
        actual = list(self.actual_points)
        start_points = desired[:-1] + actual[:-1]
        end_points = desired[1:] + actual[1:]
        colors = [(0.1, 0.45, 1.0, 1.0)] * max(0, len(desired) - 1)
        colors += [(1.0, 0.45, 0.05, 1.0)] * max(0, len(actual) - 1)
        widths = [3.0] * len(start_points)

        self._clear_draw()
        if start_points:
            self._draw.draw_lines(start_points, end_points, colors, widths)
        if desired and actual:
            self._draw.draw_points(
                [desired[-1], actual[-1]],
                [(1.0, 0.95, 0.05, 1.0), (1.0, 0.95, 0.95, 1.0)],
                [18.0, 12.0],
            )

    def _clear_draw(self):
        if self._draw is None:
            return
        if hasattr(self._draw, "clear_lines"):
            self._draw.clear_lines()
        if hasattr(self._draw, "clear_points"):
            self._draw.clear_points()

    def _update_labels(self, time_s: float, position_error: float, velocity_error: float):
        time_label = self._labels.get("time")
        error_label = self._labels.get("error")
        if time_label is not None:
            time_label.text = f"time: {time_s:.2f} s"
        if error_label is not None:
            error_label.text = f"pos err: {position_error:.3f} m | vel err: {velocity_error:.3f} m/s"

    def set_status(self, message: str):
        label = self._labels.get("time")
        if label is not None:
            label.text = message


def main():
    eval_num_envs = args_cli.num_envs or 1
    if args_cli.trajectory == "random_smooth":
        eval_num_envs = args_cli.num_envs or args_cli.random_curve_count
    eval_num_envs = max(1, eval_num_envs)

    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=eval_num_envs, use_fabric=not args_cli.disable_fabric
    )
    # Use the env target generator for fixed trajectories or random smooth
    # generalization tests.  This avoids manually editing obs["policy"].
    env_cfg.eval_mode = True
    env_cfg.trajectory_eval_mode = True
    env_cfg.trajectory_eval_type = TRAJECTORY_TYPE_IDS[args_cli.trajectory]
    env_cfg.trajectory_eval_duration_s = args_cli.duration
    if args_cli.trajectory_amp_x is not None:
        env_cfg.trajectory_eval_amp_x = args_cli.trajectory_amp_x
    if args_cli.trajectory_amp_y is not None:
        env_cfg.trajectory_eval_amp_y = args_cli.trajectory_amp_y
    if args_cli.trajectory_amp_z is not None:
        env_cfg.trajectory_eval_amp_z = args_cli.trajectory_amp_z
    if args_cli.trajectory_period is not None:
        env_cfg.trajectory_eval_period = args_cli.trajectory_period
    if args_cli.trajectory_radius_min is not None:
        env_cfg.trajectory_eval_radius_min = args_cli.trajectory_radius_min
    if args_cli.trajectory_radius_max is not None:
        env_cfg.trajectory_eval_radius_max = args_cli.trajectory_radius_max
    if args_cli.trajectory == "random_smooth":
        if args_cli.trajectory_amp_x is not None:
            env_cfg.trajectory_amp_x_range = [args_cli.trajectory_amp_x, args_cli.trajectory_amp_x]
        if args_cli.trajectory_amp_y is not None:
            env_cfg.trajectory_amp_y_range = [args_cli.trajectory_amp_y, args_cli.trajectory_amp_y]
        if args_cli.trajectory_amp_z is not None:
            env_cfg.trajectory_amp_z_range = [args_cli.trajectory_amp_z, args_cli.trajectory_amp_z]
        if args_cli.trajectory_period is not None:
            env_cfg.trajectory_period_range = [args_cli.trajectory_period, args_cli.trajectory_period]
    env_cfg.trajectory_eval_align_initial_target = args_cli.align_initial_target
    env_cfg.cap_episode_length = False
    env_cfg.use_boundaries = False
    env_cfg.domain_randomization.use_custom_randomization = False
    _apply_eval_cfg_disturbance(env_cfg)

    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)
    _apply_eval_runtime_disturbance(env.unwrapped)

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    resume_path = _resolve_checkpoint(log_root_path, agent_cfg)
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # Results mirror the existing play/eval directory layout but use a distinct
    # suffix so repeated trajectory evaluations do not overwrite pos-hold logs.
    run_name = agent_cfg.load_run or "custom_weights"
    checkpoint_name = os.path.basename(resume_path)
    checkpoint_stem = checkpoint_name[:-3]
    disturbance_label = _disturbance_label()
    result_parts = [checkpoint_stem]
    if args_cli.trajectory != "lissajous":
        result_parts.append(args_cli.trajectory)
    if disturbance_label:
        result_parts.append(disturbance_label)
    result_dir_name = "_".join(result_parts) + "_trajectory_eval"

    save_path = os.path.join(
        "source",
        "results",
        "rsl_rl",
        agent_cfg.experiment_name,
        run_name,
        result_dir_name,
    )
    os.makedirs(save_path, exist_ok=True)
    logs_csv_path = os.path.join(save_path, "logs.csv")
    summary_csv_path = os.path.join(save_path, "summary_metrics.csv")
    print(f"[INFO]: Saving trajectory eval results into: {save_path}")

    obs = env.get_observations()
    step_dt = env.unwrapped.cfg.sim.dt * env.unwrapped.cfg.decimation
    num_steps = int(math.ceil(args_cli.duration / step_dt))
    log_rows = []
    visualizer = TrajectoryEvalVisualizer(
        enabled=not args_cli.disable_trajectory_vis and not getattr(args_cli, "headless", False),
        trajectory=args_cli.trajectory,
        checkpoint_name=checkpoint_name,
        max_points=args_cli.trail_max_points,
        stride=args_cli.trail_stride,
    )

    for step in range(num_steps):
        t = step * step_dt
        with torch.inference_mode():
            # Pull target state from the env after it synchronizes to the
            # current episode time.  This keeps logs aligned with policy input.
            target_pos_w, target_lin_vel_w, target_quat_w = env.unwrapped.get_tracking_targets()
            root_pos_w = env.unwrapped._robot.data.root_pos_w
            root_quat_w = env.unwrapped._robot.data.root_quat_w
            root_lin_vel_w = quat_apply(root_quat_w, env.unwrapped._robot.data.root_lin_vel_b)

            position_errors = torch.norm(target_pos_w - root_pos_w, dim=1)
            velocity_errors = torch.norm(target_lin_vel_w - root_lin_vel_w, dim=1)
            attitude_errors = quat_error_magnitude(target_quat_w, root_quat_w)
            visualizer.update(
                step,
                t,
                target_pos_w[0],
                root_pos_w[0],
                position_errors[0].cpu().item(),
                velocity_errors[0].cpu().item(),
            )

            actions = policy(obs)
            action_norms = torch.norm(actions, dim=1)
            water_current_w = env.unwrapped.water_current_w

            for env_id in range(target_pos_w.shape[0]):
                log_rows.append(
                    {
                        "trajectory": args_cli.trajectory,
                        "disturbance": disturbance_label or "nominal",
                        "env_id": env_id,
                        "time": t,
                        "water_current_x": water_current_w[env_id, 0].cpu().item(),
                        "water_current_y": water_current_w[env_id, 1].cpu().item(),
                        "water_current_z": water_current_w[env_id, 2].cpu().item(),
                        "desired_x": target_pos_w[env_id, 0].cpu().item(),
                        "desired_y": target_pos_w[env_id, 1].cpu().item(),
                        "desired_z": target_pos_w[env_id, 2].cpu().item(),
                        "true_x": root_pos_w[env_id, 0].cpu().item(),
                        "true_y": root_pos_w[env_id, 1].cpu().item(),
                        "true_z": root_pos_w[env_id, 2].cpu().item(),
                        "desired_vx": target_lin_vel_w[env_id, 0].cpu().item(),
                        "desired_vy": target_lin_vel_w[env_id, 1].cpu().item(),
                        "desired_vz": target_lin_vel_w[env_id, 2].cpu().item(),
                        "true_vx": root_lin_vel_w[env_id, 0].cpu().item(),
                        "true_vy": root_lin_vel_w[env_id, 1].cpu().item(),
                        "true_vz": root_lin_vel_w[env_id, 2].cpu().item(),
                        "position_error": position_errors[env_id].cpu().item(),
                        "velocity_error": velocity_errors[env_id].cpu().item(),
                        "attitude_error": attitude_errors[env_id].cpu().item(),
                        "action_norm": action_norms[env_id].cpu().item(),
                    }
                )

            obs, _, _, _ = env.step(actions)

    log_df = pd.DataFrame(log_rows)
    log_df.to_csv(logs_csv_path, index=False)

    # Keep scalar summary metrics in a separate CSV for quick experiment
    # comparisons without loading the full trajectory log.
    position_errors = log_df["position_error"].to_numpy()
    velocity_errors = log_df["velocity_error"].to_numpy()
    summary = {
        "trajectory": args_cli.trajectory,
        "disturbance": disturbance_label or "nominal",
        "num_curves": int(log_df["env_id"].nunique()),
        "position_rmse": float(np.sqrt(np.mean(position_errors**2))),
        "position_mae": float(np.mean(position_errors)),
        "max_position_error": float(np.max(position_errors)),
        "velocity_rmse": float(np.sqrt(np.mean(velocity_errors**2))),
        "mean_water_current_norm": float(
            np.mean(
                np.linalg.norm(
                    log_df[["water_current_x", "water_current_y", "water_current_z"]].to_numpy(),
                    axis=1,
                )
            )
        ),
        "max_water_current_norm": float(
            np.max(
                np.linalg.norm(
                    log_df[["water_current_x", "water_current_y", "water_current_z"]].to_numpy(),
                    axis=1,
                )
            )
        ),
        "eval_damping_scale": float(args_cli.eval_damping_scale),
        "eval_thruster_scale": float(args_cli.eval_thruster_scale),
        "eval_thruster_tau_scale": float(args_cli.eval_thruster_tau_scale),
        "eval_deadband_scale": float(args_cli.eval_deadband_scale),
    }
    pd.DataFrame([summary]).to_csv(summary_csv_path, index=False)
    print(f"[INFO]: Summary metrics: {summary}")

    if args_cli.hold_open and not getattr(args_cli, "headless", False):
        visualizer.set_status("eval complete; close Isaac Sim to exit")
        print("[INFO]: Evaluation complete. Close Isaac Sim to exit.")
        while simulation_app.is_running():
            simulation_app.update()

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
