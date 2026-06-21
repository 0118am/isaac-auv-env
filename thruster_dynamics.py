"""Thruster dynamics and BlueROV2 Heavy thruster geometry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch

try:
    from .bluerov2_heavy_model import BLUEROV2_HEAVY, KGF_TO_NEWTON
except ImportError:
    from bluerov2_heavy_model import BLUEROV2_HEAVY, KGF_TO_NEWTON


# Blue Robotics T200 published full-throttle FWD/REV thrust at nominal 16 V.
T200_NOMINAL_FORWARD_THRUST_N = BLUEROV2_HEAVY.t200_nominal_forward_thrust_kgf * KGF_TO_NEWTON
T200_NOMINAL_REVERSE_THRUST_N = BLUEROV2_HEAVY.t200_nominal_reverse_thrust_kgf * KGF_TO_NEWTON
T200_REVERSE_TO_FORWARD_RATIO = BLUEROV2_HEAVY.t200_reverse_to_forward_ratio

# ArduSub vectored6dof order used by the BlueROV2 Heavy frame diagram.
BLUEROV2_HEAVY_THRUSTER_NAMES = (
    "front_right_horizontal",
    "front_left_horizontal",
    "rear_right_horizontal",
    "rear_left_horizontal",
    "front_right_vertical",
    "front_left_vertical",
    "rear_right_vertical",
    "rear_left_vertical",
)


def _quat_from_x_axis(direction: tuple[float, float, float]) -> torch.Tensor:
    """Return a wxyz quaternion that rotates the local +X axis to ``direction``."""

    dst = torch.tensor(direction, dtype=torch.float32)
    dst = dst / torch.linalg.norm(dst)
    dot = dst[0].clamp(-1.0, 1.0)

    if dot < -0.999999:
        return torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float32)

    quat = torch.tensor([1.0 + dot, 0.0, -dst[2], dst[1]], dtype=torch.float32)
    return quat / torch.linalg.norm(quat)


def get_thruster_com_and_orientations(device):
    """Return BlueROV2 Heavy T200 thruster offsets and orientations.

    Body axes are x-forward, y-left, z-up.  The four horizontal thruster
    positions and axes come from the BlueROV2 R4 public CAD assembly.  The four
    vertical thrusters follow the BlueROV2 Heavy 2D drawing and ArduSub
    ``vectored6dof`` layout, which moves the vertical thrusters to the outside
    corners of the frame.
    """

    def create_tf_direction(x, y, z, direction):
        return torch.tensor([x, y, z], dtype=torch.float32), _quat_from_x_axis(direction)

    thruster_info = {
        # CAD-derived R4 horizontal thrusters, converted from mm:
        # CAD z -> body x, CAD x -> body y, CAD y -> body z.
        "front_right_horizontal": create_tf_direction(
            0.1510971694, -0.0950141245, -0.07235, (0.7431448255, 0.6691306064, 0.0)
        ),
        "front_left_horizontal": create_tf_direction(
            0.1510971694, 0.0950141245, -0.07235, (0.7431448255, -0.6691306064, 0.0)
        ),
        "rear_right_horizontal": create_tf_direction(
            -0.1421358755, -0.0936628306, -0.07235, (0.6691306064, -0.7431448255, 0.0)
        ),
        "rear_left_horizontal": create_tf_direction(
            -0.1421358755, 0.0936628306, -0.07235, (0.6691306064, 0.7431448255, 0.0)
        ),
        # Heavy drawing dimensions: 457.1 mm length, 436.1 mm top-view width.
        # T200 guard center is approximated at one T200 radius inboard.
        "front_right_vertical": create_tf_direction(
            BLUEROV2_HEAVY.length_x_m / 2.0 - 0.05,
            -(BLUEROV2_HEAVY.top_view_width_y_m / 2.0 - 0.05),
            0.0,
            (0.0, 0.0, 1.0),
        ),
        "front_left_vertical": create_tf_direction(
            BLUEROV2_HEAVY.length_x_m / 2.0 - 0.05,
            BLUEROV2_HEAVY.top_view_width_y_m / 2.0 - 0.05,
            0.0,
            (0.0, 0.0, 1.0),
        ),
        "rear_right_vertical": create_tf_direction(
            -(BLUEROV2_HEAVY.length_x_m / 2.0 - 0.05),
            -(BLUEROV2_HEAVY.top_view_width_y_m / 2.0 - 0.05),
            0.0,
            (0.0, 0.0, 1.0),
        ),
        "rear_left_vertical": create_tf_direction(
            -(BLUEROV2_HEAVY.length_x_m / 2.0 - 0.05),
            BLUEROV2_HEAVY.top_view_width_y_m / 2.0 - 0.05,
            0.0,
            (0.0, 0.0, 1.0),
        ),
    }

    thruster_com_offsets = torch.stack([thruster_info[name][0] for name in BLUEROV2_HEAVY_THRUSTER_NAMES]).to(
        device=device,
        dtype=torch.float32,
    )
    thruster_quats = torch.stack([thruster_info[name][1] for name in BLUEROV2_HEAVY_THRUSTER_NAMES]).to(
        device=device,
        dtype=torch.float32,
    )

    return thruster_com_offsets, thruster_quats


class Dynamics(ABC):
    def __init__(self, numEnvs: int, num_thrusters_per_env: int, device: torch.device) -> None:
        self.numEnvs = numEnvs
        self.num_thrusters_per_env = num_thrusters_per_env
        self.device = device
        self.reset_all()

    def reset(self, maskArr: list | torch.Tensor):
        self.state[maskArr, :] = 0.0
        self.prevTime[maskArr] = -1.0

    def reset_all(self):
        self.state = torch.zeros(
            (self.numEnvs, self.num_thrusters_per_env),
            dtype=torch.float32,
            device=self.device,
            requires_grad=False,
        )
        self.prevTime = torch.ones((self.numEnvs), dtype=torch.float32, device=self.device, requires_grad=False) * -1.0

    @abstractmethod
    def update(self, cmd: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        pass


class DynamicsFirstOrder(Dynamics):
    def __init__(self, numEnvs: int, num_thrusters_per_env: int, tau: float, device: torch.device):
        super().__init__(numEnvs=numEnvs, num_thrusters_per_env=num_thrusters_per_env, device=device)
        self.tau = tau

    def update(self, cmd: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 0:
            t = t.repeat(self.numEnvs)

        self.prevTime[self.prevTime < 0] = t[self.prevTime < 0]
        dt = torch.clamp(t - self.prevTime, min=0.0)

        tau = torch.as_tensor(self.tau, dtype=cmd.dtype, device=cmd.device)
        if tau.ndim == 0:
            tau = tau.repeat(self.numEnvs)
        tau_safe = torch.clamp(tau, min=1.0e-6)
        alpha = torch.exp(-dt / tau_safe)
        alpha = torch.where(tau <= 0.0, torch.zeros_like(alpha), alpha)
        self.state = self.state * alpha.unsqueeze(-1) + (1.0 - alpha).unsqueeze(-1) * cmd

        self.prevTime = t
        return self.state


class ThrusterCommandProcessor:
    """Apply command delay, dropouts, rate limits, and quantization."""

    def __init__(
        self,
        numEnvs: int,
        num_thrusters_per_env: int,
        max_delay_steps: int,
        device: torch.device,
    ) -> None:
        self.numEnvs = numEnvs
        self.num_thrusters_per_env = num_thrusters_per_env
        self.device = device
        self.max_delay_steps = max(0, int(max_delay_steps))
        self.history_length = self.max_delay_steps + 1
        self.history_index = 0
        self.history = torch.zeros(
            (self.history_length, self.numEnvs, self.num_thrusters_per_env),
            dtype=torch.float32,
            device=self.device,
        )
        self.rate_limited_state = torch.zeros(
            (self.numEnvs, self.num_thrusters_per_env),
            dtype=torch.float32,
            device=self.device,
        )

    def reset(self, maskArr: list | torch.Tensor) -> None:
        self.history[:, maskArr, :] = 0.0
        self.rate_limited_state[maskArr, :] = 0.0

    def reset_all(self) -> None:
        self.history[:] = 0.0
        self.rate_limited_state[:] = 0.0
        self.history_index = 0

    def update(
        self,
        cmd: torch.Tensor,
        delay_steps: torch.Tensor | int,
        max_rate: torch.Tensor | float,
        dt: torch.Tensor | float,
        command_resolution: torch.Tensor | float = 0.0,
        dropout_probability: torch.Tensor | float = 0.0,
    ) -> torch.Tensor:
        self.history[self.history_index, :, :] = cmd

        delay_steps = torch.as_tensor(delay_steps, dtype=torch.long, device=cmd.device)
        if delay_steps.ndim == 0:
            delay_steps = delay_steps.repeat(self.numEnvs)
        delay_steps = torch.clamp(delay_steps.reshape(self.numEnvs), min=0, max=self.max_delay_steps)

        delayed_indices = (self.history_index - delay_steps) % self.history_length
        env_indices = torch.arange(self.numEnvs, dtype=torch.long, device=cmd.device)
        delayed_cmd = self.history[delayed_indices, env_indices, :]
        self.history_index = (self.history_index + 1) % self.history_length

        dropout_probability = torch.clamp(_expand_env_thruster_value(dropout_probability, cmd), min=0.0, max=1.0)
        if torch.any(dropout_probability > 0.0):
            dropout_mask = torch.rand_like(cmd) < dropout_probability
            delayed_cmd = torch.where(dropout_mask, self.rate_limited_state, delayed_cmd)

        rate = _expand_env_thruster_value(max_rate, cmd)
        dt_tensor = torch.as_tensor(dt, dtype=cmd.dtype, device=cmd.device)
        if dt_tensor.ndim == 0:
            dt_tensor = dt_tensor.reshape(1, 1)
        elif dt_tensor.ndim == 1:
            dt_tensor = dt_tensor.reshape(self.numEnvs, 1)
        max_delta = torch.clamp(rate, min=0.0) * dt_tensor

        delta = delayed_cmd - self.rate_limited_state
        limited_cmd = self.rate_limited_state + torch.clamp(delta, -max_delta, max_delta)
        processed_cmd = torch.where(rate <= 0.0, delayed_cmd, limited_cmd)

        resolution = torch.clamp(_expand_env_thruster_value(command_resolution, cmd), min=0.0)
        quantized_cmd = torch.round(processed_cmd / torch.clamp(resolution, min=1.0e-6)) * resolution
        self.rate_limited_state = torch.where(resolution > 0.0, quantized_cmd, processed_cmd)
        self.rate_limited_state = torch.clamp(self.rate_limited_state, min=-1.0, max=1.0)
        return self.rate_limited_state


def _expand_env_thruster_value(value: torch.Tensor | float, reference: torch.Tensor) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=reference.dtype, device=reference.device)
    if tensor.ndim == 0:
        return tensor.reshape(1, 1).repeat(reference.shape[0], reference.shape[1])
    if tensor.ndim == 1:
        if tensor.shape[0] == reference.shape[0]:
            return tensor.reshape(reference.shape[0], 1).repeat(1, reference.shape[1])
        if tensor.shape[0] == reference.shape[1]:
            return tensor.reshape(1, reference.shape[1]).repeat(reference.shape[0], 1)
    if tensor.ndim == 2:
        if tensor.shape == (reference.shape[0], 1):
            return tensor.repeat(1, reference.shape[1])
        if tensor.shape == (1, reference.shape[1]):
            return tensor.repeat(reference.shape[0], 1)
    if tensor.shape == reference.shape:
        return tensor
    raise ValueError(f"Cannot broadcast value with shape {tuple(tensor.shape)} to {tuple(reference.shape)}.")


def calculate_voltage_thrust_scale(
    voltage: torch.Tensor | float,
    nominal_voltage: float,
    exponent: float = 2.0,
    min_voltage: float = 0.0,
) -> torch.Tensor:
    """Return thrust scaling from battery voltage relative to nominal voltage."""

    if isinstance(voltage, torch.Tensor):
        voltage_tensor = voltage.to(dtype=torch.float32)
    else:
        voltage_tensor = torch.tensor(voltage, dtype=torch.float32)
    nominal = max(float(nominal_voltage), 1.0e-6)
    voltage_tensor = torch.clamp(voltage_tensor, min=float(min_voltage))
    return torch.pow(torch.clamp(voltage_tensor / nominal, min=0.0), float(exponent))


def calculate_axial_inflow_thrust_scale(
    axial_inflow_speed: torch.Tensor,
    loss_coefficient: float,
    reference_speed: float,
    min_scale: float,
) -> torch.Tensor:
    """Return thrust-loss scale from positive axial inflow speed.

    Positive axial inflow means water is moving into the propeller along its
    thrust axis.  The simple model reduces thrust with a quadratic factor and
    clamps to ``min_scale``; negative axial inflow never boosts thrust.
    """

    if loss_coefficient <= 0.0:
        return torch.ones_like(axial_inflow_speed)
    reference = max(float(reference_speed), 1.0e-6)
    inflow_ratio = torch.clamp(axial_inflow_speed, min=0.0) / reference
    scale = 1.0 - float(loss_coefficient) * inflow_ratio * inflow_ratio
    return torch.clamp(scale, min=float(min_scale), max=1.0)


def calculate_thruster_wake_interaction_scale(
    thruster_positions_b: torch.Tensor,
    thruster_axes_b: torch.Tensor,
    thrust: torch.Tensor,
    wake_length: float,
    wake_radius: float,
    loss_coefficient: float,
    expansion_rate: float = 0.0,
    min_scale: float = 0.7,
    reference_thrust: torch.Tensor | float | None = None,
) -> torch.Tensor:
    """Return thrust scales from simplified propeller wake interference.

    A source thruster sheds a wake in its signed thrust direction.  Any other
    thruster inside that expanding cylinder/cone receives a thrust-loss scale.
    This is a compact empirical model, not a blade-resolved propeller solver.
    """

    if loss_coefficient <= 0.0 or wake_length <= 0.0 or wake_radius <= 0.0:
        return torch.ones_like(thrust)

    if thruster_positions_b.ndim == 2:
        thruster_positions_b = thruster_positions_b.reshape(1, *thruster_positions_b.shape).repeat(
            thrust.shape[0],
            1,
            1,
        )
    if thruster_positions_b.shape != thruster_axes_b.shape:
        raise ValueError(
            "thruster_positions_b and thruster_axes_b must have matching "
            f"(num_envs, num_thrusters, 3) shapes, got {tuple(thruster_positions_b.shape)} "
            f"and {tuple(thruster_axes_b.shape)}."
        )
    if thrust.shape != thruster_axes_b.shape[:2]:
        raise ValueError(
            f"thrust must have shape {tuple(thruster_axes_b.shape[:2])}, got {tuple(thrust.shape)}."
        )

    source_pos = thruster_positions_b.unsqueeze(2)
    target_pos = thruster_positions_b.unsqueeze(1)
    rel_source_to_target = target_pos - source_pos

    signed_direction = torch.sign(thrust).unsqueeze(-1) * thruster_axes_b
    axial_distance = torch.sum(rel_source_to_target * signed_direction.unsqueeze(2), dim=-1)
    radial_vector = rel_source_to_target - axial_distance.unsqueeze(-1) * signed_direction.unsqueeze(2)
    radial_distance = torch.linalg.norm(radial_vector, dim=-1)

    wake_radius_at_target = float(wake_radius) + torch.clamp(axial_distance, min=0.0) * max(
        float(expansion_rate),
        0.0,
    )
    in_wake = (
        (axial_distance > 0.0)
        & (axial_distance <= float(wake_length))
        & (radial_distance <= wake_radius_at_target)
        & (torch.abs(thrust).unsqueeze(-1) > 1.0e-6)
    )
    num_thrusters = thrust.shape[1]
    source_is_target = torch.eye(num_thrusters, dtype=torch.bool, device=thrust.device).reshape(
        1,
        num_thrusters,
        num_thrusters,
    )
    in_wake = in_wake & ~source_is_target

    if reference_thrust is None:
        reference = torch.clamp(torch.max(torch.abs(thrust), dim=1, keepdim=True).values, min=1.0e-6)
    else:
        reference = torch.as_tensor(reference_thrust, dtype=thrust.dtype, device=thrust.device)
        if reference.ndim == 0:
            reference = reference.reshape(1, 1)
        elif reference.ndim == 1:
            if reference.shape[0] == thrust.shape[0]:
                reference = reference.reshape(thrust.shape[0], 1)
            elif reference.shape[0] == thrust.shape[1]:
                reference = reference.reshape(1, thrust.shape[1])
        reference = torch.clamp(reference, min=1.0e-6)

    source_strength = torch.clamp(torch.abs(thrust) / reference, min=0.0, max=1.0)
    radial_ratio = radial_distance / torch.clamp(wake_radius_at_target, min=1.0e-6)
    axial_fade = 1.0 - torch.clamp(axial_distance / float(wake_length), min=0.0, max=1.0)
    wake_profile = torch.exp(-(radial_ratio * radial_ratio)) * axial_fade
    loss = float(loss_coefficient) * source_strength.unsqueeze(-1) * wake_profile
    loss = torch.where(in_wake, loss, torch.zeros_like(loss))

    total_loss = torch.sum(loss, dim=1)
    return torch.clamp(1.0 - total_loss, min=float(min_scale), max=1.0)


def calculate_reaction_torques(
    thrust: torch.Tensor,
    thruster_axes_b: torch.Tensor,
    torque_coeff: float,
    spin_directions: torch.Tensor | list[float] | tuple[float, ...],
) -> torch.Tensor:
    """Return body-frame reaction torques from propeller spin."""

    if torque_coeff == 0.0:
        return torch.zeros_like(thruster_axes_b)
    spin = torch.as_tensor(spin_directions, dtype=thrust.dtype, device=thrust.device)
    if spin.ndim == 1:
        spin = spin.reshape(1, -1)
    if spin.shape[0] == 1:
        spin = spin.repeat(thrust.shape[0], 1)
    if spin.shape != thrust.shape:
        raise ValueError(f"spin_directions must broadcast to thrust shape {tuple(thrust.shape)}.")
    return -float(torque_coeff) * spin.unsqueeze(-1) * thrust.unsqueeze(-1) * thruster_axes_b


# based on https://github.com/uuvsimulator/uuv_simulator/blob/master/uuv_gazebo_plugins/uuv_gazebo_plugins/src/ThrusterConversionFcn.cc
@dataclass
class ConversionFunction(ABC):
    @abstractmethod
    def convert(self, cmd: torch.Tensor) -> torch.Tensor:
        pass


class ConversionFunctionBasic(ConversionFunction):
    rotorConstant: float

    def __init__(self, rotorConstant: float):
        super().__init__()
        self.rotorConstant = rotorConstant

    def convert(self, cmd: torch.Tensor) -> torch.Tensor:
        return self.rotorConstant * torch.abs(cmd) * cmd


class ConversionFunctionT200(ConversionFunction):
    """Convert normalized T200 throttle commands into thrust in newtons."""

    def __init__(
        self,
        max_forward_thrust: float | list[float] | tuple[float, ...] = T200_NOMINAL_FORWARD_THRUST_N,
        max_reverse_thrust: float | list[float] | tuple[float, ...] = T200_NOMINAL_REVERSE_THRUST_N,
    ):
        super().__init__()
        self.max_forward_thrust = max_forward_thrust
        self.max_reverse_thrust = max_reverse_thrust

    def convert(self, cmd: torch.Tensor) -> torch.Tensor:
        max_forward = torch.as_tensor(self.max_forward_thrust, dtype=cmd.dtype, device=cmd.device)
        max_reverse = torch.as_tensor(self.max_reverse_thrust, dtype=cmd.dtype, device=cmd.device)
        positive = torch.clamp(cmd, min=0.0)
        negative = torch.clamp(cmd, max=0.0)
        return max_forward * positive * positive - max_reverse * negative * negative


class ConversionFunctionLookupTable(ConversionFunction):
    """Convert normalized commands to thrust with a measured piecewise-linear table."""

    def __init__(
        self,
        command_points: list[float] | tuple[float, ...],
        thrust_points: list[float] | list[list[float]] | tuple[float, ...] | tuple[tuple[float, ...], ...],
        clamp: bool = True,
    ):
        super().__init__()
        self.command_points = torch.as_tensor(command_points, dtype=torch.float32)
        self.thrust_points = torch.as_tensor(thrust_points, dtype=torch.float32)
        self.clamp = clamp

        if self.command_points.ndim != 1 or self.command_points.numel() < 2:
            raise ValueError("command_points must be a 1D sequence with at least two samples.")
        if torch.any(self.command_points[1:] <= self.command_points[:-1]):
            raise ValueError("command_points must be strictly increasing.")

        if self.thrust_points.ndim == 1:
            self.thrust_points = self.thrust_points.reshape(1, -1)
        if self.thrust_points.ndim != 2 or self.thrust_points.shape[1] != self.command_points.numel():
            raise ValueError(
                "thrust_points must be shaped (num_samples,) or (num_thrusters, num_samples)."
            )

    def convert(self, cmd: torch.Tensor) -> torch.Tensor:
        command_points = self.command_points.to(device=cmd.device, dtype=cmd.dtype)
        thrust_points = self.thrust_points.to(device=cmd.device, dtype=cmd.dtype)
        query = torch.clamp(cmd, command_points[0], command_points[-1]) if self.clamp else cmd

        high = torch.bucketize(query.contiguous(), command_points)
        high = torch.clamp(high, min=1, max=command_points.numel() - 1)
        low = high - 1

        x0 = command_points[low]
        x1 = command_points[high]
        blend = (query - x0) / torch.clamp(x1 - x0, min=1.0e-6)

        if thrust_points.shape[0] == 1:
            y0 = thrust_points[0, low]
            y1 = thrust_points[0, high]
        elif thrust_points.shape[0] == cmd.shape[1]:
            thruster_indices = torch.arange(cmd.shape[1], dtype=torch.long, device=cmd.device).reshape(1, -1)
            thruster_indices = thruster_indices.repeat(cmd.shape[0], 1)
            y0 = thrust_points[thruster_indices, low]
            y1 = thrust_points[thruster_indices, high]
        else:
            raise ValueError(
                f"Lookup table has {thrust_points.shape[0]} rows but command tensor has {cmd.shape[1]} thrusters."
            )

        return y0 + blend * (y1 - y0)


class ConversionFunctionInflowLookupTable(ConversionFunction):
    """Convert commands and axial inflow speed to thrust with a measured 2D table."""

    def __init__(
        self,
        command_points: list[float] | tuple[float, ...],
        inflow_speed_points: list[float] | tuple[float, ...],
        thrust_points: list | tuple,
        clamp: bool = True,
    ):
        super().__init__()
        self.command_points = torch.as_tensor(command_points, dtype=torch.float32)
        self.inflow_speed_points = torch.as_tensor(inflow_speed_points, dtype=torch.float32)
        self.thrust_points = torch.as_tensor(thrust_points, dtype=torch.float32)
        self.clamp = clamp

        _validate_lookup_axis(self.command_points, "command_points")
        _validate_lookup_axis(self.inflow_speed_points, "inflow_speed_points")

        if self.thrust_points.ndim == 2:
            self.thrust_points = self.thrust_points.reshape(
                1,
                self.command_points.numel(),
                self.inflow_speed_points.numel(),
            )
        if (
            self.thrust_points.ndim != 3
            or self.thrust_points.shape[1] != self.command_points.numel()
            or self.thrust_points.shape[2] != self.inflow_speed_points.numel()
        ):
            raise ValueError(
                "thrust_points must be shaped (num_commands, num_inflow_speeds) "
                "or (num_thrusters, num_commands, num_inflow_speeds)."
            )

    def convert(self, cmd: torch.Tensor, axial_inflow_speed: torch.Tensor | None = None) -> torch.Tensor:
        if axial_inflow_speed is None:
            raise ValueError("axial_inflow_speed must be provided for inflow lookup conversion.")
        if axial_inflow_speed.shape != cmd.shape:
            raise ValueError(
                f"axial_inflow_speed shape {tuple(axial_inflow_speed.shape)} must match command shape {tuple(cmd.shape)}."
            )

        command_points = self.command_points.to(device=cmd.device, dtype=cmd.dtype)
        inflow_points = self.inflow_speed_points.to(device=cmd.device, dtype=cmd.dtype)
        thrust_points = self.thrust_points.to(device=cmd.device, dtype=cmd.dtype)

        command_query = torch.clamp(cmd, command_points[0], command_points[-1]) if self.clamp else cmd
        inflow_query = (
            torch.clamp(axial_inflow_speed, inflow_points[0], inflow_points[-1])
            if self.clamp
            else axial_inflow_speed
        )

        cmd_low, cmd_high, cmd_blend = _lookup_axis_indices(command_query, command_points)
        inflow_low, inflow_high, inflow_blend = _lookup_axis_indices(inflow_query, inflow_points)

        if thrust_points.shape[0] == 1:
            y00 = thrust_points[0, cmd_low, inflow_low]
            y10 = thrust_points[0, cmd_high, inflow_low]
            y01 = thrust_points[0, cmd_low, inflow_high]
            y11 = thrust_points[0, cmd_high, inflow_high]
        elif thrust_points.shape[0] == cmd.shape[1]:
            thruster_indices = torch.arange(cmd.shape[1], dtype=torch.long, device=cmd.device).reshape(1, -1)
            thruster_indices = thruster_indices.repeat(cmd.shape[0], 1)
            y00 = thrust_points[thruster_indices, cmd_low, inflow_low]
            y10 = thrust_points[thruster_indices, cmd_high, inflow_low]
            y01 = thrust_points[thruster_indices, cmd_low, inflow_high]
            y11 = thrust_points[thruster_indices, cmd_high, inflow_high]
        else:
            raise ValueError(
                f"Lookup table has {thrust_points.shape[0]} rows but command tensor has {cmd.shape[1]} thrusters."
            )

        y0 = y00 * (1.0 - cmd_blend) + y10 * cmd_blend
        y1 = y01 * (1.0 - cmd_blend) + y11 * cmd_blend
        return y0 * (1.0 - inflow_blend) + y1 * inflow_blend


def _validate_lookup_axis(points: torch.Tensor, name: str) -> None:
    if points.ndim != 1 or points.numel() < 2:
        raise ValueError(f"{name} must be a 1D sequence with at least two samples.")
    if torch.any(points[1:] <= points[:-1]):
        raise ValueError(f"{name} must be strictly increasing.")


def _lookup_axis_indices(query: torch.Tensor, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    high = torch.bucketize(query.contiguous(), points)
    high = torch.clamp(high, min=1, max=points.numel() - 1)
    low = high - 1
    x0 = points[low]
    x1 = points[high]
    blend = (query - x0) / torch.clamp(x1 - x0, min=1.0e-6)
    return low, high, blend
