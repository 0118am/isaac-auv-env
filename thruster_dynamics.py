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

        if self.tau <= 0.0:
            self.state = cmd
        else:
            alpha = torch.exp(-dt / self.tau)
            self.state = self.state * alpha.unsqueeze(-1) + (1.0 - alpha).unsqueeze(-1) * cmd

        self.prevTime = t
        return self.state


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
