"""
Fossen-style hydrodynamic wrenches for a rigid AUV body.

Isaac/PhysX integrates the rigid-body inertia, gyroscopic terms, and gravity.
This module therefore returns only the external fluid wrench to apply in the
body/link frame: buoyancy, relative-velocity damping, and optional added-mass
Coriolis terms.  Keeping that boundary explicit prevents double-counting the
rigid-body part of Fossen's 6-DOF model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch


def quat_conjugate_wxyz(q: torch.Tensor) -> torch.Tensor:
    """Quaternion conjugate for IsaacLab's (w, x, y, z) convention."""

    return torch.cat((q[..., 0:1], -q[..., 1:]), dim=-1)


def quat_apply_wxyz(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vectors with quaternions in IsaacLab's (w, x, y, z) convention."""

    q = q.reshape(-1, 4)
    v = v.reshape(-1, 3)
    xyz = q[:, 1:]
    t = 2.0 * torch.cross(xyz, v, dim=-1)
    return v + q[:, 0:1] * t + torch.cross(xyz, t, dim=-1)


def skew_symmetric(vec: torch.Tensor) -> torch.Tensor:
    """Return S(vec), where S(a) b = a x b."""

    mat = torch.zeros((*vec.shape[:-1], 3, 3), dtype=vec.dtype, device=vec.device)
    mat[..., 0, 1] = -vec[..., 2]
    mat[..., 0, 2] = vec[..., 1]
    mat[..., 1, 0] = vec[..., 2]
    mat[..., 1, 2] = -vec[..., 0]
    mat[..., 2, 0] = -vec[..., 1]
    mat[..., 2, 1] = vec[..., 0]
    return mat


def expand_6d_matrix(values: torch.Tensor, batch_size: int) -> torch.Tensor:
    """Return a batched 6x6 matrix from diagonal vectors or full matrices."""

    if values.ndim == 1:
        if values.shape[0] != 6:
            raise ValueError(f"Expected a 6-vector, got shape {tuple(values.shape)}.")
        return torch.diag_embed(values.reshape(1, 6).repeat(batch_size, 1))

    if values.ndim == 2:
        if values.shape == (6, 6):
            return values.reshape(1, 6, 6).repeat(batch_size, 1, 1)
        if values.shape[1] == 6:
            if values.shape[0] == 1:
                values = values.repeat(batch_size, 1)
            elif values.shape[0] != batch_size:
                raise ValueError(f"Expected batch size {batch_size}, got shape {tuple(values.shape)}.")
            return torch.diag_embed(values)

    if values.ndim == 3 and values.shape[1:] == (6, 6):
        if values.shape[0] == 1:
            return values.repeat(batch_size, 1, 1)
        if values.shape[0] == batch_size:
            return values

    raise ValueError(f"Expected a 6-vector, batched 6-vector, or 6x6 matrix, got shape {tuple(values.shape)}.")


def multiply_6d_matrix(values: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """Multiply diagonal-vector or full-matrix 6D coefficients by a 6-vector."""

    matrix = expand_6d_matrix(values, vector.shape[0])
    return torch.bmm(matrix, vector.unsqueeze(-1)).squeeze(-1)


def calculate_speed_dependent_damping_scale(
    nu_r: torch.Tensor,
    speed_points: torch.Tensor | list[float] | tuple[float, ...],
    scale_points: torch.Tensor | list[float] | list[list[float]] | tuple,
    clamp: bool = True,
) -> torch.Tensor:
    """Interpolate per-DOF damping scales from ``|nu_r|``.

    ``scale_points`` may be shaped ``(num_speed_points,)`` for one shared curve
    or ``(num_speed_points, 6)`` for per-DOF curves.  The result has shape
    ``(num_envs, 6)`` and is intended to multiply linear or quadratic damping
    coefficients before evaluating the Fossen damping wrench.
    """

    if nu_r.ndim != 2 or nu_r.shape[1] != 6:
        raise ValueError(f"nu_r must have shape (N, 6), got {tuple(nu_r.shape)}.")

    speeds = torch.as_tensor(speed_points, dtype=nu_r.dtype, device=nu_r.device)
    scales = torch.as_tensor(scale_points, dtype=nu_r.dtype, device=nu_r.device)
    if speeds.ndim != 1 or speeds.numel() < 2:
        raise ValueError("speed_points must be a 1D sequence with at least two samples.")
    if torch.any(speeds[1:] <= speeds[:-1]):
        raise ValueError("speed_points must be strictly increasing.")

    if scales.ndim == 1:
        if scales.shape[0] != speeds.numel():
            raise ValueError("scale_points length must match speed_points.")
        scales = scales.reshape(-1, 1).repeat(1, 6)
    if scales.ndim != 2 or scales.shape != (speeds.numel(), 6):
        raise ValueError(f"scale_points must have shape ({speeds.numel()},) or ({speeds.numel()}, 6).")

    query = torch.abs(nu_r)
    if clamp:
        query = torch.clamp(query, speeds[0], speeds[-1])

    high = torch.bucketize(query.contiguous(), speeds)
    high = torch.clamp(high, min=1, max=speeds.numel() - 1)
    low = high - 1

    x0 = speeds[low]
    x1 = speeds[high]
    blend = (query - x0) / torch.clamp(x1 - x0, min=1.0e-6)
    dof_indices = torch.arange(6, dtype=torch.long, device=nu_r.device).reshape(1, 6).repeat(nu_r.shape[0], 1)
    y0 = scales[low, dof_indices]
    y1 = scales[high, dof_indices]
    return y0 + blend * (y1 - y0)


@dataclass
class HydrodynamicForceModels:
    num_envs: int
    device: torch.device
    debug: bool = False

    def calculate_buoyancy_forces(
        self,
        root_quats_w: torch.Tensor,
        gravity_w: torch.Tensor,
        fluid_density: float,
        volumes: torch.Tensor,
        com_to_cob_offsets: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute buoyancy in body frame.

        PhysX already applies gravity at the COM.  We only add buoyancy here,
        but use the same world gravity vector so neutral buoyancy can be checked
        in one frame: F_g^w + F_b^w ~= 0 when m = rho * V.
        """

        if gravity_w.ndim == 1:
            gravity_w = gravity_w.reshape(1, 3).repeat(self.num_envs, 1)

        buoyancy_forces_w = -fluid_density * volumes * gravity_w
        buoyancy_forces_b = quat_apply_wxyz(quat_conjugate_wxyz(root_quats_w), buoyancy_forces_w)
        buoyancy_torques_b = torch.cross(com_to_cob_offsets, buoyancy_forces_b, dim=-1)

        if self.debug:
            print(f"buoyancy_forces_b={buoyancy_forces_b}, buoyancy_torques_b={buoyancy_torques_b}")

        return buoyancy_forces_b, buoyancy_torques_b

    def calculate_fossen_fluid_forces(
        self,
        root_quats_w: torch.Tensor,
        root_linvels_b: torch.Tensor,
        root_angvels_b: torch.Tensor,
        gravity_w: torch.Tensor,
        fluid_density: float,
        volumes: torch.Tensor,
        com_to_cob_offsets: torch.Tensor,
        linear_damping: torch.Tensor,
        quadratic_damping: torch.Tensor,
        water_current_w: torch.Tensor,
        added_mass_diag: torch.Tensor | None = None,
        relative_acceleration_b: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the body-frame fluid wrench.

        Damping is evaluated with relative velocity nu_r = nu - nu_c.  The
        dissipation check should therefore use nu_r^T tau_damping <= 0, not
        nu^T tau_damping, because moving water can do work on the vehicle.
        ``added_mass_diag`` is kept as the legacy argument name, but it accepts
        either a 6-vector of diagonal coefficients or a full 6x6 matrix.
        """

        buoyancy_forces_b, buoyancy_torques_b = self.calculate_buoyancy_forces(
            root_quats_w,
            gravity_w,
            fluid_density,
            volumes,
            com_to_cob_offsets,
        )

        nu_r = self.calculate_relative_velocity(
            root_quats_w,
            root_linvels_b,
            root_angvels_b,
            water_current_w,
        )

        damping_wrench = self.calculate_relative_damping_wrench(nu_r, linear_damping, quadratic_damping)

        fluid_wrench = torch.cat((buoyancy_forces_b, buoyancy_torques_b), dim=-1)
        fluid_wrench = fluid_wrench + damping_wrench

        if added_mass_diag is not None and torch.any(added_mass_diag != 0.0):
            fluid_wrench = fluid_wrench - self.calculate_added_mass_coriolis_wrench(nu_r, added_mass_diag)
            if relative_acceleration_b is not None:
                fluid_wrench = fluid_wrench + self.calculate_added_mass_inertia_wrench(
                    relative_acceleration_b,
                    added_mass_diag,
                )

        if self.debug:
            power = torch.sum(nu_r * damping_wrench, dim=-1)
            print(f"relative damping power={power}")

        return fluid_wrench[:, 0:3], fluid_wrench[:, 3:6]

    def calculate_relative_velocity(
        self,
        root_quats_w: torch.Tensor,
        root_linvels_b: torch.Tensor,
        root_angvels_b: torch.Tensor,
        water_current_w: torch.Tensor,
    ) -> torch.Tensor:
        """Return body-frame relative velocity ``nu_r = nu - nu_current``."""

        if water_current_w.ndim == 1:
            water_current_w = water_current_w.reshape(1, 3).repeat(self.num_envs, 1)
        water_current_b = quat_apply_wxyz(quat_conjugate_wxyz(root_quats_w), water_current_w)

        nu = torch.cat((root_linvels_b, root_angvels_b), dim=-1)
        nu_current = torch.zeros_like(nu)
        nu_current[:, 0:3] = water_current_b
        return nu - nu_current

    def calculate_added_mass_coriolis_wrench(self, nu_r: torch.Tensor, added_mass_diag: torch.Tensor) -> torch.Tensor:
        """Compute ``C_A(nu_r) nu_r`` for diagonal or full added mass.

        The environment applies ``-C_A nu_r`` as an external wrench.  The helper
        returns ``C_A nu_r`` so tests can directly check skew-symmetry and power.
        """

        added_momentum = multiply_6d_matrix(added_mass_diag, nu_r)
        v = nu_r[:, 0:3]
        omega = nu_r[:, 3:6]
        a_linear = added_momentum[:, 0:3]
        a_angular = added_momentum[:, 3:6]

        c_top = -torch.bmm(skew_symmetric(a_linear), omega.unsqueeze(-1)).squeeze(-1)
        c_bottom = (
            -torch.bmm(skew_symmetric(a_linear), v.unsqueeze(-1)).squeeze(-1)
            - torch.bmm(skew_symmetric(a_angular), omega.unsqueeze(-1)).squeeze(-1)
        )
        return torch.cat((c_top, c_bottom), dim=-1)

    def calculate_added_mass_inertia_wrench(
        self,
        relative_acceleration_b: torch.Tensor,
        added_mass_diag: torch.Tensor,
    ) -> torch.Tensor:
        """Return the external wrench ``-M_A dot(nu_r)``."""

        return -multiply_6d_matrix(added_mass_diag, relative_acceleration_b)

    def calculate_relative_damping_wrench(
        self,
        nu_r: torch.Tensor,
        linear_damping: torch.Tensor,
        quadratic_damping: torch.Tensor,
    ) -> torch.Tensor:
        """Standalone damping helper used by tests and diagnostics."""

        linear_wrench = multiply_6d_matrix(linear_damping, nu_r)
        quadratic_wrench = multiply_6d_matrix(quadratic_damping, torch.abs(nu_r) * nu_r)
        return -(linear_wrench + quadratic_wrench)


if __name__ == "__main__":
    try:
        from .bluerov2_heavy_model import BLUEROV2_HEAVY
    except ImportError:
        from bluerov2_heavy_model import BLUEROV2_HEAVY

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = HydrodynamicForceModels(num_envs=1, device=device)

    q_identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)
    gravity_w = torch.tensor([0.0, 0.0, -9.81], device=device)
    volume = torch.tensor([[BLUEROV2_HEAVY.neutral_buoyancy_volume_m3]], device=device)
    rho = BLUEROV2_HEAVY.water_density_kg_m3
    cob = torch.tensor([BLUEROV2_HEAVY.center_of_buoyancy_from_com_m], device=device)
    force_b, torque_b = model.calculate_buoyancy_forces(q_identity, gravity_w, rho, volume, cob)
    expected = torch.tensor([[0.0, 0.0, rho * volume.item() * 9.81]], device=device)
    assert torch.allclose(force_b, expected, atol=1.0e-5), (force_b, expected)
    assert torch.allclose(torque_b, torch.zeros_like(torque_b), atol=1.0e-5), torque_b

    nu_r = torch.tensor([[0.2, -0.1, 0.3, 0.04, -0.02, 0.01]], device=device)
    linear = torch.ones(6, device=device) * 0.1
    quadratic = torch.ones(6, device=device) * 2.0
    damping = model.calculate_relative_damping_wrench(nu_r, linear, quadratic)
    assert torch.all(torch.sum(nu_r * damping, dim=-1) <= 0.0)
    print("Hydrodynamic sanity checks passed.")
