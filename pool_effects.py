"""Simplified pool boundary effects for underwater vehicle simulation."""

from __future__ import annotations

import torch


def calculate_pool_boundary_scales(
    positions: torch.Tensor,
    bounds: torch.Tensor | list[float] | tuple[float, ...],
    effect_distance: float,
    damping_scale_at_boundary: float,
    added_mass_scale_at_boundary: float,
    thrust_scale_at_boundary: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return damping, added-mass, and thrust scales from box-boundary proximity.

    ``positions`` are expressed in the pool-local/world-aligned frame.  Bounds
    are ``[x_min, x_max, y_min, y_max, z_min, z_max]`` in the same frame.
    """

    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"positions must have shape (N, 3), got {tuple(positions.shape)}.")

    bounds_tensor = torch.as_tensor(bounds, dtype=positions.dtype, device=positions.device)
    if bounds_tensor.shape != (6,):
        raise ValueError(f"bounds must be [x_min, x_max, y_min, y_max, z_min, z_max], got {tuple(bounds_tensor.shape)}.")

    distance = torch.stack(
        [
            positions[:, 0] - bounds_tensor[0],
            bounds_tensor[1] - positions[:, 0],
            positions[:, 1] - bounds_tensor[2],
            bounds_tensor[3] - positions[:, 1],
            positions[:, 2] - bounds_tensor[4],
            bounds_tensor[5] - positions[:, 2],
        ],
        dim=-1,
    )
    min_distance = torch.min(distance, dim=-1).values
    effect_distance = max(float(effect_distance), 1.0e-6)
    proximity = torch.clamp((effect_distance - min_distance) / effect_distance, min=0.0, max=1.0)
    proximity = proximity.reshape(-1, 1)

    damping_scale = 1.0 + proximity * (float(damping_scale_at_boundary) - 1.0)
    added_mass_scale = 1.0 + proximity * (float(added_mass_scale_at_boundary) - 1.0)
    thrust_scale = 1.0 + proximity * (float(thrust_scale_at_boundary) - 1.0)
    return damping_scale, added_mass_scale, torch.clamp(thrust_scale, min=0.0)


def calculate_free_surface_scales(
    positions: torch.Tensor,
    surface_z: float,
    effect_distance: float,
    heave_damping_scale_at_surface: float,
    roll_pitch_damping_scale_at_surface: float,
    added_mass_scale_at_surface: float,
    buoyancy_scale_at_surface: float,
    thrust_scale_at_surface: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return hydrodynamic scales for proximity to a flat free surface.

    The model is local and empirical: within ``effect_distance`` of
    ``surface_z``, heave/roll/pitch damping and added mass are scaled while
    buoyancy and thrust may be reduced to approximate partial surfacing and
    thruster ventilation.  It is intentionally independent of the sign
    convention for depth because it uses distance to the configured plane.
    """

    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"positions must have shape (N, 3), got {tuple(positions.shape)}.")

    effect_distance = max(float(effect_distance), 1.0e-6)
    distance_to_surface = torch.abs(positions[:, 2] - float(surface_z))
    proximity = torch.clamp((effect_distance - distance_to_surface) / effect_distance, min=0.0, max=1.0)
    proximity = proximity * proximity * (3.0 - 2.0 * proximity)
    proximity = proximity.reshape(-1, 1)

    damping_scale = torch.ones((positions.shape[0], 6), dtype=positions.dtype, device=positions.device)
    added_mass_scale = torch.ones_like(damping_scale)

    heave_damping = 1.0 + proximity[:, 0] * (float(heave_damping_scale_at_surface) - 1.0)
    roll_pitch_damping = 1.0 + proximity[:, 0] * (float(roll_pitch_damping_scale_at_surface) - 1.0)
    damping_scale[:, 2] = heave_damping
    damping_scale[:, 3] = roll_pitch_damping
    damping_scale[:, 4] = roll_pitch_damping

    added_mass = 1.0 + proximity[:, 0] * (float(added_mass_scale_at_surface) - 1.0)
    added_mass_scale[:, 2] = added_mass
    added_mass_scale[:, 3] = added_mass
    added_mass_scale[:, 4] = added_mass

    buoyancy_scale = 1.0 + proximity * (float(buoyancy_scale_at_surface) - 1.0)
    thrust_scale = 1.0 + proximity * (float(thrust_scale_at_surface) - 1.0)
    return damping_scale, added_mass_scale, torch.clamp(buoyancy_scale, min=0.0), torch.clamp(thrust_scale, min=0.0)
