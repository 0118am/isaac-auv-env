"""Measured or prescribed pool water-current fields."""

from __future__ import annotations

import torch


def calculate_trilinear_current_field(
    positions: torch.Tensor,
    bounds: torch.Tensor | list[float] | tuple[float, ...],
    grid_shape: torch.Tensor | list[int] | tuple[int, int, int],
    grid_values: torch.Tensor | list,
) -> torch.Tensor:
    """Interpolate a world-frame current field on a regular pool-local grid.

    ``bounds`` are ``[x_min, x_max, y_min, y_max, z_min, z_max]``.  Grid values
    may be shaped ``(nx, ny, nz, 3)`` or flattened as ``(nx * ny * nz, 3)`` in
    x-major, then y, then z order.
    """

    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"positions must have shape (N, 3), got {tuple(positions.shape)}.")

    bounds_tensor = torch.as_tensor(bounds, dtype=positions.dtype, device=positions.device)
    if bounds_tensor.shape != (6,):
        raise ValueError(f"bounds must have shape (6,), got {tuple(bounds_tensor.shape)}.")
    if not (
        bounds_tensor[0] < bounds_tensor[1]
        and bounds_tensor[2] < bounds_tensor[3]
        and bounds_tensor[4] < bounds_tensor[5]
    ):
        raise ValueError("bounds must be ordered as min < max on each axis.")

    shape_tensor = torch.as_tensor(grid_shape, dtype=torch.long, device=positions.device)
    if shape_tensor.shape != (3,) or torch.any(shape_tensor <= 0):
        raise ValueError(f"grid_shape must be three positive integers, got {tuple(shape_tensor.tolist())}.")
    nx, ny, nz = (int(shape_tensor[0]), int(shape_tensor[1]), int(shape_tensor[2]))

    values = torch.as_tensor(grid_values, dtype=positions.dtype, device=positions.device)
    if values.ndim == 2 and values.shape == (nx * ny * nz, 3):
        values = values.reshape(nx, ny, nz, 3)
    if values.shape != (nx, ny, nz, 3):
        raise ValueError(
            f"grid_values must have shape ({nx}, {ny}, {nz}, 3) or ({nx * ny * nz}, 3), "
            f"got {tuple(values.shape)}."
        )

    ix0, ix1, fx = _axis_indices_and_fraction(positions[:, 0], bounds_tensor[0], bounds_tensor[1], nx)
    iy0, iy1, fy = _axis_indices_and_fraction(positions[:, 1], bounds_tensor[2], bounds_tensor[3], ny)
    iz0, iz1, fz = _axis_indices_and_fraction(positions[:, 2], bounds_tensor[4], bounds_tensor[5], nz)

    c000 = values[ix0, iy0, iz0]
    c100 = values[ix1, iy0, iz0]
    c010 = values[ix0, iy1, iz0]
    c110 = values[ix1, iy1, iz0]
    c001 = values[ix0, iy0, iz1]
    c101 = values[ix1, iy0, iz1]
    c011 = values[ix0, iy1, iz1]
    c111 = values[ix1, iy1, iz1]

    fx = fx.unsqueeze(-1)
    fy = fy.unsqueeze(-1)
    fz = fz.unsqueeze(-1)
    c00 = c000 * (1.0 - fx) + c100 * fx
    c10 = c010 * (1.0 - fx) + c110 * fx
    c01 = c001 * (1.0 - fx) + c101 * fx
    c11 = c011 * (1.0 - fx) + c111 * fx
    c0 = c00 * (1.0 - fy) + c10 * fy
    c1 = c01 * (1.0 - fy) + c11 * fy
    return c0 * (1.0 - fz) + c1 * fz


def _axis_indices_and_fraction(
    position: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    count: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if count == 1:
        zeros_i = torch.zeros_like(position, dtype=torch.long)
        zeros_f = torch.zeros_like(position)
        return zeros_i, zeros_i, zeros_f

    normalized = torch.clamp((position - lower) / (upper - lower), min=0.0, max=1.0)
    grid_coordinate = normalized * float(count - 1)
    lower_index = torch.floor(grid_coordinate).to(dtype=torch.long)
    upper_index = torch.clamp(lower_index + 1, max=count - 1)
    fraction = grid_coordinate - lower_index.to(dtype=position.dtype)
    return lower_index, upper_index, fraction
