"""Rigid-body property helpers shared by the WarpAUV environment and tests."""

from __future__ import annotations

import torch


def inertia_matrix_tensor(
    values,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Normalize 3-value diagonal, 3x3, or flat 9-value inertia to 3x3."""

    tensor = torch.as_tensor(values, dtype=dtype, device=device)
    if tensor.ndim == 1:
        if tensor.shape[0] == 3:
            return torch.diag(tensor)
        if tensor.shape[0] == 9:
            return tensor.reshape(3, 3)
    if tensor.ndim == 2 and tensor.shape == (3, 3):
        return tensor
    raise ValueError(f"inertia_diag must be a 3-vector, 3x3 matrix, or flat 9-value matrix, got {tuple(tensor.shape)}.")


def inertia_diag_tensor(
    values,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return inertia diagonal from any supported inertia tensor shape."""

    return torch.diagonal(inertia_matrix_tensor(values, device, dtype), dim1=-2, dim2=-1)
