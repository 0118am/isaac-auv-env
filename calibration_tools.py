"""Calibration helpers for turning pool experiment logs into dynamics profiles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch


@dataclass(frozen=True)
class DiagonalDampingFit:
    linear_damping: torch.Tensor
    quadratic_damping: torch.Tensor
    residual_rms: torch.Tensor
    sample_count: torch.Tensor

    def to_cfg_updates(self) -> dict[str, Any]:
        return {
            "linear_damping": self.linear_damping.detach().cpu().tolist(),
            "quadratic_damping": self.quadratic_damping.detach().cpu().tolist(),
        }


@dataclass(frozen=True)
class FullMatrixDampingFit:
    linear_damping: torch.Tensor
    quadratic_damping: torch.Tensor
    residual_rms: torch.Tensor
    sample_count: int
    regularization: float

    def to_cfg_updates(self) -> dict[str, Any]:
        return {
            "linear_damping": self.linear_damping.detach().cpu().tolist(),
            "quadratic_damping": self.quadratic_damping.detach().cpu().tolist(),
        }


@dataclass(frozen=True)
class DiagonalAddedMassDampingFit:
    added_mass: torch.Tensor
    effective_inertia: torch.Tensor
    linear_damping: torch.Tensor
    quadratic_damping: torch.Tensor
    residual_rms: torch.Tensor
    sample_count: torch.Tensor

    def to_cfg_updates(self) -> dict[str, Any]:
        return {
            "added_mass_diag": self.added_mass.detach().cpu().tolist(),
            "linear_damping": self.linear_damping.detach().cpu().tolist(),
            "quadratic_damping": self.quadratic_damping.detach().cpu().tolist(),
        }


@dataclass(frozen=True)
class FullMatrixAddedMassDampingFit:
    added_mass: torch.Tensor
    effective_inertia: torch.Tensor
    linear_damping: torch.Tensor
    quadratic_damping: torch.Tensor
    residual_rms: torch.Tensor
    sample_count: int
    regularization: float
    symmetrized_added_mass: bool

    def to_cfg_updates(self) -> dict[str, Any]:
        return {
            "added_mass_diag": self.added_mass.detach().cpu().tolist(),
            "linear_damping": self.linear_damping.detach().cpu().tolist(),
            "quadratic_damping": self.quadratic_damping.detach().cpu().tolist(),
        }


@dataclass(frozen=True)
class DampingSpeedScaleFit:
    linear_scales: torch.Tensor
    quadratic_scales: torch.Tensor
    residual_rms: torch.Tensor
    sample_count: torch.Tensor

    def to_cfg_updates(self, speed_points: Sequence[float], enabled: bool = True) -> dict[str, Any]:
        return {
            "speed_dependent_damping_enabled": bool(enabled),
            "damping_speed_points": [float(value) for value in speed_points],
            "linear_damping_speed_scales": self.linear_scales.detach().cpu().tolist(),
            "quadratic_damping_speed_scales": self.quadratic_scales.detach().cpu().tolist(),
        }


@dataclass(frozen=True)
class WaterCurrentProcessFit:
    mean_current_w: torch.Tensor
    residual_std_w: torch.Tensor
    tau_s: float
    estimated_alpha: float
    variation_std: float
    horizontal_max: float
    vertical_max: float
    sample_count: int

    def to_cfg_updates(self) -> dict[str, Any]:
        return {"water_current_w": self.mean_current_w.detach().cpu().tolist()}

    def to_domain_randomization_updates(self, stage_count: int = 1) -> dict[str, Any]:
        if int(stage_count) < 1:
            raise ValueError("stage_count must be positive.")
        tau = max(float(self.tau_s), 1.0e-6)
        return {
            "water_current_smooth": True,
            "water_current_tau_range": [tau, tau],
            "water_current_max_by_stage": [float(self.horizontal_max)] * int(stage_count),
            "water_current_vertical_max_by_stage": [float(self.vertical_max)] * int(stage_count),
            "water_current_variation_std_by_stage": [float(self.variation_std)] * int(stage_count),
        }


@dataclass(frozen=True)
class WaterCurrentFieldFit:
    bounds: torch.Tensor
    grid_shape: tuple[int, int, int]
    grid_values: torch.Tensor
    sample_count: int
    k_neighbors: int
    interpolation_power: float

    def to_cfg_updates(self, enabled: bool = True) -> dict[str, Any]:
        return {
            "water_current_field_enabled": bool(enabled),
            "water_current_field_bounds": self.bounds.detach().cpu().tolist(),
            "water_current_field_shape": list(self.grid_shape),
            "water_current_field_values": self.grid_values.reshape(-1, 3).detach().cpu().tolist(),
        }


@dataclass(frozen=True)
class MassFit:
    mass: float
    residual_rms: float
    sample_count: int

    def to_cfg_updates(self) -> dict[str, Any]:
        return {"mass": float(self.mass)}


@dataclass(frozen=True)
class InertiaTensorFit:
    inertia_tensor: torch.Tensor
    residual_rms: float
    sample_count: int
    design_rank: int
    projected_to_psd: bool
    min_eigenvalue_before_projection: float
    min_eigenvalue_after_projection: float

    def to_cfg_updates(self) -> dict[str, Any]:
        return {"inertia_diag": self.inertia_tensor.detach().cpu().tolist()}


@dataclass(frozen=True)
class BuoyancyVolumeFit:
    volume: float
    mean_buoyancy_force_w: torch.Tensor
    residual_rms: float
    sample_count: int
    water_density: float

    def to_cfg_updates(self) -> dict[str, Any]:
        return {"volume": float(self.volume), "water_rho": float(self.water_density)}


@dataclass(frozen=True)
class CenterOfBuoyancyFit:
    com_to_cob_offset: torch.Tensor
    residual_rms: float
    sample_count: int
    design_rank: int

    def to_cfg_updates(self) -> dict[str, Any]:
        return {"com_to_cob_offset": self.com_to_cob_offset.detach().cpu().tolist()}


@dataclass(frozen=True)
class PoolBoundaryScaleFit:
    bounds: torch.Tensor
    effect_distance: float
    damping_scale_at_boundary: float
    added_mass_scale_at_boundary: float
    thrust_scale_at_boundary: float
    residual_rms: torch.Tensor
    sample_count: torch.Tensor

    def to_cfg_updates(self, enabled: bool = True) -> dict[str, Any]:
        return {
            "pool_boundary_effects_enabled": bool(enabled),
            "pool_bounds": self.bounds.detach().cpu().tolist(),
            "pool_boundary_effect_distance": float(self.effect_distance),
            "pool_boundary_damping_scale": float(self.damping_scale_at_boundary),
            "pool_boundary_added_mass_scale": float(self.added_mass_scale_at_boundary),
            "pool_boundary_thrust_scale": float(self.thrust_scale_at_boundary),
        }


@dataclass(frozen=True)
class FreeSurfaceScaleFit:
    surface_z: float
    effect_distance: float
    heave_damping_scale: float
    roll_pitch_damping_scale: float
    added_mass_scale: float
    buoyancy_scale: float
    thrust_scale: float
    residual_rms: torch.Tensor
    sample_count: torch.Tensor

    def to_cfg_updates(self, enabled: bool = True) -> dict[str, Any]:
        return {
            "free_surface_effects_enabled": bool(enabled),
            "free_surface_z": float(self.surface_z),
            "free_surface_effect_distance": float(self.effect_distance),
            "free_surface_heave_damping_scale": float(self.heave_damping_scale),
            "free_surface_roll_pitch_damping_scale": float(self.roll_pitch_damping_scale),
            "free_surface_added_mass_scale": float(self.added_mass_scale),
            "free_surface_buoyancy_scale": float(self.buoyancy_scale),
            "free_surface_thrust_scale": float(self.thrust_scale),
        }


@dataclass(frozen=True)
class ThrusterStaticLookupFit:
    command_points: torch.Tensor
    thrust_points: torch.Tensor
    residual_rms: torch.Tensor
    sample_count: torch.Tensor
    estimated_deadband: float

    def to_cfg_updates(self, enabled: bool = True, include_deadband: bool = False) -> dict[str, Any]:
        thrust_points = self.thrust_points.detach().cpu()
        if thrust_points.shape[0] == 1:
            thrust_value: Any = thrust_points[0].tolist()
        else:
            thrust_value = thrust_points.tolist()
        updates = {
            "use_thruster_lookup_table": bool(enabled),
            "thruster_lookup_commands": self.command_points.detach().cpu().tolist(),
            "thruster_lookup_thrusts": thrust_value,
        }
        if include_deadband:
            updates["thruster_deadband"] = float(self.estimated_deadband)
        return updates


@dataclass(frozen=True)
class ThrusterInflowLookupFit:
    command_points: torch.Tensor
    inflow_speed_points: torch.Tensor
    thrust_points: torch.Tensor
    residual_rms: torch.Tensor
    sample_count: torch.Tensor

    def to_cfg_updates(self, enabled: bool = True) -> dict[str, Any]:
        thrust_points = self.thrust_points.detach().cpu()
        if thrust_points.shape[0] == 1:
            thrust_value: Any = thrust_points[0].tolist()
        else:
            thrust_value = thrust_points.tolist()
        return {
            "use_thruster_inflow_lookup_table": bool(enabled),
            "thruster_inflow_lookup_commands": self.command_points.detach().cpu().tolist(),
            "thruster_inflow_lookup_speeds": self.inflow_speed_points.detach().cpu().tolist(),
            "thruster_inflow_lookup_thrusts": thrust_value,
        }


@dataclass(frozen=True)
class ThrusterFirstOrderFit:
    time_constant_s: float
    response_delay_s: float
    initial_thrust: float
    steady_state_thrust: float
    residual_rms: float
    sample_count: int

    def to_cfg_updates(self, physics_dt_s: float | None = None) -> dict[str, Any]:
        updates: dict[str, Any] = {"dyn_time_constant": float(self.time_constant_s)}
        if physics_dt_s is not None:
            if float(physics_dt_s) <= 0.0:
                raise ValueError("physics_dt_s must be positive.")
            updates["thruster_command_delay_steps"] = int(round(float(self.response_delay_s) / float(physics_dt_s)))
        return updates


@dataclass(frozen=True)
class ThrusterVoltageExponentFit:
    nominal_voltage: float
    thrust_exponent: float
    residual_rms: float
    sample_count: int

    def to_cfg_updates(self) -> dict[str, Any]:
        return {
            "battery_voltage_nominal": float(self.nominal_voltage),
            "battery_voltage_thrust_exponent": float(self.thrust_exponent),
        }


@dataclass(frozen=True)
class TetherSpringDamperFit:
    slack_length: float
    stiffness: float
    damping: float
    residual_rms: float
    sample_count: int

    def to_cfg_updates(self, enabled: bool = True) -> dict[str, Any]:
        return {
            "tether_enabled": bool(enabled),
            "tether_slack_length": float(self.slack_length),
            "tether_stiffness": float(self.stiffness),
            "tether_damping": float(self.damping),
        }


@dataclass(frozen=True)
class TetherDragFit:
    drag_coeff: float
    residual_rms: float
    sample_count: int

    def to_cfg_updates(self, enabled: bool = True) -> dict[str, Any]:
        return {
            "tether_enabled": bool(enabled),
            "tether_drag_coeff": float(self.drag_coeff),
        }


@dataclass(frozen=True)
class MatrixProjectionResult:
    projected_matrix: torch.Tensor
    original_min_eigenvalue: float
    projected_min_eigenvalue: float
    correction_frobenius_norm: float
    symmetrized_input: bool
    preserved_skew: bool

    def to_cfg_value(self) -> list[list[float]]:
        return self.projected_matrix.detach().cpu().tolist()


@dataclass(frozen=True)
class _ProximityScaleEstimate:
    scale_at_boundary: float
    residual_rms: float
    sample_count: int


def finite_difference(time_s: torch.Tensor | Sequence[float], values: torch.Tensor | Sequence[Sequence[float]]) -> torch.Tensor:
    """Differentiate sampled values with first-order endpoints and central interior samples."""

    time = torch.as_tensor(time_s, dtype=torch.float32)
    samples = torch.as_tensor(values, dtype=torch.float32)
    if time.ndim != 1 or time.numel() < 2:
        raise ValueError("time_s must be a 1D sequence with at least two samples.")
    if torch.any(time[1:] <= time[:-1]):
        raise ValueError("time_s must be strictly increasing.")
    if samples.shape[0] != time.numel():
        raise ValueError("values first dimension must match time_s length.")

    derivative = torch.zeros_like(samples)
    derivative[0] = (samples[1] - samples[0]) / (time[1] - time[0])
    derivative[-1] = (samples[-1] - samples[-2]) / (time[-1] - time[-2])
    if time.numel() > 2:
        dt = (time[2:] - time[:-2]).reshape(-1, *([1] * (samples.ndim - 1)))
        derivative[1:-1] = (samples[2:] - samples[:-2]) / dt
    return derivative


def fit_mass_from_scale_readings(mass_samples: torch.Tensor | Sequence[float] | float) -> MassFit:
    """Average repeated dry or wet scale readings into a rigid-body mass parameter."""

    samples = _as_1d_float_samples(mass_samples, "mass_samples")
    if torch.any(samples <= 0.0):
        raise ValueError("mass_samples must be positive.")

    mass = torch.mean(samples)
    residual = samples - mass
    return MassFit(
        float(mass.item()),
        float(torch.sqrt(torch.mean(residual * residual)).item()),
        int(samples.numel()),
    )


def compound_pendulum_moments_from_periods(
    period_s_samples: torch.Tensor | Sequence[float] | float,
    mass: float,
    pivot_to_com_distance_samples: torch.Tensor | Sequence[float] | float,
    gravity_mps2: float = 9.81,
) -> torch.Tensor:
    """Convert small-angle compound-pendulum periods to COM moments of inertia.

    ``pivot_to_com_distance_samples`` is the perpendicular distance from the
    suspension axis to the center of mass for each measured period.
    """

    periods = _as_1d_float_samples(period_s_samples, "period_s_samples")
    distances = torch.as_tensor(
        pivot_to_com_distance_samples,
        dtype=periods.dtype,
        device=periods.device,
    )
    if distances.ndim == 0:
        distances = distances.reshape(1).repeat(periods.numel())
    elif distances.ndim == 1 and distances.numel() == periods.numel():
        pass
    else:
        raise ValueError("pivot_to_com_distance_samples must be a scalar or match period_s_samples.")
    if not torch.all(torch.isfinite(distances)):
        raise ValueError("pivot_to_com_distance_samples must contain only finite values.")
    if torch.any(periods <= 0.0):
        raise ValueError("period_s_samples must be positive.")
    if torch.any(distances <= 0.0):
        raise ValueError("pivot_to_com_distance_samples must be positive.")
    if float(mass) <= 0.0:
        raise ValueError("mass must be positive.")
    if float(gravity_mps2) <= 0.0:
        raise ValueError("gravity_mps2 must be positive.")

    period_scale = periods / (2.0 * torch.pi)
    return float(mass) * float(gravity_mps2) * distances * period_scale * period_scale - float(mass) * distances * distances


def fit_inertia_tensor_from_axis_moments(
    axis_b_samples: torch.Tensor | Sequence[Sequence[float]],
    moment_samples: torch.Tensor | Sequence[float],
    min_eigenvalue: float = 0.0,
    project_to_psd: bool = True,
) -> InertiaTensorFit:
    """Fit a symmetric 3x3 inertia tensor from moments about body-frame axes."""

    axes = _as_3d_axis_samples(axis_b_samples, "axis_b_samples")
    moments = _as_1d_float_samples(moment_samples, "moment_samples").to(dtype=axes.dtype, device=axes.device)
    if moments.numel() != axes.shape[0]:
        raise ValueError("moment_samples length must match axis_b_samples.")
    if float(min_eigenvalue) < 0.0:
        raise ValueError("min_eigenvalue must be non-negative.")

    x = axes[:, 0]
    y = axes[:, 1]
    z = axes[:, 2]
    design = torch.stack((x * x, y * y, z * z, 2.0 * x * y, 2.0 * x * z, 2.0 * y * z), dim=-1)
    coefficients = torch.linalg.pinv(design) @ moments
    fitted_tensor = torch.stack(
        (
            torch.stack((coefficients[0], coefficients[3], coefficients[4])),
            torch.stack((coefficients[3], coefficients[1], coefficients[5])),
            torch.stack((coefficients[4], coefficients[5], coefficients[2])),
        )
    )
    fitted_tensor = 0.5 * (fitted_tensor + fitted_tensor.T)
    eigenvalues_before = torch.linalg.eigvalsh(fitted_tensor)
    if project_to_psd:
        projection = project_symmetric_matrix_psd(
            fitted_tensor,
            min_eigenvalue=float(min_eigenvalue),
            symmetrize=False,
        )
        inertia_tensor = projection.projected_matrix
    else:
        inertia_tensor = fitted_tensor
    eigenvalues_after = torch.linalg.eigvalsh(inertia_tensor)

    tensor_coefficients = torch.stack(
        (
            inertia_tensor[0, 0],
            inertia_tensor[1, 1],
            inertia_tensor[2, 2],
            inertia_tensor[0, 1],
            inertia_tensor[0, 2],
            inertia_tensor[1, 2],
        )
    )
    residual = design @ tensor_coefficients - moments
    return InertiaTensorFit(
        inertia_tensor,
        float(torch.sqrt(torch.mean(residual * residual)).item()),
        int(axes.shape[0]),
        int(torch.linalg.matrix_rank(design).item()),
        bool(project_to_psd),
        float(torch.min(eigenvalues_before).item()),
        float(torch.min(eigenvalues_after).item()),
    )


def fit_inertia_tensor_from_compound_pendulum(
    axis_b_samples: torch.Tensor | Sequence[Sequence[float]],
    period_s_samples: torch.Tensor | Sequence[float],
    mass: float,
    pivot_to_com_distance_samples: torch.Tensor | Sequence[float] | float,
    gravity_mps2: float = 9.81,
    min_eigenvalue: float = 0.0,
    project_to_psd: bool = True,
) -> InertiaTensorFit:
    """Fit a 3x3 inertia tensor from small-angle compound-pendulum measurements."""

    moments = compound_pendulum_moments_from_periods(
        period_s_samples,
        mass=mass,
        pivot_to_com_distance_samples=pivot_to_com_distance_samples,
        gravity_mps2=gravity_mps2,
    )
    return fit_inertia_tensor_from_axis_moments(
        axis_b_samples,
        moments,
        min_eigenvalue=min_eigenvalue,
        project_to_psd=project_to_psd,
    )


def fit_buoyancy_volume_from_forces(
    buoyancy_force_w_samples: torch.Tensor | Sequence[Sequence[float]],
    water_density: float,
    gravity_w: torch.Tensor | Sequence[float] = (0.0, 0.0, -9.81),
    nonnegative: bool = True,
) -> BuoyancyVolumeFit:
    """Estimate displaced volume from measured buoyancy force samples.

    Samples should be the upward fluid buoyancy force in world coordinates,
    not the combined gravity+buoyancy net force.
    """

    forces = _as_3d_vector_samples(buoyancy_force_w_samples, "buoyancy_force_w_samples")
    gravity = torch.as_tensor(gravity_w, dtype=forces.dtype, device=forces.device)
    if gravity.shape != (3,):
        raise ValueError("gravity_w must have shape (3,).")
    gravity_norm = torch.linalg.norm(gravity)
    if gravity_norm <= 0.0:
        raise ValueError("gravity_w magnitude must be positive.")
    if float(water_density) <= 0.0:
        raise ValueError("water_density must be positive.")

    upward = -gravity / gravity_norm
    projected_force = forces @ upward
    volume_samples = projected_force / (float(water_density) * gravity_norm)
    if nonnegative:
        volume_samples = torch.clamp(volume_samples, min=0.0)
    volume = torch.mean(volume_samples)
    predicted_force = (float(water_density) * volume * gravity_norm) * upward.reshape(1, 3)
    residual = predicted_force - forces
    return BuoyancyVolumeFit(
        float(volume.item()),
        torch.mean(forces, dim=0),
        float(torch.sqrt(torch.mean(residual * residual)).item()),
        int(forces.shape[0]),
        float(water_density),
    )


def fit_com_to_cob_offset_from_buoyancy_wrenches(
    buoyancy_force_b_samples: torch.Tensor | Sequence[Sequence[float]],
    buoyancy_torque_b_samples: torch.Tensor | Sequence[Sequence[float]],
    min_force_norm: float = 1.0e-6,
) -> CenterOfBuoyancyFit:
    """Fit ``r_BG`` from body-frame buoyancy force/torque samples.

    The fitted equation is ``tau_b = r_BG x F_b``.
    """

    forces = _as_3d_vector_samples(buoyancy_force_b_samples, "buoyancy_force_b_samples")
    torques = _as_3d_vector_samples(buoyancy_torque_b_samples, "buoyancy_torque_b_samples").to(
        dtype=forces.dtype,
        device=forces.device,
    )
    if torques.shape != forces.shape:
        raise ValueError("buoyancy_torque_b_samples must match buoyancy_force_b_samples.")
    valid = torch.linalg.norm(forces, dim=1) >= float(min_force_norm)
    if int(torch.sum(valid).item()) == 0:
        raise ValueError("No buoyancy force samples exceed min_force_norm.")

    force_fit = forces[valid]
    torque_fit = torques[valid]
    design = -_skew_3d(force_fit)
    design_flat = design.reshape(-1, 3)
    target = torque_fit.reshape(-1)
    offset = torch.linalg.pinv(design_flat) @ target
    residual = design_flat @ offset - target
    rank = int(torch.linalg.matrix_rank(design_flat).item())
    return CenterOfBuoyancyFit(
        offset,
        float(torch.sqrt(torch.mean(residual * residual)).item()),
        int(force_fit.shape[0]),
        rank,
    )


def fit_com_to_cob_offset_from_static_torques(
    root_quats_w: torch.Tensor | Sequence[Sequence[float]],
    buoyancy_torque_b_samples: torch.Tensor | Sequence[Sequence[float]],
    volume: float,
    water_density: float,
    gravity_w: torch.Tensor | Sequence[float] = (0.0, 0.0, -9.81),
) -> CenterOfBuoyancyFit:
    """Fit ``r_BG`` from static orientations and measured body-frame buoyancy torques."""

    quats = _as_quat_wxyz(root_quats_w, "root_quats_w")
    if float(volume) <= 0.0:
        raise ValueError("volume must be positive.")
    if float(water_density) <= 0.0:
        raise ValueError("water_density must be positive.")
    gravity = torch.as_tensor(gravity_w, dtype=quats.dtype, device=quats.device)
    if gravity.shape != (3,):
        raise ValueError("gravity_w must have shape (3,).")

    buoyancy_force_w = -float(water_density) * float(volume) * gravity.reshape(1, 3)
    buoyancy_force_w = buoyancy_force_w.repeat(quats.shape[0], 1)
    buoyancy_force_b = _quat_apply_wxyz(_quat_conjugate_wxyz(quats), buoyancy_force_w)
    return fit_com_to_cob_offset_from_buoyancy_wrenches(buoyancy_force_b, buoyancy_torque_b_samples)


def project_symmetric_matrix_psd(
    matrix: torch.Tensor | Sequence[Sequence[float]],
    min_eigenvalue: float = 0.0,
    symmetrize: bool = True,
) -> MatrixProjectionResult:
    """Project a square matrix onto the symmetric positive-semidefinite cone."""

    source = _as_square_matrix(matrix, "matrix")
    if float(min_eigenvalue) < 0.0:
        raise ValueError("min_eigenvalue must be non-negative.")
    if not symmetrize and not torch.allclose(source, source.T, atol=1.0e-6, rtol=1.0e-6):
        raise ValueError("matrix must be symmetric when symmetrize=False.")

    symmetric = 0.5 * (source + source.T) if symmetrize else source
    eigenvalues, eigenvectors = torch.linalg.eigh(symmetric)
    clamped = torch.clamp(eigenvalues, min=float(min_eigenvalue))
    projected = (eigenvectors * clamped.reshape(1, -1)) @ eigenvectors.T
    projected = 0.5 * (projected + projected.T)
    projected_eigenvalues = torch.linalg.eigvalsh(projected)

    return MatrixProjectionResult(
        projected,
        float(torch.min(eigenvalues).item()),
        float(torch.min(projected_eigenvalues).item()),
        float(torch.linalg.norm(projected - source).item()),
        bool(symmetrize),
        False,
    )


def project_added_mass_to_physical(
    added_mass: torch.Tensor | Sequence[float] | Sequence[Sequence[float]] | float,
    min_eigenvalue: float = 0.0,
) -> MatrixProjectionResult:
    """Symmetrize and project added mass to a positive-semidefinite 6x6 matrix."""

    raw = torch.as_tensor(added_mass)
    dtype = raw.dtype if torch.is_floating_point(raw) else torch.float32
    matrix = _as_6d_matrix(
        added_mass,
        device=raw.device,
        dtype=dtype,
        name="added_mass",
    )
    return project_symmetric_matrix_psd(matrix, min_eigenvalue=min_eigenvalue, symmetrize=True)


def project_linear_damping_to_dissipative(
    linear_damping: torch.Tensor | Sequence[float] | Sequence[Sequence[float]] | float,
    min_eigenvalue: float = 0.0,
    preserve_skew: bool = True,
) -> MatrixProjectionResult:
    """Project linear damping so ``nu.T @ D @ nu`` is non-negative for all ``nu``."""

    raw = torch.as_tensor(linear_damping)
    dtype = raw.dtype if torch.is_floating_point(raw) else torch.float32
    matrix = _as_6d_matrix(
        linear_damping,
        device=raw.device,
        dtype=dtype,
        name="linear_damping",
    )
    symmetric = 0.5 * (matrix + matrix.T)
    skew = 0.5 * (matrix - matrix.T)
    projection = project_symmetric_matrix_psd(
        symmetric,
        min_eigenvalue=min_eigenvalue,
        symmetrize=False,
    )
    projected = projection.projected_matrix + (skew if preserve_skew else torch.zeros_like(skew))
    projected_symmetric = 0.5 * (projected + projected.T)
    projected_eigenvalues = torch.linalg.eigvalsh(projected_symmetric)

    return MatrixProjectionResult(
        projected,
        projection.original_min_eigenvalue,
        float(torch.min(projected_eigenvalues).item()),
        float(torch.linalg.norm(projected - matrix).item()),
        False,
        bool(preserve_skew),
    )


def calculate_damping_dissipated_power(
    nu_r: torch.Tensor | Sequence[Sequence[float]],
    linear_damping: torch.Tensor | Sequence[float] | Sequence[Sequence[float]] | float | None = None,
    quadratic_damping: torch.Tensor | Sequence[float] | Sequence[Sequence[float]] | float | None = None,
) -> torch.Tensor:
    """Return positive dissipated power samples for the fitted damping convention.

    The simulator applies the damping wrench with a minus sign.  Therefore a
    physically passive coefficient set should satisfy this returned value >= 0
    over the expected relative-velocity envelope.
    """

    nu = _as_6d_motion_samples(nu_r, "nu_r")
    power = torch.zeros(nu.shape[0], dtype=nu.dtype, device=nu.device)
    if linear_damping is not None:
        linear = _as_6d_matrix(linear_damping, nu.device, nu.dtype, "linear_damping")
        linear_wrench = nu @ linear.T
        power = power + torch.sum(nu * linear_wrench, dim=-1)
    if quadratic_damping is not None:
        quadratic = _as_6d_matrix(quadratic_damping, nu.device, nu.dtype, "quadratic_damping")
        signed_square = torch.abs(nu) * nu
        quadratic_wrench = signed_square @ quadratic.T
        power = power + torch.sum(nu * quadratic_wrench, dim=-1)
    return power


def damping_is_dissipative_for_samples(
    nu_r: torch.Tensor | Sequence[Sequence[float]],
    linear_damping: torch.Tensor | Sequence[float] | Sequence[Sequence[float]] | float | None = None,
    quadratic_damping: torch.Tensor | Sequence[float] | Sequence[Sequence[float]] | float | None = None,
    tolerance: float = 1.0e-6,
) -> bool:
    """Check sampled damping passivity over a measured or designed velocity set."""

    power = calculate_damping_dissipated_power(nu_r, linear_damping, quadratic_damping)
    return bool(torch.all(power >= -float(tolerance)).item())


def fit_diagonal_linear_quadratic_damping(
    time_s: torch.Tensor | Sequence[float],
    nu_r: torch.Tensor | Sequence[Sequence[float]],
    applied_wrench: torch.Tensor | Sequence[Sequence[float]],
    effective_mass: torch.Tensor | Sequence[float] | float,
    relative_acceleration: torch.Tensor | Sequence[Sequence[float]] | None = None,
    min_speed: float = 1.0e-5,
    nonnegative: bool = True,
) -> DiagonalDampingFit:
    """Fit diagonal ``D_l`` and ``D_q`` from pool log samples.

    The fitted equation is ``tau_applied - M_eff dot(nu_r) =
    D_l nu_r + D_q |nu_r| nu_r`` for each DOF independently.
    """

    nu, wrench, acceleration, mass = _prepare_damping_fit_inputs(
        time_s,
        nu_r,
        applied_wrench,
        effective_mass,
        relative_acceleration,
    )
    target = wrench - mass.reshape(1, 6) * acceleration

    linear = torch.zeros(6, dtype=nu.dtype, device=nu.device)
    quadratic = torch.zeros_like(linear)
    residual_rms = torch.zeros_like(linear)
    sample_count = torch.zeros(6, dtype=torch.long, device=nu.device)

    for dof in range(6):
        speed = nu[:, dof]
        mask = torch.isfinite(speed) & torch.isfinite(target[:, dof]) & (torch.abs(speed) >= float(min_speed))
        sample_count[dof] = int(torch.sum(mask).item())
        if sample_count[dof] == 0:
            continue

        design = torch.stack((speed[mask], torch.abs(speed[mask]) * speed[mask]), dim=-1)
        beta = _solve_two_parameter_least_squares(design, target[mask, dof], nonnegative)
        linear[dof] = beta[0]
        quadratic[dof] = beta[1]
        residual = design @ beta - target[mask, dof]
        residual_rms[dof] = torch.sqrt(torch.mean(residual * residual))

    return DiagonalDampingFit(linear, quadratic, residual_rms, sample_count)


def fit_full_matrix_linear_quadratic_damping(
    time_s: torch.Tensor | Sequence[float],
    nu_r: torch.Tensor | Sequence[Sequence[float]],
    applied_wrench: torch.Tensor | Sequence[Sequence[float]],
    effective_mass: torch.Tensor | Sequence[float] | float,
    relative_acceleration: torch.Tensor | Sequence[Sequence[float]] | None = None,
    min_speed_norm: float = 1.0e-5,
    regularization: float = 0.0,
) -> FullMatrixDampingFit:
    """Fit full 6x6 linear and quadratic damping matrices from multi-axis logs."""

    nu, wrench, acceleration, mass = _prepare_damping_fit_inputs(
        time_s,
        nu_r,
        applied_wrench,
        effective_mass,
        relative_acceleration,
    )
    target = wrench - mass.reshape(1, 6) * acceleration
    finite_rows = torch.all(torch.isfinite(nu), dim=1) & torch.all(torch.isfinite(target), dim=1)
    energetic_rows = torch.linalg.norm(nu, dim=1) >= float(min_speed_norm)
    mask = finite_rows & energetic_rows
    sample_count = int(torch.sum(mask).item())
    if sample_count == 0:
        raise ValueError("No valid samples remain for full-matrix damping fit.")

    nu_fit = nu[mask]
    target_fit = target[mask]
    design = torch.cat((nu_fit, torch.abs(nu_fit) * nu_fit), dim=1)
    coefficients = _solve_matrix_least_squares(design, target_fit, float(regularization))
    linear = coefficients[:6, :].T.contiguous()
    quadratic = coefficients[6:, :].T.contiguous()
    residual = design @ coefficients - target_fit
    residual_rms = torch.sqrt(torch.mean(residual * residual, dim=0))

    return FullMatrixDampingFit(linear, quadratic, residual_rms, sample_count, float(regularization))


def fit_diagonal_added_mass_linear_quadratic_damping(
    time_s: torch.Tensor | Sequence[float],
    nu_r: torch.Tensor | Sequence[Sequence[float]],
    applied_wrench: torch.Tensor | Sequence[Sequence[float]],
    rigid_body_inertia: torch.Tensor | Sequence[float] | float,
    relative_acceleration: torch.Tensor | Sequence[Sequence[float]] | None = None,
    min_signal: float = 1.0e-5,
    nonnegative: bool = True,
) -> DiagonalAddedMassDampingFit:
    """Jointly fit diagonal effective inertia, damping, and added mass.

    The fitted equation is ``tau_applied = M_eff dot(nu_r) +
    D_l nu_r + D_q |nu_r| nu_r``.  ``added_mass`` is returned as
    ``max(M_eff - rigid_body_inertia, 0)`` when ``nonnegative=True``.
    """

    nu, wrench, acceleration, rigid = _prepare_damping_fit_inputs(
        time_s,
        nu_r,
        applied_wrench,
        rigid_body_inertia,
        relative_acceleration,
    )
    effective_inertia = torch.zeros(6, dtype=nu.dtype, device=nu.device)
    linear = torch.zeros_like(effective_inertia)
    quadratic = torch.zeros_like(effective_inertia)
    residual_rms = torch.zeros_like(effective_inertia)
    sample_count = torch.zeros(6, dtype=torch.long, device=nu.device)

    for dof in range(6):
        design = torch.stack(
            (
                acceleration[:, dof],
                nu[:, dof],
                torch.abs(nu[:, dof]) * nu[:, dof],
            ),
            dim=-1,
        )
        valid = torch.isfinite(wrench[:, dof]) & torch.all(torch.isfinite(design), dim=1)
        energetic = torch.linalg.norm(design, dim=1) >= float(min_signal)
        mask = valid & energetic
        sample_count[dof] = int(torch.sum(mask).item())
        if sample_count[dof] == 0:
            continue

        beta = _solve_nonnegative_least_squares_by_subset(design[mask], wrench[mask, dof], nonnegative)
        effective_inertia[dof] = beta[0]
        linear[dof] = beta[1]
        quadratic[dof] = beta[2]
        residual = design[mask] @ beta - wrench[mask, dof]
        residual_rms[dof] = torch.sqrt(torch.mean(residual * residual))

    added_mass = effective_inertia - rigid
    if nonnegative:
        added_mass = torch.clamp(added_mass, min=0.0)
    return DiagonalAddedMassDampingFit(added_mass, effective_inertia, linear, quadratic, residual_rms, sample_count)


def fit_full_matrix_added_mass_linear_quadratic_damping(
    time_s: torch.Tensor | Sequence[float],
    nu_r: torch.Tensor | Sequence[Sequence[float]],
    applied_wrench: torch.Tensor | Sequence[Sequence[float]],
    rigid_body_inertia: torch.Tensor | Sequence[float] | Sequence[Sequence[float]] | float,
    relative_acceleration: torch.Tensor | Sequence[Sequence[float]] | None = None,
    min_signal_norm: float = 1.0e-5,
    regularization: float = 0.0,
    symmetrize_added_mass: bool = True,
) -> FullMatrixAddedMassDampingFit:
    """Fit full effective inertia, added mass, and damping matrices."""

    nu, wrench, acceleration = _prepare_motion_fit_inputs(time_s, nu_r, applied_wrench, relative_acceleration)
    rigid = _as_6d_matrix(rigid_body_inertia, nu.device, nu.dtype, "rigid_body_inertia")
    finite_rows = (
        torch.all(torch.isfinite(nu), dim=1)
        & torch.all(torch.isfinite(acceleration), dim=1)
        & torch.all(torch.isfinite(wrench), dim=1)
    )
    design = torch.cat((acceleration, nu, torch.abs(nu) * nu), dim=1)
    energetic_rows = torch.linalg.norm(design, dim=1) >= float(min_signal_norm)
    mask = finite_rows & energetic_rows
    sample_count = int(torch.sum(mask).item())
    if sample_count == 0:
        raise ValueError("No valid samples remain for full-matrix added-mass fit.")

    design_fit = design[mask]
    wrench_fit = wrench[mask]
    coefficients = _solve_matrix_least_squares(design_fit, wrench_fit, float(regularization))
    effective = coefficients[:6, :].T.contiguous()
    linear = coefficients[6:12, :].T.contiguous()
    quadratic = coefficients[12:18, :].T.contiguous()
    added_mass = effective - rigid
    if symmetrize_added_mass:
        added_mass = 0.5 * (added_mass + added_mass.T)
    residual = design_fit @ coefficients - wrench_fit
    residual_rms = torch.sqrt(torch.mean(residual * residual, dim=0))

    return FullMatrixAddedMassDampingFit(
        added_mass,
        effective,
        linear,
        quadratic,
        residual_rms,
        sample_count,
        float(regularization),
        bool(symmetrize_added_mass),
    )


def fit_speed_dependent_damping_scales(
    time_s: torch.Tensor | Sequence[float],
    nu_r: torch.Tensor | Sequence[Sequence[float]],
    applied_wrench: torch.Tensor | Sequence[Sequence[float]],
    effective_mass: torch.Tensor | Sequence[float] | float,
    speed_points: torch.Tensor | Sequence[float],
    nominal_linear_damping: torch.Tensor | Sequence[float],
    nominal_quadratic_damping: torch.Tensor | Sequence[float],
    relative_acceleration: torch.Tensor | Sequence[Sequence[float]] | None = None,
    min_speed: float = 1.0e-5,
    nonnegative: bool = True,
) -> DampingSpeedScaleFit:
    """Fit per-speed-bin scales for existing linear and quadratic damping.

    Bins are centered on ``speed_points`` using midpoints as boundaries.  The
    returned ``(num_speed_points, 6)`` scales can be written directly to
    ``linear_damping_speed_scales`` and ``quadratic_damping_speed_scales``.
    """

    nu, wrench, acceleration, mass = _prepare_damping_fit_inputs(
        time_s,
        nu_r,
        applied_wrench,
        effective_mass,
        relative_acceleration,
    )
    speeds = torch.as_tensor(speed_points, dtype=nu.dtype, device=nu.device)
    if speeds.ndim != 1 or speeds.numel() < 2:
        raise ValueError("speed_points must be a 1D sequence with at least two samples.")
    if torch.any(speeds[1:] <= speeds[:-1]):
        raise ValueError("speed_points must be strictly increasing.")

    nominal_linear = _as_6_vector(nominal_linear_damping, nu.device, nu.dtype, "nominal_linear_damping")
    nominal_quadratic = _as_6_vector(nominal_quadratic_damping, nu.device, nu.dtype, "nominal_quadratic_damping")
    target = wrench - mass.reshape(1, 6) * acceleration

    num_bins = speeds.numel()
    linear_scales = torch.ones((num_bins, 6), dtype=nu.dtype, device=nu.device)
    quadratic_scales = torch.ones_like(linear_scales)
    residual_rms = torch.zeros_like(linear_scales)
    sample_count = torch.zeros((num_bins, 6), dtype=torch.long, device=nu.device)
    edges = (speeds[1:] + speeds[:-1]) * 0.5

    for bin_index in range(num_bins):
        lower = -torch.inf if bin_index == 0 else edges[bin_index - 1]
        upper = torch.inf if bin_index == num_bins - 1 else edges[bin_index]
        for dof in range(6):
            speed = torch.abs(nu[:, dof])
            signed_speed = nu[:, dof]
            mask = (
                torch.isfinite(speed)
                & torch.isfinite(target[:, dof])
                & (speed >= lower)
                & (speed < upper)
                & (speed >= float(min_speed))
            )
            sample_count[bin_index, dof] = int(torch.sum(mask).item())
            if sample_count[bin_index, dof] == 0:
                continue

            linear_column = nominal_linear[dof] * signed_speed[mask]
            quadratic_column = nominal_quadratic[dof] * speed[mask] * signed_speed[mask]
            design = torch.stack((linear_column, quadratic_column), dim=-1)
            beta = _solve_two_parameter_least_squares(design, target[mask, dof], nonnegative)
            if torch.any(torch.abs(linear_column) > 0.0):
                linear_scales[bin_index, dof] = beta[0]
            if torch.any(torch.abs(quadratic_column) > 0.0):
                quadratic_scales[bin_index, dof] = beta[1]
            residual = design @ beta - target[mask, dof]
            residual_rms[bin_index, dof] = torch.sqrt(torch.mean(residual * residual))

    return DampingSpeedScaleFit(linear_scales, quadratic_scales, residual_rms, sample_count)


def fit_thruster_static_lookup_table(
    command_samples: torch.Tensor | Sequence[float] | Sequence[Sequence[float]],
    thrust_samples: torch.Tensor | Sequence[float] | Sequence[Sequence[float]],
    command_points: torch.Tensor | Sequence[float],
    deadband_thrust_threshold: float = 0.05,
    min_samples_per_bin: int = 1,
) -> ThrusterStaticLookupFit:
    """Fit a measured command-to-thrust lookup table from thrust-stand samples."""

    commands, thrust = _prepare_thruster_sample_matrices(command_samples, thrust_samples)
    points = _as_increasing_1d_tensor(command_points, commands.device, commands.dtype, "command_points")
    bin_indices = _bin_indices_for_axis(commands, points)

    num_thrusters = thrust.shape[1]
    num_points = points.numel()
    table = torch.zeros((num_thrusters, num_points), dtype=thrust.dtype, device=thrust.device)
    residual_rms = torch.zeros_like(table)
    sample_count = torch.zeros((num_thrusters, num_points), dtype=torch.long, device=thrust.device)

    for thruster_index in range(num_thrusters):
        for point_index in range(num_points):
            mask = (
                (bin_indices[:, thruster_index] == point_index)
                & torch.isfinite(thrust[:, thruster_index])
                & torch.isfinite(commands[:, thruster_index])
            )
            count = int(torch.sum(mask).item())
            sample_count[thruster_index, point_index] = count
            if count < int(min_samples_per_bin):
                raise ValueError(
                    "Not enough samples for thruster static lookup bin "
                    f"thruster={thruster_index}, command_point={float(points[point_index])}."
                )
            table[thruster_index, point_index] = torch.mean(thrust[mask, thruster_index])
            residual = thrust[mask, thruster_index] - table[thruster_index, point_index]
            residual_rms[thruster_index, point_index] = torch.sqrt(torch.mean(residual * residual))

    deadband = _estimate_deadband_from_static_samples(commands, thrust, float(deadband_thrust_threshold))
    return ThrusterStaticLookupFit(points, table, residual_rms, sample_count, deadband)


def fit_thruster_inflow_lookup_table(
    command_samples: torch.Tensor | Sequence[float] | Sequence[Sequence[float]],
    inflow_speed_samples: torch.Tensor | Sequence[float] | Sequence[Sequence[float]],
    thrust_samples: torch.Tensor | Sequence[float] | Sequence[Sequence[float]],
    command_points: torch.Tensor | Sequence[float],
    inflow_speed_points: torch.Tensor | Sequence[float],
    min_samples_per_cell: int = 1,
) -> ThrusterInflowLookupFit:
    """Fit a command x axial-inflow thrust surface from measured samples."""

    commands, thrust = _prepare_thruster_sample_matrices(command_samples, thrust_samples)
    inflow, _ = _prepare_thruster_sample_matrices(inflow_speed_samples, thrust_samples)
    if inflow.shape != commands.shape:
        raise ValueError("inflow_speed_samples must broadcast to the same shape as command_samples.")

    command_axis = _as_increasing_1d_tensor(command_points, commands.device, commands.dtype, "command_points")
    inflow_axis = _as_increasing_1d_tensor(
        inflow_speed_points,
        commands.device,
        commands.dtype,
        "inflow_speed_points",
    )
    command_bins = _bin_indices_for_axis(commands, command_axis)
    inflow_bins = _bin_indices_for_axis(inflow, inflow_axis)

    num_thrusters = thrust.shape[1]
    table = torch.zeros(
        (num_thrusters, command_axis.numel(), inflow_axis.numel()),
        dtype=thrust.dtype,
        device=thrust.device,
    )
    residual_rms = torch.zeros_like(table)
    sample_count = torch.zeros(table.shape, dtype=torch.long, device=thrust.device)

    for thruster_index in range(num_thrusters):
        for command_index in range(command_axis.numel()):
            for inflow_index in range(inflow_axis.numel()):
                mask = (
                    (command_bins[:, thruster_index] == command_index)
                    & (inflow_bins[:, thruster_index] == inflow_index)
                    & torch.isfinite(thrust[:, thruster_index])
                    & torch.isfinite(commands[:, thruster_index])
                    & torch.isfinite(inflow[:, thruster_index])
                )
                count = int(torch.sum(mask).item())
                sample_count[thruster_index, command_index, inflow_index] = count
                if count < int(min_samples_per_cell):
                    raise ValueError(
                        "Not enough samples for thruster inflow lookup cell "
                        f"thruster={thruster_index}, command={float(command_axis[command_index])}, "
                        f"inflow={float(inflow_axis[inflow_index])}."
                    )
                table[thruster_index, command_index, inflow_index] = torch.mean(thrust[mask, thruster_index])
                residual = thrust[mask, thruster_index] - table[thruster_index, command_index, inflow_index]
                residual_rms[thruster_index, command_index, inflow_index] = torch.sqrt(torch.mean(residual * residual))

    return ThrusterInflowLookupFit(command_axis, inflow_axis, table, residual_rms, sample_count)


def fit_thruster_first_order_response(
    time_s: torch.Tensor | Sequence[float],
    measured_thrust: torch.Tensor | Sequence[float],
    command_step_time_s: float = 0.0,
    initial_thrust: float | None = None,
    steady_state_thrust: float | None = None,
    tail_fraction: float = 0.2,
    min_progress: float = 0.02,
    max_progress: float = 0.98,
    delay_candidate_count: int = 64,
) -> ThrusterFirstOrderFit:
    """Fit first-order thruster response ``tau`` and response delay from a step log."""

    time = torch.as_tensor(time_s, dtype=torch.float32)
    thrust = torch.as_tensor(measured_thrust, dtype=torch.float32)
    if time.ndim != 1 or thrust.ndim != 1 or time.shape[0] != thrust.shape[0]:
        raise ValueError("time_s and measured_thrust must be 1D tensors with matching length.")
    if time.numel() < 4:
        raise ValueError("At least four samples are required for first-order response fitting.")
    if torch.any(time[1:] <= time[:-1]):
        raise ValueError("time_s must be strictly increasing.")
    if not torch.all(torch.isfinite(thrust)):
        raise ValueError("measured_thrust must contain only finite values.")
    if not 0.0 < float(tail_fraction) <= 1.0:
        raise ValueError("tail_fraction must be in (0, 1].")

    command_step_time = float(command_step_time_s)
    before_step = time < command_step_time
    if initial_thrust is None:
        if torch.any(before_step):
            initial = torch.mean(thrust[before_step])
        else:
            initial = thrust[0]
    else:
        initial = torch.tensor(float(initial_thrust), dtype=thrust.dtype, device=thrust.device)

    if steady_state_thrust is None:
        tail_count = max(1, int(round(float(tail_fraction) * thrust.numel())))
        steady = torch.mean(thrust[-tail_count:])
    else:
        steady = torch.tensor(float(steady_state_thrust), dtype=thrust.dtype, device=thrust.device)

    amplitude = steady - initial
    if torch.abs(amplitude) <= 1.0e-6:
        raise ValueError("Step response amplitude is too small to fit.")
    progress = (thrust - initial) / amplitude
    valid_progress = torch.isfinite(progress) & (progress > float(min_progress)) & (progress < float(max_progress))
    after_step = time >= command_step_time
    fit_mask = valid_progress & after_step
    if int(torch.sum(fit_mask).item()) < 2:
        raise ValueError("Not enough samples in the first-order response fitting window.")

    first_motion_time = torch.min(time[fit_mask])
    delay_upper = max(command_step_time, float(first_motion_time.item()))
    candidates = torch.linspace(
        command_step_time,
        delay_upper,
        max(1, int(delay_candidate_count)),
        dtype=time.dtype,
        device=time.device,
    )

    best_delay = candidates[0]
    best_tau = torch.tensor(float("inf"), dtype=time.dtype, device=time.device)
    best_error = torch.tensor(float("inf"), dtype=time.dtype, device=time.device)
    best_count = 0
    for candidate_delay in candidates:
        x = time[fit_mask] - candidate_delay
        candidate_mask = x > 0.0
        if int(torch.sum(candidate_mask).item()) < 2:
            continue
        x = x[candidate_mask]
        y = -torch.log(torch.clamp(1.0 - progress[fit_mask][candidate_mask], min=1.0e-6))
        denom = torch.sum(x * x)
        if denom <= 0.0:
            continue
        slope = torch.sum(x * y) / denom
        if slope <= 0.0:
            continue
        tau = 1.0 / slope
        predicted_progress = torch.where(
            time <= candidate_delay,
            torch.zeros_like(time),
            1.0 - torch.exp(-(time - candidate_delay) / tau),
        )
        predicted = initial + amplitude * predicted_progress
        residual = predicted[after_step] - thrust[after_step]
        error = torch.mean(residual * residual)
        if error < best_error:
            best_error = error
            best_delay = candidate_delay
            best_tau = tau
            best_count = int(torch.sum(candidate_mask).item())

    if not torch.isfinite(best_tau):
        raise ValueError("Could not fit a positive thruster time constant.")

    residual_rms = torch.sqrt(best_error)
    return ThrusterFirstOrderFit(
        float(best_tau.item()),
        max(0.0, float((best_delay - command_step_time).item())),
        float(initial.item()),
        float(steady.item()),
        float(residual_rms.item()),
        best_count,
    )


def fit_thruster_voltage_exponent(
    voltage_samples: torch.Tensor | Sequence[float],
    thrust_scale_samples: torch.Tensor | Sequence[float],
    nominal_voltage: float,
) -> ThrusterVoltageExponentFit:
    """Fit the exponent in ``thrust_scale = (voltage / nominal_voltage) ** exponent``."""

    voltage = torch.as_tensor(voltage_samples, dtype=torch.float32)
    scale = torch.as_tensor(thrust_scale_samples, dtype=torch.float32)
    if voltage.shape != scale.shape:
        raise ValueError("voltage_samples and thrust_scale_samples must have matching shapes.")
    if float(nominal_voltage) <= 0.0:
        raise ValueError("nominal_voltage must be positive.")
    valid = (
        torch.isfinite(voltage)
        & torch.isfinite(scale)
        & (voltage > 0.0)
        & (scale > 0.0)
        & (torch.abs(voltage - float(nominal_voltage)) > 1.0e-6)
    )
    sample_count = int(torch.sum(valid).item())
    if sample_count == 0:
        raise ValueError("No valid non-nominal voltage samples remain.")

    x = torch.log(voltage[valid] / float(nominal_voltage))
    y = torch.log(scale[valid])
    denom = torch.sum(x * x)
    if denom <= 0.0:
        raise ValueError("Voltage samples do not span a usable range.")
    exponent = torch.sum(x * y) / denom
    predicted = torch.pow(voltage[valid] / float(nominal_voltage), exponent)
    residual = predicted - scale[valid]
    residual_rms = torch.sqrt(torch.mean(residual * residual))

    return ThrusterVoltageExponentFit(
        float(nominal_voltage),
        float(exponent.item()),
        float(residual_rms.item()),
        sample_count,
    )


def fit_tether_spring_damper(
    tether_length_samples: torch.Tensor | Sequence[float],
    tension_samples: torch.Tensor | Sequence[float],
    velocity_along_tether_samples: torch.Tensor | Sequence[float] | None = None,
    slack_length_candidates: torch.Tensor | Sequence[float] | None = None,
    min_stretch: float = 1.0e-5,
    nonnegative: bool = True,
) -> TetherSpringDamperFit:
    """Fit slack length, spring stiffness, and damping for the tether model.

    ``velocity_along_tether_samples`` uses the runtime convention:
    ``body_velocity dot direction_to_anchor``.  Negative values mean the
    vehicle is moving away from the anchor and increase damping tension.
    """

    length = torch.as_tensor(tether_length_samples, dtype=torch.float32).reshape(-1)
    tension = torch.as_tensor(tension_samples, dtype=torch.float32).reshape(-1)
    if length.shape != tension.shape:
        raise ValueError("tether_length_samples and tension_samples must have matching shapes.")
    if velocity_along_tether_samples is None:
        velocity = torch.zeros_like(length)
    else:
        velocity = torch.as_tensor(velocity_along_tether_samples, dtype=length.dtype, device=length.device).reshape(-1)
        if velocity.shape != length.shape:
            raise ValueError("velocity_along_tether_samples must match tether_length_samples.")
    finite = torch.isfinite(length) & torch.isfinite(tension) & torch.isfinite(velocity)
    finite = finite & (length >= 0.0) & (tension >= 0.0)
    if int(torch.sum(finite).item()) < 2:
        raise ValueError("At least two finite tether samples are required.")
    length = length[finite]
    tension = tension[finite]
    velocity = velocity[finite]

    if slack_length_candidates is None:
        lower = torch.min(length)
        upper = torch.max(length)
        candidates = torch.linspace(lower, upper, 64, dtype=length.dtype, device=length.device)
    else:
        candidates = torch.as_tensor(slack_length_candidates, dtype=length.dtype, device=length.device).reshape(-1)
        if candidates.numel() == 0:
            raise ValueError("slack_length_candidates must not be empty.")
        if not torch.all(torch.isfinite(candidates)):
            raise ValueError("slack_length_candidates must contain only finite values.")

    damping_column = torch.clamp(-velocity, min=0.0)
    best_error = torch.tensor(float("inf"), dtype=length.dtype, device=length.device)
    best_slack = candidates[0]
    best_beta = torch.zeros(2, dtype=length.dtype, device=length.device)
    best_count = 0
    for slack in candidates:
        stretch = torch.clamp(length - slack, min=0.0)
        active = (stretch >= float(min_stretch)) | (damping_column > 0.0)
        if int(torch.sum(active).item()) < 2:
            continue
        design = torch.stack((stretch[active], damping_column[active]), dim=-1)
        beta = _solve_nonnegative_least_squares_by_subset(design, tension[active], nonnegative)
        residual = design @ beta - tension[active]
        error = torch.mean(residual * residual)
        if error < best_error:
            best_error = error
            best_slack = slack
            best_beta = beta
            best_count = int(torch.sum(active).item())

    if not torch.isfinite(best_error):
        raise ValueError("Could not fit tether spring-damper parameters.")
    return TetherSpringDamperFit(
        float(best_slack.item()),
        float(best_beta[0].item()),
        float(best_beta[1].item()),
        float(torch.sqrt(best_error).item()),
        best_count,
    )


def fit_tether_drag_coefficient(
    relative_velocity_w_samples: torch.Tensor | Sequence[Sequence[float]],
    drag_force_w_samples: torch.Tensor | Sequence[Sequence[float]],
    nonnegative: bool = True,
) -> TetherDragFit:
    """Fit ``drag_coeff`` for ``force = -drag_coeff * ||v_rel|| * v_rel``."""

    velocity = torch.as_tensor(relative_velocity_w_samples, dtype=torch.float32)
    force = torch.as_tensor(drag_force_w_samples, dtype=torch.float32)
    if velocity.ndim != 2 or velocity.shape[1] != 3:
        raise ValueError("relative_velocity_w_samples must have shape (N, 3).")
    if force.shape != velocity.shape:
        raise ValueError("drag_force_w_samples must have the same shape as relative_velocity_w_samples.")
    finite = torch.all(torch.isfinite(velocity), dim=1) & torch.all(torch.isfinite(force), dim=1)
    speed = torch.linalg.norm(velocity, dim=1, keepdim=True)
    design_vector = -speed * velocity
    energetic = torch.linalg.norm(design_vector, dim=1) > 1.0e-8
    mask = finite & energetic
    sample_count = int(torch.sum(mask).item())
    if sample_count == 0:
        raise ValueError("No finite nonzero relative-velocity samples remain for tether drag fitting.")

    design = design_vector[mask].reshape(-1)
    target = force[mask].reshape(-1)
    denom = torch.sum(design * design)
    coeff = torch.sum(design * target) / denom
    if nonnegative:
        coeff = torch.clamp(coeff, min=0.0)
    residual = design * coeff - target
    return TetherDragFit(
        float(coeff.item()),
        float(torch.sqrt(torch.mean(residual * residual)).item()),
        sample_count,
    )


def fit_water_current_process(
    time_s: torch.Tensor | Sequence[float],
    water_current_w: torch.Tensor | Sequence[Sequence[float]],
    mean_current_w: torch.Tensor | Sequence[float] | None = None,
) -> WaterCurrentProcessFit:
    """Estimate smooth-current process parameters from measured water velocity.

    ``water_current_w`` is expected to be an ``(N, 3)`` world-frame current log
    from a drift marker, ADV, DVL water-track estimate, or equivalent source.
    """

    time = torch.as_tensor(time_s, dtype=torch.float32)
    currents = torch.as_tensor(water_current_w, dtype=torch.float32)
    if time.ndim != 1 or time.numel() < 2:
        raise ValueError("time_s must be a 1D sequence with at least two samples.")
    if torch.any(time[1:] <= time[:-1]):
        raise ValueError("time_s must be strictly increasing.")
    if currents.ndim != 2 or currents.shape != (time.numel(), 3):
        raise ValueError(f"water_current_w must have shape ({time.numel()}, 3), got {tuple(currents.shape)}.")

    finite_rows = torch.all(torch.isfinite(currents), dim=1)
    if torch.sum(finite_rows) < 2:
        raise ValueError("water_current_w must contain at least two finite samples.")
    time = time[finite_rows]
    currents = currents[finite_rows]

    if mean_current_w is None:
        mean_current = torch.mean(currents, dim=0)
    else:
        mean_current = torch.as_tensor(mean_current_w, dtype=currents.dtype, device=currents.device)
        if mean_current.shape != (3,):
            raise ValueError(f"mean_current_w must have shape (3,), got {tuple(mean_current.shape)}.")
    residual = currents - mean_current.reshape(1, 3)
    residual_std = torch.sqrt(torch.mean(residual * residual, dim=0))

    previous = residual[:-1]
    current = residual[1:]
    denom = torch.sum(previous * previous)
    if denom <= 0.0:
        estimated_alpha = 0.0
        tau_s = 0.0
    else:
        estimated_alpha = float(torch.sum(previous * current) / denom)
        estimated_alpha = max(0.0, min(estimated_alpha, 0.999999))
        if estimated_alpha <= 0.0:
            tau_s = 0.0
        else:
            mean_dt = float(torch.mean(time[1:] - time[:-1]))
            tau_s = -mean_dt / torch.log(torch.tensor(estimated_alpha)).item()

    horizontal_norm = torch.linalg.norm(currents[:, 0:2], dim=1)
    variation_std = float(torch.mean(residual_std[0:2]))
    return WaterCurrentProcessFit(
        mean_current,
        residual_std,
        tau_s,
        estimated_alpha,
        variation_std,
        float(torch.max(horizontal_norm)),
        float(torch.max(torch.abs(currents[:, 2]))),
        int(currents.shape[0]),
    )


def fit_water_current_field_grid(
    sample_positions: torch.Tensor | Sequence[Sequence[float]],
    sample_currents_w: torch.Tensor | Sequence[Sequence[float]],
    grid_shape: Sequence[int],
    bounds: torch.Tensor | Sequence[float] | None = None,
    k_neighbors: int = 8,
    interpolation_power: float = 2.0,
    exact_distance: float = 1.0e-6,
) -> WaterCurrentFieldFit:
    """Build a regular pool-local current grid from scattered measurements.

    The output order matches ``calculate_trilinear_current_field``:
    ``(nx, ny, nz, 3)`` internally and flattened in x-major, then y, then z
    order when exported to cfg updates.
    """

    positions = torch.as_tensor(sample_positions, dtype=torch.float32)
    currents = torch.as_tensor(sample_currents_w, dtype=torch.float32)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"sample_positions must have shape (N, 3), got {tuple(positions.shape)}.")
    if currents.shape != positions.shape:
        raise ValueError(f"sample_currents_w must have shape {tuple(positions.shape)}, got {tuple(currents.shape)}.")
    finite_rows = torch.all(torch.isfinite(positions), dim=1) & torch.all(torch.isfinite(currents), dim=1)
    if torch.sum(finite_rows) == 0:
        raise ValueError("At least one finite current sample is required.")
    positions = positions[finite_rows]
    currents = currents[finite_rows]

    shape = _validate_grid_shape(grid_shape)
    if bounds is None:
        bounds_tensor = _bounds_from_positions(positions)
    else:
        bounds_tensor = torch.as_tensor(bounds, dtype=positions.dtype, device=positions.device)
        _validate_bounds(bounds_tensor)

    k = min(max(1, int(k_neighbors)), positions.shape[0])
    power = float(interpolation_power)
    if power <= 0.0:
        raise ValueError("interpolation_power must be positive.")

    grid_nodes = _regular_grid_nodes(bounds_tensor, shape)
    grid_values = torch.zeros((grid_nodes.shape[0], 3), dtype=positions.dtype, device=positions.device)
    for node_index, node in enumerate(grid_nodes):
        distances = torch.linalg.norm(positions - node.reshape(1, 3), dim=1)
        exact_mask = distances <= float(exact_distance)
        if torch.any(exact_mask):
            grid_values[node_index] = torch.mean(currents[exact_mask], dim=0)
            continue

        nearest_distances, nearest_indices = torch.topk(distances, k=k, largest=False)
        weights = 1.0 / torch.clamp(nearest_distances, min=float(exact_distance)) ** power
        grid_values[node_index] = torch.sum(currents[nearest_indices] * weights.reshape(-1, 1), dim=0) / torch.sum(weights)

    return WaterCurrentFieldFit(
        bounds_tensor,
        shape,
        grid_values.reshape(*shape, 3),
        int(positions.shape[0]),
        k,
        power,
    )


def fit_pool_boundary_effect_scales(
    sample_positions: torch.Tensor | Sequence[Sequence[float]],
    bounds: torch.Tensor | Sequence[float],
    effect_distance: float,
    damping_scale_samples: torch.Tensor | Sequence[float] | Sequence[Sequence[float]] | None = None,
    added_mass_scale_samples: torch.Tensor | Sequence[float] | Sequence[Sequence[float]] | None = None,
    thrust_scale_samples: torch.Tensor | Sequence[float] | Sequence[Sequence[float]] | None = None,
    min_proximity: float = 1.0e-4,
) -> PoolBoundaryScaleFit:
    """Fit empirical near-wall scale factors from repeated pool experiments.

    The samples should be measured ratios against an open-water baseline, for
    example ``D_near_wall / D_open_water`` or
    ``thrust_near_wall / thrust_open_water`` at each logged position.
    """

    positions = _as_3d_position_samples(sample_positions, "sample_positions")
    bounds_tensor = torch.as_tensor(bounds, dtype=positions.dtype, device=positions.device)
    _validate_bounds(bounds_tensor)
    proximity = _calculate_pool_boundary_proximity(positions, bounds_tensor, effect_distance)

    damping = _fit_single_proximity_scale(
        proximity,
        damping_scale_samples,
        "damping_scale_samples",
        min_proximity=min_proximity,
        lower_bound=1.0e-6,
        default_scale=1.0,
    )
    added_mass = _fit_single_proximity_scale(
        proximity,
        added_mass_scale_samples,
        "added_mass_scale_samples",
        min_proximity=min_proximity,
        lower_bound=1.0e-6,
        default_scale=1.0,
    )
    thrust = _fit_single_proximity_scale(
        proximity,
        thrust_scale_samples,
        "thrust_scale_samples",
        min_proximity=min_proximity,
        lower_bound=0.0,
        default_scale=1.0,
    )

    return PoolBoundaryScaleFit(
        bounds_tensor,
        float(effect_distance),
        damping.scale_at_boundary,
        added_mass.scale_at_boundary,
        thrust.scale_at_boundary,
        torch.tensor(
            [damping.residual_rms, added_mass.residual_rms, thrust.residual_rms],
            dtype=positions.dtype,
            device=positions.device,
        ),
        torch.tensor(
            [damping.sample_count, added_mass.sample_count, thrust.sample_count],
            dtype=torch.long,
            device=positions.device,
        ),
    )


def fit_free_surface_effect_scales(
    sample_positions: torch.Tensor | Sequence[Sequence[float]],
    surface_z: float,
    effect_distance: float,
    heave_damping_scale_samples: torch.Tensor | Sequence[float] | Sequence[Sequence[float]] | None = None,
    roll_pitch_damping_scale_samples: torch.Tensor | Sequence[float] | Sequence[Sequence[float]] | None = None,
    added_mass_scale_samples: torch.Tensor | Sequence[float] | Sequence[Sequence[float]] | None = None,
    buoyancy_scale_samples: torch.Tensor | Sequence[float] | Sequence[Sequence[float]] | None = None,
    thrust_scale_samples: torch.Tensor | Sequence[float] | Sequence[Sequence[float]] | None = None,
    min_proximity: float = 1.0e-4,
) -> FreeSurfaceScaleFit:
    """Fit flat free-surface empirical scale factors from depth sweeps."""

    positions = _as_3d_position_samples(sample_positions, "sample_positions")
    proximity = _calculate_free_surface_proximity(positions, surface_z, effect_distance)

    heave_damping = _fit_single_proximity_scale(
        proximity,
        heave_damping_scale_samples,
        "heave_damping_scale_samples",
        min_proximity=min_proximity,
        lower_bound=1.0e-6,
        default_scale=1.0,
    )
    roll_pitch_damping = _fit_single_proximity_scale(
        proximity,
        roll_pitch_damping_scale_samples,
        "roll_pitch_damping_scale_samples",
        min_proximity=min_proximity,
        lower_bound=1.0e-6,
        default_scale=1.0,
    )
    added_mass = _fit_single_proximity_scale(
        proximity,
        added_mass_scale_samples,
        "added_mass_scale_samples",
        min_proximity=min_proximity,
        lower_bound=1.0e-6,
        default_scale=1.0,
    )
    buoyancy = _fit_single_proximity_scale(
        proximity,
        buoyancy_scale_samples,
        "buoyancy_scale_samples",
        min_proximity=min_proximity,
        lower_bound=0.0,
        default_scale=1.0,
    )
    thrust = _fit_single_proximity_scale(
        proximity,
        thrust_scale_samples,
        "thrust_scale_samples",
        min_proximity=min_proximity,
        lower_bound=0.0,
        default_scale=1.0,
    )

    return FreeSurfaceScaleFit(
        float(surface_z),
        float(effect_distance),
        heave_damping.scale_at_boundary,
        roll_pitch_damping.scale_at_boundary,
        added_mass.scale_at_boundary,
        buoyancy.scale_at_boundary,
        thrust.scale_at_boundary,
        torch.tensor(
            [
                heave_damping.residual_rms,
                roll_pitch_damping.residual_rms,
                added_mass.residual_rms,
                buoyancy.residual_rms,
                thrust.residual_rms,
            ],
            dtype=positions.dtype,
            device=positions.device,
        ),
        torch.tensor(
            [
                heave_damping.sample_count,
                roll_pitch_damping.sample_count,
                added_mass.sample_count,
                buoyancy.sample_count,
                thrust.sample_count,
            ],
            dtype=torch.long,
            device=positions.device,
        ),
    )


def _prepare_damping_fit_inputs(
    time_s: torch.Tensor | Sequence[float],
    nu_r: torch.Tensor | Sequence[Sequence[float]],
    applied_wrench: torch.Tensor | Sequence[Sequence[float]],
    effective_mass: torch.Tensor | Sequence[float] | float,
    relative_acceleration: torch.Tensor | Sequence[Sequence[float]] | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    nu = torch.as_tensor(nu_r, dtype=torch.float32)
    if nu.ndim != 2 or nu.shape[1] != 6:
        raise ValueError(f"nu_r must have shape (N, 6), got {tuple(nu.shape)}.")
    wrench = torch.as_tensor(applied_wrench, dtype=nu.dtype, device=nu.device)
    if wrench.shape != nu.shape:
        raise ValueError(f"applied_wrench must have shape {tuple(nu.shape)}, got {tuple(wrench.shape)}.")
    if relative_acceleration is None:
        acceleration = finite_difference(time_s, nu).to(device=nu.device, dtype=nu.dtype)
    else:
        acceleration = torch.as_tensor(relative_acceleration, dtype=nu.dtype, device=nu.device)
        if acceleration.shape != nu.shape:
            raise ValueError(
                f"relative_acceleration must have shape {tuple(nu.shape)}, got {tuple(acceleration.shape)}."
            )
    mass = _as_6_vector(effective_mass, nu.device, nu.dtype, "effective_mass")
    return nu, wrench, acceleration, mass


def _prepare_motion_fit_inputs(
    time_s: torch.Tensor | Sequence[float],
    nu_r: torch.Tensor | Sequence[Sequence[float]],
    applied_wrench: torch.Tensor | Sequence[Sequence[float]],
    relative_acceleration: torch.Tensor | Sequence[Sequence[float]] | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    nu = torch.as_tensor(nu_r, dtype=torch.float32)
    if nu.ndim != 2 or nu.shape[1] != 6:
        raise ValueError(f"nu_r must have shape (N, 6), got {tuple(nu.shape)}.")
    wrench = torch.as_tensor(applied_wrench, dtype=nu.dtype, device=nu.device)
    if wrench.shape != nu.shape:
        raise ValueError(f"applied_wrench must have shape {tuple(nu.shape)}, got {tuple(wrench.shape)}.")
    if relative_acceleration is None:
        acceleration = finite_difference(time_s, nu).to(device=nu.device, dtype=nu.dtype)
    else:
        acceleration = torch.as_tensor(relative_acceleration, dtype=nu.dtype, device=nu.device)
        if acceleration.shape != nu.shape:
            raise ValueError(
                f"relative_acceleration must have shape {tuple(nu.shape)}, got {tuple(acceleration.shape)}."
            )
    return nu, wrench, acceleration


def _validate_grid_shape(grid_shape: Sequence[int]) -> tuple[int, int, int]:
    if len(grid_shape) != 3:
        raise ValueError("grid_shape must contain three entries.")
    shape = tuple(int(value) for value in grid_shape)
    if any(value <= 0 for value in shape):
        raise ValueError("grid_shape entries must be positive.")
    if any(float(raw) != float(value) for raw, value in zip(grid_shape, shape)):
        raise ValueError("grid_shape entries must be integers.")
    return shape


def _validate_bounds(bounds: torch.Tensor) -> None:
    if bounds.shape != (6,):
        raise ValueError(f"bounds must have shape (6,), got {tuple(bounds.shape)}.")
    if not (bounds[0] < bounds[1] and bounds[2] < bounds[3] and bounds[4] < bounds[5]):
        raise ValueError("bounds must be ordered as min < max on each axis.")


def _bounds_from_positions(positions: torch.Tensor) -> torch.Tensor:
    lower = torch.min(positions, dim=0).values
    upper = torch.max(positions, dim=0).values
    extent = upper - lower
    padding = torch.where(extent > 0.0, 0.05 * extent, torch.full_like(extent, 1.0e-3))
    return torch.stack(
        (
            lower[0] - padding[0],
            upper[0] + padding[0],
            lower[1] - padding[1],
            upper[1] + padding[1],
            lower[2] - padding[2],
            upper[2] + padding[2],
        )
    )


def _regular_grid_nodes(bounds: torch.Tensor, shape: tuple[int, int, int]) -> torch.Tensor:
    xs = torch.linspace(bounds[0], bounds[1], shape[0], dtype=bounds.dtype, device=bounds.device)
    ys = torch.linspace(bounds[2], bounds[3], shape[1], dtype=bounds.dtype, device=bounds.device)
    zs = torch.linspace(bounds[4], bounds[5], shape[2], dtype=bounds.dtype, device=bounds.device)
    x_grid, y_grid, z_grid = torch.meshgrid(xs, ys, zs, indexing="ij")
    return torch.stack((x_grid, y_grid, z_grid), dim=-1).reshape(-1, 3)


def _as_1d_float_samples(value: torch.Tensor | Sequence[float] | float, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(value)
    if not torch.is_floating_point(tensor):
        tensor = tensor.to(dtype=torch.float32)
    tensor = tensor.reshape(-1)
    if tensor.numel() == 0:
        raise ValueError(f"{name} must contain at least one sample.")
    if not torch.all(torch.isfinite(tensor)):
        raise ValueError(f"{name} must contain only finite values.")
    return tensor


def _as_3d_vector_samples(value: torch.Tensor | Sequence[float] | Sequence[Sequence[float]], name: str) -> torch.Tensor:
    tensor = torch.as_tensor(value)
    if not torch.is_floating_point(tensor):
        tensor = tensor.to(dtype=torch.float32)
    if tensor.ndim == 1 and tensor.shape[0] == 3:
        tensor = tensor.reshape(1, 3)
    if tensor.ndim != 2 or tensor.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3) or (3,), got {tuple(tensor.shape)}.")
    if not torch.all(torch.isfinite(tensor)):
        raise ValueError(f"{name} must contain only finite values.")
    return tensor


def _as_3d_axis_samples(value: torch.Tensor | Sequence[Sequence[float]], name: str) -> torch.Tensor:
    axes = _as_3d_vector_samples(value, name)
    norm = torch.linalg.norm(axes, dim=-1, keepdim=True)
    if torch.any(norm <= 0.0):
        raise ValueError(f"{name} contains a zero axis.")
    return axes / norm


def _as_quat_wxyz(value: torch.Tensor | Sequence[float] | Sequence[Sequence[float]], name: str) -> torch.Tensor:
    quat = torch.as_tensor(value)
    if not torch.is_floating_point(quat):
        quat = quat.to(dtype=torch.float32)
    if quat.ndim == 1 and quat.shape[0] == 4:
        quat = quat.reshape(1, 4)
    if quat.ndim != 2 or quat.shape[1] != 4:
        raise ValueError(f"{name} must have shape (N, 4) or (4,), got {tuple(quat.shape)}.")
    if not torch.all(torch.isfinite(quat)):
        raise ValueError(f"{name} must contain only finite values.")
    norm = torch.linalg.norm(quat, dim=-1, keepdim=True)
    if torch.any(norm <= 0.0):
        raise ValueError(f"{name} contains a zero quaternion.")
    return quat / norm


def _quat_conjugate_wxyz(quat: torch.Tensor) -> torch.Tensor:
    return torch.cat((quat[:, 0:1], -quat[:, 1:]), dim=-1)


def _quat_apply_wxyz(quat: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    quat_xyz = quat[:, 1:]
    quat_w = quat[:, 0:1]
    uv = torch.cross(quat_xyz, vector, dim=-1)
    uuv = torch.cross(quat_xyz, uv, dim=-1)
    return vector + 2.0 * (quat_w * uv + uuv)


def _skew_3d(vec: torch.Tensor) -> torch.Tensor:
    mat = torch.zeros((vec.shape[0], 3, 3), dtype=vec.dtype, device=vec.device)
    mat[:, 0, 1] = -vec[:, 2]
    mat[:, 0, 2] = vec[:, 1]
    mat[:, 1, 0] = vec[:, 2]
    mat[:, 1, 2] = -vec[:, 0]
    mat[:, 2, 0] = -vec[:, 1]
    mat[:, 2, 1] = vec[:, 0]
    return mat


def _prepare_thruster_sample_matrices(
    command_samples: torch.Tensor | Sequence[float] | Sequence[Sequence[float]],
    thrust_samples: torch.Tensor | Sequence[float] | Sequence[Sequence[float]],
) -> tuple[torch.Tensor, torch.Tensor]:
    commands = torch.as_tensor(command_samples)
    thrust = torch.as_tensor(thrust_samples)
    if not torch.is_floating_point(commands):
        commands = commands.to(dtype=torch.float32)
    if not torch.is_floating_point(thrust):
        thrust = thrust.to(dtype=torch.float32)

    if commands.ndim == 1:
        commands = commands.reshape(-1, 1)
    if thrust.ndim == 1:
        thrust = thrust.reshape(-1, 1)
    if commands.ndim != 2 or thrust.ndim != 2:
        raise ValueError("thruster samples must be shaped (N,) or (N, num_thrusters).")
    if commands.shape[0] != thrust.shape[0]:
        raise ValueError("thruster sample arrays must have matching first dimension.")
    if commands.shape[1] == 1 and thrust.shape[1] > 1:
        commands = commands.repeat(1, thrust.shape[1])
    elif thrust.shape[1] == 1 and commands.shape[1] > 1:
        thrust = thrust.repeat(1, commands.shape[1])
    if commands.shape != thrust.shape:
        raise ValueError(f"thruster samples could not be broadcast together: {tuple(commands.shape)} vs {tuple(thrust.shape)}.")
    if not torch.all(torch.isfinite(commands)) or not torch.all(torch.isfinite(thrust)):
        raise ValueError("thruster samples must contain only finite values.")
    return commands, thrust.to(device=commands.device, dtype=commands.dtype)


def _as_increasing_1d_tensor(
    value: torch.Tensor | Sequence[float],
    device: torch.device,
    dtype: torch.dtype,
    name: str,
) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=dtype, device=device)
    if tensor.ndim != 1 or tensor.numel() < 2:
        raise ValueError(f"{name} must be a 1D sequence with at least two samples.")
    if not torch.all(torch.isfinite(tensor)):
        raise ValueError(f"{name} must contain only finite values.")
    if torch.any(tensor[1:] <= tensor[:-1]):
        raise ValueError(f"{name} must be strictly increasing.")
    return tensor


def _bin_indices_for_axis(samples: torch.Tensor, axis_points: torch.Tensor) -> torch.Tensor:
    edges = 0.5 * (axis_points[1:] + axis_points[:-1])
    indices = torch.bucketize(samples.contiguous(), edges)
    return torch.clamp(indices, min=0, max=axis_points.numel() - 1)


def _estimate_deadband_from_static_samples(
    commands: torch.Tensor,
    thrust: torch.Tensor,
    thrust_threshold: float,
) -> float:
    if float(thrust_threshold) < 0.0:
        raise ValueError("deadband_thrust_threshold must be non-negative.")
    inactive = torch.abs(thrust) <= float(thrust_threshold)
    if not torch.any(inactive):
        return 0.0
    return float(torch.max(torch.abs(commands[inactive])).item())


def _calculate_pool_boundary_proximity(
    positions: torch.Tensor,
    bounds: torch.Tensor,
    effect_distance: float,
) -> torch.Tensor:
    if float(effect_distance) <= 0.0:
        raise ValueError("effect_distance must be positive.")
    distance = torch.stack(
        (
            positions[:, 0] - bounds[0],
            bounds[1] - positions[:, 0],
            positions[:, 1] - bounds[2],
            bounds[3] - positions[:, 1],
            positions[:, 2] - bounds[4],
            bounds[5] - positions[:, 2],
        ),
        dim=-1,
    )
    min_distance = torch.min(distance, dim=-1).values
    return torch.clamp((float(effect_distance) - min_distance) / float(effect_distance), min=0.0, max=1.0)


def _calculate_free_surface_proximity(
    positions: torch.Tensor,
    surface_z: float,
    effect_distance: float,
) -> torch.Tensor:
    if float(effect_distance) <= 0.0:
        raise ValueError("effect_distance must be positive.")
    distance = torch.abs(positions[:, 2] - float(surface_z))
    proximity = torch.clamp((float(effect_distance) - distance) / float(effect_distance), min=0.0, max=1.0)
    return proximity * proximity * (3.0 - 2.0 * proximity)


def _fit_single_proximity_scale(
    proximity: torch.Tensor,
    measured_scale: torch.Tensor | Sequence[float] | Sequence[Sequence[float]] | None,
    name: str,
    *,
    min_proximity: float,
    lower_bound: float | None,
    default_scale: float,
) -> _ProximityScaleEstimate:
    if measured_scale is None:
        return _ProximityScaleEstimate(float(default_scale), 0.0, 0)
    if float(min_proximity) < 0.0:
        raise ValueError("min_proximity must be non-negative.")

    samples = torch.as_tensor(measured_scale, dtype=proximity.dtype, device=proximity.device)
    if samples.ndim == 0:
        raise ValueError(f"{name} must have shape (N,) or (N, K).")
    if samples.shape[0] != proximity.shape[0]:
        raise ValueError(f"{name} first dimension must match sample_positions.")
    if samples.ndim == 1:
        samples = samples.reshape(-1, 1)
    elif samples.ndim > 2:
        samples = samples.reshape(samples.shape[0], -1)
    if not torch.all(torch.isfinite(samples)):
        raise ValueError(f"{name} must contain only finite values.")

    design = proximity.reshape(-1, 1).expand_as(samples)
    valid = torch.isfinite(samples) & torch.isfinite(design) & (design >= float(min_proximity))
    sample_count = int(torch.sum(valid).item())
    if sample_count == 0:
        raise ValueError(f"{name} has no samples inside the modeled effect distance.")

    design_flat = design[valid]
    target_flat = samples[valid] - 1.0
    denom = torch.sum(design_flat * design_flat)
    if denom <= 0.0:
        raise ValueError(f"{name} has zero proximity energy.")
    scale = 1.0 + torch.sum(design_flat * target_flat) / denom
    if lower_bound is not None:
        scale = torch.clamp(scale, min=float(lower_bound))

    residual = 1.0 + design_flat * (scale - 1.0) - samples[valid]
    residual_rms = torch.sqrt(torch.mean(residual * residual))
    return _ProximityScaleEstimate(float(scale.item()), float(residual_rms.item()), sample_count)


def _as_3d_position_samples(value: torch.Tensor | Sequence[Sequence[float]], name: str) -> torch.Tensor:
    tensor = torch.as_tensor(value)
    if not torch.is_floating_point(tensor):
        tensor = tensor.to(dtype=torch.float32)
    if tensor.ndim != 2 or tensor.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3), got {tuple(tensor.shape)}.")
    if not torch.all(torch.isfinite(tensor)):
        raise ValueError(f"{name} must contain only finite values.")
    return tensor


def _as_square_matrix(value: torch.Tensor | Sequence[Sequence[float]], name: str) -> torch.Tensor:
    tensor = torch.as_tensor(value)
    if not torch.is_floating_point(tensor):
        tensor = tensor.to(dtype=torch.float32)
    if tensor.ndim != 2 or tensor.shape[0] != tensor.shape[1]:
        raise ValueError(f"{name} must be a square matrix, got shape {tuple(tensor.shape)}.")
    if not torch.all(torch.isfinite(tensor)):
        raise ValueError(f"{name} must contain only finite values.")
    return tensor


def _as_6d_motion_samples(value: torch.Tensor | Sequence[Sequence[float]], name: str) -> torch.Tensor:
    tensor = torch.as_tensor(value)
    if not torch.is_floating_point(tensor):
        tensor = tensor.to(dtype=torch.float32)
    if tensor.ndim == 1 and tensor.shape[0] == 6:
        tensor = tensor.reshape(1, 6)
    if tensor.ndim != 2 or tensor.shape[1] != 6:
        raise ValueError(f"{name} must have shape (N, 6) or (6,), got {tuple(tensor.shape)}.")
    if not torch.all(torch.isfinite(tensor)):
        raise ValueError(f"{name} must contain only finite values.")
    return tensor


def _as_6_vector(value: torch.Tensor | Sequence[float] | float, device: torch.device, dtype: torch.dtype, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=dtype, device=device)
    if not torch.all(torch.isfinite(tensor)):
        raise ValueError(f"{name} must contain only finite values.")
    if tensor.ndim == 0:
        return tensor.reshape(1).repeat(6)
    if tensor.ndim == 1 and tensor.shape[0] == 6:
        return tensor
    raise ValueError(f"{name} must be a scalar or 6-vector, got shape {tuple(tensor.shape)}.")


def _as_6d_matrix(
    value: torch.Tensor | Sequence[float] | Sequence[Sequence[float]] | float,
    device: torch.device,
    dtype: torch.dtype,
    name: str,
) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=dtype, device=device)
    if not torch.all(torch.isfinite(tensor)):
        raise ValueError(f"{name} must contain only finite values.")
    if tensor.ndim == 0:
        return torch.eye(6, dtype=dtype, device=device) * tensor
    if tensor.ndim == 1 and tensor.shape[0] == 6:
        return torch.diag(tensor)
    if tensor.ndim == 2 and tensor.shape == (6, 6):
        return tensor
    raise ValueError(f"{name} must be a scalar, 6-vector, or 6x6 matrix, got shape {tuple(tensor.shape)}.")


def _solve_two_parameter_least_squares(design: torch.Tensor, target: torch.Tensor, nonnegative: bool) -> torch.Tensor:
    if design.ndim != 2 or design.shape[1] != 2:
        raise ValueError("design must have shape (N, 2).")
    if target.ndim != 1 or target.shape[0] != design.shape[0]:
        raise ValueError("target must have shape (N,).")

    if not nonnegative:
        return torch.linalg.pinv(design) @ target

    candidates = []
    unconstrained = torch.linalg.pinv(design) @ target
    if torch.all(unconstrained >= 0.0):
        candidates.append(unconstrained)

    zero = torch.zeros(2, dtype=design.dtype, device=design.device)
    candidates.append(zero)
    for index in range(2):
        column = design[:, index]
        denom = torch.sum(column * column)
        beta = zero.clone()
        if denom > 0.0:
            beta[index] = torch.clamp(torch.sum(column * target) / denom, min=0.0)
        candidates.append(beta)

    best = candidates[0]
    best_error = torch.sum((design @ best - target) ** 2)
    for candidate in candidates[1:]:
        error = torch.sum((design @ candidate - target) ** 2)
        if error < best_error:
            best = candidate
            best_error = error
    return best


def _solve_nonnegative_least_squares_by_subset(
    design: torch.Tensor,
    target: torch.Tensor,
    nonnegative: bool,
) -> torch.Tensor:
    if design.ndim != 2 or target.ndim != 1 or design.shape[0] != target.shape[0]:
        raise ValueError("design must be (N, P) and target must be (N,).")
    parameter_count = design.shape[1]
    if not nonnegative:
        return torch.linalg.pinv(design) @ target

    best = torch.zeros(parameter_count, dtype=design.dtype, device=design.device)
    best_error = torch.sum(target * target)
    for mask_bits in range(1, 1 << parameter_count):
        active_indices = [index for index in range(parameter_count) if mask_bits & (1 << index)]
        active_design = design[:, active_indices]
        active_beta = torch.linalg.pinv(active_design) @ target
        if torch.any(active_beta < 0.0):
            continue
        candidate = torch.zeros_like(best)
        candidate[active_indices] = active_beta
        error = torch.sum((design @ candidate - target) ** 2)
        if error < best_error:
            best = candidate
            best_error = error
    return best


def _solve_matrix_least_squares(design: torch.Tensor, target: torch.Tensor, regularization: float) -> torch.Tensor:
    if regularization < 0.0:
        raise ValueError("regularization must be non-negative.")
    if design.ndim != 2 or target.ndim != 2 or design.shape[0] != target.shape[0]:
        raise ValueError("design and target must be 2D matrices with matching sample counts.")
    if regularization == 0.0:
        return torch.linalg.pinv(design) @ target

    normal = design.T @ design
    ridge = float(regularization) * torch.eye(normal.shape[0], dtype=design.dtype, device=design.device)
    return torch.linalg.solve(normal + ridge, design.T @ target)
