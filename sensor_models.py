"""Lightweight sensor/estimator models used by WarpAUV observations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SensorChannelMeasurement:
    value: torch.Tensor
    valid: torch.Tensor


@dataclass(frozen=True)
class IMUSensorMeasurement:
    accelerometer_b: torch.Tensor
    gyroscope_b: torch.Tensor
    valid: torch.Tensor


@dataclass(frozen=True)
class DepthSensorMeasurement:
    depth: torch.Tensor
    valid: torch.Tensor


@dataclass(frozen=True)
class DVLSensorMeasurement:
    velocity_b: torch.Tensor
    valid: torch.Tensor


@dataclass(frozen=True)
class PositionSensorMeasurement:
    position_w: torch.Tensor
    valid: torch.Tensor


def expand_observation_parameter(value: torch.Tensor | float, reference: torch.Tensor) -> torch.Tensor:
    """Broadcast scalar, per-env, or per-observation parameters to observation shape."""

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


def _observation_group_indices(
    selector: slice | int | Sequence[int],
    obs_dim: int,
    device: torch.device,
) -> torch.Tensor:
    if isinstance(selector, slice):
        return torch.arange(obs_dim, dtype=torch.long, device=device)[selector]
    if isinstance(selector, int):
        return torch.tensor([selector], dtype=torch.long, device=device)
    return torch.as_tensor(list(selector), dtype=torch.long, device=device)


def build_observation_group_parameter(
    group_values: Mapping[str, torch.Tensor | float | Sequence[float]],
    group_slices: Mapping[str, slice | int | Sequence[int]],
    reference: torch.Tensor,
) -> torch.Tensor:
    """Build an observation-shaped parameter tensor from semantic groups."""

    result = torch.zeros_like(reference)
    obs_dim = reference.shape[1]
    for group_name, value in group_values.items():
        if group_name not in group_slices:
            known = ", ".join(sorted(group_slices))
            raise ValueError(f"Unknown observation group '{group_name}'. Known groups: {known}.")
        indices = _observation_group_indices(group_slices[group_name], obs_dim, reference.device)
        if torch.any((indices < 0) | (indices >= obs_dim)):
            raise ValueError(f"Observation group '{group_name}' contains indices outside [0, {obs_dim}).")

        tensor = torch.as_tensor(value, dtype=reference.dtype, device=reference.device)
        if tensor.ndim == 0:
            result[:, indices] = tensor
        elif tensor.ndim == 1:
            if tensor.shape[0] == indices.numel():
                result[:, indices] = tensor.reshape(1, -1)
            elif tensor.shape[0] == reference.shape[0]:
                result[:, indices] = tensor.reshape(-1, 1)
            else:
                raise ValueError(
                    f"Observation group '{group_name}' has {indices.numel()} entries, "
                    f"but value shape is {tuple(tensor.shape)}."
                )
        elif tensor.ndim == 2 and tensor.shape == (reference.shape[0], indices.numel()):
            result[:, indices] = tensor
        elif tensor.ndim == 2 and tensor.shape == (1, indices.numel()):
            result[:, indices] = tensor.repeat(reference.shape[0], 1)
        else:
            raise ValueError(
                f"Cannot broadcast observation group '{group_name}' value with shape "
                f"{tuple(tensor.shape)} to {(reference.shape[0], indices.numel())}."
            )
    return result


def apply_sensor_channel_model(
    true_value: torch.Tensor,
    bias: torch.Tensor | float = 0.0,
    scale: torch.Tensor | float = 1.0,
    noise_std: torch.Tensor | float = 0.0,
    min_value: torch.Tensor | float | None = None,
    max_value: torch.Tensor | float | None = None,
    dropout_probability: torch.Tensor | float = 0.0,
    previous_measurement: torch.Tensor | None = None,
) -> SensorChannelMeasurement:
    """Apply common sensor channel effects to a physical value tensor."""

    value = torch.as_tensor(true_value)
    if not torch.is_floating_point(value):
        value = value.to(dtype=torch.float32)
    if value.ndim == 1:
        value = value.reshape(-1, 1)
    if value.ndim != 2:
        raise ValueError(f"true_value must have shape (N, D) or (N,), got {tuple(value.shape)}.")
    if not torch.all(torch.isfinite(value)):
        raise ValueError("true_value must contain only finite values.")

    modeled = value * expand_observation_parameter(scale, value) + expand_observation_parameter(bias, value)
    std = torch.clamp(expand_observation_parameter(noise_std, value), min=0.0)
    if torch.any(std > 0.0):
        modeled = modeled + torch.randn_like(modeled) * std

    if min_value is not None:
        modeled = torch.maximum(modeled, expand_observation_parameter(min_value, modeled))
    if max_value is not None:
        modeled = torch.minimum(modeled, expand_observation_parameter(max_value, modeled))

    dropout_probability = torch.clamp(expand_observation_parameter(dropout_probability, modeled), min=0.0, max=1.0)
    valid = torch.rand_like(modeled) >= dropout_probability
    if previous_measurement is not None:
        previous = torch.as_tensor(previous_measurement, dtype=modeled.dtype, device=modeled.device)
        if previous.shape != modeled.shape:
            previous = expand_observation_parameter(previous, modeled)
        modeled = torch.where(valid, modeled, previous)
    return SensorChannelMeasurement(modeled, valid)


def calculate_imu_measurement(
    body_quat_w: torch.Tensor,
    linear_acceleration_w: torch.Tensor,
    angular_velocity_b: torch.Tensor,
    gravity_w: torch.Tensor | Sequence[float] = (0.0, 0.0, -9.81),
    accelerometer_bias: torch.Tensor | float = 0.0,
    accelerometer_scale: torch.Tensor | float = 1.0,
    accelerometer_noise_std: torch.Tensor | float = 0.0,
    gyroscope_bias: torch.Tensor | float = 0.0,
    gyroscope_scale: torch.Tensor | float = 1.0,
    gyroscope_noise_std: torch.Tensor | float = 0.0,
) -> IMUSensorMeasurement:
    """Return ideal/noisy IMU accelerometer and gyro measurements.

    Quaternions use the repository's ``wxyz`` convention.  The accelerometer
    output is body-frame specific force: ``R_bw * (a_w - gravity_w)``.
    """

    quat = _as_quat_wxyz(body_quat_w, "body_quat_w")
    acceleration = _as_sensor_matrix(linear_acceleration_w, 3, "linear_acceleration_w").to(
        dtype=quat.dtype,
        device=quat.device,
    )
    gyro = _as_sensor_matrix(angular_velocity_b, 3, "angular_velocity_b").to(dtype=quat.dtype, device=quat.device)
    if acceleration.shape[0] == 1 and quat.shape[0] > 1:
        acceleration = acceleration.repeat(quat.shape[0], 1)
    if gyro.shape[0] == 1 and quat.shape[0] > 1:
        gyro = gyro.repeat(quat.shape[0], 1)
    if acceleration.shape[0] != quat.shape[0] or gyro.shape[0] != quat.shape[0]:
        raise ValueError("IMU inputs must have matching environment counts.")

    gravity = torch.as_tensor(gravity_w, dtype=quat.dtype, device=quat.device)
    if gravity.ndim == 1:
        if gravity.shape[0] != 3:
            raise ValueError("gravity_w must have length 3.")
        gravity = gravity.reshape(1, 3).repeat(quat.shape[0], 1)
    elif gravity.shape != acceleration.shape:
        raise ValueError(f"gravity_w must have shape (3,) or {tuple(acceleration.shape)}.")

    specific_force_w = acceleration - gravity
    specific_force_b = _quat_apply_wxyz(_quat_conjugate_wxyz(quat), specific_force_w)
    accel = apply_sensor_channel_model(
        specific_force_b,
        bias=accelerometer_bias,
        scale=accelerometer_scale,
        noise_std=accelerometer_noise_std,
    )
    gyro_meas = apply_sensor_channel_model(
        gyro,
        bias=gyroscope_bias,
        scale=gyroscope_scale,
        noise_std=gyroscope_noise_std,
    )
    valid = torch.all(accel.valid, dim=-1, keepdim=True) & torch.all(gyro_meas.valid, dim=-1, keepdim=True)
    return IMUSensorMeasurement(accel.value, gyro_meas.value, valid)


def calculate_depth_sensor_measurement(
    position_w: torch.Tensor,
    surface_z: float,
    depth_axis_sign: float = 1.0,
    depth_bias: torch.Tensor | float = 0.0,
    depth_scale: torch.Tensor | float = 1.0,
    depth_noise_std: torch.Tensor | float = 0.0,
    max_depth: float | None = None,
    dropout_probability: torch.Tensor | float = 0.0,
    previous_depth: torch.Tensor | None = None,
) -> DepthSensorMeasurement:
    """Return depth from a flat water-surface plane.

    ``depth_axis_sign=1`` matches pool coordinates where larger z is deeper.
    Use ``-1`` for z-up worlds where depth is ``surface_z - z``.
    """

    position = _as_sensor_matrix(position_w, 3, "position_w")
    raw_depth = float(depth_axis_sign) * (position[:, 2:3] - float(surface_z))
    raw_depth = torch.clamp(raw_depth, min=0.0)
    measurement = apply_sensor_channel_model(
        raw_depth,
        bias=depth_bias,
        scale=depth_scale,
        noise_std=depth_noise_std,
        min_value=0.0,
        max_value=max_depth,
        dropout_probability=dropout_probability,
        previous_measurement=previous_depth,
    )
    return DepthSensorMeasurement(measurement.value, measurement.valid)


def calculate_dvl_velocity_measurement(
    linear_velocity_b: torch.Tensor,
    altitude: torch.Tensor | float,
    max_range: float,
    min_range: float = 0.0,
    water_velocity_b: torch.Tensor | None = None,
    velocity_bias: torch.Tensor | float = 0.0,
    velocity_scale: torch.Tensor | float = 1.0,
    velocity_noise_std: torch.Tensor | float = 0.0,
    dropout_probability: torch.Tensor | float = 0.0,
    previous_velocity_b: torch.Tensor | None = None,
) -> DVLSensorMeasurement:
    """Return DVL body-frame velocity and a bottom/water-lock validity flag."""

    velocity = _as_sensor_matrix(linear_velocity_b, 3, "linear_velocity_b")
    if water_velocity_b is not None:
        water_velocity = _as_sensor_matrix(water_velocity_b, 3, "water_velocity_b").to(
            dtype=velocity.dtype,
            device=velocity.device,
        )
        if water_velocity.shape[0] == 1 and velocity.shape[0] > 1:
            water_velocity = water_velocity.repeat(velocity.shape[0], 1)
        if water_velocity.shape != velocity.shape:
            raise ValueError("water_velocity_b must broadcast to linear_velocity_b.")
        velocity = velocity - water_velocity

    measurement = apply_sensor_channel_model(
        velocity,
        bias=velocity_bias,
        scale=velocity_scale,
        noise_std=velocity_noise_std,
        dropout_probability=0.0,
    )
    altitude_tensor = _as_sensor_column(altitude, velocity.shape[0], velocity.device, velocity.dtype, "altitude")
    if float(max_range) < float(min_range):
        raise ValueError("max_range must be >= min_range.")
    lock_valid = (altitude_tensor >= float(min_range)) & (altitude_tensor <= float(max_range))
    dropout_probability = torch.clamp(
        expand_observation_parameter(dropout_probability, altitude_tensor),
        min=0.0,
        max=1.0,
    )
    dropout_valid = torch.rand_like(altitude_tensor) >= dropout_probability
    valid = lock_valid & dropout_valid
    value = measurement.value
    if previous_velocity_b is not None:
        previous = torch.as_tensor(previous_velocity_b, dtype=value.dtype, device=value.device)
        if previous.shape != value.shape:
            previous = expand_observation_parameter(previous, value)
        value = torch.where(valid, value, previous)
    return DVLSensorMeasurement(value, valid)


def calculate_position_sensor_measurement(
    position_w: torch.Tensor,
    reference_position_w: torch.Tensor | Sequence[float] | None = None,
    max_range: float | None = None,
    min_range: float = 0.0,
    position_bias: torch.Tensor | float = 0.0,
    position_scale: torch.Tensor | float = 1.0,
    position_noise_std: torch.Tensor | float = 0.0,
    dropout_probability: torch.Tensor | float = 0.0,
    previous_position_w: torch.Tensor | None = None,
) -> PositionSensorMeasurement:
    """Return an external visual/acoustic position measurement with range validity."""

    position = _as_sensor_matrix(position_w, 3, "position_w")
    measurement = apply_sensor_channel_model(
        position,
        bias=position_bias,
        scale=position_scale,
        noise_std=position_noise_std,
        dropout_probability=0.0,
    )

    valid = torch.ones((position.shape[0], 1), dtype=torch.bool, device=position.device)
    if reference_position_w is not None or max_range is not None:
        if reference_position_w is None:
            reference = torch.zeros((1, 3), dtype=position.dtype, device=position.device)
        else:
            reference = _as_sensor_matrix(reference_position_w, 3, "reference_position_w").to(
                dtype=position.dtype,
                device=position.device,
            )
        if reference.shape[0] == 1 and position.shape[0] > 1:
            reference = reference.repeat(position.shape[0], 1)
        if reference.shape != position.shape:
            raise ValueError("reference_position_w must broadcast to position_w.")
        distance = torch.linalg.norm(position - reference, dim=-1, keepdim=True)
        valid = valid & (distance >= float(min_range))
        if max_range is not None:
            valid = valid & (distance <= float(max_range))

    dropout_probability = torch.clamp(
        expand_observation_parameter(dropout_probability, valid.to(dtype=position.dtype)),
        min=0.0,
        max=1.0,
    )
    valid = valid & (torch.rand_like(dropout_probability) >= dropout_probability)
    value = measurement.value
    if previous_position_w is not None:
        previous = torch.as_tensor(previous_position_w, dtype=value.dtype, device=value.device)
        if previous.shape != value.shape:
            previous = expand_observation_parameter(previous, value)
        value = torch.where(valid, value, previous)
    return PositionSensorMeasurement(value, valid)


class ObservationDelayBuffer:
    """Per-environment fixed-step delay buffer for policy observations."""

    def __init__(self, num_envs: int, obs_dim: int, max_delay_steps: int, device: torch.device) -> None:
        self.num_envs = num_envs
        self.obs_dim = obs_dim
        self.device = device
        self.max_delay_steps = max(0, int(max_delay_steps))
        self.history_length = self.max_delay_steps + 1
        self.history_index = 0
        self.history = torch.zeros(
            (self.history_length, self.num_envs, self.obs_dim),
            dtype=torch.float32,
            device=self.device,
        )
        self.valid_counts = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

    def reset(self, env_ids: list | torch.Tensor) -> None:
        self.history[:, env_ids, :] = 0.0
        self.valid_counts[env_ids] = 0

    def reset_all(self) -> None:
        self.history[:] = 0.0
        self.valid_counts[:] = 0
        self.history_index = 0

    def update(self, obs: torch.Tensor, delay_steps: torch.Tensor | int) -> torch.Tensor:
        self.history[self.history_index, :, :] = obs
        self.valid_counts = torch.clamp(self.valid_counts + 1, max=self.history_length)

        delay_steps = torch.as_tensor(delay_steps, dtype=torch.long, device=obs.device)
        if delay_steps.ndim == 0:
            delay_steps = delay_steps.repeat(self.num_envs)
        delay_steps = torch.clamp(delay_steps.reshape(self.num_envs), min=0, max=self.max_delay_steps)
        available_delay = torch.clamp(self.valid_counts - 1, min=0)
        effective_delay = torch.minimum(delay_steps, available_delay)

        delayed_indices = (self.history_index - effective_delay) % self.history_length
        env_indices = torch.arange(self.num_envs, dtype=torch.long, device=obs.device)
        delayed_obs = self.history[delayed_indices, env_indices, :]
        self.history_index = (self.history_index + 1) % self.history_length
        return delayed_obs


class ObservationFilterState:
    """Stateful estimator effects: sample hold, dropouts, low-pass, and bias drift."""

    def __init__(self, num_envs: int, obs_dim: int, device: torch.device) -> None:
        self.num_envs = num_envs
        self.obs_dim = obs_dim
        self.device = device
        self.previous_measurement = torch.zeros(
            (self.num_envs, self.obs_dim),
            dtype=torch.float32,
            device=self.device,
        )
        self.bias_drift = torch.zeros_like(self.previous_measurement)
        self.step_counts = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.has_measurement = torch.zeros_like(self.previous_measurement, dtype=torch.bool)

    def reset(self, env_ids: list | torch.Tensor) -> None:
        self.previous_measurement[env_ids] = 0.0
        self.bias_drift[env_ids] = 0.0
        self.step_counts[env_ids] = 0
        self.has_measurement[env_ids] = False

    def reset_all(self) -> None:
        self.previous_measurement[:] = 0.0
        self.bias_drift[:] = 0.0
        self.step_counts[:] = 0
        self.has_measurement[:] = False

    def update(
        self,
        obs: torch.Tensor,
        fixed_bias: torch.Tensor | float,
        noise_std: torch.Tensor | float,
        update_period_steps: torch.Tensor | int = 1,
        dropout_probability: torch.Tensor | float = 0.0,
        lowpass_alpha: torch.Tensor | float = 1.0,
        bias_drift_std: torch.Tensor | float = 0.0,
        dt: float = 1.0,
    ) -> torch.Tensor:
        """Return a filtered measurement while updating persistent estimator state."""

        update_period_steps = torch.as_tensor(update_period_steps, dtype=torch.long, device=obs.device)
        if update_period_steps.ndim == 0:
            update_period_steps = update_period_steps.repeat(self.num_envs)
        update_period_steps = torch.clamp(update_period_steps.reshape(self.num_envs), min=1)

        bias_drift_std = torch.clamp(expand_observation_parameter(bias_drift_std, obs), min=0.0)
        if torch.any(bias_drift_std > 0.0):
            self.bias_drift = self.bias_drift + torch.randn_like(self.bias_drift) * bias_drift_std * (dt**0.5)

        fixed_bias = expand_observation_parameter(fixed_bias, obs)
        noise_std = torch.clamp(expand_observation_parameter(noise_std, obs), min=0.0)
        raw_measurement = obs + fixed_bias + self.bias_drift
        if torch.any(noise_std > 0.0):
            raw_measurement = raw_measurement + torch.randn_like(raw_measurement) * noise_std

        alpha = torch.clamp(expand_observation_parameter(lowpass_alpha, obs), min=0.0, max=1.0)
        previous_for_filter = torch.where(
            self.has_measurement,
            self.previous_measurement,
            raw_measurement,
        )
        filtered_measurement = alpha * raw_measurement + (1.0 - alpha) * previous_for_filter

        should_update = (self.step_counts % update_period_steps == 0).unsqueeze(-1)
        dropout_probability = torch.clamp(expand_observation_parameter(dropout_probability, obs), min=0.0, max=1.0)
        dropout = torch.rand_like(obs) < dropout_probability
        accept_update = should_update & (~dropout | ~self.has_measurement)

        measurement = torch.where(accept_update, filtered_measurement, self.previous_measurement)
        self.previous_measurement[:] = measurement
        self.has_measurement[:] = self.has_measurement | accept_update
        self.step_counts[:] = self.step_counts + 1
        return measurement


def apply_observation_sensor_model(
    obs: torch.Tensor,
    delay_buffer: ObservationDelayBuffer,
    delay_steps: torch.Tensor | int,
    noise_std: torch.Tensor | float,
    bias: torch.Tensor | float,
    filter_state: ObservationFilterState | None = None,
    update_period_steps: torch.Tensor | int = 1,
    dropout_probability: torch.Tensor | float = 0.0,
    lowpass_alpha: torch.Tensor | float = 1.0,
    bias_drift_std: torch.Tensor | float = 0.0,
    dt: float = 1.0,
) -> torch.Tensor:
    """Apply delay, additive bias/noise, and optional estimator dynamics."""

    delayed_obs = delay_buffer.update(obs, delay_steps)
    if filter_state is None:
        bias_tensor = expand_observation_parameter(bias, delayed_obs)
        std_tensor = torch.clamp(expand_observation_parameter(noise_std, delayed_obs), min=0.0)
        if torch.any(std_tensor > 0.0):
            delayed_obs = delayed_obs + torch.randn_like(delayed_obs) * std_tensor
        return delayed_obs + bias_tensor

    return filter_state.update(
        delayed_obs,
        bias,
        noise_std,
        update_period_steps,
        dropout_probability,
        lowpass_alpha,
        bias_drift_std,
        dt,
    )


def _as_sensor_matrix(value: torch.Tensor | Sequence[float], width: int, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(value)
    if not torch.is_floating_point(tensor):
        tensor = tensor.to(dtype=torch.float32)
    if tensor.ndim == 1:
        if tensor.shape[0] != width:
            raise ValueError(f"{name} must have length {width}.")
        tensor = tensor.reshape(1, width)
    if tensor.ndim != 2 or tensor.shape[1] != width:
        raise ValueError(f"{name} must have shape (N, {width}) or ({width},), got {tuple(tensor.shape)}.")
    if not torch.all(torch.isfinite(tensor)):
        raise ValueError(f"{name} must contain only finite values.")
    return tensor


def _as_sensor_column(
    value: torch.Tensor | float,
    num_envs: int,
    device: torch.device,
    dtype: torch.dtype,
    name: str,
) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=dtype, device=device)
    if tensor.ndim == 0:
        tensor = tensor.reshape(1, 1).repeat(num_envs, 1)
    elif tensor.ndim == 1:
        if tensor.shape[0] == num_envs:
            tensor = tensor.reshape(num_envs, 1)
        elif tensor.shape[0] == 1:
            tensor = tensor.reshape(1, 1).repeat(num_envs, 1)
        else:
            raise ValueError(f"{name} must be scalar, shape (N,), or shape (N, 1).")
    elif tensor.ndim == 2:
        if tensor.shape == (1, 1):
            tensor = tensor.repeat(num_envs, 1)
        elif tensor.shape != (num_envs, 1):
            raise ValueError(f"{name} must be scalar, shape (N,), or shape (N, 1).")
    else:
        raise ValueError(f"{name} must be scalar, shape (N,), or shape (N, 1).")
    if not torch.all(torch.isfinite(tensor)):
        raise ValueError(f"{name} must contain only finite values.")
    return tensor


def _as_quat_wxyz(value: torch.Tensor, name: str) -> torch.Tensor:
    quat = _as_sensor_matrix(value, 4, name)
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
