"""Math checks for Fossen-style hydrodynamics.

Run directly with the IsaacLab Python environment, for example:
    /home/jining_yang/miniconda3/envs/env_isaaclab/bin/python tests/test_dynamics_math.py
"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_module(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


hydro = _load_module("rigid_body_hydrodynamics", "rigid_body_hydrodynamics.py")
model_params = _load_module("bluerov2_heavy_model", "bluerov2_heavy_model.py")
thrusters = _load_module("thruster_dynamics", "thruster_dynamics.py")
sensors = _load_module("sensor_models", "sensor_models.py")
pool_effects = _load_module("pool_effects", "pool_effects.py")
tether = _load_module("tether_dynamics", "tether_dynamics.py")
profiles = _load_module("pool_dynamics_profile", "pool_dynamics_profile.py")
current_fields = _load_module("water_current_fields", "water_current_fields.py")
rigid_body_properties = _load_module("rigid_body_properties", "rigid_body_properties.py")
calibration = _load_module("calibration_tools", "calibration_tools.py")
audit_cli = _load_module("audit_pool_profile", "custom_workflows/audit_pool_profile.py")
profile_builder_cli = _load_module(
    "build_pool_profile_from_calibration",
    "custom_workflows/build_pool_profile_from_calibration.py",
)
static_fit_cli = _load_module("fit_pool_static_logs", "custom_workflows/fit_pool_static_logs.py")
thruster_fit_cli = _load_module("fit_pool_thruster_logs", "custom_workflows/fit_pool_thruster_logs.py")
environment_fit_cli = _load_module(
    "fit_pool_environment_logs",
    "custom_workflows/fit_pool_environment_logs.py",
)
hydrodynamics_fit_cli = _load_module(
    "fit_pool_hydrodynamics_logs",
    "custom_workflows/fit_pool_hydrodynamics_logs.py",
)
tether_fit_cli = _load_module("fit_pool_tether_logs", "custom_workflows/fit_pool_tether_logs.py")


def _assert_raises(error_type, callback, *args, **kwargs):
    try:
        callback(*args, **kwargs)
    except error_type:
        return
    raise AssertionError(f"Expected {error_type.__name__}.")


def _write_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    lines = [",".join(header)]
    lines.extend(",".join(str(value) for value in row) for row in rows)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_relative_damping_dissipates_relative_motion():
    model = hydro.HydrodynamicForceModels(num_envs=2, device=torch.device("cpu"))
    nu_r = torch.tensor(
        [
            [0.2, -0.1, 0.3, 0.04, -0.02, 0.01],
            [-0.3, 0.2, -0.1, -0.03, 0.05, -0.02],
        ]
    )
    linear = torch.tensor([0.00526, 0.00526, 0.00526, 0.00032, 0.00032, 0.00032])
    quadratic = torch.tensor([39.196, 68.272, 135.402, 0.277, 1.387, 0.770])
    damping = model.calculate_relative_damping_wrench(nu_r, linear, quadratic)
    assert torch.all(torch.sum(nu_r * damping, dim=-1) <= 0.0)


def test_full_matrix_linear_damping_dissipates_relative_motion():
    model = hydro.HydrodynamicForceModels(num_envs=2, device=torch.device("cpu"))
    nu_r = torch.tensor(
        [
            [0.2, -0.1, 0.3, 0.04, -0.02, 0.01],
            [-0.3, 0.2, -0.1, -0.03, 0.05, -0.02],
        ]
    )
    base = torch.tensor(
        [
            [0.8, 0.1, 0.0, 0.0, 0.0, 0.02],
            [0.1, 1.0, 0.03, 0.0, 0.0, 0.0],
            [0.0, 0.03, 1.2, 0.0, 0.04, 0.0],
            [0.0, 0.0, 0.0, 0.2, 0.01, 0.0],
            [0.0, 0.0, 0.04, 0.01, 0.3, 0.02],
            [0.02, 0.0, 0.0, 0.0, 0.02, 0.25],
        ]
    )
    linear = base.T @ base + 0.01 * torch.eye(6)
    quadratic = torch.zeros(6)
    damping = model.calculate_relative_damping_wrench(nu_r, linear, quadratic)
    assert torch.all(torch.sum(nu_r * damping, dim=-1) <= 0.0)


def test_speed_dependent_damping_scale_interpolates_shared_curve():
    nu_r = torch.tensor([[0.0, 0.5, 1.0, 1.5, -2.0, 3.0]])

    scale = hydro.calculate_speed_dependent_damping_scale(
        nu_r,
        speed_points=[0.0, 2.0],
        scale_points=[1.0, 3.0],
    )

    expected = torch.tensor([[1.0, 1.5, 2.0, 2.5, 3.0, 3.0]])
    assert torch.allclose(scale, expected)


def test_speed_dependent_damping_scale_interpolates_per_dof_curves():
    nu_r = torch.tensor([[0.5, -0.5, 0.5, -0.5, 0.5, -0.5]])

    scale = hydro.calculate_speed_dependent_damping_scale(
        nu_r,
        speed_points=[0.0, 1.0],
        scale_points=[
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [2.0, 4.0, 6.0, 8.0, 10.0, 12.0],
        ],
    )

    expected = torch.tensor([[1.5, 3.0, 4.5, 6.0, 7.5, 9.0]])
    assert torch.allclose(scale, expected)


def test_calibration_fits_diagonal_linear_quadratic_damping_from_synthetic_log():
    speeds = torch.tensor([-1.0, -0.7, -0.4, -0.2, 0.2, 0.4, 0.7, 1.0])
    nu_r = speeds.reshape(-1, 1).repeat(1, 6)
    linear = torch.tensor([1.0, 1.5, 2.0, 0.2, 0.3, 0.4])
    quadratic = torch.tensor([4.0, 5.0, 6.0, 0.7, 0.8, 0.9])
    applied_wrench = linear.reshape(1, 6) * nu_r + quadratic.reshape(1, 6) * torch.abs(nu_r) * nu_r

    fit = calibration.fit_diagonal_linear_quadratic_damping(
        time_s=torch.arange(len(speeds), dtype=torch.float32),
        nu_r=nu_r,
        applied_wrench=applied_wrench,
        effective_mass=torch.ones(6),
        relative_acceleration=torch.zeros_like(nu_r),
    )

    assert torch.allclose(fit.linear_damping, linear, atol=1.0e-5)
    assert torch.allclose(fit.quadratic_damping, quadratic, atol=1.0e-5)
    assert fit.to_cfg_updates()["linear_damping"] == fit.linear_damping.tolist()


def test_calibration_fits_full_matrix_linear_quadratic_damping_from_synthetic_log():
    torch.manual_seed(2)
    nu_r = torch.randn(80, 6)
    linear = torch.tensor(
        [
            [1.0, 0.2, 0.0, 0.0, 0.0, 0.1],
            [0.1, 1.4, 0.2, 0.0, 0.0, 0.3],
            [0.0, 0.1, 2.0, 0.0, 0.2, 0.0],
            [0.0, 0.0, 0.0, 0.4, 0.1, 0.0],
            [0.0, 0.0, 0.2, 0.1, 0.6, 0.0],
            [0.1, 0.3, 0.0, 0.0, 0.0, 0.8],
        ]
    )
    quadratic = torch.tensor(
        [
            [2.0, 0.1, 0.0, 0.0, 0.0, 0.2],
            [0.2, 2.5, 0.1, 0.0, 0.0, 0.4],
            [0.0, 0.2, 3.0, 0.0, 0.3, 0.0],
            [0.0, 0.0, 0.0, 0.5, 0.1, 0.0],
            [0.0, 0.0, 0.3, 0.1, 0.7, 0.0],
            [0.2, 0.4, 0.0, 0.0, 0.0, 0.9],
        ]
    )
    applied_wrench = nu_r @ linear.T + (torch.abs(nu_r) * nu_r) @ quadratic.T

    fit = calibration.fit_full_matrix_linear_quadratic_damping(
        time_s=torch.arange(nu_r.shape[0], dtype=torch.float32),
        nu_r=nu_r,
        applied_wrench=applied_wrench,
        effective_mass=torch.ones(6),
        relative_acceleration=torch.zeros_like(nu_r),
    )

    assert torch.allclose(fit.linear_damping, linear, atol=1.0e-4)
    assert torch.allclose(fit.quadratic_damping, quadratic, atol=1.0e-4)
    assert fit.sample_count == nu_r.shape[0]
    assert fit.to_cfg_updates()["quadratic_damping"] == fit.quadratic_damping.tolist()


def test_calibration_fits_diagonal_added_mass_and_damping_from_synthetic_log():
    torch.manual_seed(3)
    time_s = torch.arange(80, dtype=torch.float32)
    acceleration = torch.randn(80, 6)
    nu_r = torch.randn(80, 6)
    rigid_inertia = torch.tensor([11.5, 11.5, 11.5, 0.3, 0.4, 0.5])
    added_mass = torch.tensor([1.2, 1.4, 1.6, 0.05, 0.06, 0.07])
    linear = torch.tensor([0.8, 0.9, 1.0, 0.10, 0.11, 0.12])
    quadratic = torch.tensor([2.0, 2.2, 2.4, 0.20, 0.22, 0.24])
    applied_wrench = (
        (rigid_inertia + added_mass).reshape(1, 6) * acceleration
        + linear.reshape(1, 6) * nu_r
        + quadratic.reshape(1, 6) * torch.abs(nu_r) * nu_r
    )

    fit = calibration.fit_diagonal_added_mass_linear_quadratic_damping(
        time_s,
        nu_r,
        applied_wrench,
        rigid_body_inertia=rigid_inertia,
        relative_acceleration=acceleration,
    )

    assert torch.allclose(fit.added_mass, added_mass, atol=1.0e-4)
    assert torch.allclose(fit.effective_inertia, rigid_inertia + added_mass, atol=1.0e-4)
    assert torch.allclose(fit.linear_damping, linear, atol=1.0e-4)
    assert torch.allclose(fit.quadratic_damping, quadratic, atol=1.0e-4)
    assert fit.to_cfg_updates()["added_mass_diag"] == fit.added_mass.tolist()


def test_calibration_fits_full_matrix_added_mass_and_damping_from_synthetic_log():
    torch.manual_seed(4)
    time_s = torch.arange(140, dtype=torch.float32)
    acceleration = torch.randn(140, 6)
    nu_r = torch.randn(140, 6)
    rigid_inertia = torch.diag(torch.tensor([11.5, 11.5, 11.5, 0.3, 0.4, 0.5]))
    added_mass = torch.tensor(
        [
            [1.2, 0.1, 0.0, 0.0, 0.0, 0.05],
            [0.1, 1.4, 0.08, 0.0, 0.0, 0.04],
            [0.0, 0.08, 1.6, 0.0, 0.06, 0.0],
            [0.0, 0.0, 0.0, 0.05, 0.01, 0.0],
            [0.0, 0.0, 0.06, 0.01, 0.06, 0.02],
            [0.05, 0.04, 0.0, 0.0, 0.02, 0.07],
        ]
    )
    linear = torch.tensor(
        [
            [0.8, 0.05, 0.0, 0.0, 0.0, 0.02],
            [0.03, 0.9, 0.04, 0.0, 0.0, 0.02],
            [0.0, 0.04, 1.0, 0.0, 0.03, 0.0],
            [0.0, 0.0, 0.0, 0.10, 0.01, 0.0],
            [0.0, 0.0, 0.03, 0.01, 0.11, 0.0],
            [0.02, 0.02, 0.0, 0.0, 0.0, 0.12],
        ]
    )
    quadratic = torch.tensor(
        [
            [2.0, 0.1, 0.0, 0.0, 0.0, 0.03],
            [0.1, 2.2, 0.08, 0.0, 0.0, 0.04],
            [0.0, 0.08, 2.4, 0.0, 0.05, 0.0],
            [0.0, 0.0, 0.0, 0.20, 0.02, 0.0],
            [0.0, 0.0, 0.05, 0.02, 0.22, 0.01],
            [0.03, 0.04, 0.0, 0.0, 0.01, 0.24],
        ]
    )
    effective = rigid_inertia + added_mass
    applied_wrench = acceleration @ effective.T + nu_r @ linear.T + (torch.abs(nu_r) * nu_r) @ quadratic.T

    fit = calibration.fit_full_matrix_added_mass_linear_quadratic_damping(
        time_s,
        nu_r,
        applied_wrench,
        rigid_body_inertia=rigid_inertia,
        relative_acceleration=acceleration,
    )

    assert torch.allclose(fit.added_mass, added_mass, atol=1.0e-4)
    assert torch.allclose(fit.effective_inertia, effective, atol=1.0e-4)
    assert torch.allclose(fit.linear_damping, linear, atol=1.0e-4)
    assert torch.allclose(fit.quadratic_damping, quadratic, atol=1.0e-4)
    assert fit.symmetrized_added_mass is True
    assert fit.to_cfg_updates()["added_mass_diag"] == fit.added_mass.tolist()


def test_hydrodynamics_calibration_log_pipeline_fits_full_physical_matrices():
    torch.manual_seed(17)
    sample_count = 180
    time_s = torch.arange(sample_count, dtype=torch.float32) * 0.05
    acceleration = torch.randn(sample_count, 6)
    nu_r = torch.randn(sample_count, 6)
    profile = profiles.NOMINAL_POOL_DYNAMICS_PROFILE
    rigid_mass = torch.zeros(6, 6)
    rigid_mass[0:3, 0:3] = torch.eye(3) * profile.rigid_body.mass
    rigid_mass[3:6, 3:6] = rigid_body_properties.inertia_matrix_tensor(
        profile.rigid_body.inertia_diag,
        torch.device("cpu"),
    )
    added_base = torch.tensor(
        [
            [1.2, 0.1, 0.0, 0.0, 0.0, 0.04],
            [0.1, 1.4, 0.08, 0.0, 0.0, 0.03],
            [0.0, 0.08, 1.6, 0.0, 0.05, 0.0],
            [0.0, 0.0, 0.0, 0.08, 0.01, 0.0],
            [0.0, 0.0, 0.05, 0.01, 0.10, 0.02],
            [0.04, 0.03, 0.0, 0.0, 0.02, 0.12],
        ]
    )
    added_mass = 0.5 * (added_base + added_base.T)
    linear_seed = torch.tensor(
        [
            [0.8, 0.05, 0.0, 0.0, 0.0, 0.02],
            [0.03, 0.9, 0.04, 0.0, 0.0, 0.02],
            [0.0, 0.04, 1.0, 0.0, 0.03, 0.0],
            [0.0, 0.0, 0.0, 0.10, 0.01, 0.0],
            [0.0, 0.0, 0.03, 0.01, 0.11, 0.0],
            [0.02, 0.02, 0.0, 0.0, 0.0, 0.12],
        ]
    )
    linear_damping = linear_seed.T @ linear_seed + 0.05 * torch.eye(6)
    quadratic_damping = torch.diag(torch.tensor([2.0, 2.2, 2.4, 0.2, 0.22, 0.24]))
    wrench = (
        acceleration @ (rigid_mass + added_mass).T
        + nu_r @ linear_damping.T
        + (torch.abs(nu_r) * nu_r) @ quadratic_damping.T
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        header = [
            "time_s",
            *hydrodynamics_fit_cli.NU_COLUMNS,
            *hydrodynamics_fit_cli.WRENCH_COLUMNS,
            *hydrodynamics_fit_cli.ACCEL_COLUMNS,
        ]
        _write_csv(
            root / hydrodynamics_fit_cli.MOTION_LOG_FILENAME,
            header,
            [
                [
                    float(time_s[index]),
                    *nu_r[index].tolist(),
                    *wrench[index].tolist(),
                    *acceleration[index].tolist(),
                ]
                for index in range(sample_count)
            ],
        )

        result = hydrodynamics_fit_cli.fit_hydrodynamics_calibration_logs(root, fit_mode="full")
        output_path = root / "hydrodynamics_updates.json"
        report_path = root / "hydrodynamics_report.json"
        exit_code = hydrodynamics_fit_cli.main(
            [
                str(root),
                "--fit-mode",
                "full",
                "--output",
                str(output_path),
                "--report",
                str(report_path),
            ]
        )
        output_updates, output_domain = profile_builder_cli.load_update_payload(output_path)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        merged = profiles.merge_pool_dynamics_cfg_updates(cfg_updates=output_updates)

    assert torch.allclose(torch.tensor(result.cfg_updates["added_mass_diag"]), added_mass, atol=2.0e-4)
    assert torch.allclose(torch.tensor(result.cfg_updates["linear_damping"]), linear_damping, atol=2.0e-4)
    assert torch.allclose(torch.tensor(result.cfg_updates["quadratic_damping"]), quadratic_damping, atol=2.0e-4)
    assert result.diagnostics["design_rank"] == 18
    assert result.diagnostics["sampled_passivity"]["is_passive"] is True
    assert result.diagnostics["added_mass_projection"]["projected_min_eigenvalue"] >= -1.0e-6
    assert exit_code == 0
    assert output_updates == result.cfg_updates
    assert output_domain == {}
    assert merged.hydrodynamics.added_mass == result.cfg_updates["added_mass_diag"]
    assert report["source_files"] == [hydrodynamics_fit_cli.MOTION_LOG_FILENAME]


def test_calibration_projects_added_mass_to_symmetric_psd():
    added_mass = torch.diag(torch.tensor([1.2, 1.0, -0.15, 0.2, 0.3, 0.4]))
    added_mass[0, 1] = 0.2
    added_mass[1, 0] = -0.1

    projection = calibration.project_added_mass_to_physical(added_mass, min_eigenvalue=0.05)
    projected = projection.projected_matrix

    assert projection.original_min_eigenvalue < 0.0
    assert projection.projected_min_eigenvalue >= 0.05 - 1.0e-5
    assert projection.symmetrized_input is True
    assert torch.allclose(projected, projected.T, atol=1.0e-6)
    assert torch.all(torch.linalg.eigvalsh(projected) >= 0.05 - 1.0e-5)
    assert projection.to_cfg_value() == projected.tolist()


def test_calibration_projects_linear_damping_to_dissipative_preserving_skew():
    torch.manual_seed(5)
    symmetric = torch.diag(torch.tensor([1.0, 0.7, -0.2, 0.1, 0.2, 0.3]))
    symmetric[0, 1] = 0.1
    symmetric[1, 0] = 0.1
    skew = torch.zeros(6, 6)
    skew[0, 5] = 0.4
    skew[5, 0] = -0.4
    damping = symmetric + skew

    projection = calibration.project_linear_damping_to_dissipative(damping, preserve_skew=True)
    projected = projection.projected_matrix

    assert projection.original_min_eigenvalue < 0.0
    assert projection.projected_min_eigenvalue >= -1.0e-6
    assert projection.preserved_skew is True
    assert torch.allclose(0.5 * (projected - projected.T), skew, atol=1.0e-6)
    assert torch.all(torch.linalg.eigvalsh(0.5 * (projected + projected.T)) >= -1.0e-6)

    nu_r = torch.randn(64, 6)
    dissipated_power = calibration.calculate_damping_dissipated_power(nu_r, linear_damping=projected)
    assert torch.all(dissipated_power >= -1.0e-5)
    assert calibration.damping_is_dissipative_for_samples(nu_r, linear_damping=projected)


def test_calibration_checks_sampled_quadratic_damping_power():
    nu_r = torch.tensor(
        [
            [0.5, -0.2, 0.1, 0.03, -0.04, 0.02],
            [-0.4, 0.3, -0.2, -0.02, 0.05, -0.01],
        ]
    )
    quadratic = torch.tensor([2.0, 2.5, 3.0, 0.2, 0.25, 0.3])
    bad_quadratic = torch.tensor([-2.0, 2.5, 3.0, 0.2, 0.25, 0.3])

    assert torch.all(calibration.calculate_damping_dissipated_power(nu_r, quadratic_damping=quadratic) > 0.0)
    assert calibration.damping_is_dissipative_for_samples(nu_r, quadratic_damping=quadratic)
    assert not calibration.damping_is_dissipative_for_samples(nu_r, quadratic_damping=bad_quadratic)


def test_calibration_fits_speed_dependent_damping_scales_from_synthetic_log():
    speed_samples = torch.tensor([0.1, -0.2, 0.3, -0.4, 0.7, -0.8, 0.9, -1.0])
    nu_r = speed_samples.reshape(-1, 1).repeat(1, 6)
    nominal_linear = torch.tensor([1.0, 1.5, 2.0, 0.2, 0.3, 0.4])
    nominal_quadratic = torch.tensor([4.0, 5.0, 6.0, 0.7, 0.8, 0.9])
    speed_points = [0.25, 0.85]
    linear_scales = torch.tensor(
        [
            [1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
            [1.6, 1.5, 1.4, 1.3, 1.2, 1.1],
        ]
    )
    quadratic_scales = torch.tensor(
        [
            [0.8, 0.9, 1.0, 1.1, 1.2, 1.3],
            [1.4, 1.3, 1.2, 1.1, 1.0, 0.9],
        ]
    )

    applied_wrench = torch.zeros_like(nu_r)
    for sample_index, speed in enumerate(torch.abs(speed_samples)):
        bin_index = 0 if speed < 0.55 else 1
        applied_wrench[sample_index] = (
            nominal_linear * linear_scales[bin_index] * nu_r[sample_index]
            + nominal_quadratic * quadratic_scales[bin_index] * speed * nu_r[sample_index]
        )

    fit = calibration.fit_speed_dependent_damping_scales(
        time_s=torch.arange(len(speed_samples), dtype=torch.float32),
        nu_r=nu_r,
        applied_wrench=applied_wrench,
        effective_mass=torch.ones(6),
        speed_points=speed_points,
        nominal_linear_damping=nominal_linear,
        nominal_quadratic_damping=nominal_quadratic,
        relative_acceleration=torch.zeros_like(nu_r),
    )

    assert torch.allclose(fit.linear_scales, linear_scales, atol=1.0e-4)
    assert torch.allclose(fit.quadratic_scales, quadratic_scales, atol=1.0e-4)
    updates = fit.to_cfg_updates(speed_points)
    assert updates["speed_dependent_damping_enabled"] is True
    assert updates["damping_speed_points"] == speed_points


def test_calibration_fits_water_current_process_from_synthetic_log():
    alpha = 0.8
    time_s = torch.arange(8, dtype=torch.float32)
    powers = alpha ** torch.arange(len(time_s), dtype=torch.float32)
    mean_current = torch.tensor([0.1, -0.02, 0.01])
    residual = torch.stack((powers, -0.5 * powers, 0.25 * powers), dim=-1)
    current = mean_current.reshape(1, 3) + residual

    fit = calibration.fit_water_current_process(time_s, current, mean_current_w=mean_current)

    assert torch.allclose(fit.mean_current_w, mean_current)
    assert abs(fit.estimated_alpha - alpha) < 1.0e-6
    assert abs(fit.tau_s - (-1.0 / torch.log(torch.tensor(alpha)).item())) < 1.0e-5
    assert fit.sample_count == len(time_s)
    assert fit.to_cfg_updates()["water_current_w"] == mean_current.tolist()
    updates = fit.to_domain_randomization_updates(stage_count=2)
    assert updates["water_current_smooth"] is True
    assert updates["water_current_tau_range"][0] == updates["water_current_tau_range"][1]
    assert len(updates["water_current_max_by_stage"]) == 2


def test_calibration_fits_buoyancy_volume_from_force_samples():
    rho = 997.0
    volume = 0.0123
    gravity_w = torch.tensor([0.0, 0.0, -9.81])
    force_w = -rho * volume * gravity_w
    samples = force_w.reshape(1, 3).repeat(5, 1)

    fit = calibration.fit_buoyancy_volume_from_forces(samples, water_density=rho, gravity_w=gravity_w)
    updates = fit.to_cfg_updates()

    assert abs(fit.volume - volume) < 1.0e-8
    assert torch.allclose(fit.mean_buoyancy_force_w, force_w, atol=1.0e-6)
    assert fit.residual_rms < 1.0e-6
    assert fit.sample_count == 5
    assert updates["volume"] == fit.volume
    assert updates["water_rho"] == rho


def test_calibration_fits_com_to_cob_from_buoyancy_wrenches():
    offset = torch.tensor([0.08, -0.04, 0.025])
    forces_b = torch.tensor(
        [
            [0.0, 0.0, 120.0],
            [0.0, 120.0, 0.0],
            [120.0, 0.0, 0.0],
            [60.0, 80.0, 100.0],
        ]
    )
    torques_b = torch.cross(offset.reshape(1, 3).repeat(forces_b.shape[0], 1), forces_b, dim=-1)

    fit = calibration.fit_com_to_cob_offset_from_buoyancy_wrenches(forces_b, torques_b)
    updates = fit.to_cfg_updates()

    assert torch.allclose(fit.com_to_cob_offset, offset, atol=1.0e-6)
    assert fit.residual_rms < 1.0e-5
    assert fit.sample_count == forces_b.shape[0]
    assert fit.design_rank == 3
    assert updates["com_to_cob_offset"] == fit.com_to_cob_offset.tolist()


def test_calibration_fits_com_to_cob_from_static_orientation_torques():
    offset = torch.tensor([0.04, -0.03, 0.02])
    rho = 997.0
    volume = 0.0115
    gravity_w = torch.tensor([0.0, 0.0, -9.81])
    half_sqrt = torch.sqrt(torch.tensor(0.5))
    quats = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [half_sqrt, half_sqrt, 0.0, 0.0],
            [half_sqrt, 0.0, half_sqrt, 0.0],
            [half_sqrt, 0.0, 0.0, half_sqrt],
        ]
    )
    force_w = (-rho * volume * gravity_w).reshape(1, 3).repeat(quats.shape[0], 1)
    force_b = hydro.quat_apply_wxyz(hydro.quat_conjugate_wxyz(quats), force_w)
    torques_b = torch.cross(offset.reshape(1, 3).repeat(quats.shape[0], 1), force_b, dim=-1)

    fit = calibration.fit_com_to_cob_offset_from_static_torques(
        root_quats_w=quats,
        buoyancy_torque_b_samples=torques_b,
        volume=volume,
        water_density=rho,
        gravity_w=gravity_w,
    )

    assert torch.allclose(fit.com_to_cob_offset, offset, atol=1.0e-6)
    assert fit.residual_rms < 1.0e-5
    assert fit.sample_count == quats.shape[0]
    assert fit.design_rank == 3


def test_calibration_fits_mass_from_scale_readings():
    readings = torch.tensor([11.48, 11.52, 11.50, 11.50])

    fit = calibration.fit_mass_from_scale_readings(readings)
    updates = fit.to_cfg_updates()

    assert abs(fit.mass - 11.5) < 1.0e-6
    assert fit.residual_rms > 0.0
    assert fit.sample_count == readings.numel()
    assert updates["mass"] == fit.mass


def test_calibration_fits_inertia_tensor_from_axis_moments():
    inertia = torch.tensor(
        [
            [0.32, 0.018, -0.012],
            [0.018, 0.41, 0.015],
            [-0.012, 0.015, 0.53],
        ]
    )
    axes = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
        ]
    )
    axes = axes / torch.linalg.norm(axes, dim=-1, keepdim=True)
    moments = torch.sum((axes @ inertia) * axes, dim=-1)

    fit = calibration.fit_inertia_tensor_from_axis_moments(axes, moments)
    updates = fit.to_cfg_updates()

    assert torch.allclose(fit.inertia_tensor, inertia, atol=1.0e-6)
    assert fit.residual_rms < 1.0e-6
    assert fit.sample_count == axes.shape[0]
    assert fit.design_rank == 6
    assert fit.min_eigenvalue_after_projection > 0.0
    assert updates["inertia_diag"] == fit.inertia_tensor.tolist()


def test_calibration_fits_inertia_tensor_from_compound_pendulum_periods():
    inertia = torch.tensor(
        [
            [0.28, 0.012, 0.006],
            [0.012, 0.37, -0.009],
            [0.006, -0.009, 0.49],
        ]
    )
    axes = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
        ]
    )
    axes = axes / torch.linalg.norm(axes, dim=-1, keepdim=True)
    moments = torch.sum((axes @ inertia) * axes, dim=-1)
    mass = 11.5
    gravity = 9.81
    distances = torch.tensor([0.35, 0.36, 0.37, 0.38, 0.39, 0.40])
    periods = 2.0 * torch.pi * torch.sqrt((moments + mass * distances * distances) / (mass * gravity * distances))

    recovered_moments = calibration.compound_pendulum_moments_from_periods(periods, mass, distances, gravity)
    fit = calibration.fit_inertia_tensor_from_compound_pendulum(
        axes,
        period_s_samples=periods,
        mass=mass,
        pivot_to_com_distance_samples=distances,
        gravity_mps2=gravity,
    )

    assert torch.allclose(recovered_moments, moments, atol=1.0e-6)
    assert torch.allclose(fit.inertia_tensor, inertia, atol=1.0e-6)
    assert fit.residual_rms < 1.0e-6
    assert fit.design_rank == 6


def test_static_calibration_log_pipeline_builds_rigid_body_updates():
    rho = 997.0
    gravity_z = -9.81
    volume = 0.0118
    offset = torch.tensor([0.04, -0.03, 0.02])
    inertia = torch.tensor(
        [
            [0.31, 0.012, -0.006],
            [0.012, 0.42, 0.009],
            [-0.006, 0.009, 0.51],
        ]
    )
    axes = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
        ]
    )
    axes = axes / torch.linalg.norm(axes, dim=-1, keepdim=True)
    moments = torch.sum((axes @ inertia) * axes, dim=-1)
    half_sqrt = torch.sqrt(torch.tensor(0.5))
    quats = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [half_sqrt, half_sqrt, 0.0, 0.0],
            [half_sqrt, 0.0, half_sqrt, 0.0],
            [half_sqrt, 0.0, 0.0, half_sqrt],
        ]
    )
    force_w = torch.tensor([[0.0, 0.0, -rho * volume * gravity_z]]).repeat(quats.shape[0], 1)
    force_b = hydro.quat_apply_wxyz(hydro.quat_conjugate_wxyz(quats), force_w)
    torques_b = torch.cross(offset.reshape(1, 3).repeat(quats.shape[0], 1), force_b, dim=-1)

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _write_csv(
            root / "rigid_body_mass_readings.csv",
            ["sample_id", "mass_kg", "configuration"],
            [["m1", 11.48, "pool"], ["m2", 11.52, "pool"]],
        )
        _write_csv(
            root / "rigid_body_buoyancy_forces.csv",
            [
                "sample_id",
                "buoyancy_force_w_x_n",
                "buoyancy_force_w_y_n",
                "buoyancy_force_w_z_n",
                "water_density_kg_m3",
                "gravity_w_z_mps2",
            ],
            [[f"b{index}", 0.0, 0.0, -rho * volume * gravity_z, rho, gravity_z] for index in range(3)],
        )
        _write_csv(
            root / "rigid_body_axis_moments.csv",
            ["sample_id", "axis_b_x", "axis_b_y", "axis_b_z", "moment_kg_m2"],
            [
                [f"i{index}", *axes[index].tolist(), float(moments[index])]
                for index in range(axes.shape[0])
            ],
        )
        _write_csv(
            root / "rigid_body_static_buoyancy_torques.csv",
            [
                "sample_id",
                "quat_w",
                "quat_x",
                "quat_y",
                "quat_z",
                "buoyancy_torque_b_x_nm",
                "buoyancy_torque_b_y_nm",
                "buoyancy_torque_b_z_nm",
                "volume_m3",
                "water_density_kg_m3",
            ],
            [
                [f"c{index}", *quats[index].tolist(), *torques_b[index].tolist(), volume, rho]
                for index in range(quats.shape[0])
            ],
        )

        result = static_fit_cli.fit_static_calibration_logs(root, gravity_z=gravity_z)
        output_path = root / "static_updates.json"
        report_path = root / "static_report.json"
        exit_code = static_fit_cli.main(
            [str(root), "--output", str(output_path), "--report", str(report_path)]
        )
        output_updates, output_domain = profile_builder_cli.load_update_payload(output_path)
        report = json.loads(report_path.read_text(encoding="utf-8"))

    assert abs(result.cfg_updates["mass"] - 11.5) < 1.0e-6
    assert abs(result.cfg_updates["volume"] - volume) < 1.0e-7
    assert result.cfg_updates["water_rho"] == rho
    assert torch.allclose(torch.tensor(result.cfg_updates["com_to_cob_offset"]), offset, atol=1.0e-6)
    assert torch.allclose(torch.tensor(result.cfg_updates["inertia_diag"]), inertia, atol=1.0e-6)
    assert result.diagnostics["center_of_buoyancy"]["design_rank"] == 3
    assert result.diagnostics["inertia"]["design_rank"] == 6
    assert exit_code == 0
    assert output_updates == result.cfg_updates
    assert output_domain == {}
    assert report["source_files"] == list(result.source_files)


def test_buoyancy_uses_world_gravity_then_body_frame():
    model = hydro.HydrodynamicForceModels(num_envs=1, device=torch.device("cpu"))
    q_identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    gravity_w = torch.tensor([0.0, 0.0, -9.81])
    rho = 997.0
    volume = torch.tensor([[model_params.BLUEROV2_HEAVY.neutral_buoyancy_volume_m3]])
    cob = torch.tensor([[0.0, 0.0, 0.01]])

    force_b, torque_b = model.calculate_buoyancy_forces(q_identity, gravity_w, rho, volume, cob)
    expected_force_b = torch.tensor([[0.0, 0.0, rho * volume.item() * 9.81]])
    assert torch.allclose(force_b, expected_force_b, atol=1.0e-5)
    assert torch.allclose(torque_b, torch.cross(cob, expected_force_b, dim=-1), atol=1.0e-5)


def test_added_mass_coriolis_is_power_preserving():
    model = hydro.HydrodynamicForceModels(num_envs=2, device=torch.device("cpu"))
    nu_r = torch.tensor(
        [
            [0.3, -0.2, 0.1, 0.04, -0.05, 0.02],
            [-0.2, 0.1, 0.4, -0.03, 0.02, -0.06],
        ]
    )
    added_mass_diag = torch.tensor([1.0, 1.2, 1.4, 0.2, 0.25, 0.3])
    c_nu = model.calculate_added_mass_coriolis_wrench(nu_r, added_mass_diag)
    assert torch.allclose(torch.sum(nu_r * c_nu, dim=-1), torch.zeros(2), atol=1.0e-7)


def test_full_matrix_added_mass_coriolis_is_power_preserving():
    model = hydro.HydrodynamicForceModels(num_envs=2, device=torch.device("cpu"))
    nu_r = torch.tensor(
        [
            [0.3, -0.2, 0.1, 0.04, -0.05, 0.02],
            [-0.2, 0.1, 0.4, -0.03, 0.02, -0.06],
        ]
    )
    base = torch.tensor(
        [
            [1.0, 0.2, 0.0, 0.0, 0.0, 0.04],
            [0.2, 1.1, 0.1, 0.0, 0.0, 0.03],
            [0.0, 0.1, 1.4, 0.0, 0.05, 0.0],
            [0.0, 0.0, 0.0, 0.25, 0.02, 0.0],
            [0.0, 0.0, 0.05, 0.02, 0.3, 0.01],
            [0.04, 0.03, 0.0, 0.0, 0.01, 0.35],
        ]
    )
    added_mass = 0.5 * (base + base.T)
    c_nu = model.calculate_added_mass_coriolis_wrench(nu_r, added_mass)
    assert torch.allclose(torch.sum(nu_r * c_nu, dim=-1), torch.zeros(2), atol=1.0e-7)


def test_added_mass_inertia_wrench_is_negative_mass_times_relative_acceleration():
    model = hydro.HydrodynamicForceModels(num_envs=2, device=torch.device("cpu"))
    nu_r_dot = torch.tensor(
        [
            [0.2, -0.1, 0.05, 0.01, -0.02, 0.03],
            [-0.3, 0.2, -0.04, -0.01, 0.03, -0.02],
        ]
    )
    added_mass_diag = torch.tensor([1.0, 1.2, 1.4, 0.2, 0.25, 0.3])
    wrench = model.calculate_added_mass_inertia_wrench(nu_r_dot, added_mass_diag)
    expected = -added_mass_diag.reshape(1, 6) * nu_r_dot
    assert torch.allclose(wrench, expected)


def test_fossen_fluid_forces_include_added_mass_inertia():
    model = hydro.HydrodynamicForceModels(num_envs=1, device=torch.device("cpu"))
    q_identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    zeros_3 = torch.zeros(1, 3)
    zeros_6 = torch.zeros(6)
    gravity_w = torch.tensor([0.0, 0.0, -9.81])
    volume = torch.zeros(1, 1)
    cob = torch.zeros(1, 3)
    added_mass_diag = torch.tensor([1.0, 1.2, 1.4, 0.2, 0.25, 0.3])
    nu_r_dot = torch.tensor([[0.2, -0.1, 0.05, 0.01, -0.02, 0.03]])

    force, torque = model.calculate_fossen_fluid_forces(
        q_identity,
        zeros_3,
        zeros_3,
        gravity_w,
        997.0,
        volume,
        cob,
        zeros_6,
        zeros_6,
        zeros_3,
        added_mass_diag,
        nu_r_dot,
    )
    expected = -added_mass_diag.reshape(1, 6) * nu_r_dot
    assert torch.allclose(torch.cat((force, torque), dim=-1), expected)


def test_bluerov2_heavy_thruster_geometry_is_eight_t200s():
    offsets, quats = thrusters.get_thruster_com_and_orientations(torch.device("cpu"))
    assert offsets.shape == (8, 3)
    assert quats.shape == (8, 4)
    assert torch.allclose(torch.linalg.norm(quats, dim=-1), torch.ones(8), atol=1.0e-6)

    unit_x = torch.tensor([[1.0, 0.0, 0.0]]).repeat(8, 1)
    directions = hydro.quat_apply_wxyz(quats, unit_x)
    assert torch.allclose(directions[:4, 2], torch.zeros(4), atol=1.0e-6)
    assert torch.allclose(directions[:4].sum(dim=0)[1:], torch.zeros(2), atol=1.0e-6)
    assert directions[:4].sum(dim=0)[0] > 2.8
    assert torch.allclose(directions[4:], torch.tensor([[0.0, 0.0, 1.0]]).repeat(4, 1), atol=1.0e-6)


def test_bluerov2_heavy_model_parameters_are_consistent():
    model = model_params.BLUEROV2_HEAVY
    assert model.mass_kg == 11.5
    assert model.forward_bollard_thrust_kgf == 9.0
    assert model.vertical_bollard_thrust_kgf == 14.0
    assert abs(model.neutral_buoyancy_volume_m3 - model.mass_kg / model.water_density_kg_m3) < 1.0e-12

    expected_inertia = model_params.solid_box_inertia_diag(
        model.mass_kg,
        model.length_x_m,
        model.width_y_m,
        model.height_z_m,
    )
    assert model.inertia_diag_kg_m2 == expected_inertia
    assert model.physx_inertia_matrix_kg_m2 == (
        expected_inertia[0],
        0.0,
        0.0,
        0.0,
        expected_inertia[1],
        0.0,
        0.0,
        0.0,
        expected_inertia[2],
    )


def test_inertia_tensor_helper_accepts_diagonal_matrix_and_flat_values():
    diag = [1.0, 2.0, 3.0]
    matrix = [[1.0, 0.1, 0.0], [0.1, 2.0, 0.2], [0.0, 0.2, 3.0]]
    flat = [1.0, 0.1, 0.0, 0.1, 2.0, 0.2, 0.0, 0.2, 3.0]

    diag_matrix = rigid_body_properties.inertia_matrix_tensor(diag, torch.device("cpu"))
    full_matrix = rigid_body_properties.inertia_matrix_tensor(matrix, torch.device("cpu"))
    flat_matrix = rigid_body_properties.inertia_matrix_tensor(flat, torch.device("cpu"))

    assert torch.allclose(diag_matrix, torch.diag(torch.tensor(diag)))
    assert torch.allclose(full_matrix, torch.tensor(matrix))
    assert torch.allclose(flat_matrix, torch.tensor(matrix))
    assert torch.allclose(rigid_body_properties.inertia_diag_tensor(matrix, torch.device("cpu")), torch.tensor(diag))


def test_rigid_body_profile_accepts_full_symmetric_inertia_tensor():
    profile = profiles.PoolDynamicsProfile(
        rigid_body=profiles.RigidBodyProfile(
            inertia_diag=[
                [0.3, 0.01, 0.0],
                [0.01, 0.4, 0.02],
                [0.0, 0.02, 0.5],
            ]
        )
    )

    updates = profiles.pool_dynamics_profile_to_cfg_updates(profile)

    assert updates["inertia_diag"] == [
        [0.3, 0.01, 0.0],
        [0.01, 0.4, 0.02],
        [0.0, 0.02, 0.5],
    ]


def test_rigid_body_profile_rejects_nonsymmetric_inertia_tensor():
    bad_profile = profiles.PoolDynamicsProfile(
        rigid_body=profiles.RigidBodyProfile(
            inertia_diag=[
                [0.3, 0.01, 0.0],
                [0.02, 0.4, 0.02],
                [0.0, 0.02, 0.5],
            ]
        )
    )

    _assert_raises(ValueError, bad_profile.validate)


def test_nominal_pool_dynamics_profile_matches_vehicle_defaults():
    profile = profiles.NOMINAL_POOL_DYNAMICS_PROFILE
    updates = profiles.pool_dynamics_profile_to_cfg_updates(profile)

    model = model_params.BLUEROV2_HEAVY
    assert updates["mass"] == model.mass_kg
    assert updates["volume"] == model.neutral_buoyancy_volume_m3
    assert updates["water_rho"] == model.water_density_kg_m3
    assert updates["center_of_mass_offset"] == list(model.center_of_mass_offset_m)
    assert updates["com_to_cob_offset"] == list(model.center_of_buoyancy_from_com_m)
    assert len(updates["t200_max_forward_thrust"]) == 8
    assert len(updates["t200_max_reverse_thrust"]) == 8
    assert updates["use_thruster_lookup_table"] is False
    assert profiles.pool_dynamics_domain_randomization_updates(profile) == {}


class _DummyDomainRandomization:
    pass


class _DummyCfg:
    def __init__(self):
        self.domain_randomization = _DummyDomainRandomization()


def test_pool_dynamics_profile_applies_measured_parameters_to_cfg():
    full_linear_damping = [
        [1.0, 0.1, 0.0, 0.0, 0.0, 0.0],
        [0.1, 1.2, 0.0, 0.0, 0.0, 0.02],
        [0.0, 0.0, 1.4, 0.0, 0.03, 0.0],
        [0.0, 0.0, 0.0, 0.2, 0.0, 0.0],
        [0.0, 0.0, 0.03, 0.0, 0.25, 0.0],
        [0.0, 0.02, 0.0, 0.0, 0.0, 0.3],
    ]
    lookup_rows = [[-4.0, 0.0, 3.0, 6.0] for _ in range(8)]
    profile = profiles.PoolDynamicsProfile(
        name="measured-pool",
        hydrodynamics=profiles.HydrodynamicsProfile(
            linear_damping=full_linear_damping,
            quadratic_damping=[10.0, 12.0, 14.0, 0.4, 0.5, 0.6],
            speed_dependent_damping_enabled=True,
            damping_speed_points=[0.0, 0.5, 1.0],
            linear_damping_speed_scales=[
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [1.1, 1.0, 1.2, 1.0, 1.1, 1.0],
                [1.2, 1.1, 1.4, 1.0, 1.2, 1.1],
            ],
            quadratic_damping_speed_scales=[1.0, 1.15, 1.3],
            added_mass=[1.0, 1.1, 1.2, 0.1, 0.2, 0.3],
            water_current_w=[0.02, -0.01, 0.0],
            water_current_field_enabled=True,
            water_current_field_bounds=[0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
            water_current_field_shape=[2, 1, 1],
            water_current_field_values=[[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]],
        ),
        thrusters=profiles.ThrusterProfile(
            use_lookup_table=True,
            lookup_commands=[-1.0, 0.0, 0.5, 1.0],
            lookup_thrusts=lookup_rows,
            use_inflow_lookup_table=True,
            inflow_lookup_commands=[-1.0, 0.0, 1.0],
            inflow_lookup_speeds=[-0.5, 0.5],
            inflow_lookup_thrusts=[
                [-6.0, -4.0],
                [0.0, 0.0],
                [4.0, 6.0],
            ],
            command_delay_steps=2,
            max_command_rate=4.0,
            command_resolution=0.02,
            command_dropout_probability=0.05,
            wake_interaction_enabled=True,
            wake_loss_coefficient=0.2,
            wake_length=0.5,
            reaction_torque_coeff=0.02,
        ),
        pool_boundary=profiles.PoolBoundaryProfile(
            enabled=True,
            bounds=[-3.0, 3.0, -2.0, 2.0, 0.3, 2.2],
            effect_distance=0.4,
        ),
        free_surface=profiles.FreeSurfaceProfile(
            enabled=True,
            surface_z=0.3,
            effect_distance=0.5,
            heave_damping_scale=1.6,
            roll_pitch_damping_scale=1.25,
            added_mass_scale=1.2,
            buoyancy_scale=0.9,
            thrust_scale=0.75,
        ),
        tether=profiles.TetherProfile(
            enabled=True,
            slack_length=1.5,
            stiffness=15.0,
            num_segments=4,
            segment_diameter=0.006,
            segment_density=1200.0,
            segment_buoyancy_density=997.0,
        ),
        observation=profiles.ObservationProfile(
            noise_std=[0.01] * 17,
            bias_range=0.02,
            delay_steps=1,
            update_period_steps=2,
            dropout_probability={"linear_velocity_b": 0.1},
            lowpass_alpha={"linear_velocity_b": 0.4, "angular_velocity_b": 0.6},
            bias_drift_std={"position_error_b": 0.001},
        ),
        sensors=profiles.SensorProfile(
            imu=profiles.IMUSensorProfile(
                accelerometer_noise_std=[0.01, 0.01, 0.02],
                accelerometer_bias=[0.0, 0.0, 0.05],
                gyroscope_noise_std=0.001,
                gyroscope_scale=[1.0, 1.0, 0.99],
            ),
            depth=profiles.DepthSensorProfile(
                surface_z=0.3,
                depth_axis_sign=1.0,
                noise_std=0.02,
                bias=0.01,
                max_depth=20.0,
                dropout_probability=0.05,
            ),
            dvl=profiles.DVLSensorProfile(
                min_range=0.2,
                max_range=8.0,
                velocity_noise_std=[0.02, 0.02, 0.03],
                dropout_probability=0.1,
            ),
            position=profiles.PositionSensorProfile(
                reference_position_w=[0.0, 0.0, 0.3],
                max_range=6.0,
                position_noise_std=0.01,
                position_bias=[0.01, -0.01, 0.0],
            ),
        ),
        domain_randomization=profiles.DomainRandomizationProfile(
            use_custom_randomization=True,
            mass_range=[11.0, 12.0],
            thruster_command_resolution_range=[0.0, 0.02],
            thruster_command_dropout_probability_range=[0.0, 0.1],
            observation_delay_steps_range=[0, 2],
            observation_update_period_steps_range=[1, 3],
            observation_dropout_probability_range=[0.0, 0.2],
            observation_lowpass_alpha_range=[0.4, 1.0],
            observation_bias_drift_std_range=[0.0, 0.002],
            disturbance_curriculum=True,
            disturbance_curriculum_stage_steps=[10, 20],
            water_current_smooth=True,
            water_current_tau_range=[4.0, 8.0],
            water_current_max_by_stage=[0.02, 0.06, 0.10],
            water_current_vertical_max_by_stage=[0.005, 0.01, 0.02],
            water_current_variation_std_by_stage=[0.001, 0.003, 0.006],
        ),
    )

    cfg = profiles.apply_pool_dynamics_profile(_DummyCfg(), profile)

    assert cfg.linear_damping == full_linear_damping
    assert cfg.added_mass_diag == [1.0, 1.1, 1.2, 0.1, 0.2, 0.3]
    assert cfg.speed_dependent_damping_enabled is True
    assert cfg.damping_speed_points == [0.0, 0.5, 1.0]
    assert cfg.linear_damping_speed_scales[1] == [1.1, 1.0, 1.2, 1.0, 1.1, 1.0]
    assert cfg.quadratic_damping_speed_scales == [1.0, 1.15, 1.3]
    assert cfg.water_current_field_enabled is True
    assert cfg.water_current_field_shape == [2, 1, 1]
    assert cfg.water_current_field_values == [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]]
    assert cfg.use_thruster_lookup_table is True
    assert cfg.thruster_lookup_thrusts == lookup_rows
    assert cfg.use_thruster_inflow_lookup_table is True
    assert cfg.thruster_inflow_lookup_commands == [-1.0, 0.0, 1.0]
    assert cfg.thruster_inflow_lookup_speeds == [-0.5, 0.5]
    assert cfg.thruster_inflow_lookup_thrusts == [[-6.0, -4.0], [0.0, 0.0], [4.0, 6.0]]
    assert cfg.thruster_command_delay_steps == 2
    assert cfg.thruster_command_resolution == 0.02
    assert cfg.thruster_command_dropout_probability == 0.05
    assert cfg.thruster_wake_interaction_enabled is True
    assert cfg.thruster_wake_loss_coefficient == 0.2
    assert cfg.thruster_wake_length == 0.5
    assert cfg.pool_boundary_effects_enabled is True
    assert cfg.free_surface_effects_enabled is True
    assert cfg.free_surface_z == 0.3
    assert cfg.free_surface_heave_damping_scale == 1.6
    assert cfg.free_surface_thrust_scale == 0.75
    assert cfg.tether_enabled is True
    assert cfg.tether_num_segments == 4
    assert cfg.tether_segment_diameter == 0.006
    assert cfg.tether_segment_density == 1200.0
    assert cfg.tether_segment_buoyancy_density == 997.0
    assert cfg.observation_noise_std == [0.01] * 17
    assert cfg.observation_update_period_steps == 2
    assert cfg.observation_dropout_probability == {"linear_velocity_b": 0.1}
    assert cfg.observation_lowpass_alpha == {"linear_velocity_b": 0.4, "angular_velocity_b": 0.6}
    assert cfg.observation_bias_drift_std == {"position_error_b": 0.001}
    assert cfg.imu_accelerometer_noise_std == [0.01, 0.01, 0.02]
    assert cfg.imu_accelerometer_bias == [0.0, 0.0, 0.05]
    assert cfg.imu_gyroscope_noise_std == 0.001
    assert cfg.imu_gyroscope_scale == [1.0, 1.0, 0.99]
    assert cfg.depth_sensor_surface_z == 0.3
    assert cfg.depth_sensor_max_depth == 20.0
    assert cfg.depth_sensor_dropout_probability == 0.05
    assert cfg.dvl_min_range == 0.2
    assert cfg.dvl_max_range == 8.0
    assert cfg.dvl_velocity_noise_std == [0.02, 0.02, 0.03]
    assert cfg.dvl_dropout_probability == 0.1
    assert cfg.position_sensor_reference_position_w == [0.0, 0.0, 0.3]
    assert cfg.position_sensor_max_range == 6.0
    assert cfg.position_sensor_bias == [0.01, -0.01, 0.0]
    assert cfg.domain_randomization.use_custom_randomization is True
    assert cfg.domain_randomization.mass_range == [11.0, 12.0]
    assert cfg.domain_randomization.thruster_command_resolution_range == [0.0, 0.02]
    assert cfg.domain_randomization.thruster_command_dropout_probability_range == [0.0, 0.1]
    assert cfg.domain_randomization.observation_delay_steps_range == [0, 2]
    assert cfg.domain_randomization.observation_update_period_steps_range == [1, 3]
    assert cfg.domain_randomization.observation_dropout_probability_range == [0.0, 0.2]
    assert cfg.domain_randomization.observation_lowpass_alpha_range == [0.4, 1.0]
    assert cfg.domain_randomization.observation_bias_drift_std_range == [0.0, 0.002]
    assert cfg.domain_randomization.disturbance_curriculum is True
    assert cfg.domain_randomization.disturbance_curriculum_stage_steps == [10, 20]
    assert cfg.domain_randomization.water_current_smooth is True
    assert cfg.domain_randomization.water_current_tau_range == [4.0, 8.0]
    assert cfg.domain_randomization.water_current_max_by_stage == [0.02, 0.06, 0.10]
    assert cfg.domain_randomization.water_current_vertical_max_by_stage == [0.005, 0.01, 0.02]
    assert cfg.domain_randomization.water_current_variation_std_by_stage == [0.001, 0.003, 0.006]


def test_pool_dynamics_profile_rejects_missing_required_lookup_table():
    bad_profile = profiles.PoolDynamicsProfile(
        thrusters=profiles.ThrusterProfile(
            use_lookup_table=True,
            lookup_commands=[-1.0, 0.0, 1.0],
            lookup_thrusts=[],
        )
    )

    _assert_raises(ValueError, bad_profile.validate)


def test_pool_dynamics_profile_rejects_bad_damping_speed_curve():
    bad_profile = profiles.PoolDynamicsProfile(
        hydrodynamics=profiles.HydrodynamicsProfile(
            speed_dependent_damping_enabled=True,
            damping_speed_points=[0.0, 1.0],
            linear_damping_speed_scales=[1.0],
        )
    )

    _assert_raises(ValueError, bad_profile.validate)


def test_pool_dynamics_profile_rejects_bad_observation_estimator_parameters():
    bad_profile = profiles.PoolDynamicsProfile(
        observation=profiles.ObservationProfile(
            update_period_steps=0,
            dropout_probability=1.2,
            lowpass_alpha=-0.1,
            bias_drift_std=-0.01,
        )
    )

    _assert_raises(ValueError, bad_profile.validate)


def test_pool_dynamics_profile_rejects_bad_physical_sensor_parameters():
    bad_profile = profiles.PoolDynamicsProfile(
        sensors=profiles.SensorProfile(
            depth=profiles.DepthSensorProfile(depth_axis_sign=0.0),
            dvl=profiles.DVLSensorProfile(min_range=5.0, max_range=2.0),
            position=profiles.PositionSensorProfile(dropout_probability=1.5),
        )
    )

    _assert_raises(ValueError, bad_profile.validate)


def test_pool_dynamics_profile_rejects_bad_water_current_randomization_parameters():
    bad_profile = profiles.PoolDynamicsProfile(
        domain_randomization=profiles.DomainRandomizationProfile(
            water_current_tau_range=[0.0, 2.0],
            water_current_max_by_stage=[0.02, 0.04],
            water_current_vertical_max_by_stage=[0.01],
            disturbance_curriculum_stage_steps=[10],
        )
    )

    _assert_raises(ValueError, bad_profile.validate)


def test_pool_dynamics_profile_accepts_grouped_observation_parameters():
    profile = profiles.PoolDynamicsProfile(
        observation=profiles.ObservationProfile(
            noise_std={"linear_velocity_b": 0.01, "angular_velocity_b": [0.02, 0.02, 0.03]},
            bias_range={"position_error_b": 0.05},
            dropout_probability={"linear_velocity_b": 0.1},
            lowpass_alpha={"angular_velocity_b": 0.5},
            bias_drift_std={"position_error_b": 0.001},
        )
    )

    updates = profiles.pool_dynamics_profile_to_cfg_updates(profile)

    assert updates["observation_noise_std"] == {
        "linear_velocity_b": 0.01,
        "angular_velocity_b": [0.02, 0.02, 0.03],
    }
    assert updates["observation_bias_range"] == {"position_error_b": 0.05}
    assert updates["observation_dropout_probability"] == {"linear_velocity_b": 0.1}
    assert updates["observation_lowpass_alpha"] == {"angular_velocity_b": 0.5}
    assert updates["observation_bias_drift_std"] == {"position_error_b": 0.001}


def test_pool_dynamics_profile_round_trips_dict_and_json():
    profile = profiles.PoolDynamicsProfile(
        name="round-trip-pool",
        hydrodynamics=profiles.HydrodynamicsProfile(
            water_current_w=[0.01, 0.02, 0.0],
            added_mass=[1.0, 1.1, 1.2, 0.1, 0.2, 0.3],
        ),
        thrusters=profiles.ThrusterProfile(
            use_inflow_lookup_table=True,
            inflow_lookup_commands=[-1.0, 0.0, 1.0],
            inflow_lookup_speeds=[-0.5, 0.5],
            inflow_lookup_thrusts=[
                [-6.0, -4.0],
                [0.0, 0.0],
                [4.0, 6.0],
            ],
        ),
        sensors=profiles.SensorProfile(
            imu=profiles.IMUSensorProfile(accelerometer_noise_std=[0.01, 0.01, 0.02]),
            dvl=profiles.DVLSensorProfile(max_range=12.0, dropout_probability=0.05),
        ),
    )

    data = profiles.pool_dynamics_profile_to_dict(profile)
    restored = profiles.pool_dynamics_profile_from_dict(data)
    assert profiles.pool_dynamics_profile_to_dict(restored) == data

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "pool_profile.json"
        profiles.write_pool_dynamics_profile_json(profile, path)
        loaded = profiles.load_pool_dynamics_profile_json(path)
    assert profiles.pool_dynamics_profile_to_dict(loaded) == data


def test_pool_dynamics_profile_merges_flat_calibration_updates():
    lookup_rows = [[-4.0, 0.0, 6.0] for _ in range(8)]
    profile = profiles.merge_pool_dynamics_cfg_updates(
        cfg_updates=[
            {
                "mass": 11.7,
                "volume": 0.0118,
                "inertia_diag": [
                    [0.32, 0.01, 0.0],
                    [0.01, 0.41, 0.02],
                    [0.0, 0.02, 0.53],
                ],
                "com_to_cob_offset": [0.0, 0.0, 0.02],
            },
            {
                "added_mass_diag": [1.0, 1.1, 1.2, 0.1, 0.2, 0.3],
                "linear_damping": [1.0, 1.1, 1.2, 0.1, 0.2, 0.3],
                "use_thruster_lookup_table": True,
                "thruster_lookup_commands": [-1.0, 0.0, 1.0],
                "thruster_lookup_thrusts": lookup_rows,
                "depth_sensor_noise_std": 0.01,
            },
            {"mass_range": [11.5, 11.9]},
        ],
        domain_randomization_updates={"volume_range": [0.0116, 0.0120]},
        name="measured-pool",
        description="Merged from calibration updates.",
    )
    updates = profiles.pool_dynamics_profile_to_cfg_updates(profile)
    randomization_updates = profiles.pool_dynamics_domain_randomization_updates(profile)

    assert profile.name == "measured-pool"
    assert profile.description == "Merged from calibration updates."
    assert profile.rigid_body.mass == 11.7
    assert profile.rigid_body.inertia_diag[0][1] == 0.01
    assert profile.hydrodynamics.added_mass == [1.0, 1.1, 1.2, 0.1, 0.2, 0.3]
    assert profile.thrusters.use_lookup_table is True
    assert profile.sensors.depth.noise_std == 0.01
    assert updates["added_mass_diag"] == profile.hydrodynamics.added_mass
    assert updates["thruster_lookup_thrusts"] == lookup_rows
    assert randomization_updates["mass_range"] == [11.5, 11.9]
    assert randomization_updates["volume_range"] == [0.0116, 0.0120]


def test_pool_profile_builder_cli_merges_update_json_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        updates_path = root / "calibration_updates.json"
        randomization_path = root / "randomization_updates.json"
        output_path = root / "measured_profile.json"
        updates_path.write_text(
            json.dumps(
                {
                    "cfg_updates": {
                        "mass": 11.6,
                        "pool_boundary_effects_enabled": True,
                        "pool_bounds": [-3.0, 3.0, -2.0, 2.0, 0.2, 2.0],
                        "pool_boundary_damping_scale": 1.4,
                    },
                    "domain_randomization_updates": {"mass_range": [11.4, 11.8]},
                }
            ),
            encoding="utf-8",
        )
        randomization_path.write_text(
            json.dumps({"water_current_max_by_stage": [0.02], "water_current_vertical_max_by_stage": [0.005]}),
            encoding="utf-8",
        )

        profile = profile_builder_cli.build_profile_from_files(
            base_profile_path=None,
            update_paths=[updates_path],
            domain_randomization_update_paths=[randomization_path],
            name="cli-measured-pool",
            strict=True,
        )
        exit_code = profile_builder_cli.main(
            [
                "--updates",
                str(updates_path),
                "--domain-randomization-updates",
                str(randomization_path),
                "--name",
                "cli-measured-pool",
                "--output",
                str(output_path),
            ]
        )
        loaded = profiles.load_pool_dynamics_profile_json(output_path)

    assert profile.pool_boundary.enabled is True
    assert profile.pool_boundary.bounds == [-3.0, 3.0, -2.0, 2.0, 0.2, 2.0]
    assert profile.domain_randomization.mass_range == [11.4, 11.8]
    assert profile.domain_randomization.water_current_max_by_stage == [0.02]
    assert exit_code == 0
    assert loaded.name == "cli-measured-pool"
    assert loaded.rigid_body.mass == 11.6


def test_pool_dynamics_profile_rejects_unknown_json_fields():
    _assert_raises(
        ValueError,
        profiles.pool_dynamics_profile_from_dict,
        {"name": "bad-profile", "thrusters": {"not_a_thruster_parameter": 1.0}},
    )


def test_pool_dynamics_profile_audit_flags_nominal_high_fidelity_gaps():
    report = profiles.audit_pool_dynamics_profile(
        profiles.NOMINAL_POOL_DYNAMICS_PROFILE,
        profiles.PoolProfileAuditOptions(
            near_boundaries_expected=True,
            near_surface_expected=True,
            tether_expected=True,
            spatial_current_expected=True,
            physical_sensors_expected=True,
        ),
    )
    warning_sections = {finding.section for finding in report.findings if finding.severity == "warning"}

    assert "hydrodynamics.added_mass" in warning_sections
    assert "thrusters.lookup_table" in warning_sections
    assert "pool_boundary.enabled" in warning_sections
    assert "free_surface.enabled" in warning_sections
    assert "tether.enabled" in warning_sections
    assert "sensors" in warning_sections
    assert "domain_randomization" in warning_sections
    assert report.counts_by_severity["critical"] == 0
    assert report.readiness_score < 1.0
    assert report.to_dict()["profile_name"] == profiles.NOMINAL_POOL_DYNAMICS_PROFILE.name


def test_pool_profile_calibration_tasks_include_experiment_metadata():
    tasks = profiles.pool_profile_calibration_tasks(
        profiles.NOMINAL_POOL_DYNAMICS_PROFILE,
        profiles.PoolProfileAuditOptions(
            near_boundaries_expected=True,
            near_surface_expected=True,
            tether_expected=True,
            spatial_current_expected=True,
            physical_sensors_expected=True,
        ),
    )
    by_section = {task.section: task for task in tasks}

    assert "rigid_body.static_properties" in by_section
    assert "hydrodynamics.added_mass" in by_section
    assert "thrusters.lookup_table" in by_section
    assert by_section["rigid_body.static_properties"].priority == "P0"
    assert "fit_buoyancy_volume_from_forces" in by_section["rigid_body.static_properties"].calibration_functions
    assert "volume" in by_section["rigid_body.static_properties"].update_keys
    assert by_section["hydrodynamics.added_mass"].severity == "warning"
    assert "fit_full_matrix_added_mass_linear_quadratic_damping" in by_section[
        "hydrodynamics.added_mass"
    ].calibration_functions
    assert by_section["thrusters.lookup_table"].update_keys[0] == "use_thruster_lookup_table"
    assert all(task.to_dict()["section"] == task.section for task in tasks)


def test_pool_profile_calibration_update_template_groups_missing_fields():
    template = profiles.pool_profile_calibration_update_template(
        profiles.NOMINAL_POOL_DYNAMICS_PROFILE,
        profiles.PoolProfileAuditOptions(
            near_boundaries_expected=True,
            spatial_current_expected=True,
            physical_sensors_expected=True,
        ),
    )
    payload = template["update_payload"]

    assert template["template_type"] == "pool_calibration_update_template"
    assert "cfg_updates" not in template
    assert "domain_randomization_updates" not in template
    assert payload["cfg_updates"]["mass"] is None
    assert payload["cfg_updates"]["added_mass_diag"] is None
    assert payload["cfg_updates"]["use_thruster_lookup_table"] is None
    assert payload["cfg_updates"]["water_current_field_enabled"] is None
    assert payload["domain_randomization_updates"]["mass_range"] is None
    assert payload["domain_randomization_updates"]["water_current_max_by_stage"] is None
    assert template["unmapped_update_keys"] == []
    assert any(task["section"] == "hydrodynamics.water_current_field" for task in template["tasks"])


def test_pool_profile_calibration_log_schemas_describe_required_csv_inputs():
    schemas = profiles.pool_profile_calibration_log_schemas(
        profiles.NOMINAL_POOL_DYNAMICS_PROFILE,
        profiles.PoolProfileAuditOptions(
            near_boundaries_expected=True,
            spatial_current_expected=True,
            tether_expected=True,
            physical_sensors_expected=True,
        ),
    )
    by_filename = {schema.filename: schema for schema in schemas}

    assert "rigid_body_mass_readings.csv" in by_filename
    assert "hydrodynamics_motion_wrench_log.csv" in by_filename
    assert "thruster_static_stand.csv" in by_filename
    assert "water_current_field_samples.csv" in by_filename
    assert "sensor_reference_log.csv" in by_filename
    assert by_filename["rigid_body_mass_readings.csv"].csv_header == ("sample_id", "mass_kg", "configuration")
    assert "fit_mass_from_scale_readings" in by_filename["rigid_body_mass_readings.csv"].calibration_functions
    assert "nu_r_u_mps" in by_filename["hydrodynamics_motion_wrench_log.csv"].csv_header
    assert any(
        column.name == "nu_r_dot_u_mps2" and column.required is False
        for column in by_filename["hydrodynamics_motion_wrench_log.csv"].columns
    )
    assert by_filename["thruster_static_stand.csv"].to_dict()["columns"][0]["name"] == "thruster_index"


def test_pool_calibration_log_validator_detects_bad_values_and_missing_files():
    schemas = profiles.pool_profile_calibration_log_schemas(profiles.NOMINAL_POOL_DYNAMICS_PROFILE)
    mass_schema = next(schema for schema in schemas if schema.filename == "rigid_body_mass_readings.csv")

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        mass_path = root / mass_schema.filename
        mass_path.write_text("sample_id,mass_kg,configuration\nscale-1,11.5,pool\n", encoding="utf-8")
        valid_report = profiles.validate_pool_calibration_log_directory(root, (mass_schema,))

        mass_path.write_text("sample_id,mass_kg,configuration\nscale-1,nan,pool\n", encoding="utf-8")
        invalid_report = profiles.validate_pool_calibration_log_directory(root, (mass_schema,))

        mass_path.unlink()
        missing_report = profiles.validate_pool_calibration_log_directory(root, (mass_schema,))

    assert valid_report.is_valid
    assert valid_report.row_counts[mass_schema.filename] == 1
    assert valid_report.to_dict()["error_count"] == 0
    assert audit_cli.exit_code_for_log_validation(valid_report) == 0
    assert not invalid_report.is_valid
    assert any(issue.column == "mass_kg" and "valid float" in issue.message for issue in invalid_report.issues)
    assert audit_cli.exit_code_for_log_validation(invalid_report) == 2
    assert not missing_report.is_valid
    assert any("missing" in issue.message for issue in missing_report.issues)


def test_pool_dynamics_profile_audit_accepts_configured_pool_profile_without_warnings():
    full_linear = [[0.0 for _ in range(6)] for _ in range(6)]
    full_added_mass = [[0.0 for _ in range(6)] for _ in range(6)]
    for index in range(6):
        full_linear[index][index] = 1.0 + 0.1 * index
        full_added_mass[index][index] = 0.2 + 0.01 * index

    profile = profiles.PoolDynamicsProfile(
        hydrodynamics=profiles.HydrodynamicsProfile(
            linear_damping=full_linear,
            quadratic_damping=full_linear,
            speed_dependent_damping_enabled=True,
            damping_speed_points=[0.0, 1.0],
            linear_damping_speed_scales=[[1.0] * 6, [1.1] * 6],
            added_mass=full_added_mass,
            water_current_field_enabled=True,
            water_current_field_shape=[1, 1, 1],
            water_current_field_values=[[0.01, 0.0, 0.0]],
        ),
        thrusters=profiles.ThrusterProfile(
            command_delay_steps=1,
            max_command_rate=5.0,
            command_resolution=0.01,
            command_dropout_probability=0.01,
            use_lookup_table=True,
            lookup_commands=[-1.0, 0.0, 1.0],
            lookup_thrusts=[[-5.0, 0.0, 7.0] for _ in range(8)],
            use_inflow_lookup_table=True,
            inflow_lookup_commands=[-1.0, 0.0, 1.0],
            inflow_lookup_speeds=[-0.5, 0.5],
            inflow_lookup_thrusts=[
                [[-5.0, -4.0], [0.0, 0.0], [6.0, 5.0]]
                for _ in range(8)
            ],
        ),
        battery=profiles.BatteryProfile(voltage_drop_per_s=0.01),
        pool_boundary=profiles.PoolBoundaryProfile(enabled=True),
        free_surface=profiles.FreeSurfaceProfile(enabled=True),
        tether=profiles.TetherProfile(enabled=True, num_segments=3),
        observation=profiles.ObservationProfile(noise_std=0.01, delay_steps=1),
        sensors=profiles.SensorProfile(
            imu=profiles.IMUSensorProfile(accelerometer_noise_std=0.01),
            depth=profiles.DepthSensorProfile(noise_std=0.01, max_depth=20.0),
            dvl=profiles.DVLSensorProfile(velocity_noise_std=0.01, dropout_probability=0.01),
            position=profiles.PositionSensorProfile(position_noise_std=0.01, max_range=10.0),
        ),
        domain_randomization=profiles.DomainRandomizationProfile(
            water_current_max_by_stage=[0.05],
            water_current_vertical_max_by_stage=[0.01],
            water_current_variation_std_by_stage=[0.005],
        ),
    )

    report = profiles.audit_pool_dynamics_profile(
        profile,
        profiles.PoolProfileAuditOptions(
            near_boundaries_expected=True,
            near_surface_expected=True,
            tether_expected=True,
            spatial_current_expected=True,
            physical_sensors_expected=True,
        ),
    )

    assert report.counts_by_severity["warning"] == 0
    assert not report.has_blocking_findings()
    assert report.readiness_score > 0.8


def test_pool_profile_audit_cli_loads_profile_json_and_sets_exit_code():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "nominal_pool_profile.json"
        profiles.write_pool_dynamics_profile_json(profiles.NOMINAL_POOL_DYNAMICS_PROFILE, path)
        options = profiles.PoolProfileAuditOptions(physical_sensors_expected=True)
        report = audit_cli.load_and_audit_profile(path, options)
        text = audit_cli.format_audit_report(report)
        tasks = audit_cli.load_calibration_tasks(path, options)
        checklist_text = audit_cli.format_calibration_tasks(report.profile_name, tasks)
        template = audit_cli.load_calibration_update_template(path, options)
        log_schemas = audit_cli.load_calibration_log_schemas(path, options)
        log_template_dir = Path(tmpdir) / "log_templates"
        audit_cli.write_calibration_log_templates(log_template_dir, log_schemas)
        mass_csv = log_template_dir / "rigid_body_mass_readings.csv"
        schemas_json = log_template_dir / "schemas.json"
        mass_csv_text = mass_csv.read_text(encoding="utf-8")
        schemas_json_data = json.loads(schemas_json.read_text(encoding="utf-8"))
        mass_schema = next(schema for schema in log_schemas if schema.filename == mass_csv.name)
        mass_csv.write_text(mass_csv_text + "scale-1,11.5,pool\n", encoding="utf-8")
        log_validation = profiles.validate_pool_calibration_log_directory(log_template_dir, (mass_schema,))
        log_validation_text = audit_cli.format_calibration_log_validation_report(log_validation)

    assert "Profile: bluerov2-heavy-nominal-pool" in text
    assert "Readiness score:" in text
    assert report.to_dict()["counts_by_severity"]["warning"] > 0
    assert audit_cli.exit_code_for_report(report, fail_on_warning=True, fail_on_critical=False) == 2
    assert audit_cli.exit_code_for_report(report, fail_on_warning=False, fail_on_critical=True) == 0
    assert "Calibration tasks:" in checklist_text
    assert "rigid_body.static_properties" in checklist_text
    assert any(task.section == "sensors" for task in tasks)
    assert template["update_payload"]["cfg_updates"]["mass"] is None
    assert any(task["section"] == "sensors" for task in template["tasks"])
    assert any(schema.filename == "sensor_reference_log.csv" for schema in log_schemas)
    assert mass_csv_text.startswith("sample_id,mass_kg,configuration")
    assert schemas_json_data[0]["filename"]
    assert log_validation.is_valid
    assert "Valid: yes" in log_validation_text


def test_bluerov2_heavy_vehicle_thrust_calibration():
    model = model_params.BLUEROV2_HEAVY
    offsets, quats = thrusters.get_thruster_com_and_orientations(torch.device("cpu"))
    del offsets

    unit_x = torch.tensor([[1.0, 0.0, 0.0]]).repeat(8, 1)
    directions = hydro.quat_apply_wxyz(quats, unit_x)
    horizontal_thrust = model.forward_bollard_thrust_kgf * model_params.KGF_TO_NEWTON / (
        2.0 * (0.7431448255 + 0.6691306064)
    )
    vertical_thrust = model.vertical_bollard_thrust_kgf * model_params.KGF_TO_NEWTON / 4.0
    forward_force = torch.sum(directions[:4] * horizontal_thrust, dim=0)
    vertical_force = torch.sum(directions[4:] * vertical_thrust, dim=0)
    assert torch.allclose(forward_force[0], torch.tensor(model.forward_bollard_thrust_kgf * model_params.KGF_TO_NEWTON))
    assert torch.allclose(forward_force[1:], torch.zeros(2), atol=1.0e-5)
    assert torch.allclose(
        vertical_force,
        torch.tensor([0.0, 0.0, model.vertical_bollard_thrust_kgf * model_params.KGF_TO_NEWTON]),
        atol=1.0e-4,
    )


def test_t200_conversion_is_asymmetric_and_quadratic():
    conversion = thrusters.ConversionFunctionT200(
        max_forward_thrust=[10.0, 20.0],
        max_reverse_thrust=[5.0, 8.0],
    )
    commands = torch.tensor([[1.0, -1.0], [0.5, -0.5], [0.0, 0.0]])
    thrust = conversion.convert(commands)
    expected = torch.tensor([[10.0, -8.0], [2.5, -2.0], [0.0, 0.0]])
    assert torch.allclose(thrust, expected)


def test_calibration_fits_thruster_static_lookup_table_from_stand_samples():
    command_samples = torch.tensor([-1.0, -0.05, 0.0, 0.05, 1.0])
    thrust_samples = torch.tensor(
        [
            [-4.0, -8.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [6.0, 10.0],
        ]
    )

    fit = calibration.fit_thruster_static_lookup_table(
        command_samples,
        thrust_samples,
        command_points=[-1.0, 0.0, 1.0],
        deadband_thrust_threshold=0.1,
    )
    updates = fit.to_cfg_updates(include_deadband=True)

    assert torch.allclose(fit.thrust_points, torch.tensor([[-4.0, 0.0, 6.0], [-8.0, 0.0, 10.0]]))
    assert torch.equal(fit.sample_count, torch.tensor([[1, 3, 1], [1, 3, 1]]))
    assert abs(fit.estimated_deadband - 0.05) < 1.0e-6
    assert updates["use_thruster_lookup_table"] is True
    assert updates["thruster_lookup_commands"] == [-1.0, 0.0, 1.0]
    assert updates["thruster_lookup_thrusts"] == fit.thrust_points.tolist()
    assert updates["thruster_deadband"] == fit.estimated_deadband


def test_calibration_fits_thruster_inflow_lookup_table_from_grid_samples():
    command_points = torch.tensor([-1.0, 0.0, 1.0])
    inflow_points = torch.tensor([-0.5, 0.5])
    table = torch.tensor(
        [
            [[-6.0, -4.0], [0.0, 0.0], [4.0, 6.0]],
            [[-8.0, -7.0], [0.0, 0.0], [8.0, 7.0]],
        ]
    )
    command_samples = []
    inflow_samples = []
    thrust_samples = []
    for command_index, command in enumerate(command_points):
        for inflow_index, inflow in enumerate(inflow_points):
            command_samples.append(float(command))
            inflow_samples.append(float(inflow))
            thrust_samples.append(table[:, command_index, inflow_index].tolist())

    fit = calibration.fit_thruster_inflow_lookup_table(
        command_samples,
        inflow_samples,
        thrust_samples,
        command_points=command_points,
        inflow_speed_points=inflow_points,
    )
    updates = fit.to_cfg_updates()

    assert torch.allclose(fit.thrust_points, table)
    assert torch.equal(fit.sample_count, torch.ones_like(table, dtype=torch.long))
    assert updates["use_thruster_inflow_lookup_table"] is True
    assert updates["thruster_inflow_lookup_commands"] == command_points.tolist()
    assert updates["thruster_inflow_lookup_speeds"] == inflow_points.tolist()
    assert updates["thruster_inflow_lookup_thrusts"] == table.tolist()


def test_calibration_fits_thruster_first_order_response_from_step_log():
    time_s = torch.arange(0.0, 2.0, 0.01)
    delay = 0.2
    tau = 0.4
    steady = 10.0
    progress = torch.where(
        time_s <= delay,
        torch.zeros_like(time_s),
        1.0 - torch.exp(-(time_s - delay) / tau),
    )
    thrust = steady * progress

    fit = calibration.fit_thruster_first_order_response(
        time_s,
        thrust,
        command_step_time_s=0.0,
        initial_thrust=0.0,
        steady_state_thrust=steady,
        delay_candidate_count=128,
    )
    updates = fit.to_cfg_updates(physics_dt_s=0.05)

    assert abs(fit.time_constant_s - tau) < 5.0e-3
    assert abs(fit.response_delay_s - delay) < 5.0e-3
    assert fit.residual_rms < 1.0e-3
    assert updates["dyn_time_constant"] == fit.time_constant_s
    assert updates["thruster_command_delay_steps"] == 4


def test_calibration_fits_thruster_voltage_exponent():
    voltage = torch.tensor([12.0, 14.0, 16.0, 18.0])
    exponent = 2.3
    scale = (voltage / 16.0) ** exponent

    fit = calibration.fit_thruster_voltage_exponent(voltage, scale, nominal_voltage=16.0)
    updates = fit.to_cfg_updates()

    assert abs(fit.thrust_exponent - exponent) < 1.0e-6
    assert fit.sample_count == 3
    assert fit.residual_rms < 1.0e-6
    assert updates["battery_voltage_nominal"] == 16.0
    assert updates["battery_voltage_thrust_exponent"] == fit.thrust_exponent


def test_calibration_fits_linear_battery_voltage_sag():
    time_s = torch.tensor([10.0, 11.0, 12.0, 13.0])
    voltage = torch.tensor([16.0, 15.9, 15.8, 15.7])

    fit = calibration.fit_battery_voltage_sag(time_s, voltage)
    updates = fit.to_cfg_updates()

    assert abs(fit.initial_voltage - 16.0) < 1.0e-6
    assert abs(fit.min_observed_voltage - 15.7) < 1.0e-6
    assert abs(fit.voltage_drop_per_s - 0.1) < 1.0e-6
    assert fit.residual_rms < 1.0e-6
    assert fit.sample_count == 4
    assert fit.time_origin_s == 10.0
    assert updates["battery_voltage"] == fit.initial_voltage
    assert updates["battery_min_voltage"] == fit.min_observed_voltage


def test_thruster_calibration_log_pipeline_builds_lookup_and_response_updates():
    command_points = [-1.0, 0.0, 1.0]
    inflow_points = [-0.5, 0.5]
    inflow_table = [
        [-6.0, -4.0],
        [0.0, 0.0],
        [4.0, 6.0],
    ]
    time_s = torch.arange(0.0, 4.0, 0.02)
    step_time = 0.5
    response_delay = 0.14
    tau = 0.35
    steady_thrust = 6.0
    commands = torch.where(time_s < step_time, torch.zeros_like(time_s), torch.ones_like(time_s))
    response_start = step_time + response_delay
    thrust = torch.where(
        time_s <= response_start,
        torch.zeros_like(time_s),
        steady_thrust * (1.0 - torch.exp(-(time_s - response_start) / tau)),
    )
    battery_time = torch.tensor([0.0, 1.0, 2.0, 3.0])
    battery_voltage = torch.tensor([16.0, 15.9, 15.8, 15.7])
    voltage_exponent = 2.3
    battery_thrust_scale = (battery_voltage / 16.0) ** voltage_exponent

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _write_csv(
            root / "thruster_static_stand.csv",
            ["thruster_index", "command", "thrust_n", "voltage_v", "current_a"],
            [
                ["shared", -1.0, -4.0, 16.0, 5.0],
                ["shared", 0.0, 0.0, 16.0, 0.0],
                ["shared", 1.0, 6.0, 16.0, 7.0],
            ],
        )
        _write_csv(
            root / "thruster_inflow_stand.csv",
            ["thruster_index", "command", "axial_inflow_speed_mps", "thrust_n"],
            [
                ["shared", command, inflow, inflow_table[command_index][inflow_index]]
                for command_index, command in enumerate(command_points)
                for inflow_index, inflow in enumerate(inflow_points)
            ],
        )
        _write_csv(
            root / "thruster_step_response.csv",
            ["time_s", "command", "measured_thrust_n", "voltage_v"],
            [
                [float(time_s[index]), float(commands[index]), float(thrust[index]), 16.0]
                for index in range(time_s.numel())
            ],
        )
        _write_csv(
            root / "battery_voltage_thrust_samples.csv",
            ["time_s", "voltage_v", "thrust_scale"],
            [
                [float(battery_time[index]), float(battery_voltage[index]), float(battery_thrust_scale[index])]
                for index in range(battery_time.numel())
            ],
        )

        result = thruster_fit_cli.fit_thruster_calibration_logs(root, physics_dt_s=0.02)
        output_path = root / "thruster_updates.json"
        report_path = root / "thruster_report.json"
        exit_code = thruster_fit_cli.main(
            [
                str(root),
                "--physics-dt",
                "0.02",
                "--output",
                str(output_path),
                "--report",
                str(report_path),
            ]
        )
        output_updates, output_domain = profile_builder_cli.load_update_payload(output_path)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        profile = profiles.merge_pool_dynamics_cfg_updates(cfg_updates=output_updates)

    assert result.cfg_updates["thruster_lookup_commands"] == command_points
    assert result.cfg_updates["thruster_lookup_thrusts"] == [-4.0, 0.0, 6.0]
    assert result.cfg_updates["thruster_inflow_lookup_speeds"] == inflow_points
    assert result.cfg_updates["thruster_inflow_lookup_thrusts"] == inflow_table
    assert abs(result.cfg_updates["dyn_time_constant"] - tau) < 0.02
    assert abs(result.diagnostics["first_order_response"]["response_delay_s"] - response_delay) < 0.03
    assert result.cfg_updates["thruster_command_delay_steps"] in (6, 7, 8)
    assert abs(result.cfg_updates["battery_voltage"] - 16.0) < 1.0e-6
    assert abs(result.cfg_updates["battery_voltage_drop_per_s"] - 0.1) < 1.0e-6
    assert abs(result.cfg_updates["battery_voltage_thrust_exponent"] - voltage_exponent) < 1.0e-5
    assert exit_code == 0
    assert output_updates == result.cfg_updates
    assert output_domain == {}
    assert profile.thrusters.use_lookup_table is True
    assert profile.thrusters.use_inflow_lookup_table is True
    assert report["source_files"] == list(result.source_files)


def test_lookup_table_conversion_interpolates_shared_curve():
    conversion = thrusters.ConversionFunctionLookupTable(
        command_points=[-1.0, 0.0, 1.0],
        thrust_points=[-4.0, 0.0, 6.0],
    )
    commands = torch.tensor([[-0.5, 0.5, 2.0]])
    thrust = conversion.convert(commands)
    expected = torch.tensor([[-2.0, 3.0, 6.0]])
    assert torch.allclose(thrust, expected)


def test_lookup_table_conversion_supports_per_thruster_curves():
    conversion = thrusters.ConversionFunctionLookupTable(
        command_points=[-1.0, 0.0, 1.0],
        thrust_points=[
            [-4.0, 0.0, 6.0],
            [-8.0, 0.0, 10.0],
        ],
    )
    commands = torch.tensor([[-0.5, 0.5], [1.0, -1.0]])
    thrust = conversion.convert(commands)
    expected = torch.tensor([[-2.0, 5.0], [6.0, -8.0]])
    assert torch.allclose(thrust, expected)


def test_inflow_lookup_table_conversion_interpolates_shared_surface():
    conversion = thrusters.ConversionFunctionInflowLookupTable(
        command_points=[-1.0, 0.0, 1.0],
        inflow_speed_points=[-1.0, 1.0],
        thrust_points=[
            [-8.0, -12.0],
            [2.0, -2.0],
            [12.0, 8.0],
        ],
    )
    commands = torch.tensor([[0.5, -0.5, 2.0]])
    axial_inflow = torch.tensor([[0.0, 1.0, -2.0]])

    thrust = conversion.convert(commands, axial_inflow)

    expected = torch.tensor([[5.0, -7.0, 12.0]])
    assert torch.allclose(thrust, expected)


def test_inflow_lookup_table_conversion_supports_per_thruster_surfaces():
    conversion = thrusters.ConversionFunctionInflowLookupTable(
        command_points=[0.0, 1.0],
        inflow_speed_points=[0.0, 1.0],
        thrust_points=[
            [[0.0, -1.0], [10.0, 8.0]],
            [[0.0, -2.0], [20.0, 16.0]],
        ],
    )
    commands = torch.tensor([[0.5, 0.5]])
    axial_inflow = torch.tensor([[0.5, 0.5]])

    thrust = conversion.convert(commands, axial_inflow)

    expected = torch.tensor([[4.25, 8.5]])
    assert torch.allclose(thrust, expected)


def test_thruster_command_processor_applies_step_delay():
    processor = thrusters.ThrusterCommandProcessor(
        numEnvs=1,
        num_thrusters_per_env=2,
        max_delay_steps=2,
        device=torch.device("cpu"),
    )
    delay_steps = torch.tensor([2])
    max_rate = torch.tensor([0.0])

    out_1 = processor.update(torch.tensor([[1.0, -1.0]]), delay_steps, max_rate, 0.1)
    out_2 = processor.update(torch.tensor([[0.5, 0.5]]), delay_steps, max_rate, 0.1)
    out_3 = processor.update(torch.tensor([[0.0, 0.0]]), delay_steps, max_rate, 0.1)

    assert torch.allclose(out_1, torch.zeros(1, 2))
    assert torch.allclose(out_2, torch.zeros(1, 2))
    assert torch.allclose(out_3, torch.tensor([[1.0, -1.0]]))


def test_thruster_command_processor_applies_rate_limit():
    processor = thrusters.ThrusterCommandProcessor(
        numEnvs=1,
        num_thrusters_per_env=2,
        max_delay_steps=0,
        device=torch.device("cpu"),
    )
    delay_steps = torch.tensor([0])
    max_rate = torch.tensor([2.0])

    out_1 = processor.update(torch.tensor([[1.0, -1.0]]), delay_steps, max_rate, 0.1)
    out_2 = processor.update(torch.tensor([[1.0, -1.0]]), delay_steps, max_rate, 0.1)
    out_3 = processor.update(torch.tensor([[-1.0, 1.0]]), delay_steps, max_rate, 0.1)

    assert torch.allclose(out_1, torch.tensor([[0.2, -0.2]]))
    assert torch.allclose(out_2, torch.tensor([[0.4, -0.4]]))
    assert torch.allclose(out_3, torch.tensor([[0.2, -0.2]]))


def test_thruster_command_processor_broadcasts_per_env_rate_limit():
    processor = thrusters.ThrusterCommandProcessor(
        numEnvs=2,
        num_thrusters_per_env=2,
        max_delay_steps=0,
        device=torch.device("cpu"),
    )
    commands = torch.tensor([[1.0, -1.0], [1.0, -1.0]])
    delay_steps = torch.tensor([0, 0])
    max_rate = torch.tensor([[1.0], [3.0]])

    out = processor.update(commands, delay_steps, max_rate, 0.1)

    assert torch.allclose(out, torch.tensor([[0.1, -0.1], [0.3, -0.3]]))


def test_thruster_command_processor_quantizes_commands():
    processor = thrusters.ThrusterCommandProcessor(
        numEnvs=1,
        num_thrusters_per_env=2,
        max_delay_steps=0,
        device=torch.device("cpu"),
    )

    out = processor.update(
        torch.tensor([[0.26, -0.24]]),
        delay_steps=torch.tensor([0]),
        max_rate=torch.tensor([0.0]),
        dt=0.1,
        command_resolution=torch.tensor([0.1]),
    )

    assert torch.allclose(out, torch.tensor([[0.3, -0.2]]))


def test_thruster_command_processor_dropout_holds_previous_command():
    processor = thrusters.ThrusterCommandProcessor(
        numEnvs=1,
        num_thrusters_per_env=2,
        max_delay_steps=0,
        device=torch.device("cpu"),
    )

    out_1 = processor.update(
        torch.tensor([[0.8, -0.8]]),
        delay_steps=torch.tensor([0]),
        max_rate=torch.tensor([0.0]),
        dt=0.1,
        dropout_probability=torch.tensor([0.0]),
    )
    out_2 = processor.update(
        torch.tensor([[-0.5, 0.5]]),
        delay_steps=torch.tensor([0]),
        max_rate=torch.tensor([0.0]),
        dt=0.1,
        dropout_probability=torch.tensor([1.0]),
    )

    assert torch.allclose(out_1, torch.tensor([[0.8, -0.8]]))
    assert torch.allclose(out_2, out_1)


def test_voltage_thrust_scale_uses_nominal_voltage_ratio():
    voltage = torch.tensor([[16.0], [14.0]])
    scale = thrusters.calculate_voltage_thrust_scale(voltage, nominal_voltage=16.0, exponent=2.0)
    expected = torch.tensor([[1.0], [(14.0 / 16.0) ** 2]])
    assert torch.allclose(scale, expected)


def test_axial_inflow_thrust_scale_reduces_positive_inflow_only():
    axial_inflow = torch.tensor([[-1.0, 0.0, 0.5, 2.0]])
    scale = thrusters.calculate_axial_inflow_thrust_scale(
        axial_inflow,
        loss_coefficient=0.5,
        reference_speed=1.0,
        min_scale=0.4,
    )
    expected = torch.tensor([[1.0, 1.0, 0.875, 0.4]])
    assert torch.allclose(scale, expected)


def test_thruster_wake_interaction_reduces_downstream_thruster_only():
    positions = torch.tensor([[[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.0, 0.3, 0.0]]])
    axes = torch.tensor([[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]])
    thrust = torch.tensor([[10.0, 0.0, 0.0]])

    scale = thrusters.calculate_thruster_wake_interaction_scale(
        positions,
        axes,
        thrust,
        wake_length=1.0,
        wake_radius=0.1,
        loss_coefficient=0.5,
        expansion_rate=0.0,
        min_scale=0.2,
        reference_thrust=10.0,
    )

    expected = torch.tensor([[1.0, 0.6, 1.0]])
    assert torch.allclose(scale, expected, atol=1.0e-6)


def test_thruster_reaction_torques_follow_spin_direction_and_signed_thrust():
    thrust = torch.tensor([[10.0, -5.0]])
    axes = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]])

    torques = thrusters.calculate_reaction_torques(
        thrust,
        axes,
        torque_coeff=0.1,
        spin_directions=[1.0, -1.0],
    )

    expected = torch.tensor([[[-1.0, 0.0, 0.0], [0.0, 0.0, -0.5]]])
    assert torch.allclose(torques, expected)


def test_observation_delay_buffer_returns_current_until_history_is_available():
    buffer = sensors.ObservationDelayBuffer(num_envs=1, obs_dim=2, max_delay_steps=2, device=torch.device("cpu"))

    out_1 = buffer.update(torch.tensor([[1.0, 2.0]]), delay_steps=torch.tensor([2]))
    out_2 = buffer.update(torch.tensor([[3.0, 4.0]]), delay_steps=torch.tensor([2]))
    out_3 = buffer.update(torch.tensor([[5.0, 6.0]]), delay_steps=torch.tensor([2]))

    assert torch.allclose(out_1, torch.tensor([[1.0, 2.0]]))
    assert torch.allclose(out_2, torch.tensor([[1.0, 2.0]]))
    assert torch.allclose(out_3, torch.tensor([[1.0, 2.0]]))


def test_observation_sensor_model_adds_bias_without_noise():
    buffer = sensors.ObservationDelayBuffer(num_envs=2, obs_dim=3, max_delay_steps=0, device=torch.device("cpu"))
    obs = torch.tensor([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]])
    bias = torch.tensor([[0.1, 0.2, 0.3], [-0.1, -0.2, -0.3]])

    measured = sensors.apply_observation_sensor_model(
        obs,
        buffer,
        delay_steps=torch.tensor([0, 0]),
        noise_std=0.0,
        bias=bias,
    )

    assert torch.allclose(measured, obs + bias)


def test_observation_filter_state_holds_between_update_periods():
    state = sensors.ObservationFilterState(num_envs=1, obs_dim=2, device=torch.device("cpu"))
    zeros = torch.zeros(1, 2)

    out_1 = state.update(torch.tensor([[1.0, 2.0]]), zeros, 0.0, update_period_steps=torch.tensor([2]))
    out_2 = state.update(torch.tensor([[3.0, 4.0]]), zeros, 0.0, update_period_steps=torch.tensor([2]))
    out_3 = state.update(torch.tensor([[5.0, 6.0]]), zeros, 0.0, update_period_steps=torch.tensor([2]))

    assert torch.allclose(out_1, torch.tensor([[1.0, 2.0]]))
    assert torch.allclose(out_2, out_1)
    assert torch.allclose(out_3, torch.tensor([[5.0, 6.0]]))


def test_observation_filter_state_dropout_holds_previous_measurement():
    state = sensors.ObservationFilterState(num_envs=1, obs_dim=2, device=torch.device("cpu"))
    zeros = torch.zeros(1, 2)

    out_1 = state.update(torch.tensor([[1.0, 2.0]]), zeros, 0.0, dropout_probability=0.0)
    out_2 = state.update(torch.tensor([[3.0, 4.0]]), zeros, 0.0, dropout_probability=1.0)

    assert torch.allclose(out_1, torch.tensor([[1.0, 2.0]]))
    assert torch.allclose(out_2, out_1)


def test_observation_filter_state_lowpass_filters_updates():
    state = sensors.ObservationFilterState(num_envs=1, obs_dim=1, device=torch.device("cpu"))

    out_1 = state.update(torch.tensor([[0.0]]), 0.0, 0.0, lowpass_alpha=0.5)
    out_2 = state.update(torch.tensor([[2.0]]), 0.0, 0.0, lowpass_alpha=0.5)
    out_3 = state.update(torch.tensor([[2.0]]), 0.0, 0.0, lowpass_alpha=0.5)

    assert torch.allclose(out_1, torch.tensor([[0.0]]))
    assert torch.allclose(out_2, torch.tensor([[1.0]]))
    assert torch.allclose(out_3, torch.tensor([[1.5]]))


def test_observation_filter_state_bias_drift_changes_measurement():
    torch.manual_seed(0)
    state = sensors.ObservationFilterState(num_envs=1, obs_dim=2, device=torch.device("cpu"))

    measured = state.update(
        torch.zeros(1, 2),
        fixed_bias=0.0,
        noise_std=0.0,
        bias_drift_std=torch.ones(1, 2),
        dt=0.25,
    )

    assert not torch.allclose(measured, torch.zeros(1, 2))


def test_observation_group_parameter_builds_semantic_vector():
    reference = torch.zeros(2, 7)
    groups = {
        "position_error_b": slice(0, 3),
        "linear_velocity_b": slice(3, 6),
        "depth": 6,
    }

    parameter = sensors.build_observation_group_parameter(
        {
            "position_error_b": [0.1, 0.2, 0.3],
            "linear_velocity_b": 0.4,
            "depth": [1.0, 2.0],
        },
        groups,
        reference,
    )

    expected = torch.tensor(
        [
            [0.1, 0.2, 0.3, 0.4, 0.4, 0.4, 1.0],
            [0.1, 0.2, 0.3, 0.4, 0.4, 0.4, 2.0],
        ]
    )
    assert torch.allclose(parameter, expected)


def test_sensor_channel_model_applies_scale_bias_clamp_and_hold():
    true_value = torch.tensor([[1.0, -2.0]])
    previous = torch.tensor([[9.0, 9.0]])

    measured = sensors.apply_sensor_channel_model(
        true_value,
        scale=torch.tensor([2.0, 3.0]),
        bias=torch.tensor([0.5, -0.5]),
        min_value=-5.0,
        max_value=3.0,
        dropout_probability=0.0,
    )
    dropped = sensors.apply_sensor_channel_model(
        true_value,
        dropout_probability=1.0,
        previous_measurement=previous,
    )

    assert torch.allclose(measured.value, torch.tensor([[2.5, -5.0]]))
    assert torch.equal(measured.valid, torch.ones(1, 2, dtype=torch.bool))
    assert torch.allclose(dropped.value, previous)
    assert torch.equal(dropped.valid, torch.zeros(1, 2, dtype=torch.bool))


def test_imu_sensor_model_reports_specific_force_and_gyro():
    measurement = sensors.calculate_imu_measurement(
        body_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        linear_acceleration_w=torch.zeros(1, 3),
        angular_velocity_b=torch.tensor([[0.1, -0.2, 0.3]]),
        gravity_w=[0.0, 0.0, -9.81],
        accelerometer_bias=torch.tensor([0.0, 0.0, 0.1]),
        gyroscope_scale=torch.tensor([2.0, 1.0, 0.5]),
    )

    assert torch.allclose(measurement.accelerometer_b, torch.tensor([[0.0, 0.0, 9.91]]))
    assert torch.allclose(measurement.gyroscope_b, torch.tensor([[0.2, -0.2, 0.15]]))
    assert torch.equal(measurement.valid, torch.ones(1, 1, dtype=torch.bool))


def test_depth_sensor_model_supports_positive_down_axis_and_hold():
    position_w = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 3.0]])
    previous = torch.tensor([[4.0], [5.0]])

    measured = sensors.calculate_depth_sensor_measurement(
        position_w,
        surface_z=1.0,
        depth_axis_sign=1.0,
        depth_scale=2.0,
        depth_bias=0.1,
        max_depth=3.0,
    )
    dropped = sensors.calculate_depth_sensor_measurement(
        position_w,
        surface_z=1.0,
        dropout_probability=1.0,
        previous_depth=previous,
    )

    assert torch.allclose(measured.depth, torch.tensor([[0.1], [3.0]]))
    assert torch.equal(measured.valid, torch.ones(2, 1, dtype=torch.bool))
    assert torch.allclose(dropped.depth, previous)
    assert torch.equal(dropped.valid, torch.zeros(2, 1, dtype=torch.bool))


def test_dvl_sensor_model_subtracts_water_velocity_and_range_validity():
    velocity = torch.tensor([[1.0, 0.0, -0.2], [0.5, 0.2, 0.0]])
    water_velocity = torch.tensor([[0.2, -0.1, 0.0], [0.1, 0.1, 0.0]])
    previous = torch.tensor([[9.0, 9.0, 9.0], [8.0, 8.0, 8.0]])

    measurement = sensors.calculate_dvl_velocity_measurement(
        velocity,
        altitude=torch.tensor([1.0, 5.0]),
        max_range=3.0,
        water_velocity_b=water_velocity,
        velocity_bias=torch.tensor([0.0, 0.0, 0.1]),
        previous_velocity_b=previous,
    )

    assert torch.allclose(measurement.velocity_b[0], torch.tensor([0.8, 0.1, -0.1]))
    assert torch.allclose(measurement.velocity_b[1], previous[1])
    assert torch.equal(measurement.valid, torch.tensor([[True], [False]]))


def test_position_sensor_model_range_dropouts_hold_previous():
    position = torch.tensor([[1.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
    previous = torch.tensor([[9.0, 9.0, 9.0], [8.0, 8.0, 8.0]])

    measurement = sensors.calculate_position_sensor_measurement(
        position,
        reference_position_w=[0.0, 0.0, 0.0],
        max_range=2.0,
        position_bias=torch.tensor([0.1, 0.0, -0.1]),
        previous_position_w=previous,
    )

    assert torch.allclose(measurement.position_w[0], torch.tensor([1.1, 0.0, -0.1]))
    assert torch.allclose(measurement.position_w[1], previous[1])
    assert torch.equal(measurement.valid, torch.tensor([[True], [False]]))


def test_trilinear_water_current_field_interpolates_regular_grid():
    values = []
    for ix in range(2):
        for iy in range(2):
            for iz in range(2):
                values.append([float(ix), float(iy), float(iz)])

    currents = current_fields.calculate_trilinear_current_field(
        positions=torch.tensor([[0.5, 0.5, 0.5], [1.0, 0.0, 0.0]]),
        bounds=[0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        grid_shape=[2, 2, 2],
        grid_values=values,
    )

    expected = torch.tensor([[0.5, 0.5, 0.5], [1.0, 0.0, 0.0]])
    assert torch.allclose(currents, expected)


def test_calibration_builds_water_current_field_grid_from_samples():
    positions = []
    currents = []
    for ix in range(2):
        for iy in range(2):
            for iz in range(2):
                position = [float(ix), float(iy), float(iz)]
                positions.append(position)
                currents.append([position[0] + 0.1, position[1] - 0.2, position[2] + 0.3])

    fit = calibration.fit_water_current_field_grid(
        positions,
        currents,
        grid_shape=[2, 2, 2],
        bounds=[0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        k_neighbors=4,
    )
    updates = fit.to_cfg_updates()
    interpolated = current_fields.calculate_trilinear_current_field(
        positions=torch.tensor([[0.5, 0.5, 0.5], [1.0, 0.0, 0.0]]),
        bounds=updates["water_current_field_bounds"],
        grid_shape=updates["water_current_field_shape"],
        grid_values=updates["water_current_field_values"],
    )

    assert updates["water_current_field_enabled"] is True
    assert updates["water_current_field_shape"] == [2, 2, 2]
    assert fit.sample_count == 8
    assert torch.allclose(interpolated, torch.tensor([[0.6, 0.3, 0.8], [1.1, -0.2, 0.3]]))


def test_calibration_fits_pool_boundary_effect_scales_from_synthetic_samples():
    positions = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.6, 0.0, 0.0],
            [1.75, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, -1.8, 0.0],
        ]
    )
    bounds = [-2.0, 2.0, -2.0, 2.0, -2.0, 2.0]
    effect_distance = 0.5
    damping, added_mass, thrust = pool_effects.calculate_pool_boundary_scales(
        positions,
        bounds,
        effect_distance,
        damping_scale_at_boundary=1.8,
        added_mass_scale_at_boundary=1.25,
        thrust_scale_at_boundary=0.7,
    )

    fit = calibration.fit_pool_boundary_effect_scales(
        positions,
        bounds,
        effect_distance,
        damping_scale_samples=damping,
        added_mass_scale_samples=added_mass,
        thrust_scale_samples=thrust,
    )
    updates = fit.to_cfg_updates()

    assert abs(fit.damping_scale_at_boundary - 1.8) < 1.0e-6
    assert abs(fit.added_mass_scale_at_boundary - 1.25) < 1.0e-6
    assert abs(fit.thrust_scale_at_boundary - 0.7) < 1.0e-6
    assert torch.equal(fit.sample_count, torch.tensor([4, 4, 4]))
    assert torch.allclose(fit.residual_rms, torch.zeros(3), atol=1.0e-6)
    assert updates["pool_boundary_effects_enabled"] is True
    assert updates["pool_boundary_damping_scale"] == fit.damping_scale_at_boundary


def test_calibration_fits_free_surface_effect_scales_from_synthetic_samples():
    positions = torch.tensor(
        [
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.85],
            [0.0, 0.0, 0.7],
            [0.0, 0.0, 0.6],
            [0.0, 0.0, 0.0],
        ]
    )
    damping, added_mass, buoyancy, thrust = pool_effects.calculate_free_surface_scales(
        positions,
        surface_z=1.0,
        effect_distance=0.5,
        heave_damping_scale_at_surface=1.6,
        roll_pitch_damping_scale_at_surface=1.25,
        added_mass_scale_at_surface=1.35,
        buoyancy_scale_at_surface=0.82,
        thrust_scale_at_surface=0.65,
    )

    fit = calibration.fit_free_surface_effect_scales(
        positions,
        surface_z=1.0,
        effect_distance=0.5,
        heave_damping_scale_samples=damping[:, 2],
        roll_pitch_damping_scale_samples=damping[:, 3:5],
        added_mass_scale_samples=added_mass[:, 2:5],
        buoyancy_scale_samples=buoyancy,
        thrust_scale_samples=thrust,
    )
    updates = fit.to_cfg_updates()

    assert abs(fit.heave_damping_scale - 1.6) < 1.0e-6
    assert abs(fit.roll_pitch_damping_scale - 1.25) < 1.0e-6
    assert abs(fit.added_mass_scale - 1.35) < 1.0e-6
    assert abs(fit.buoyancy_scale - 0.82) < 1.0e-6
    assert abs(fit.thrust_scale - 0.65) < 1.0e-6
    assert torch.equal(fit.sample_count, torch.tensor([4, 8, 12, 4, 4]))
    assert torch.allclose(fit.residual_rms, torch.zeros(5), atol=1.0e-6)
    assert updates["free_surface_effects_enabled"] is True
    assert updates["free_surface_added_mass_scale"] == fit.added_mass_scale


def test_environment_calibration_log_pipeline_builds_current_and_proximity_updates():
    alpha = 0.8
    time_s = torch.arange(8, dtype=torch.float32)
    powers = alpha ** torch.arange(len(time_s), dtype=torch.float32)
    mean_current = torch.tensor([0.1, -0.02, 0.01])
    current = mean_current.reshape(1, 3) + torch.stack((powers, -0.5 * powers, 0.25 * powers), dim=-1)
    field_positions = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
        ]
    )
    field_currents = torch.stack(
        (field_positions[:, 0], field_positions[:, 1], torch.zeros(field_positions.shape[0])),
        dim=-1,
    )
    bounds = [0.0, 10.0, 0.0, 10.0, 0.0, 10.0]
    boundary_positions = torch.tensor([[1.0, 5.0, 5.0], [0.0, 5.0, 5.0], [5.0, 5.0, 5.0]])
    boundary_damping, boundary_added_mass, boundary_thrust = pool_effects.calculate_pool_boundary_scales(
        boundary_positions,
        bounds=bounds,
        effect_distance=2.0,
        damping_scale_at_boundary=1.5,
        added_mass_scale_at_boundary=1.25,
        thrust_scale_at_boundary=0.8,
    )
    surface_positions = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 0.75], [0.0, 0.0, 0.5], [0.0, 0.0, 0.0]])
    surface_damping, surface_added_mass, surface_buoyancy, surface_thrust = (
        pool_effects.calculate_free_surface_scales(
            surface_positions,
            surface_z=1.0,
            effect_distance=0.5,
            heave_damping_scale_at_surface=1.6,
            roll_pitch_damping_scale_at_surface=1.25,
            added_mass_scale_at_surface=1.35,
            buoyancy_scale_at_surface=0.82,
            thrust_scale_at_surface=0.65,
        )
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _write_csv(
            root / "water_current_timeseries.csv",
            ["time_s", "current_w_x_mps", "current_w_y_mps", "current_w_z_mps"],
            [[float(time_s[index]), *current[index].tolist()] for index in range(time_s.numel())],
        )
        _write_csv(
            root / "water_current_field_samples.csv",
            ["pos_x_m", "pos_y_m", "pos_z_m", "current_w_x_mps", "current_w_y_mps", "current_w_z_mps"],
            [
                [*field_positions[index].tolist(), *field_currents[index].tolist()]
                for index in range(field_positions.shape[0])
            ],
        )
        _write_csv(
            root / "pool_boundary_effect_samples.csv",
            ["pos_x_m", "pos_y_m", "pos_z_m", "damping_scale", "added_mass_scale", "thrust_scale"],
            [
                [
                    *boundary_positions[index].tolist(),
                    float(boundary_damping[index, 0]),
                    float(boundary_added_mass[index, 0]),
                    float(boundary_thrust[index, 0]),
                ]
                for index in range(boundary_positions.shape[0])
            ],
        )
        _write_csv(
            root / "free_surface_effect_samples.csv",
            [
                "pos_z_m",
                "heave_damping_scale",
                "roll_pitch_damping_scale",
                "added_mass_scale",
                "buoyancy_scale",
                "thrust_scale",
            ],
            [
                [
                    float(surface_positions[index, 2]),
                    float(surface_damping[index, 2]),
                    float(surface_damping[index, 3]),
                    float(surface_added_mass[index, 2]),
                    float(surface_buoyancy[index, 0]),
                    float(surface_thrust[index, 0]),
                ]
                for index in range(surface_positions.shape[0])
            ],
        )

        result = environment_fit_cli.fit_environment_calibration_logs(
            root,
            current_stage_count=2,
            current_grid_shape=(2, 2, 1),
            current_bounds=(0.0, 1.0, 0.0, 1.0, 0.0, 1.0),
            pool_bounds=bounds,
            boundary_effect_distance=2.0,
            surface_z=1.0,
            surface_effect_distance=0.5,
        )
        output_path = root / "environment_updates.json"
        report_path = root / "environment_report.json"
        exit_code = environment_fit_cli.main(
            [
                str(root),
                "--current-stages",
                "2",
                "--current-grid-shape",
                "2",
                "2",
                "1",
                "--current-bounds",
                "0",
                "1",
                "0",
                "1",
                "0",
                "1",
                "--pool-bounds",
                *[str(value) for value in bounds],
                "--boundary-effect-distance",
                "2.0",
                "--surface-z",
                "1.0",
                "--surface-effect-distance",
                "0.5",
                "--output",
                str(output_path),
                "--report",
                str(report_path),
            ]
        )
        output_updates, output_domain = profile_builder_cli.load_update_payload(output_path)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        profile = profiles.merge_pool_dynamics_cfg_updates(
            cfg_updates=output_updates,
            domain_randomization_updates=output_domain,
        )

    assert torch.allclose(torch.tensor(result.cfg_updates["water_current_w"]), torch.mean(current, dim=0))
    assert result.cfg_updates["water_current_field_shape"] == [2, 2, 1]
    assert len(result.cfg_updates["water_current_field_values"]) == 4
    assert abs(result.cfg_updates["pool_boundary_damping_scale"] - 1.5) < 1.0e-6
    assert abs(result.cfg_updates["pool_boundary_added_mass_scale"] - 1.25) < 1.0e-6
    assert abs(result.cfg_updates["pool_boundary_thrust_scale"] - 0.8) < 1.0e-6
    assert abs(result.cfg_updates["free_surface_heave_damping_scale"] - 1.6) < 1.0e-6
    assert abs(result.cfg_updates["free_surface_roll_pitch_damping_scale"] - 1.25) < 1.0e-6
    assert abs(result.cfg_updates["free_surface_added_mass_scale"] - 1.35) < 1.0e-6
    assert abs(result.cfg_updates["free_surface_buoyancy_scale"] - 0.82) < 1.0e-6
    assert abs(result.cfg_updates["free_surface_thrust_scale"] - 0.65) < 1.0e-6
    assert len(result.domain_randomization_updates["water_current_max_by_stage"]) == 2
    assert exit_code == 0
    assert output_updates == result.cfg_updates
    assert output_domain == result.domain_randomization_updates
    assert profile.pool_boundary.enabled is True
    assert profile.free_surface.enabled is True
    assert report["source_files"] == list(result.source_files)


def test_calibration_fits_tether_spring_damper_from_synthetic_samples():
    length = torch.tensor([1.8, 2.0, 2.2, 2.5, 3.0, 2.4])
    velocity_along_tether = torch.tensor([0.0, 0.0, 0.0, -0.2, -0.4, 0.3])
    slack = 2.0
    stiffness = 20.0
    damping = 5.0
    tension = stiffness * torch.clamp(length - slack, min=0.0) + damping * torch.clamp(
        -velocity_along_tether,
        min=0.0,
    )

    fit = calibration.fit_tether_spring_damper(
        length,
        tension,
        velocity_along_tether,
        slack_length_candidates=[1.8, 2.0, 2.2],
    )
    updates = fit.to_cfg_updates()

    assert abs(fit.slack_length - slack) < 1.0e-6
    assert abs(fit.stiffness - stiffness) < 3.0e-5
    assert abs(fit.damping - damping) < 3.0e-5
    assert fit.residual_rms < 1.0e-5
    assert updates["tether_enabled"] is True
    assert updates["tether_slack_length"] == fit.slack_length
    assert updates["tether_stiffness"] == fit.stiffness
    assert updates["tether_damping"] == fit.damping


def test_calibration_fits_tether_drag_coefficient_from_synthetic_samples():
    relative_velocity = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, -2.0, 0.0],
            [0.0, 0.0, 0.5],
        ]
    )
    drag_coeff = 0.25
    speed = torch.linalg.norm(relative_velocity, dim=-1, keepdim=True)
    drag_force = -drag_coeff * speed * relative_velocity

    fit = calibration.fit_tether_drag_coefficient(relative_velocity, drag_force)
    updates = fit.to_cfg_updates()

    assert abs(fit.drag_coeff - drag_coeff) < 1.0e-6
    assert fit.residual_rms < 1.0e-7
    assert fit.sample_count == 3
    assert updates["tether_enabled"] is True
    assert updates["tether_drag_coeff"] == fit.drag_coeff


def test_tether_calibration_log_pipeline_builds_multisegment_updates():
    length = torch.tensor([1.8, 2.0, 2.2, 2.5, 3.0, 2.4])
    velocity_along_tether = torch.tensor([0.0, 0.0, 0.0, -0.2, -0.4, 0.3])
    slack = 2.0
    stiffness = 20.0
    damping = 5.0
    tension = stiffness * torch.clamp(length - slack, min=0.0) + damping * torch.clamp(
        -velocity_along_tether,
        min=0.0,
    )
    relative_velocity = torch.tensor([[1.0, 0.0, 0.0], [0.0, -2.0, 0.0], [0.0, 0.0, 0.5]])
    drag_coeff = 0.25
    drag_force = -drag_coeff * torch.linalg.norm(relative_velocity, dim=-1, keepdim=True) * relative_velocity

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _write_csv(
            root / "tether_tension_samples.csv",
            ["length_m", "tension_n", "velocity_along_tether_mps"],
            [
                [float(length[index]), float(tension[index]), float(velocity_along_tether[index])]
                for index in range(length.numel())
            ],
        )
        _write_csv(
            root / "tether_drag_samples.csv",
            [
                "relative_velocity_x_mps",
                "relative_velocity_y_mps",
                "relative_velocity_z_mps",
                "drag_force_x_n",
                "drag_force_y_n",
                "drag_force_z_n",
            ],
            [
                [*relative_velocity[index].tolist(), *drag_force[index].tolist()]
                for index in range(relative_velocity.shape[0])
            ],
        )

        result = tether_fit_cli.fit_tether_calibration_logs(
            root,
            anchor_pos_w=(1.0, 2.0, 3.0),
            attach_offset_b=(-0.25, 0.0, 0.0),
            num_segments=4,
            segment_diameter=0.006,
            segment_density=1200.0,
            segment_buoyancy_density=997.0,
            slack_length_candidates=(1.8, 2.0, 2.2),
        )
        output_path = root / "tether_updates.json"
        report_path = root / "tether_report.json"
        exit_code = tether_fit_cli.main(
            [
                str(root),
                "--anchor-pos-w",
                "1",
                "2",
                "3",
                "--attach-offset-b",
                "-0.25",
                "0",
                "0",
                "--num-segments",
                "4",
                "--segment-diameter",
                "0.006",
                "--segment-density",
                "1200",
                "--slack-candidates",
                "1.8",
                "2.0",
                "2.2",
                "--output",
                str(output_path),
                "--report",
                str(report_path),
            ]
        )
        output_updates, output_domain = profile_builder_cli.load_update_payload(output_path)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        merged = profiles.merge_pool_dynamics_cfg_updates(cfg_updates=output_updates)

    assert abs(result.cfg_updates["tether_slack_length"] - slack) < 1.0e-6
    assert abs(result.cfg_updates["tether_stiffness"] - stiffness) < 3.0e-5
    assert abs(result.cfg_updates["tether_damping"] - damping) < 3.0e-5
    assert abs(result.cfg_updates["tether_drag_coeff"] - drag_coeff) < 1.0e-6
    assert result.cfg_updates["tether_num_segments"] == 4
    assert result.cfg_updates["tether_segment_diameter"] == 0.006
    assert exit_code == 0
    assert output_updates == result.cfg_updates
    assert output_domain == {}
    assert merged.tether.enabled is True
    assert merged.tether.anchor_pos_w == [1.0, 2.0, 3.0]
    assert report["source_files"] == list(result.source_files)


def test_pool_boundary_scales_are_one_away_from_walls():
    positions = torch.tensor([[0.0, 0.0, 0.0]])
    bounds = [-2.0, 2.0, -2.0, 2.0, -2.0, 2.0]

    damping, added_mass, thrust = pool_effects.calculate_pool_boundary_scales(
        positions,
        bounds,
        effect_distance=0.5,
        damping_scale_at_boundary=1.5,
        added_mass_scale_at_boundary=1.2,
        thrust_scale_at_boundary=0.8,
    )

    assert torch.allclose(damping, torch.ones(1, 1))
    assert torch.allclose(added_mass, torch.ones(1, 1))
    assert torch.allclose(thrust, torch.ones(1, 1))


def test_pool_boundary_scales_increase_near_boundary():
    positions = torch.tensor([[1.75, 0.0, 0.0], [2.0, 0.0, 0.0]])
    bounds = [-2.0, 2.0, -2.0, 2.0, -2.0, 2.0]

    damping, added_mass, thrust = pool_effects.calculate_pool_boundary_scales(
        positions,
        bounds,
        effect_distance=0.5,
        damping_scale_at_boundary=1.5,
        added_mass_scale_at_boundary=1.2,
        thrust_scale_at_boundary=0.8,
    )

    assert torch.allclose(damping, torch.tensor([[1.25], [1.5]]))
    assert torch.allclose(added_mass, torch.tensor([[1.1], [1.2]]))
    assert torch.allclose(thrust, torch.tensor([[0.9], [0.8]]))


def test_free_surface_scales_affect_heave_roll_pitch_near_surface():
    positions = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 0.75], [0.0, 0.0, 0.0]])

    damping, added_mass, buoyancy, thrust = pool_effects.calculate_free_surface_scales(
        positions,
        surface_z=1.0,
        effect_distance=0.5,
        heave_damping_scale_at_surface=1.4,
        roll_pitch_damping_scale_at_surface=1.2,
        added_mass_scale_at_surface=1.3,
        buoyancy_scale_at_surface=0.8,
        thrust_scale_at_surface=0.6,
    )

    assert torch.allclose(damping[0], torch.tensor([1.0, 1.0, 1.4, 1.2, 1.2, 1.0]))
    assert torch.allclose(added_mass[0], torch.tensor([1.0, 1.0, 1.3, 1.3, 1.3, 1.0]))
    assert torch.allclose(buoyancy[0], torch.tensor([0.8]))
    assert torch.allclose(thrust[0], torch.tensor([0.6]))
    assert torch.all(damping[1, [2, 3, 4]] > torch.ones(3))
    assert torch.allclose(damping[1, [0, 1, 5]], torch.ones(3))
    assert torch.allclose(damping[2], torch.ones(6))
    assert torch.allclose(buoyancy[2], torch.ones(1))


def test_tether_wrench_is_zero_inside_slack_without_drag():
    force_b, torque_b = tether.calculate_tether_wrench(
        body_pos_w=torch.tensor([[0.0, 0.0, 0.0]]),
        body_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        body_linvel_w=torch.zeros(1, 3),
        water_current_w=torch.zeros(1, 3),
        anchor_pos_w=[1.0, 0.0, 0.0],
        attach_offset_b=[0.0, 0.0, 0.0],
        slack_length=2.0,
        stiffness=10.0,
        damping=0.0,
        drag_coeff=0.0,
        quat_conjugate_fn=hydro.quat_conjugate_wxyz,
        quat_apply_fn=hydro.quat_apply_wxyz,
    )

    assert torch.allclose(force_b, torch.zeros(1, 3))
    assert torch.allclose(torque_b, torch.zeros(1, 3))


def test_tether_wrench_pulls_toward_anchor_after_slack():
    force_b, torque_b = tether.calculate_tether_wrench(
        body_pos_w=torch.tensor([[0.0, 0.0, 0.0]]),
        body_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        body_linvel_w=torch.zeros(1, 3),
        water_current_w=torch.zeros(1, 3),
        anchor_pos_w=[3.0, 0.0, 0.0],
        attach_offset_b=[0.0, 1.0, 0.0],
        slack_length=1.0,
        stiffness=10.0,
        damping=0.0,
        drag_coeff=0.0,
        quat_conjugate_fn=hydro.quat_conjugate_wxyz,
        quat_apply_fn=hydro.quat_apply_wxyz,
    )

    expected_force = torch.tensor([[3.0, -1.0, 0.0]]) / torch.sqrt(torch.tensor(10.0)) * (torch.sqrt(torch.tensor(10.0)) - 1.0) * 10.0
    expected_torque = torch.cross(torch.tensor([[0.0, 1.0, 0.0]]), expected_force, dim=-1)
    assert torch.allclose(force_b, expected_force)
    assert torch.allclose(torque_b, expected_torque)


def test_multisegment_tether_matches_single_segment_without_distributed_loads():
    common_args = dict(
        body_pos_w=torch.tensor([[0.0, 0.0, 0.0]]),
        body_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        body_linvel_w=torch.zeros(1, 3),
        water_current_w=torch.zeros(1, 3),
        anchor_pos_w=[3.0, 0.0, 0.0],
        attach_offset_b=[0.0, 0.0, 0.0],
        slack_length=1.0,
        stiffness=10.0,
        damping=0.0,
        drag_coeff=0.0,
        quat_conjugate_fn=hydro.quat_conjugate_wxyz,
        quat_apply_fn=hydro.quat_apply_wxyz,
    )
    single_force, single_torque = tether.calculate_tether_wrench(**common_args)
    multi_force, multi_torque = tether.calculate_multisegment_tether_wrench(
        **common_args,
        num_segments=4,
        segment_diameter=0.01,
        segment_density=1000.0,
        segment_buoyancy_density=1000.0,
        gravity_w=[0.0, 0.0, -9.81],
    )

    assert torch.allclose(multi_force, single_force)
    assert torch.allclose(multi_torque, single_torque)


def test_multisegment_tether_adds_negative_buoyancy_load():
    force_b, torque_b = tether.calculate_multisegment_tether_wrench(
        body_pos_w=torch.tensor([[0.0, 0.0, 0.0]]),
        body_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        body_linvel_w=torch.zeros(1, 3),
        water_current_w=torch.zeros(1, 3),
        anchor_pos_w=[2.0, 0.0, 0.0],
        attach_offset_b=[0.0, 0.0, 0.0],
        slack_length=2.0,
        stiffness=10.0,
        damping=0.0,
        drag_coeff=0.0,
        num_segments=4,
        segment_diameter=0.1,
        segment_density=1100.0,
        segment_buoyancy_density=1000.0,
        gravity_w=[0.0, 0.0, -10.0],
        quat_conjugate_fn=hydro.quat_conjugate_wxyz,
        quat_apply_fn=hydro.quat_apply_wxyz,
    )

    expected_weight = 0.5 * (1100.0 - 1000.0) * torch.pi * (0.05**2) * 2.0 * -10.0
    assert torch.allclose(force_b, torch.tensor([[0.0, 0.0, expected_weight]]), atol=1.0e-5)
    assert torch.allclose(torque_b, torch.zeros(1, 3), atol=1.0e-6)


def test_multisegment_tether_drag_opposes_relative_motion():
    force_b, _ = tether.calculate_multisegment_tether_wrench(
        body_pos_w=torch.tensor([[0.0, 0.0, 0.0]]),
        body_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        body_linvel_w=torch.tensor([[2.0, 0.0, 0.0]]),
        water_current_w=torch.zeros(1, 3),
        anchor_pos_w=[0.0, 2.0, 0.0],
        attach_offset_b=[0.0, 0.0, 0.0],
        slack_length=2.0,
        stiffness=10.0,
        damping=0.0,
        drag_coeff=0.25,
        num_segments=4,
        segment_diameter=0.01,
        segment_density=1000.0,
        segment_buoyancy_density=1000.0,
        gravity_w=[0.0, 0.0, -9.81],
        quat_conjugate_fn=hydro.quat_conjugate_wxyz,
        quat_apply_fn=hydro.quat_apply_wxyz,
    )

    assert force_b[0, 0] < 0.0
    assert torch.allclose(force_b[:, 1:], torch.zeros(1, 2), atol=1.0e-6)


if __name__ == "__main__":
    test_relative_damping_dissipates_relative_motion()
    test_full_matrix_linear_damping_dissipates_relative_motion()
    test_speed_dependent_damping_scale_interpolates_shared_curve()
    test_speed_dependent_damping_scale_interpolates_per_dof_curves()
    test_calibration_fits_diagonal_linear_quadratic_damping_from_synthetic_log()
    test_calibration_fits_full_matrix_linear_quadratic_damping_from_synthetic_log()
    test_calibration_fits_diagonal_added_mass_and_damping_from_synthetic_log()
    test_calibration_fits_full_matrix_added_mass_and_damping_from_synthetic_log()
    test_hydrodynamics_calibration_log_pipeline_fits_full_physical_matrices()
    test_calibration_projects_added_mass_to_symmetric_psd()
    test_calibration_projects_linear_damping_to_dissipative_preserving_skew()
    test_calibration_checks_sampled_quadratic_damping_power()
    test_calibration_fits_speed_dependent_damping_scales_from_synthetic_log()
    test_calibration_fits_water_current_process_from_synthetic_log()
    test_calibration_fits_buoyancy_volume_from_force_samples()
    test_calibration_fits_com_to_cob_from_buoyancy_wrenches()
    test_calibration_fits_com_to_cob_from_static_orientation_torques()
    test_calibration_fits_mass_from_scale_readings()
    test_calibration_fits_inertia_tensor_from_axis_moments()
    test_calibration_fits_inertia_tensor_from_compound_pendulum_periods()
    test_static_calibration_log_pipeline_builds_rigid_body_updates()
    test_buoyancy_uses_world_gravity_then_body_frame()
    test_added_mass_coriolis_is_power_preserving()
    test_full_matrix_added_mass_coriolis_is_power_preserving()
    test_added_mass_inertia_wrench_is_negative_mass_times_relative_acceleration()
    test_fossen_fluid_forces_include_added_mass_inertia()
    test_bluerov2_heavy_thruster_geometry_is_eight_t200s()
    test_bluerov2_heavy_model_parameters_are_consistent()
    test_inertia_tensor_helper_accepts_diagonal_matrix_and_flat_values()
    test_rigid_body_profile_accepts_full_symmetric_inertia_tensor()
    test_rigid_body_profile_rejects_nonsymmetric_inertia_tensor()
    test_nominal_pool_dynamics_profile_matches_vehicle_defaults()
    test_pool_dynamics_profile_applies_measured_parameters_to_cfg()
    test_pool_dynamics_profile_rejects_missing_required_lookup_table()
    test_pool_dynamics_profile_rejects_bad_damping_speed_curve()
    test_pool_dynamics_profile_rejects_bad_observation_estimator_parameters()
    test_pool_dynamics_profile_rejects_bad_physical_sensor_parameters()
    test_pool_dynamics_profile_rejects_bad_water_current_randomization_parameters()
    test_pool_dynamics_profile_accepts_grouped_observation_parameters()
    test_pool_dynamics_profile_round_trips_dict_and_json()
    test_pool_dynamics_profile_merges_flat_calibration_updates()
    test_pool_profile_builder_cli_merges_update_json_files()
    test_pool_dynamics_profile_rejects_unknown_json_fields()
    test_pool_dynamics_profile_audit_flags_nominal_high_fidelity_gaps()
    test_pool_profile_calibration_tasks_include_experiment_metadata()
    test_pool_profile_calibration_update_template_groups_missing_fields()
    test_pool_profile_calibration_log_schemas_describe_required_csv_inputs()
    test_pool_calibration_log_validator_detects_bad_values_and_missing_files()
    test_pool_dynamics_profile_audit_accepts_configured_pool_profile_without_warnings()
    test_pool_profile_audit_cli_loads_profile_json_and_sets_exit_code()
    test_bluerov2_heavy_vehicle_thrust_calibration()
    test_t200_conversion_is_asymmetric_and_quadratic()
    test_calibration_fits_thruster_static_lookup_table_from_stand_samples()
    test_calibration_fits_thruster_inflow_lookup_table_from_grid_samples()
    test_calibration_fits_thruster_first_order_response_from_step_log()
    test_calibration_fits_thruster_voltage_exponent()
    test_calibration_fits_linear_battery_voltage_sag()
    test_thruster_calibration_log_pipeline_builds_lookup_and_response_updates()
    test_lookup_table_conversion_interpolates_shared_curve()
    test_lookup_table_conversion_supports_per_thruster_curves()
    test_inflow_lookup_table_conversion_interpolates_shared_surface()
    test_inflow_lookup_table_conversion_supports_per_thruster_surfaces()
    test_thruster_command_processor_applies_step_delay()
    test_thruster_command_processor_applies_rate_limit()
    test_thruster_command_processor_broadcasts_per_env_rate_limit()
    test_thruster_command_processor_quantizes_commands()
    test_thruster_command_processor_dropout_holds_previous_command()
    test_voltage_thrust_scale_uses_nominal_voltage_ratio()
    test_axial_inflow_thrust_scale_reduces_positive_inflow_only()
    test_thruster_wake_interaction_reduces_downstream_thruster_only()
    test_thruster_reaction_torques_follow_spin_direction_and_signed_thrust()
    test_observation_delay_buffer_returns_current_until_history_is_available()
    test_observation_sensor_model_adds_bias_without_noise()
    test_observation_filter_state_holds_between_update_periods()
    test_observation_filter_state_dropout_holds_previous_measurement()
    test_observation_filter_state_lowpass_filters_updates()
    test_observation_filter_state_bias_drift_changes_measurement()
    test_observation_group_parameter_builds_semantic_vector()
    test_sensor_channel_model_applies_scale_bias_clamp_and_hold()
    test_imu_sensor_model_reports_specific_force_and_gyro()
    test_depth_sensor_model_supports_positive_down_axis_and_hold()
    test_dvl_sensor_model_subtracts_water_velocity_and_range_validity()
    test_position_sensor_model_range_dropouts_hold_previous()
    test_trilinear_water_current_field_interpolates_regular_grid()
    test_calibration_builds_water_current_field_grid_from_samples()
    test_calibration_fits_pool_boundary_effect_scales_from_synthetic_samples()
    test_calibration_fits_free_surface_effect_scales_from_synthetic_samples()
    test_environment_calibration_log_pipeline_builds_current_and_proximity_updates()
    test_calibration_fits_tether_spring_damper_from_synthetic_samples()
    test_calibration_fits_tether_drag_coefficient_from_synthetic_samples()
    test_tether_calibration_log_pipeline_builds_multisegment_updates()
    test_pool_boundary_scales_are_one_away_from_walls()
    test_pool_boundary_scales_increase_near_boundary()
    test_free_surface_scales_affect_heave_roll_pitch_near_surface()
    test_tether_wrench_is_zero_inside_slack_without_drag()
    test_tether_wrench_pulls_toward_anchor_after_slack()
    test_multisegment_tether_matches_single_segment_without_distributed_loads()
    test_multisegment_tether_adds_negative_buoyancy_load()
    test_multisegment_tether_drag_opposes_relative_motion()
    print("Dynamics math checks passed.")
