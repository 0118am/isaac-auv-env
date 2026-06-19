"""Math checks for Fossen-style hydrodynamics.

Run directly with the IsaacLab Python environment, for example:
    /home/jining_yang/miniconda3/envs/env_isaaclab/bin/python tests/test_dynamics_math.py
"""

from __future__ import annotations

import importlib.util
import sys
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


if __name__ == "__main__":
    test_relative_damping_dissipates_relative_motion()
    test_buoyancy_uses_world_gravity_then_body_frame()
    test_added_mass_coriolis_is_power_preserving()
    test_bluerov2_heavy_thruster_geometry_is_eight_t200s()
    test_bluerov2_heavy_model_parameters_are_consistent()
    test_bluerov2_heavy_vehicle_thrust_calibration()
    test_t200_conversion_is_asymmetric_and_quadratic()
    print("Dynamics math checks passed.")
