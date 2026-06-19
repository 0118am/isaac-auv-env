"""Nominal BlueROV2 Heavy + T200 physical parameters.

The values here are the single source of truth for the simulated vehicle model.
They are kept separate from the IsaacLab environment so tests, dynamics code,
and asset configuration all agree on the same body frame and units.
"""

from __future__ import annotations

from dataclasses import dataclass


KGF_TO_NEWTON = 9.80665


def solid_box_inertia_diag(mass_kg: float, length_x_m: float, width_y_m: float, height_z_m: float) -> tuple[float, float, float]:
    """Return a conservative box inertia estimate about the body-frame COM."""

    i_xx = mass_kg * (width_y_m**2 + height_z_m**2) / 12.0
    i_yy = mass_kg * (length_x_m**2 + height_z_m**2) / 12.0
    i_zz = mass_kg * (length_x_m**2 + width_y_m**2) / 12.0
    return (i_xx, i_yy, i_zz)


def diagonal_inertia_to_physx_matrix(diag: tuple[float, float, float]) -> tuple[float, ...]:
    """Return PhysX's flattened 3x3 inertia tensor order for a diagonal tensor."""

    return (diag[0], 0.0, 0.0, 0.0, diag[1], 0.0, 0.0, 0.0, diag[2])


@dataclass(frozen=True)
class BlueROV2HeavyModel:
    """Body-frame parameters for the BlueROV2 Heavy configuration.

    Body axes are x-forward, y-left, z-up.  The link frame is treated as the
    rigid-body center of mass; restoring stability is modeled with the COB
    offset below.
    """

    mass_kg: float = 11.5
    water_density_kg_m3: float = 997.0
    length_x_m: float = 0.4571
    width_y_m: float = 0.5750
    top_view_width_y_m: float = 0.4361
    height_z_m: float = 0.2539
    center_of_mass_offset_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    center_of_buoyancy_from_com_m: tuple[float, float, float] = (0.0, 0.0, 0.01)
    forward_bollard_thrust_kgf: float = 9.0
    lateral_bollard_thrust_kgf: float = 9.0
    vertical_bollard_thrust_kgf: float = 14.0
    t200_nominal_forward_thrust_kgf: float = 5.25
    t200_nominal_reverse_thrust_kgf: float = 4.10

    @property
    def neutral_buoyancy_volume_m3(self) -> float:
        return self.mass_kg / self.water_density_kg_m3

    @property
    def inertia_diag_kg_m2(self) -> tuple[float, float, float]:
        return solid_box_inertia_diag(self.mass_kg, self.length_x_m, self.width_y_m, self.height_z_m)

    @property
    def physx_inertia_matrix_kg_m2(self) -> tuple[float, ...]:
        return diagonal_inertia_to_physx_matrix(self.inertia_diag_kg_m2)

    @property
    def t200_reverse_to_forward_ratio(self) -> float:
        return self.t200_nominal_reverse_thrust_kgf / self.t200_nominal_forward_thrust_kgf


BLUEROV2_HEAVY = BlueROV2HeavyModel()

