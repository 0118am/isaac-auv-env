import isaaclab.sim as sim_utils

from isaaclab.assets import RigidObjectCfg

import os

from ..bluerov2_heavy_model import BLUEROV2_HEAVY

USD_PATH = os.path.join(os.path.dirname(__file__), "../data/warpauv/warpauv.usd")

WARPAUV_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=USD_PATH,
        mass_props=sim_utils.MassPropertiesCfg(
            mass=BLUEROV2_HEAVY.mass_kg,
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=10.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            articulation_enabled=False,
        ),

        copy_from_source=False,
    ),
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(0.0, 0.0, 5),
    )
)
"""Configuration for the WarpAUV."""
