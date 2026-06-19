# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Quacopter environment.
"""

import gymnasium as gym

from . import agents
from .warpauv_env import WarpAUVEnv, WarpAUVEnvCfg, WarpAUVTrajEnv, WarpAUVTrajEnvCfg

##
# Register Gym environments.
##

gym.register(
    id="Isaac-WarpAUV-Direct-v1",
    entry_point="isaaclab_tasks.direct.isaac-auv-env:WarpAUVEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": WarpAUVEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.WarpAUVPPORunnerCfg
    },
)

# Trajectory-tracking variant with a 28-D observation space and moving target
# command distribution.  Keeping this as a new task avoids invalidating old
# pos-hold checkpoints trained on the 17-D observation.
gym.register(
    id="Isaac-WarpAUV-Traj-Direct-v1",
    entry_point="isaaclab_tasks.direct.isaac-auv-env:WarpAUVTrajEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": WarpAUVTrajEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.WarpAUVTrajPPORunnerCfg
    },
)
