import gymnasium as gym

from . import agents
from .uavs_setpoint_env import QuadcopterEnv, QuadcopterEnvCfg
from .traverse import UAVsTraversalEnv, UAVsTraversalEnvCfg
from .import controller
##
# Register Gym environments.
##

gym.register(
    id = "Iot-Quadcopter-Direct-v0",
    entry_point = "omni.isaac.lab_tasks.direct.Isaaclab_caric_mission:QuadcopterEnv",
    disable_env_checker = True,
    kwargs={
        "env_cfg_entry_point": QuadcopterEnvCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "skrl_ippo_cfg_entry_point": f"{agents.__name__}:skrl_ippo_cfg.yaml",
        "skrl_mappo_cfg_entry_point": f"{agents.__name__}:skrl_mappo_cfg.yaml",
    },
)

gym.register(
    id = "Iot-Quadcopter-Traverse-v0",
    entry_point = "omni.isaac.lab_tasks.direct.Isaaclab_caric_mission:UAVsTraversalEnv",
    disable_env_checker = True,
    kwargs={
        "env_cfg_entry_point": UAVsTraversalEnvCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "skrl_ippo_cfg_entry_point": f"{agents.__name__}:skrl_ippo_cfg.yaml",
        "skrl_mappo_cfg_entry_point": f"{agents.__name__}:skrl_mappo_cfg.yaml",
    },
)