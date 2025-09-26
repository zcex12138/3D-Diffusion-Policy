from pathlib import Path
from typing import Literal

import numpy as np
from dphand.mujoco.utils import *
import yaml

from dphand.envs.base_env.dphand_panda_env import DphandPandaEnv

_CONFIG_PATH = Path(__file__).parent.parent / "configs"

class PandaPickAndPlaceEnv(DphandPandaEnv):
    def __init__(self, 
        config: str, 
        render_mode: Literal["rgb_array", "human"] = "rgb_array"
        ):

        with open(_CONFIG_PATH / str(config + '.yaml'), "r") as f:
            self.cfg = yaml.safe_load(f)
        
        super().__init__(self.cfg, render_mode)

    def reset(self, seed=None, **kwargs):
        self._lift_flag = False
        return super().reset(seed, **kwargs)
    
    def _get_done(self):
        object_pos = self._obs["state"]["object_pos"]
        target_pos = self._obs["state"]["target_pos"]
        
        goal_dist = np.linalg.norm(object_pos - target_pos)
        # success
        if goal_dist < 0.1:
            return True, False, {"success": 1}
        # lift
        if object_pos[2] > 0.05 and not self._lift_flag:
            self._lift_flag = True
        # fall down
        elif object_pos[2] < 0.02 and self._lift_flag:
            self._lift_flag = False
            return True, False, {"success": 0}
        # time out
        truncated = self.time_limit_exceeded()
        return False, False, {"success": 0}

if __name__ == "__main__":
    from dphand.mujoco.wrappers import TeleopIntervention
    env = PandaPickAndPlaceEnv(config="panda_pick_cube_env_cfg", render_mode="human")
    env = TeleopIntervention(env, ip="192.168.3.11", test=True, use_relative_pose=True)
    env.reset()
    # 'esc' to quit
    while True:
        env.step(env.action_space.sample())
        env.render()