from pathlib import Path
from typing import Any, Literal, Tuple, Dict

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from dphand.mujoco.utils import *
import yaml

from dphand.envs.base_env.dphand_env import DphandEnv

_PROJECT_ROOT = Path(__file__).parent.parent

class DphandPickAndPlaceEnv(DphandEnv):
    def __init__(self, 
        config: str, 
        render_mode: Literal["rgb_array", "human"] = "rgb_array"
        ):

        with open(_PROJECT_ROOT / "configs" / str(config + '.yaml'), "r") as f:
            self.cfg = yaml.safe_load(f)
        
        super().__init__(self.cfg, render_mode)

    def _get_done(self):
        hand_pos = self._obs["state"]["dphand/joint_pos"][:3]
        object_pos = self._obs["state"]["object_pos"]
        target_pos = self._obs["state"]["target_pos"]
        
        goal_dist = np.linalg.norm(object_pos - target_pos)
        hand_dist = np.linalg.norm(hand_pos - object_pos)
        # success
        if goal_dist < 0.15 and hand_dist > 0.4:
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
    env = DphandPickAndPlaceEnv(config="dphand_pick_cube_env_cfg", render_mode="human")
    env = TeleopIntervention(env, ip="192.168.3.11", test=True, use_relative_pose=True)
    env.reset()
    while True:
        env.step(env.action_space.sample())
        env.render()