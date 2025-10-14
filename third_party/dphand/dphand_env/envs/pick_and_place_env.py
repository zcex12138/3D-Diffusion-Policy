from pathlib import Path
from typing import Literal
import mujoco

import numpy as np
from dphand_env.mujoco.utils import *
import yaml

from dphand_env.envs.base_env.dphand_panda_env import DphandPandaEnv

_CONFIG_PATH = Path(__file__).parent.parent / "configs"

class PickAndPlaceEnv(DphandPandaEnv):
    def __init__(self, 
        config: str, 
        render_mode: Literal["rgb_array", "human"] = "rgb_array"
        ):

        with open(_CONFIG_PATH / str(config + '.yaml'), "r") as f:
            self.cfg = yaml.safe_load(f)
        
        super().__init__(self.cfg, render_mode)

        self._goal_dist = self.cfg['reset']['goal_dist']

    def reset(self, seed=None, **kwargs):
        """Reset the environment."""
        # Reset hand to initial position.
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        
        if self.cfg['reset']['domain_rand']:
            object_pos_range = self.cfg['reset']['object_pos_range']
            target_pos_range = self.cfg['reset']['target_pos_range']
            
            # 随机化位置
            self.data.jnt("object").qpos[:2] += object_pos_range * np.random.uniform(-1, 1, 2)
            self.data.jnt("target").qpos[:2] += target_pos_range * np.random.uniform(-1, 1, 2)

        if "init_state" in kwargs:
            self.data.jnt("object").qpos[:] = kwargs["init_state"][:7]
            self.data.jnt("target").qpos[:] = kwargs["init_state"][7:]
        
        # 在设置完随机位置后调用forward
        mujoco.mj_forward(self.model, self.data)

        # 运行几个物理步骤让系统稳定
        for _ in range(5):
            mujoco.mj_step(self.model, self.data)

        self._init_state = np.concatenate([self.data.jnt("object").qpos[:].copy(), self.data.jnt("target").qpos[:].copy()])
        self._lift_flag = False
        self._success_steps = 0
        self._obs = self._compute_observation()
        return self._obs, {}
    
    def _get_done(self):
        object_pos = self._obs["state"]["object_pos"]
        target_pos = self._obs["state"]["target_pos"]
        
        goal_dist = np.linalg.norm(object_pos - target_pos)
        # success
        if goal_dist < self._goal_dist:
            self._success_steps += 1
            if self._success_steps >= 20:
                return True, False, {"success": 1, "init_state": self._init_state}
        # lift
        if object_pos[2] > 0.05 and not self._lift_flag:
            self._lift_flag = True
        # fall down
        elif object_pos[2] < 0.02 and self._lift_flag:
            self._lift_flag = False
            return True, False, {"success": 0, "init_state": self._init_state}
        # time out
        truncated = self.time_limit_exceeded()
        return False, False, {"success": 0, "init_state": self._init_state}

if __name__ == "__main__":
    from dphand.mujoco.wrappers import TeleopIntervention
    env = PickAndPlaceEnv(config="pick_cube_env_cfg", render_mode="human")
    env = TeleopIntervention(env, ip="192.168.3.11", test=True, use_relative_pose=True)
    env.reset()
    # 'esc' to quit
    while True:
        env.step(env.action_space.sample())
        env.render()