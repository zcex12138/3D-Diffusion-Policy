from pathlib import Path
from typing import Any, Literal, Tuple, Dict

import gymnasium as gym
import mujoco
import numpy as np
from dphand.mujoco.utils import *

from dphand.mujoco.mujoco_env import MujocoGymEnv

_PROJECT_ROOT = Path(__file__).parent.parent.parent

class BaseEnv(MujocoGymEnv):

    def __init__(
        self,
        cfg: dict,
        render_mode: Literal["rgb_array", "human"] = "rgb_array",
    ):
        # config
        self.cfg = cfg

        super().__init__(
            xml_path=_PROJECT_ROOT / self.cfg['env']['xml_path'],
            seed=self.cfg['env']['seed'],
            control_dt=self.cfg['sim']['control_dt'],
            physics_dt=self.cfg['sim']['physics_dt'],
            image_size=self.cfg['env']['image_size'],
            time_limit=self.cfg["reset"]["time_limit"],
            render_mode=render_mode,
        )

        self.cam_names = self.cfg['env']['camera_names']
        self.cam_ids = dict({
            cam_name: self.model.camera(cam_name).id for cam_name in self.cam_names
        })
        self.use_image_obs = self.cfg['env']['use_image_obs']
        self._n_steps = int(self.control_dt // self.physics_dt)


        self.observation_space = gym.spaces.Dict({"state": gym.spaces.Dict(),})
        self._set_action_space()

        if self.use_image_obs:
            self.observation_space = gym.spaces.Dict({
                "state": gym.spaces.Dict(),
                "image": gym.spaces.Dict()
                })


    def reset(
        self, seed=None, **kwargs
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset the environment."""
        # Reset hand to initial position.
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        mujoco.mj_forward(self.model, self.data)
        
        # 运行几个物理步骤让系统稳定
        for _ in range(5):
            mujoco.mj_step(self.model, self.data)

        self._obs = self._compute_observation()
        return self._obs, {}

    def physics_step(
        self, ctrl: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        
        # physics step
        for step in range(self._n_steps):
            self.data.ctrl = ctrl
            mujoco.mj_step(self.model, self.data)
    
    def update_mocap(self, mocap_name, pos, quat):
        mocap_id = self.model.body_mocapid[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, mocap_name)]
        self.data.mocap_pos[mocap_id, :] = pos
        self.data.mocap_quat[mocap_id, :] = quat
