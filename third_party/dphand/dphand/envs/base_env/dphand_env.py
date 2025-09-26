from pathlib import Path
from typing import Any, Literal, Tuple, Dict
import gymnasium as gym

import numpy as np
from dphand.mujoco.utils import *

from dphand.envs.base_env.base_env import BaseEnv

class DphandEnv(BaseEnv):

    def __init__(
        self,
        cfg: dict,
        render_mode: Literal["rgb_array", "human"] = "rgb_array",
    ):
        self.cfg = cfg

        super().__init__(
            cfg=self.cfg,
            render_mode=render_mode,
        )

        # store last state
        self._rew = None
        self._action = None
        self._ctrl = None

        # Caching.
        self._dphand_dof_ids = np.arange(28)
        self._dphand_ctrl_ids = np.arange(28)

        
        self._ctrl_scale = self.cfg['action']['ctrl_scale']
        self.action_space, self._ctrl_space = self._set_action_space()

    def _set_action_space(self):
        """
        自定义action_space设置, 将第[3:7]维度从欧拉角改为四元数
        """
        # 获取原始的控制范围
        bounds = self.model.actuator_ctrlrange.copy().astype(np.float32)
        low, high = bounds.T
        # 创建新的action_space，总维度为原来的28+1=29（欧拉角3维变四元数4维）
        new_low = np.zeros(29, dtype=np.float32)
        new_high = np.zeros(29, dtype=np.float32)
        # 前3维（位置）保持不变
        new_low[:3] = low[:3]
        new_high[:3] = high[:3]
        # 第[3:7]维度设为四元数范围[-1, 1]
        new_low[3:7] = -1.0
        new_high[3:7] = 1.0
        # 剩余维度（关节角度）保持不变，但索引需要调整
        new_low[7:] = low[6:]
        new_high[7:] = high[6:]
        return gym.spaces.Box(low=new_low, high=new_high, dtype=np.float32), gym.spaces.Box(low=low, high=high, dtype=np.float32)
    
    def step(
        self, action: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        
        self._action = action
        # mujoco physics step
        self.physics_step(self._pre_physics_step(action))
        # compute observation and reward
        self._obs = self._compute_observation()
        self._rew = self._compute_reward()
        terminated, truncated, info = self._get_done()
        return self._obs, self._rew, terminated, truncated, info

    def _pre_physics_step(self, action: np.ndarray):
        ctrl = np.zeros(28)
        ctrl[:3] = action[:3]
        ctrl[3:6] = quat2rpy(action[3:7])
        ctrl[6:] = action[7:]
        if self._ctrl is not None:
            ctrl = np.clip(
                self._ctrl + (ctrl - self._ctrl) * self._ctrl_scale,
                self._ctrl_space.low,
                self._ctrl_space.high,
            )
        self._ctrl = ctrl
        return ctrl

    def _compute_observation(self) -> dict:
        obs = {}
        obs["state"] = {}
        obs["state"]['dphand/joint_pos'] = self.data.qpos[self._dphand_dof_ids].astype(np.float32)
        obs["state"]['dphand/joint_vel'] = self.data.qvel[self._dphand_dof_ids].astype(np.float32)
        if self.use_image_obs:
            obs['image'] = {}
            obs['image']['input'] = self._viewer.render_rgb_cam("rgb_array", self.cam_ids["input"])

        for obs_key in self.cfg["obs"]["extra_obs_keys"]:
            obs["state"][obs_key] = self.data.sensor(obs_key).data.astype(np.float32)
        return obs

    def _compute_reward(self):
        return 0.01
    
    def _get_done(self):
        return False, False, {"success": 0}
