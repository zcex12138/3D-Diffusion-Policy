from pathlib import Path
from typing import Any, Literal, Tuple, Dict

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np
import yaml

try:
    import mujoco_py
except ImportError as e:
    MUJOCO_PY_IMPORT_ERROR = e
else:
    MUJOCO_PY_IMPORT_ERROR = None

from dphand.envs.base_env.base_env import BaseEnv
from dphand.mujoco.utils import *
from dphand.mujoco.franka_controller import opspace

class DphandPandaEnv(BaseEnv):

    def __init__(
        self,
        cfg: str,
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

        # franka panda control
        self.site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "pinch")
        self.pos_gains, self.ori_gains = self.cfg['franka']['pos_gains'], self.cfg['franka']['ori_gains']
        self.damping_ratio = self.cfg['franka']['damping_ratio']

        self._panda_dof_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i}") 
            for i in range(1, 8)
        ]
        self._panda_ctrl_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"actuator{i}") 
            for i in range(1, 8)
        ]
        # 除panda外的
        self._dphand_dof_ids = np.arange(
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "Forearm_Wrist-Flexion"),
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "Little-PM_Little-DIP-Flexion") + 1
        )
        # 验证 id 是否正确
        # mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, self._dphand_dof_ids[0])
        self._dphand_ctrl_ids = np.arange(
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "Wrist"),
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "Little-PD") + 1
        )
        
        self.ctrl_scale = self.cfg['action']['ctrl_scale']
        self.action_space, self._ctrl_space = self._set_action_space()

    def step(
        self, action: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        
        self._action = action
        # mujoco physics step
        self.physics_step(self._pre_physics_step(action))
        self._obs = self._compute_observation()
        self._rew = self._compute_reward()
        terminated, truncated, info = self._get_done()
        # update mocap
        self.update_mocap('pinch', self._obs["state"]['panda/ee_pos'], self._obs["state"]['panda/ee_quat'])
        return self._obs, self._rew, terminated, truncated, info

    def _pre_physics_step(self, action):
        ctrl = action.copy()
        # franka cartesian impedance control
        tau = opspace(
            model=self.model,
            data=self.data,
            site_id=self.site_id,
            dof_ids=self._panda_dof_ids,
            pos=action[:3], # delta pos
            ori=action[3:7], # rot
            pos_gains=self.pos_gains,
            ori_gains=self.ori_gains,
            damping_ratio=self.damping_ratio
        )
        ctrl[:7] = tau
        if self._ctrl is not None:
            ctrl = np.clip(
                self._ctrl + (action - self._ctrl) * self.ctrl_scale,
                self._ctrl_space.low,
                self._ctrl_space.high,
            )
        return ctrl

    def _compute_observation(self) -> dict:
        obs = {}
        obs["state"] = {}
        obs["state"]['panda/ee_pos'] = self.data.sensor('tcp_pos').data.astype(np.float32)
        obs["state"]['panda/ee_quat'] = self.data.sensor('tcp_quat').data.astype(np.float32)
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

    def _set_action_space(self):
        """
        将前3维位置改为delta pos, 第[3:7]维度从欧拉角改为四元数
        """
        bounds = self.model.actuator_ctrlrange.copy().astype(np.float32)
        low, high = bounds.T
        new_low, new_high = low.copy(), high.copy()
        new_low[:3] = -0.03
        new_high[:3] = 0.03
        new_low[3:7] = -1.0
        new_high[3:7] = 1.0
        return gym.spaces.Box(low=new_low, high=new_high, dtype=np.float32), gym.spaces.Box(low=low, high=high, dtype=np.float32)