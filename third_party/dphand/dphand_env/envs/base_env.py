from pathlib import Path
from typing import Any, Literal, Tuple, Dict

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np
from dphand_env.mujoco.utils import *

from dphand_env.mujoco.mujoco_env import MujocoGymEnv
from dphand_env.mujoco.franka_controller import opspace

_PROJECT_ROOT = Path(__file__).parent.parent

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

        # use image obs
        self.use_image_obs = self.cfg['env']['use_image_obs']
        self.cam_names = self.cfg['env']['camera_names']
        self.cam_ids = dict({
            cam_name: self.model.camera(cam_name).id for cam_name in self.cam_names
        })

        # use tactile obs
        self.use_tactile_obs = self.cfg['env']['use_tactile_obs']
        if self.use_tactile_obs:
            self.tactile_cam_names = self.cfg['env']['tactile_camera_names']
            self.tactile_cam_ids = dict({
                cam_name: self.model.camera(cam_name).id for cam_name in self.tactile_cam_names
            })
            self.tactile_image_size = self.cfg['env']['tactile_image_size']


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
        # 验证 id 是否正确: mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, self._dphand_dof_ids[0])
        self._dphand_ctrl_ids = np.arange(
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "Wrist"),
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "Little-PD") + 1
        )
        
        # physics step
        self._n_steps = int(self.control_dt // self.physics_dt)
        # obs and action space
        self.observation_space = gym.spaces.Dict({"state": gym.spaces.Dict(),})
        self.action_space, self._ctrl_space = self._set_action_space()

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

        # 重置控制信号状态
        self._ctrl = None
        self._action = None
        
        self._obs = self._compute_observation()
        return self._obs, {}

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
        return ctrl

    def physics_step(
        self, ctrl: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        
        # physics step
        for step in range(self._n_steps):
            self.data.ctrl = ctrl
            mujoco.mj_step(self.model, self.data)
        self._ctrl = ctrl.copy()
    
    def _compute_observation(self) -> dict:
        obs = {}
        obs["state"] = {}
        obs["state"]['panda/ee_pos'] = self.data.sensor('tcp_pos').data.astype(np.float32)
        obs["state"]['panda/ee_quat'] = self.data.sensor('tcp_quat').data.astype(np.float32)
        obs["state"]['dphand/joint_pos'] = self.data.qpos[self._dphand_dof_ids].astype(np.float32)
        obs["state"]['dphand/joint_vel'] = self.data.qvel[self._dphand_dof_ids].astype(np.float32)
        if self.use_image_obs:
            obs['image'] = {}
            for cam_name in self.cam_names:
                obs['image'][cam_name] = self._viewer.render_rgb_cam("rgb_array", self.cam_ids[cam_name])
        if self.use_tactile_obs:
            obs['tactile'] = {}
            for cam_name in self.tactile_cam_names:
                _ , obs['tactile'][cam_name] = self._viewer.render_segment_depth(self.tactile_cam_ids[cam_name], size=(self.tactile_image_size, self.tactile_image_size),Target_geom_id=self.model.geom(name = self.cfg["env"]["Target_name"]).id)

        for obs_key in self.cfg["obs"]["extra_obs_keys"]:
            obs["state"][obs_key] = self.data.sensor(obs_key).data.astype(np.float32)
        return obs

    def _compute_reward(self):
        return 0.01
    
    def _get_done(self):
        return False, False, {"success": 0}
    
    def update_mocap(self, mocap_name, pos, quat):
        mocap_id = self.model.body_mocapid[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, mocap_name)]
        self.data.mocap_pos[mocap_id, :] = pos
        self.data.mocap_quat[mocap_id, :] = quat

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
