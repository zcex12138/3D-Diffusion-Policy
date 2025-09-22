from pathlib import Path
from typing import Any, Literal, Tuple, Dict

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np
from dphand.mujoco.utils import *
import yaml

try:
    import mujoco_py
except ImportError as e:
    MUJOCO_PY_IMPORT_ERROR = e
else:
    MUJOCO_PY_IMPORT_ERROR = None

from dphand.mujoco.mujoco_env import MujocoGymEnv

_HERE = Path(__file__).parent.parent
_XML_PATH = _HERE / "assets" / "xmls" / "dphand" /"dphand_grasp.xml"
_CONFIG_PATH = _HERE / "assets" / "configs" / "dphand_pick_cube_env_cfg.yaml"

class DphandPickCubeEnv(MujocoGymEnv):

    def __init__(
        self,
        config_path: str = _CONFIG_PATH,
        seed: int = 0,
        control_dt: float = 0.02, # n-substeps = control_dt // physics_dt
        physics_dt: float = 0.002, # dt
        render_mode: Literal["rgb_array", "human"] = "rgb_array",
        image_size: int = 128,
        image_obs: bool = False,
    ):
        # config
        with open(config_path, "r") as f:
            self.cfg = yaml.safe_load(f)


        super().__init__(
            xml_path=_XML_PATH,
            seed=seed,
            control_dt=control_dt,
            physics_dt=physics_dt,
            time_limit=self.cfg["reset"]["time_limit"],
            image_size=image_size,
            render_mode=render_mode
        )

        self.cam_names = ["input", "fixed_cam", "front"]
        self.cam_ids = dict({
            cam_name: self.model.camera(cam_name).id for cam_name in self.cam_names
        })
        self.image_obs = image_obs

        # store last state
        self._obs = None
        self._rew = None
        self._action = None

        # Caching.
        self._dphand_dof_ids = np.arange(28)
        self._dphand_ctrl_ids = np.arange(28)

        self._lift_flag = False

        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Dict(
                    {
                        "dphand/joint_pos": spaces.Box(
                            -np.inf, np.inf, shape=(28,), dtype=np.float32
                        ),
                        "dphand/joint_vel": spaces.Box(
                            -np.inf, np.inf, shape=(28,), dtype=np.float32
                        ),
                        # "dphand/joint_torque": specs.Array(shape=(21,), dtype=np.float32),
                        "block_pos": spaces.Box(
                            -np.inf, np.inf, shape=(3,), dtype=np.float32
                        ),
                        "block_rot": spaces.Box(
                            -np.inf, np.inf, shape=(4,), dtype=np.float32
                        )
                    }
                ),
            }
        )

        if self.image_obs:
            self.observation_space = gym.spaces.Dict(
                {
                    "state": gym.spaces.Dict(
                        {
                            "dphand/joint_pos": spaces.Box(
                                -np.inf, np.inf, shape=(28,), dtype=np.float32
                            ),
                            "dphand/joint_vel": spaces.Box(
                                -np.inf, np.inf, shape=(28,), dtype=np.float32
                            ),
                            "block_pos": spaces.Box(
                                -np.inf, np.inf, shape=(3,), dtype=np.float32
                            ),
                            "block_rot": spaces.Box(
                                -np.inf, np.inf, shape=(4,), dtype=np.float32
                            )
                        }
                    ),
                    "images": gym.spaces.Dict(
                        {
                            "input": gym.spaces.Box(
                                low=0,
                                high=255,
                                shape=(self._image_size, self._image_size, 3),
                                dtype=np.uint8,
                            ),
                            "fixed_cam": gym.spaces.Box(
                                low=0,
                                high=255,
                                shape=(self._image_size, self._image_size, 3),
                                dtype=np.uint8,
                            ),
                        }
                    ),
                }
            )

        self.action_space = self._set_action_space()
        self.action_scale = 0.3

    def reset(
        self, seed=None, **kwargs
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset the environment."""
        # Reset hand to initial position.
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        # Reset the block to initial position.
        self.data.jnt("block").qpos[:2] += np.random.uniform(-0.02, 0.02, size=2)
        self.data.jnt("block2").qpos[:2] += np.random.uniform(-0.04, 0.04, size=2)

        # 确保所有关节速度为零，避免突然的运动
        self.data.qvel[:] = 0.0
        
        # 前向计算物理状态
        mujoco.mj_forward(self.model, self.data)
        
        # 运行几个物理步骤让系统稳定
        for _ in range(5):
            mujoco.mj_step(self.model, self.data)

        self._obs = self._compute_observation()
        return self._obs, {}
    
    def _set_goal(self, goal_pos: np.ndarray, goal_rot: np.ndarray):
        """
        Set the goal position and orientation for the block.
        Params:
            goal_pos: np.ndarray, shape (3,)
            goal_rot: np.ndarray, shape (4,)
        """
        self.goal_pos = goal_pos
        self.goal_rot = goal_rot
        # set the mocap pose
        if self.data.mocap_pos.shape[0] > 0:
            self.data.mocap_pos[0] = self.goal_pos
            self.data.mocap_quat[0] = self.goal_rot

    def step(
        self, action: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """
        take a step in the environment.
        Params:
            action: np.ndarray

        Returns:
            observation: dict[str, np.ndarray],
            reward: float,
            done: bool,
            truncated: bool,
            info: dict[str, Any]
        """
        action = np.asarray(action, dtype=np.float32)
        action_raw = action.copy()
        # clip the action to the action space
        if self._action is not None:
            action = np.clip(
            self._action + (action - self._action) * self.action_scale,
            self.action_space.low,
            self.action_space.high,
        )

        self._action = action_raw
        # physics step
        for i in range(int(self.control_dt // self.physics_dt)):
            self.data.ctrl[self._dphand_ctrl_ids] = action
            mujoco.mj_step(self.model, self.data)
        
        # compute observation and reward
        self._obs = self._compute_observation()
        self._rew = self._get_reward(self._obs, np.asarray(action))
        terminated, trucated, info = self._get_done(self._obs)
        self._action = action
        info["lift"] = self._lift_flag
        return self._obs, self._rew, terminated, trucated, info

    # Helper methods.

    def _compute_observation(self) -> dict:
        obs = {}
        obs["state"] = {}

        obs["state"]["dphand/joint_pos"] = self.data.qpos[self._dphand_dof_ids].astype(np.float32)
        obs["state"]["dphand/joint_vel"] = self.data.qvel[self._dphand_dof_ids].astype(np.float32)
        obs["state"]["block_pos"] = self.data.sensor("block_pos").data.astype(np.float32)
        obs["state"]["block_rot"] = self.data.sensor("block_quat").data.astype(np.float32)
        obs["state"]["block2_pos"] = self.data.sensor("block2_pos").data.astype(np.float32)
        obs["state"]["block2_rot"] = self.data.sensor("block2_quat").data.astype(np.float32)

        if self.image_obs:
            obs["images"] = {}
            obs["images"]["input"] = self._viewer.render_rgb_cam("rgb_array", self.cam_ids["input"])
            obs["images"]["fixed_cam"] = self._viewer.render_rgb_cam("rgb_array", self.cam_ids["front"])
        return obs

    def _get_reward(
        self,
        obs: Dict[str, np.ndarray],
        actions: np.ndarray
    ) -> Tuple[float, bool, bool]:
        """
        Compute the reward based on the observation and action.
        """
        return self._compute_reward(
            obs,
            actions,
            self.cfg["reward"]["dist_reward_scale"],
            self.cfg["reward"]["rot_reward_scale"],
            self.cfg["reward"]["rot_eps"],
            self.cfg["reward"]["action_penalty_scale"],
            self.cfg["reward"]["reach_goal_bonus"],
            self.cfg["reward"]["success_tolerance"],
            self.cfg["reset"]["fall_dist"],
            self.cfg["reward"]["fall_penalty"]
        )

    def _compute_reward(
            self, 
            obs: Dict,
            actions: np.ndarray,
            dist_reward_scale: float,
            rot_reward_scale: float,
            rot_eps: float,
            action_penalty_scale: float,
            reach_goal_bonus: float,
            success_tolerance: float,
            fall_dist: float,
            fall_penalty: float
            ):
        reward = 1.0
        return reward
    
    def _get_done(self, obs):
        hand_pos = obs["state"]["dphand/joint_pos"][:3]
        block1_pos = obs["state"]["block_pos"]
        block2_pos = obs["state"]["block2_pos"]
        goal_dist = np.linalg.norm(block1_pos - block2_pos)
        hand_dist = np.linalg.norm(hand_pos - block1_pos)
        # success
        if goal_dist < 0.15 and hand_dist > 0.4:
            return True, False, {"success": 1}
        # lift
        if block1_pos[2] > 0.05 and not self._lift_flag:
            self._lift_flag = True
        # fall down
        elif block1_pos[2] < 0.02 and self._lift_flag:
            self._lift_flag = False
            return True, False, {"success": 0}
        # time out
        truncated = self.time_limit_exceeded()
        return False, False, {"success": 0}

if __name__ == "__main__":
    from dphand.mujoco.wrappers import Fix6DPoseWrapper, TeleopIntervention
    env = gym.make("DphandPickCube-v0", render_mode="human", image_obs=True)

    env = gym.wrappers.FlattenObservation(env)
    env = TeleopIntervention(env, ip="192.168.3.44", test=True)
    env = Fix6DPoseWrapper(env, pose=[0, 0, 0.3, -1.5707, 1.5707, 0])
    env_unwrapped = env.unwrapped
    obs, _ = env.reset()
    import time
    # env._viewer.viewer.vopt.frame = mujoco.mjtFrame.mjFRAME_SITE
    flag = True
    cnt = 0
    import cv2
    start_time = time.time()
    while True:
        data_time_left = env_unwrapped.data.time
        cnt += 1
        if cnt % 100 == 0:
            flag = not flag
        action = flag * env.action_space.low + (not flag) * env.action_space.high
        # random action
        # action = np.random.uniform(low=env.action_space.low, 
        #                         high=env.action_space.high, 
        #                         size=env.action_space.shape)
        obs, reward, done, truncated, info = env.step(action)
        obs_unflatten = env.unwrapped._obs
        if env_unwrapped.image_obs:
            image = np.hstack(
                [obs_unflatten["images"]["front"], obs_unflatten["images"]["wrist"]]
            )
            cv2.imshow("image_obs", image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        # reset
        # if done | truncated:
        #     print(env_unwrapped.data.time)
        #     obs, _ = env.reset()
        #     start_time = time.time()

        env.render()
        real_time = time.time() - start_time
        # print("physics_fps:" , cnt / real_time)
        # print(env_unwrapped.data.time - real_time)
        time.sleep(max(0, env_unwrapped.data.time - real_time))
    env.close()
