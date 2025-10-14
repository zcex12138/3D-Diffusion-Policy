import gymnasium as gym
import numpy as np
from pynput import keyboard

from dphand_teleop.dphand_teleoperator import DPhandTeleoperator
from dphand_env.mujoco.utils import *
 
import mujoco

class Fix6DPoseWrapper(gym.ActionWrapper):
    """
    A wrapper to fix the 6D pose of the dexterous hand in the environment.
    This wrapper modifies the action space to only allow changes in the gripper's
    position and orientation, while keeping the 6D pose fixed.
    """
    def __init__(self, env, pose=np.zeros(6)):
        super().__init__(env)
        self.model = env.unwrapped.model
        self.data = env.unwrapped.data
        self._viewer = env.unwrapped._viewer
        # Assuming the action space is a Box with shape (n,), where n is the number of actions
        if not isinstance(env.action_space, gym.spaces.Box):
            raise ValueError("The action space must be a Box space.")
        self.action_space = gym.spaces.Box(
            low=env.action_space.low[6:], 
            high=env.action_space.high[6:],
            shape=(self.env.action_space.shape[0] - 6,), 
            dtype=np.float32
        )
        self.pose = np.array(pose, dtype=np.float32)

    def action(self, action):
        return np.concatenate((self.pose, action))
    
    def step(self, action):
        obs, rew, done, truncated, info = self.env.step(self.action(action))
        # info["action"] = info["action"][6:]  # Keep only the action part that is not fixed
        return obs, rew, done, truncated, info

class TeleopIntervention(gym.ActionWrapper):
    """
    A wrapper to allow teleoperation intervention in the environment.
    When the user presses 'p', the teleoperation is toggled on/off.
    """
    def __init__(self, env, ip="192.168.3.11", test=True, use_relative_pose=True):
        super().__init__(env)
        env.reset()
        self.expert = DPhandTeleoperator(ip, test=test, n_step=5, use_relative_pose=use_relative_pose)
        self.intervened = True
        self.keyboard = None
        self.use_relative_pose = use_relative_pose
        self.listener = keyboard.Listener(on_press=self.on_press)
        self.listener.start()

    def on_press(self, key):
        self.keyboard = key.char if hasattr(key, 'char') else key.name
        if self.keyboard == 'p':
            self.intervened = not self.intervened
    
    def action(self, action: np.ndarray) -> np.ndarray:
        """
        Input:
        - action: policy action
        Output:
        - action: teleop action if nonezero; else, policy action
        """
        if self.intervened:
            arm_pos, arm_rot, angles = self.expert.get_target_action_j2j()
            if self.use_relative_pose:
                action[:3] = self._init_pos + 2.0 * arm_pos
                action[3:7] = arm_rot
                action[7:] = angles
        return action, False

    def step(self, action):
        new_action, replaced = self.action(action)
        obs, rew, done, truncated, info = self.env.step(new_action)
        info["replaced"] = True if replaced else False
        info["action"] = new_action
        # update mocap
        self.env.update_mocap('teleop', new_action[:3], new_action[3:7])
        return obs, rew, done, truncated, info
    
    def close(self):
        self.listener.stop()
    #     self.expert.close()
    #     super().close()

    def reset(self, seed=None, **kwargs):
        obs, info = self.env.reset()
        if self.use_relative_pose:
            self._init_pos = self.env.unwrapped.data.sensor('tcp_pos').data.astype(np.float32)
        self.keyboard = None
        for i in range(20):
            self.step(np.zeros(self.action_space.shape))  # 让expert更新初始位置
        return obs, info
