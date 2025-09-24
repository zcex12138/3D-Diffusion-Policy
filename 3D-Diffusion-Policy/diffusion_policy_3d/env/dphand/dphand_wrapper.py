import gymnasium as gym
from gymnasium import spaces
import numpy as np
from natsort import natsorted
from termcolor import cprint
from diffusion_policy_3d.gym_util_dphand.mujoco_point_cloud import PointCloudGenerator
from diffusion_policy_3d.gym_util_dphand.mjpc_wrapper import point_cloud_sampling
from dphand.dphand_env import DphandPickCubeEnv

# TASK_BOUNDS = {
#     'dphand_pick_cube': [-0.5, -0.5, 0.0, 0.5, 0.5, 1.0],
# }

class DphandEnvWrapper(gym.ObservationWrapper):
    def __init__(self, env: DphandPickCubeEnv, device="cuda:0", num_points=1024, use_point_cloud=True):
        super().__init__(env)
        self._viewer = self.env.unwrapped._viewer
        self.device_id = int(device.split(":")[-1]) if ":" in device else 0
        self.image_size = self.env.unwrapped._image_size
        self.num_points = num_points
        self.use_point_cloud = use_point_cloud
        self.cam_name = "input"  # ['input', 'fixed_cam']

        self.use_point_crop = False
        cprint(f"[DphandEnv] use_point_crop: {self.use_point_crop}, use_point_cloud: {self.use_point_cloud}", "cyan")

        # 设置点云生成器
        self.pc_generator = PointCloudGenerator(
            model=self.env.unwrapped.model, 
            viewer=self._viewer,
            cam_name=self.cam_name,
            img_size=self.image_size
        )
        

        # 设置任务边界
        # x_min, y_min, z_min, x_max, y_max, z_max = TASK_BOUNDS['dphand_pick_cube']
        # self.min_bound, self.max_bound = [x_min, y_min, z_min], [x_max, y_max, z_max]

        # 环境参数
        self.episode_length = self._max_episode_steps = 1000
        self.cur_step = 0
        
        # 设置observation space
        self._update_observation_space()

    def _update_observation_space(self):
        """更新观察空间，在原始环境观察基础上添加点云数据"""
        obs_space = {}
        obs_space['image'] = spaces.Box(0, 255, (self.image_size, self.image_size, 3), np.float32)
        obs_space['point_cloud'] = spaces.Box(-np.inf, np.inf, (self.num_points, 6), np.float32)  # 修改为6通道以支持RGB颜色
        obs_space['depth'] = spaces.Box(0, 255, (self.image_size, self.image_size), np.float32)
        # 添加机器人状态观察空间（从原始状态中提取）
        obs_sensor_dim = len(self.env.unwrapped._dphand_dof_ids) + 3
        obs_space['agent_pos'] = spaces.Box(-np.inf, np.inf, (obs_sensor_dim,), np.float32)
        # 添加完整状态观察空间
        obs_space['full_state'] = spaces.Box(-np.inf, np.inf, (obs_sensor_dim * 2 + 14,), np.float32)
        self.observation_space = spaces.Dict(obs_space)

    def get_point_cloud(self, use_rgb=True):
        """生成点云数据"""
        point_cloud, depth = self.pc_generator.generateCroppedPointCloud()
        if not use_rgb:
            point_cloud = point_cloud[..., :3]
        # 裁剪点云
        # if self.use_point_crop:
        #     if self.min_bound is not None:
        #         mask = np.all(point_cloud[:, :3] > self.min_bound, axis=1)
        #         point_cloud = point_cloud[mask]
        #     if self.max_bound is not None:
        #         mask = np.all(point_cloud[:, :3] < self.max_bound, axis=1)
        #         point_cloud = point_cloud[mask]
        # 采样点云
        point_cloud = point_cloud_sampling(point_cloud, self.num_points, 'fps')
        depth = depth[::-1].copy() # 翻转会导致深度数据内存不连续，需要copy一下
        return point_cloud, depth

    def observation(self, obs):
        """重写observation方法, 在原始观察基础上添加点云数据"""
        if self.use_point_cloud:
            # 生成点云数据
            point_cloud, depth = self.get_point_cloud()
        else:
            # 不生成点云，设置为零
            point_cloud = np.zeros((self.num_points, 3), dtype=np.float32)
            # 仍然需要获取深度图像用于其他用途
            depth = self.pc_generator.captureImage(self.pc_generator.cam_id, capture_depth=True)
        
        new_obs = {}
        # 在原始观察基础上添加新的观察数据
        new_obs['image'] = obs['images'][self.cam_name]
        new_obs['point_cloud'] = point_cloud
        new_obs['depth'] = depth
        new_obs['agent_pos'] = np.concatenate([
            obs["state"]["dphand/joint_pos"],
            obs["state"]["block_pos"]
        ])
        # 构建完整状态：包含原有的state部分
        new_obs['full_state'] = np.concatenate([
            obs["state"]["dphand/joint_pos"],
            obs["state"]["dphand/joint_vel"],
            obs["state"]["block_pos"],
            obs["state"]["block_rot"],
            obs["state"]["block2_pos"],
            obs["state"]["block2_rot"]
        ])
        return new_obs

    def step(self, action):
        """执行动作"""
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = self.observation(obs)
        return obs, reward, terminated | truncated, info

    def reset(self, **kwargs):
        """重置环境"""
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs)

    def seed(self, seed=None):
        """设置随机种子（兼容性方法）"""
        pass

    def set_seed(self, seed=None):
        """设置随机种子"""
        pass

    def render(self, mode='rgb_array'):
        """渲染环境"""
        self.env.render()
        # 获取front相机的图像
        return self._viewer.render_rgb_cam("rgb_array", 0)

    def close(self):
        """关闭环境"""
        if hasattr(self.env, 'close'):
            self.env.close()

    def __getattr__(self, name):
        """代理属性访问到原始环境"""
        return getattr(self.env, name)