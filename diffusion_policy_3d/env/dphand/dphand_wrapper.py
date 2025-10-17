import gymnasium as gym
from gymnasium import spaces
import numpy as np
from termcolor import cprint
from diffusion_policy_3d.gym_util.mujoco_point_cloud import PointCloudGenerator, point_cloud_sampling


class DphandPointCloudEnvWrapper(gym.ObservationWrapper):
    def __init__(self, env, num_points=1024, use_point_cloud=True):
        super().__init__(env)
        self._viewer = self.env.unwrapped._viewer
        self.image_size = self.env.unwrapped._image_size
        self.num_points = num_points
        self.use_point_cloud = use_point_cloud
        self.cam_names = ["front", "wrist"]

        self.use_point_crop = False

        # 设置点云生成器
        self.pc_generator = PointCloudGenerator(
            model=self.env.unwrapped.model, 
            viewer=self._viewer,
            cam_name=self.cam_names[0],
            img_size=self.image_size,
            filter_geom_id = 1 # 渲染时滤掉 geom_id <= 1 的物体
        )
        
        # 环境参数
        self.episode_length = self._max_episode_steps = 1000
        self.cur_step = 0
        
        # 设置observation space
        self._update_observation_space()

    def _update_observation_space(self):
        """更新观察空间，在原始环境观察基础上添加点云数据"""
        obs_space = {}
        for cam_name in self.cam_names:
            obs_space[cam_name] = spaces.Box(0, 255, (self.image_size, self.image_size, 3), np.float32) # image
        obs_space['depth'] = spaces.Box(0, 255, (self.image_size, self.image_size), np.float32)
        obs_space['point_cloud'] = spaces.Box(-np.inf, np.inf, (self.num_points, 6), np.float32)  # 修改为6通道以支持RGB颜色
        # 添加机器人状态观察空间（从原始状态中提取）
        obs_sensor_dim = len(self.env.unwrapped._dphand_dof_ids) + 3
        obs_space['agent_pos'] = spaces.Box(-np.inf, np.inf, (obs_sensor_dim,), np.float32)
        # 添加完整状态观察空间
        obs_space['full_state'] = spaces.Box(-np.inf, np.inf, (obs_sensor_dim * 2 + 14,), np.float32)
        self.observation_space = spaces.Dict(obs_space)

    def observation(self, obs):
        """重写observation方法, 在原始观察基础上添加点云数据"""
        image_dict = {cam_name: obs['image'][cam_name] for cam_name in self.cam_names}

        depth = self.pc_generator.captureImage(self.pc_generator.cam_id, capture_depth=True)
        if self.use_point_cloud:
            # 生成点云数据
            point_cloud = self.pc_generator.generatePointCloudFromImages(
                        color_img=image_dict[self.cam_names[0]],
                        depth=depth,
                        use_rgb=True
                    )
            point_cloud = point_cloud_sampling(point_cloud, self.num_points, 'fps')
        else:
            point_cloud = np.zeros((self.num_points, 3), dtype=np.float32)

        new_obs = {}
        # 在原始观察基础上添加新的观察数据
        for cam_name in self.cam_names:
            new_obs[cam_name] = obs['image'][cam_name]
        new_obs['point_cloud'] = point_cloud
        new_obs['depth'] = depth
        new_obs['agent_pos'] = np.concatenate([
            obs["state"]["panda/ee_pos"],
            obs["state"]["panda/ee_quat"],
            obs["state"]["dphand/joint_pos"]
        ])

        # 构建完整状态：包含原有的state部分
        new_obs['full_state'] = np.concatenate([
            obs["state"]["panda/ee_pos"], # 3
            obs["state"]["panda/ee_quat"], # 4
            obs["state"]["dphand/joint_pos"], # 22
            obs["state"]["dphand/joint_vel"], # 22
            obs["state"]["object_pos"], # 3
            obs["state"]["object_quat"], # 4
            obs["state"]["target_pos"], # 3
            obs["state"]["target_quat"] # 4
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

    def close(self):
        """关闭环境"""
        if hasattr(self.env, 'close'):
            self.env.close()

    def __getattr__(self, name):
        """代理属性访问到原始环境"""
        return getattr(self.env, name)


class DphandImageEnvWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self._viewer = self.env.unwrapped._viewer
        self.image_size = self.env.unwrapped._image_size
        self.cam_names = ['front', 'wrist']

        # 环境参数
        self.episode_length = self._max_episode_steps = 1000
        self.cur_step = 0
        
        # 设置observation space
        self._update_observation_space()

    def _update_observation_space(self):
        """更新观察空间，在原始环境观察基础上添加点云数据"""
        obs_space = {}
        for cam_name in self.cam_names:
            obs_space[cam_name] = spaces.Box(0, 255, (self.image_size, self.image_size, 3), np.float32) # image
        obs_sensor_dim = len(self.env.unwrapped._dphand_dof_ids) + 3
        obs_space['agent_pos'] = spaces.Box(-np.inf, np.inf, (obs_sensor_dim,), np.float32)
        self.observation_space = spaces.Dict(obs_space)

    def observation(self, obs):
        """ 重新组装obs字典 """
        new_obs = {}
        for cam_name in self.cam_names:
            new_obs[cam_name] = obs['image'][cam_name]

        new_obs['agent_pos'] = np.concatenate([
            obs["state"]["panda/ee_pos"],
            obs["state"]["panda/ee_quat"],
            obs["state"]["dphand/joint_pos"]
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

    def close(self):
        """关闭环境"""
        if hasattr(self.env, 'close'):
            self.env.close()

    def __getattr__(self, name):
        """代理属性访问到原始环境"""
        return getattr(self.env, name)