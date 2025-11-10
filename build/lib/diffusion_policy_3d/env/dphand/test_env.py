"""
DphandEnvWrapper使用示例
展示如何使用wrapper来获取标准化的观察和动作
"""
import os
import sys
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

try:
    from diffusion_policy_3d.env.dphand.dphand_wrapper import DphandPointCloudEnvWrapper
except ImportError:
    from diffusion_policy_3d.env.dphand.dphand_wrapper import DphandPandaEnvWrapper as DphandPointCloudEnvWrapper
from diffusion_policy_3d.gym_util.mujoco_point_cloud import point_cloud_sampling
from dphand_env.envs.pick_and_place_env import PickAndPlaceEnv
from dphand_env.mujoco.wrappers import TeleopIntervention

import visualizer
np.set_printoptions(4)

show_point_cloud = True
if show_point_cloud:
    visualizer_3d = visualizer.RealTime3DVisualizer()
    visualizer_3d.start_visualization(port=5000)

def main():
    """主函数: 演示wrapper的基本用法"""
    env = PickAndPlaceEnv(config="pick_cube_env_cfg", render_mode="human")
    env = TeleopIntervention(env, ip="192.168.3.80", test=True, use_relative_pose=True)
    # 创建环境wrapper
    env = DphandPointCloudEnvWrapper(
        env=env,
        use_point_cloud=False,
        num_points=1024
    )
    obs = env.reset()

    # 运行几个episode
    import cv2
    import time

    for episode in range(100):
        print(f"\n=== Episode {episode + 1} ===")

        total_reward = 0
        step_count = 0

        # 重置环境
        obs = env.reset()
        pc = env.pc_generator
        start_time = time.time()
        done = False
        while not done:  # 限制每episode的步数
            # 生成随机动作
            action = np.random.uniform(
                low=env.action_space.low,
                high=env.action_space.high,
                size=env.action_space.shape
            )

            # 执行动作
            obs, reward, done, info = env.step(action)
            total_reward += reward
            step_count += 1

            # 显示图像（如果可用）
            # 转换为BGR格式用于OpenCV显示
            # depth = obs['depth'].copy()
            _, depth = env.unwrapped._viewer.render_segment_depth(env.unwrapped.cam_ids["little_tip_cam"], width=env.unwrapped.image_size, height=env.unwrapped.image_size)
            img = obs['front']

            depth[depth>0] = (depth[depth>0] - depth[depth>0].min()) / (depth[depth>0].max() - depth[depth>0].min() + 1e-8) * 255
            depth_uint8 = depth.astype(np.uint8)
            depth_uint8 = cv2.cvtColor(depth_uint8, cv2.COLOR_GRAY2BGR)
            
            all_img = np.concatenate([img, depth_uint8], axis=0)
            cv2.imshow('all', all_img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            env.render()

            if show_point_cloud:
                # point_cloud = obs['point_cloud']
                point_cloud = pc.generatePointCloudFromImages(obs['front'], obs['depth'], use_rgb=True)
                point_cloud = point_cloud_sampling(point_cloud, 512, 'fps')
                # index = (np.abs(point_cloud[:,0])<0.001) & (np.abs(point_cloud[:,1])<0.001) & (np.abs(point_cloud[:,2])<0.001)
                visualizer_3d.update_point_cloud(point_cloud)

            # 打印步数信息
            if step_count % 10 == 0:
                print(f"fps: {step_count / (time.time() - start_time):.2f}, step: {step_count}, reward: {reward:.3f}, total_reward: {total_reward:.3f}")

            if info['success']:
                env.reset()
                episode += 1

    # 关闭环境
    env.close()
    cv2.destroyAllWindows()
    print("\n环境已关闭")

if __name__ == "__main__":

    main()
