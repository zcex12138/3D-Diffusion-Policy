"""
DphandEnvWrapper使用示例
展示如何使用wrapper来获取标准化的观察和动作
"""
import numpy as np

from dphand.dphand_env import DphandPickCubeEnv
from dphand.mujoco.wrappers import TeleopIntervention
from diffusion_policy_3d.env.dphand.dphand_wrapper import DphandEnvWrapper

import visualizer

show_point_cloud = False
if show_point_cloud:
    visualizer_3d = visualizer.RealTime3DVisualizer()
    visualizer_3d.start_visualization(port=5000)

def main():
    """主函数: 演示wrapper的基本用法"""
    env = DphandPickCubeEnv(render_mode="human", image_obs=True, image_size=256, control_dt=0.02)
    env = TeleopIntervention(env, ip="192.168.3.11", test=True, use_relative_pose=True)
    # env = Fix6DPoseWrapper(env, pose=[0, 0, 0.3, -1.5707, 1.5707, 0])  # 固定手腕位置
    # 创建环境wrapper
    env = DphandEnvWrapper(
        env=env,
        device="cuda:0",  # 或 "cpu"
        use_point_cloud=False,
        num_points=1024
    )
    obs = env.reset()

    # 运行几个episode
    import cv2
    import time

    for episode in range(10):
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
            #     # 转换为BGR格式用于OpenCV显示
            depth = obs['depth']
            img = obs['image']
            img2 = env._viewer.render_rgb_cam("rgb_array", 2)
            # 放大img2
            img2 = cv2.resize(img2, (img2.shape[1] * 3, img2.shape[0] * 3))

            depth[depth>0] = (depth[depth>0] - depth[depth>0].min()) / (depth[depth>0].max() - depth[depth>0].min() + 1e-8) * 255
            depth_uint8 = depth.astype(np.uint8)
            depth_uint8 = cv2.cvtColor(depth_uint8, cv2.COLOR_GRAY2BGR)

            all_img = np.concatenate([img2], axis=0)
            cv2.imshow('all', all_img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            env.render()

            if show_point_cloud:
                point_cloud = pc.generatePointCloudFromImages(img, depth, use_rgb=True)
                visualizer_3d.update_point_cloud(point_cloud)

            # 打印步数信息
            if step_count % 100 == 0:
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
