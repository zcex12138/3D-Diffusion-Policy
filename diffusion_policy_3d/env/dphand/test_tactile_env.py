"""
DphandEnvWrapper使用示例
展示如何使用wrapper来获取标准化的观察和动作
"""
from pathlib import Path
import numpy as np

from diffusion_policy_3d.env.dphand.env_factory import make_env
from diffusion_policy_3d.gym_util.mujoco_point_cloud import point_cloud_sampling

import cv2
import time
import visualizer

def main():
    """主函数: 演示wrapper的基本用法"""
    # 使用配置文件创建环境
    
    cfg_path = Path(__file__).parent / "configs" / "pick_and_place_pc.yaml"
    env = make_env(
        str(cfg_path),
        use_tactile_obs=True,
        wrapper_overrides={}
    )
    obs = env.reset()

    show_point_cloud = True
    if show_point_cloud:
        pc = env.pc_generator
        visualizer_3d = visualizer.RealTime3DVisualizer()
        visualizer_3d.start_visualization(port=5000)

    def depth_to_uint8(depth):
        valid = depth > 0
        depth_scaled = np.zeros_like(depth, dtype=np.float32)
        d_min = 0.0
        d_max = 0.0
        if np.any(valid):
            d = depth[valid].astype(np.float32)
            d_min = float(d.min())
            d_max = float(d.max())
            denom = (d_max - d_min) + 1e-8
            depth_scaled[valid] = (d - d_min) / denom * 255.0
        depth_uint8 = depth_scaled.astype(np.uint8)
        depth_uint8 = cv2.cvtColor(depth_uint8, cv2.COLOR_GRAY2BGR)
        return depth_uint8, d_min, d_max

    for episode in range(100): 
        print(f"\n=== Episode {episode + 1} ===")

        total_reward = 0
        step_count = 0

        # 重置环境
        obs, _ = env.reset()
        start_time = time.time()
        done = False
        while not done:
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
            tip_depth_imgs = []
            for cam_name in obs['tactile'].keys():
                depth = obs['tactile'][cam_name]
                depth_uint8, d_min, d_max = depth_to_uint8(depth)
                cv2.putText(depth_uint8, cam_name, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
                # print(f"{cam_name} depth min: {d_min:.4f}, max: {d_max:.4f}")
                tip_depth_imgs.append(depth_uint8)

            tip_depth_viz = np.concatenate(tip_depth_imgs, axis=1)
            cv2.imshow('front_rgb', obs['front'])
            cv2.imshow('tip_depths', tip_depth_viz)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            env.render()

            if show_point_cloud:
                # point_cloud = pc.generatePointCloudFromImages(obs['front'], obs['depth'], use_rgb=True)
                # point_cloud = point_cloud_sampling(point_cloud, 512, 'fps')
                # index = (np.abs(point_cloud[:,0])<0.001) & (np.abs(point_cloud[:,1])<0.001) & (np.abs(point_cloud[:,2])<0.001)
                point_cloud = obs['point_cloud']
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
