"""
DphandEnvWrapper使用示例
展示如何使用wrapper来获取标准化的观察和动作
"""
import numpy as np

from diffusion_policy_3d.gym_util.mujoco_point_cloud import point_cloud_sampling
from diffusion_policy_3d.env.dphand.env_factory import make_env
from pathlib import Path
import visualizer
np.set_printoptions(4)

def main():
    """主函数: 演示wrapper的基本用法"""
    cfg_path = Path(__file__).parent / "configs" / "pick_and_place_pc.yaml"
    env = make_env(
        str(cfg_path),
        wrapper_overrides={
            "TeleopIntervention": {
                "enabled": True,
                "ip": "192.168.3.80",
                "test": True,
                "use_relative_pose": True,
            }
        }
    )

    # 只有使用 pick_and_place_pc 创建环境才能显示点云
    show_point_cloud = True
    if show_point_cloud:
        pc = env.pc_generator
        visualizer_3d = visualizer.RealTime3DVisualizer()
        visualizer_3d.start_visualization(port=5000)

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

            # 显示图像
            images = []
            for cam_name in env.cam_names:
                image = obs['image'][cam_name].transpose(1,2,0)
                images.append(image)
            images = np.concatenate(images, axis=1)
            cv2.imshow('images', images)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            env.render()

            if show_point_cloud:
                # point_cloud = obs['point_cloud']
                point_cloud = pc.generatePointCloudFromImages(obs['image']['front'].transpose(1,2,0).copy(), obs['depth'].squeeze(), use_rgb=True)
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
