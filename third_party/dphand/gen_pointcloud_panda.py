#!/usr/bin/env python3
"""
后处理脚本：从RGB和深度图像生成点云数据
用于在数据收集完成后批量生成点云，提高数据收集效率
"""

import argparse
import os
import zarr
import numpy as np
from termcolor import cprint
from diffusion_policy_3d.gym_util_dphand.mujoco_point_cloud import PointCloudGenerator
from diffusion_policy_3d.gym_util_dphand.mjpc_wrapper import point_cloud_sampling
from dphand.envs.panda_pick_and_place_env import PandaPickAndPlaceEnv
import visualizer
import cv2


def test_playback_images(zarr_path):
    """
    简单的测试函数：播放存储的RGB和深度图像
    """
    cprint(f"开始播放图像: {zarr_path}", "yellow")
    
    # 读取zarr文件
    zarr_root = zarr.open(zarr_path, mode='r')
    
    # 读取RGB和深度数据
    rgb_arrays = zarr_root['data']['img'][:]
    depth_arrays = zarr_root['data']['depth'][:]
    
    cprint(f"读取到 {len(rgb_arrays)} 个RGB图像", "green")
    cprint(f"读取到 {len(depth_arrays)} 个深度图像", "green")
    
    # 创建窗口
    cv2.namedWindow('RGB + Depth', cv2.WINDOW_NORMAL)
    
    # 播放图像
    for i in range(len(rgb_arrays)):
        rgb_img = rgb_arrays[i]
        depth_img = depth_arrays[i]
        
        # 处理深度图像显示
        depth_normalized = depth_img.copy()
        if depth_normalized.max() > depth_normalized.min():
            depth_normalized = (depth_normalized - depth_normalized.min()) / (depth_normalized.max() - depth_normalized.min()) * 255
        depth_uint8 = depth_normalized.astype(np.uint8)
        depth_colored = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_JET)
        
        # 调整图像大小以匹配
        target_size = (rgb_img.shape[1], rgb_img.shape[0])
        depth_resized = cv2.resize(depth_colored, target_size)
        
        # 水平拼接RGB和深度图像
        combined_img = np.concatenate([rgb_img, depth_resized], axis=1)
        
        # 添加帧信息
        frame_text = f"Frame: {i}/{len(rgb_arrays)-1}"
        cv2.putText(combined_img, frame_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        cv2.imshow('RGB + Depth', combined_img)
        
        # 按任意键继续，按'q'退出
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break
    
    cv2.destroyAllWindows()
    cprint("播放结束", "green")


def save_frame(rgb_arrays, depth_arrays, frame_idx, save_count, show_rgb, show_depth):
    """保存当前帧"""
    save_dir = "saved_frames"
    os.makedirs(save_dir, exist_ok=True)
    
    if show_rgb:
        rgb_img = rgb_arrays[frame_idx]
        rgb_path = os.path.join(save_dir, f"frame_{save_count:03d}_rgb.png")
        cv2.imwrite(rgb_path, rgb_img)
        cprint(f"保存RGB图像: {rgb_path}", "green")
    
    if show_depth and depth_arrays is not None:
        depth_img = depth_arrays[frame_idx]
        depth_normalized = depth_img.copy()
        if depth_normalized.max() > depth_normalized.min():
            depth_normalized = (depth_normalized - depth_normalized.min()) / (depth_normalized.max() - depth_normalized.min()) * 255
        depth_uint8 = depth_normalized.astype(np.uint8)
        depth_colored = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_JET)
        
        depth_path = os.path.join(save_dir, f"frame_{save_count:03d}_depth.png")
        cv2.imwrite(depth_path, depth_colored)
        cprint(f"保存深度图像: {depth_path}", "green")





def check_depth_quality(depth_img, min_valid_ratio=0.1):
    """
    检查深度图像质量
    """
    valid_pixels = (depth_img > 0) & (depth_img < 10.0)  # 假设深度范围0-10米
    valid_ratio = np.sum(valid_pixels) / depth_img.size
    
    if valid_ratio < min_valid_ratio:
        print(f"Warning: Low depth quality, valid ratio: {valid_ratio:.3f}")
        return False
    return True

def point_cloud_sampling_fixed(point_cloud: np.ndarray, num_points: int, method: str = 'uniform'):
    """
    修复版本的点云采样函数，避免零填充
    """
    if num_points == 'all':
        return point_cloud
    
    if point_cloud.shape[0] <= num_points:
        # 使用重复采样而不是零填充
        if point_cloud.shape[0] == 0:
            # 如果点云为空，返回默认点云
            return np.array([[0.1, 0.1, 0.1, 0, 0, 0]] * num_points)
        
        # 重复采样现有点
        repeat_times = num_points // point_cloud.shape[0]
        remainder = num_points % point_cloud.shape[0]
        
        repeated_cloud = np.tile(point_cloud, (repeat_times, 1))
        if remainder > 0:
            additional_indices = np.random.choice(point_cloud.shape[0], remainder, replace=True)
            additional_points = point_cloud[additional_indices]
            repeated_cloud = np.concatenate([repeated_cloud, additional_points], axis=0)
        
        return repeated_cloud
    
    # 正常采样逻辑
    if method == 'uniform':
        sampled_indices = np.random.choice(point_cloud.shape[0], num_points, replace=False)
        return point_cloud[sampled_indices]
    elif method == 'fps':
        # FPS采样逻辑保持不变
        import torch
        import pytorch3d.ops as torch3d_ops
        
        point_cloud_tensor = torch.from_numpy(point_cloud).unsqueeze(0).cuda()
        num_points_tensor = torch.tensor([num_points]).cuda()
        _, sampled_indices = torch3d_ops.sample_farthest_points(
            points=point_cloud_tensor[..., :3], K=num_points_tensor)
        point_cloud = point_cloud_tensor.squeeze(0).cpu().numpy()
        return point_cloud[sampled_indices.squeeze(0).cpu().numpy()]
    else:
        raise NotImplementedError(f"point cloud sampling method {method} not implemented")


def main(args):
    # 读取zarr文件
    zarr_path = args.zarr_path
    cprint(f"读取zarr文件: {zarr_path}", "yellow")
    zarr_root = zarr.open(zarr_path, mode='r')
    
    # 读取RGB和深度数据
    rgb_arrays = zarr_root['data']['img'][:]
    depth_arrays = zarr_root['data']['depth'][:]
    
    cprint(f"读取到 {len(rgb_arrays)} 个RGB图像", "green")
    cprint(f"读取到 {len(depth_arrays)} 个深度图像", "green")
    cprint(f"RGB图像形状: {rgb_arrays.shape}", "green")
    cprint(f"深度图像形状: {depth_arrays.shape}", "green")
    
    # 创建环境来获取相机参数
    env = PandaPickAndPlaceEnv(config="panda_pick_cube_env_cfg", render_mode="rgb_array")
    
    # 创建点云生成器
    pc_generator = PointCloudGenerator(
        model=env.model, 
        viewer=env._viewer,
        cam_name='input',
        img_size=env.cfg['env']['image_size']
    )
    cprint(f"相机内参矩阵:\n{pc_generator.cam_mat}", "green")

    visualizer_3d = visualizer.RealTime3DVisualizer()
    visualizer_3d.start_visualization(port=5000)

    # 存储所有点云数据
    all_point_clouds = []
    
    # 逐个处理每张图像
    for i in range(len(rgb_arrays)):
        rgb_img = rgb_arrays[i]
        depth_img = depth_arrays[i]
        
        if i % 100 == 0:  # 每100张图像打印一次进度
            cprint(f"处理图像 {i}/{len(rgb_arrays)}", "cyan")

        depth_img[depth_img>0] = (depth_img[depth_img>0] - depth_img[depth_img>0].min()) / (depth_img[depth_img>0].max() - depth_img[depth_img>0].min() + 1e-8) * 255
        depth_uint8 = depth_img.astype(np.uint8)
        depth_uint8 = cv2.cvtColor(depth_uint8, cv2.COLOR_GRAY2BGR)
        all_img = np.concatenate([depth_uint8, rgb_img], axis=0)
        cv2.imshow('all', all_img)
        cv2.waitKey(1)
        
        # 生成点云
        point_cloud = pc_generator.generatePointCloudFromImages(
            rgb_img, depth_img, use_rgb=True
        )
        
        
        # 使用修复版本的采样函数
        point_cloud = point_cloud_sampling_fixed(point_cloud, args.num_points, method='fps')
        
        all_point_clouds.append(point_cloud)
        
        # 可视化（如果启用）
        visualizer_3d.update_point_cloud(point_cloud)

    # 转换为numpy数组
    all_point_clouds = np.stack(all_point_clouds, axis=0)
    cprint(f"生成的点云数据形状: {all_point_clouds.shape}", "green")

    # 保存点云数据到新的zarr文件
    output_path = args.output_path
    if output_path is None:
        output_path = zarr_path.replace('.zarr', '_with_pointcloud.zarr')
    
    cprint(f"保存点云数据到: {output_path}", "yellow")
    
    # 创建新的zarr文件
    zarr_output = zarr.group(output_path)
    zarr_data = zarr_output.create_group('data')
    zarr_meta = zarr_output.create_group('meta')
    
    # 复制原有数据
    for key in zarr_root['data'].keys():
        zarr_data.create_dataset(key, data=zarr_root['data'][key][:], 
                               chunks=zarr_root['data'][key].chunks,
                               dtype=zarr_root['data'][key].dtype,
                               compressor=zarr_root['data'][key].compressor)
    
    # 保存新的点云数据
    compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=1)
    point_cloud_chunk_size = (100, all_point_clouds.shape[1], all_point_clouds.shape[2])
    zarr_data.create_dataset('point_cloud', data=all_point_clouds, 
                           chunks=point_cloud_chunk_size, dtype='float32', 
                           overwrite=True, compressor=compressor)
    
    # 复制元数据
    for key in zarr_root['meta'].keys():
        zarr_meta.create_dataset(key, data=zarr_root['meta'][key][:],
                               dtype=zarr_root['meta'][key].dtype,
                               compressor=zarr_root['meta'][key].compressor)
    
    cprint(f'-'*50, 'cyan')
    cprint(f'点云数据形状: {all_point_clouds.shape}, 范围: [{np.min(all_point_clouds)}, {np.max(all_point_clouds)}]', 'green')
    cprint(f'保存完成: {output_path}', 'green')

    # 清理
    del env, pc_generator
    del zarr_root, zarr_output, zarr_data, zarr_meta

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='从RGB和深度图像生成点云数据')
    parser.add_argument('--zarr_path', 
        default="./3D-Diffusion-Policy/data/panda/panda_pick_and_place_0926_1.zarr", 
        type=str)
    parser.add_argument('--num_points', type=int, default=1024, help='点云采样点数')
    parser.add_argument('--output_path', type=str, default=None, help='输出文件路径')
    
    args = parser.parse_args()
    # test_playback_images(args.zarr_path)
    main(args)
