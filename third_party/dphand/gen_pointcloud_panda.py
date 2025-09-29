"""
后处理脚本: 从RGB和深度图像生成点云数据
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
        rgb_img = rgb_arrays[i].copy()
        depth_img = depth_arrays[i].copy()
        
        if i % 100 == 0:  # 每100张图像打印一次进度
            cprint(f"处理图像 {i}/{len(rgb_arrays)}", "cyan")

        depth_img[depth_img>0] = (depth_img[depth_img>0] - depth_img[depth_img>0].min()) / (depth_img[depth_img>0].max() - depth_img[depth_img>0].min() + 1e-8) * 255
        depth_uint8 = depth_img.astype(np.uint8)
        depth_uint8 = cv2.cvtColor(depth_uint8, cv2.COLOR_GRAY2BGR)
        all_img = np.concatenate([depth_uint8, rgb_img], axis=0)
        cv2.imshow('all', all_img)
        cv2.waitKey(1)
        
        # 生成点云
        pc = pc_generator.generatePointCloudFromImages(rgb_arrays[i], depth_arrays[i], use_rgb=True)
        point_cloud = point_cloud_sampling(pc, args.num_points, method='fps')
        
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
