#!/usr/bin/env python3
"""
Dphand 指尖深度导出脚本（带 episode 重置）

功能：
- 读取 zarr 数据，按 episode 顺序回放（每个 demo 结束后强制 reset）
- 对每一步，用动作驱动环境一步
- 从 5 个指尖相机渲染深度，转成灰度图 (uint8, 3 通道 BGR)
- 写入克隆出的 zarr 文件 data/tip_depth/<cam_name>
- 可选实时可视化 (--visualize)
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm
import zarr
import numcodecs
import cv2

# 添加项目路径
ROOT_DIR = str(Path(__file__).parent.parent)
sys.path.append(ROOT_DIR)

from diffusion_policy_3d.env.dphand.dphand_wrapper import DphandImageEnvWrapper
from dphand_env.envs.pick_and_place_env import PickAndPlaceEnv

TIP_CAM_NAMES = [
    "thumb_tip_cam",
    "index_tip_cam",
    "middle_tip_cam",
    "ring_tip_cam",
    "little_tip_cam",
]


def depth_to_uint8(depth, max_depth=0.00007):
    """
    将深度 [0, max_depth] 等比例映射到灰度 [0, 255]。
    大于 max_depth 或 无穷远（=0）的部分统一设为背景色（黑）。
    返回:
        depth_uint8: 3 通道 BGR 灰度图 (uint8)
        d_min: 当前帧深度的最小值（裁剪后）
        d_max: 当前帧深度的最大值（裁剪后，<= max_depth）
    """
    depth_scaled = np.zeros_like(depth, dtype=np.float32)
    valid = (depth > 0) & (depth <= max_depth)
    depth_scaled[valid] = (depth[valid] / max_depth) * 255.0
    depth_uint8 = depth_scaled.astype(np.uint8)
    depth_uint8 = cv2.cvtColor(depth_uint8, cv2.COLOR_GRAY2BGR)

    if np.any(valid):
        d_min = float(depth[valid].min())
        d_max = float(depth[valid].max())
    else:
        d_min = 0.0
        d_max = 0.0
    return depth_uint8, d_min, d_max


class TipDepthZarrWriter:
    """
    克隆原始 zarr，并在 data/tip_depth/ 下写入灰度指尖深度图
    每个相机一个数组：shape = (N, H, W, 3), dtype = uint8
    """

    def __init__(self, src_zarr_path: str, dst_zarr_path: str, cam_names: List[str]):
        self.cam_names = cam_names
        self.dst_zarr_path = os.path.abspath(os.path.expanduser(dst_zarr_path))
        self.src_zarr_path = os.path.abspath(os.path.expanduser(src_zarr_path))
        if self.dst_zarr_path == self.src_zarr_path:
            raise ValueError("输出路径不能与原路径相同，否则会破坏源数据。")

        # 拷贝源 zarr 结构到新目录
        src_root = zarr.open(self.src_zarr_path, mode="r")
        dst_store = zarr.DirectoryStore(self.dst_zarr_path)
        zarr.copy_store(src_root.store, dst_store, if_exists="replace")

        self.root = zarr.open(dst_store, mode="a")
        self.tip_group = self.root.require_group("data").require_group("tip_depth")
        self.arrays: Dict[str, zarr.Array] = {}
        self.img_shape = None
        self.total_steps = 0
        self.compressor = numcodecs.Blosc(
            cname="zstd", clevel=5, shuffle=numcodecs.Blosc.BITSHUFFLE
        )

    def _ensure_arrays(self, img_shape):
        """第一次写入时，根据图像形状初始化各个相机数组"""
        if self.img_shape is not None:
            return
        self.img_shape = img_shape  # (H, W, 3)
        for cam_name in self.cam_names:
            self.arrays[cam_name] = self.tip_group.zeros(
                name=cam_name,
                shape=(0,) + img_shape,  # (N, H, W, 3)
                chunks=(1,) + img_shape,
                dtype=np.uint8,
                compressor=self.compressor,
            )

    def append_step(self, img_dict: Dict[str, np.ndarray]):
        """追加一帧所有相机的灰度图"""
        if not img_dict:
            return
        if self.img_shape is None:
            self._ensure_arrays(next(iter(img_dict.values())).shape)

        self.total_steps += 1
        for cam_name in self.cam_names:
            arr = self.arrays[cam_name]
            arr.resize((self.total_steps,) + arr.shape[1:])
            arr[-1] = img_dict[cam_name]

    def finalize(self):
        # 目前没有额外操作，留接口方便将来扩展
        pass


class DphandDepthExporter:
    """
    不基于 episode 的深度导出器：
    - 从原始 zarr 中直接读取 data/action（按时间展开）
    - 用环境按顺序执行这些动作
    - 每一步渲染指尖相机深度，转灰度图，写入新的 zarr
    """

    def __init__(self, zarr_path: str, depth_zarr_path: Optional[str] = None):
        self.zarr_path = zarr_path
        self.zarr_root = zarr.open(zarr_path, mode="r")

        # 动作序列：按时间展开
        if "action" not in self.zarr_root["data"]:
            raise KeyError("zarr['data']['action'] 不存在，请检查数据格式。")
        self.actions = self.zarr_root["data"]["action"][:]  # (N, act_dim)

        # episode meta 信息
        if "episode_ends" not in self.zarr_root["meta"]:
            raise KeyError("zarr['meta']['episode_ends'] 不存在，无法确定 demo 边界。")
        self.episode_ends = self.zarr_root["meta"]["episode_ends"][:].astype(np.int64)
        if self.episode_ends.ndim != 1 or len(self.episode_ends) == 0:
            raise ValueError("episode_ends 格式不正确。")
        self.num_episodes = len(self.episode_ends)

        if "init_state" in self.zarr_root["meta"]:
            self.init_states = self.zarr_root["meta"]["init_state"][:]
        else:
            self.init_states = None

        # 如果有 front 图像，用于可视化时拼接
        self.has_front = (
            "image" in self.zarr_root["data"]
            and "front" in self.zarr_root["data"]["image"]
        )
        if self.has_front:
            self.front_imgs = self.zarr_root["data"]["image"]["front"][:]  # (N, H, W, 3)
        else:
            self.front_imgs = None

        # 创建环境
        self.env = DphandImageEnvWrapper(
            PickAndPlaceEnv(config="pick_cube_env_cfg", render_mode="human")
        )
        self.tip_cam_names = TIP_CAM_NAMES

        # 创建写入器
        self.depth_writer = (
            TipDepthZarrWriter(self.zarr_path, depth_zarr_path, self.tip_cam_names)
            if depth_zarr_path
            else None
        )

    def _get_episode_init_state(self, episode_idx: int) -> Optional[np.ndarray]:
        if self.init_states is None:
            return None
        init = self.init_states
        if init.ndim == 2:
            return np.asarray(init[episode_idx], dtype=np.float32)
        flat = init.reshape(-1)
        start = episode_idx * 14
        end = start + 14
        if end > flat.shape[0]:
            raise IndexError("init_state 数据长度不足。")
        return np.asarray(flat[start:end], dtype=np.float32)

    def _reset_env_for_episode(self, episode_idx: int):
        reset_kwargs = {}
        if episode_idx < self.num_episodes:
            init_state = self._get_episode_init_state(episode_idx)
            if init_state is not None:
                reset_kwargs["init_state"] = init_state
        obs = self.env.reset(**reset_kwargs)
        return obs

    def run(self, max_steps: Optional[int] = None, visualize: bool = False):
        num_steps_total = self.actions.shape[0]
        if max_steps is not None:
            num_steps = min(max_steps, num_steps_total)
        else:
            num_steps = num_steps_total

        print(f"总 step 数: {num_steps_total}，本次处理: {num_steps} 步")

        # 初次 reset（第 0 个 demo）
        episode_idx = 0
        next_episode_end = self.episode_ends[episode_idx]
        obs = self._reset_env_for_episode(episode_idx)

        for step_idx in tqdm(range(num_steps), desc="Processing steps"):
            action = self.actions[step_idx]
            obs, reward, done, info = self.env.step(action)

            img_dict: Dict[str, np.ndarray] = {}
            vis_imgs = []

            # 渲染每个指尖相机的深度并转成灰度 BGR 图
            for cam_name in self.tip_cam_names:
                _, depth = self.env.unwrapped._viewer.render_segment_depth(
                    self.env.unwrapped.cam_ids[cam_name]
                )
                depth_uint8, d_min, d_max = depth_to_uint8(depth)
                img_dict[cam_name] = depth_uint8
                if visualize:
                    vis_imgs.append(depth_uint8)

            # 写入 zarr
            if self.depth_writer is not None:
                self.depth_writer.append_step(img_dict)

            # 可视化
            if visualize:
                if self.has_front:
                    tip_panel = np.concatenate(vis_imgs, axis=0)
                    all_img = np.concatenate(
                        [self.front_imgs[step_idx], tip_panel], axis=0
                    )
                else:
                    all_img = np.concatenate(vis_imgs, axis=0)
                cv2.imshow("front_and_tip_depths", all_img)
                self.env.render()
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("检测到按键 'q'，提前结束。")
                    break

            # 判断 demo 边界
            reached_episode_end = (
                next_episode_end is not None and (step_idx + 1) == next_episode_end
            )
            need_reset = done or reached_episode_end

            if reached_episode_end and episode_idx + 1 < self.num_episodes:
                episode_idx += 1
                next_episode_end = self.episode_ends[episode_idx]
            elif reached_episode_end:
                next_episode_end = None

            if need_reset and (step_idx + 1 < num_steps):
                if reached_episode_end and episode_idx < self.num_episodes:
                    obs = self._reset_env_for_episode(episode_idx)
                else:
                    obs = self.env.reset()

        print("深度导出完成。")

    def finalize(self):
        if self.depth_writer:
            self.depth_writer.finalize()


def main():
    parser = argparse.ArgumentParser(description="Dphand 指尖灰度深度生成脚本（无 episode）")
    parser.add_argument(
        "--zarr_path",
        type=str,
        default="data/1014/pick_and_place_47demos_1014_pc.zarr",
        help="原始 zarr 数据路径",
    )
    parser.add_argument(
        "--depth_zarr_path",
        type=str,
        default=None,
        help="输出 zarr 路径（默认自动加后缀 _tip_gray.zarr）",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="最多处理多少步（默认处理完所有 step）",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="是否实时显示回放（默认不显示）",
    )
    args = parser.parse_args()

    print(f"加载数据: {args.zarr_path}")

    depth_zarr_path = (
        args.depth_zarr_path
        if args.depth_zarr_path
        else args.zarr_path.replace(".zarr", "_tip_gray.zarr")
    )

    exporter = DphandDepthExporter(
        zarr_path=args.zarr_path,
        depth_zarr_path=depth_zarr_path,
    )

    exporter.run(max_steps=args.max_steps, visualize=args.visualize)
    exporter.finalize()

    if args.visualize:
        cv2.destroyAllWindows()

    print(f"所有 step 处理完成，输出保存到: {depth_zarr_path}")


if __name__ == "__main__":
    main()
