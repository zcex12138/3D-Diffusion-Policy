#!/usr/bin/env python3
"""
Replay demos with live visualization to verify tip-depth frames stored inside a Zarr dataset.

Shows (per step):
  - Current rendered front camera from the simulator vs the saved RGB frame.
  - For each fingertip camera: freshly rendered depth (converted to uint8 BGR)
    next to the saved Zarr image (data/tip_depth/<cam>).

Goal: quickly confirm that the exported fingertip depth images match what the simulator sees.
"""

import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import zarr
from tqdm import tqdm

import sys

# Add repo root to import env wrappers
ROOT_DIR = str(Path(__file__).resolve().parents[2])
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from diffusion_policy_3d.env.dphand.dphand_wrapper import DphandImageEnvWrapper  # noqa: E402
from dphand_env.envs.pick_and_place_env import PickAndPlaceEnv  # noqa: E402

TIP_CAM_NAMES = [
    "thumb_tip_cam",
    "index_tip_cam",
    "middle_tip_cam",
    "ring_tip_cam",
    "little_tip_cam",
]


def depth_to_uint8(depth: np.ndarray, max_depth: float = 0.000005) -> np.ndarray:
    """Convert MuJoCo depth map to a uint8 BGR visualization."""
    depth_scaled = np.zeros_like(depth, dtype=np.float32)
    valid = (depth > 0) & (depth <= max_depth)
    depth_scaled[valid] = (depth[valid] / max_depth) * 255.0
    depth_uint8 = depth_scaled.astype(np.uint8)
    return cv2.cvtColor(depth_uint8, cv2.COLOR_GRAY2BGR)


def ensure_uint8(img: np.ndarray) -> np.ndarray:
    """Clamp/convert any array to uint8 for visualization."""
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img


class TipDepthReplayChecker:
    def __init__(self, zarr_path: str):
        self.zarr_path = os.path.expanduser(zarr_path)
        self.root = zarr.open(self.zarr_path, mode="r")

        data = self.root["data"]
        meta = self.root["meta"]

        if "action" not in data:
            raise KeyError("Missing 'data/action' in zarr dataset.")
        self.actions = data["action"][:]  # (N, act_dim)

        # RGB frames
        self.front_imgs = None
        if "image" in data and "front" in data["image"]:
            self.front_imgs = data["image"]["front"][:]

        # Tip depth images (expected to exist after export)
        if "tip_depth" not in data:
            raise KeyError("Missing 'data/tip_depth' group. Did you export tip depth?")
        self.tip_depth_arrays = {
            cam: data["tip_depth"][cam][:]
            for cam in TIP_CAM_NAMES
            if cam in data["tip_depth"]
        }
        missing = [cam for cam in TIP_CAM_NAMES if cam not in self.tip_depth_arrays]
        if missing:
            raise KeyError(f"Tip depth arrays missing for cams: {missing}")

        # Episode boundaries
        if "episode_ends" not in meta:
            raise KeyError("Missing 'meta/episode_ends' in zarr dataset.")
        self.episode_ends = meta["episode_ends"][:].astype(np.int64)
        self.num_episodes = len(self.episode_ends)

        # Optional init states (object/target pose)
        self.init_states = None
        if "init_state" in meta:
            init_arr = meta["init_state"]
            self.init_states = init_arr[:]

        self.env = DphandImageEnvWrapper(
            PickAndPlaceEnv(config="pick_cube_env_cfg", render_mode="human")
        )
        self.tip_cam_names = TIP_CAM_NAMES

    def _episode_slice(self, episode_idx: int) -> Tuple[int, int]:
        start = 0 if episode_idx == 0 else int(self.episode_ends[episode_idx - 1])
        end = int(self.episode_ends[episode_idx])
        return start, end

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
            raise IndexError("init_state length mismatch with episodes.")
        return np.asarray(flat[start:end], dtype=np.float32)

    def _reset_episode(self, episode_idx: int):
        kwargs = {}
        init_state = self._get_episode_init_state(episode_idx)
        if init_state is not None:
            kwargs["init_state"] = init_state
        return self.env.reset(**kwargs)

    def run(
        self,
        start_episode: int = 0,
        max_episodes: Optional[int] = None,
        slow: bool = False,
    ):
        assert 0 <= start_episode < self.num_episodes, "start_episode out of range"
        if max_episodes is None:
            max_episodes = self.num_episodes - start_episode
        end_episode = min(start_episode + max_episodes, self.num_episodes)

        step_offset = 0 if start_episode == 0 else int(self.episode_ends[start_episode - 1])

        for ep in range(start_episode, end_episode):
            start, end = self._episode_slice(ep)
            start = max(start, step_offset)
            print(f"\n=== Episode {ep} | steps [{start}, {end}) ===")
            obs = self._reset_episode(ep)

            for step_idx in tqdm(range(start, end), desc=f"Episode {ep}"):
                action = self.actions[step_idx]
                obs, _, done, _ = self.env.step(action)

                live_tip_imgs: Dict[str, np.ndarray] = {}
                stored_tip_imgs: Dict[str, np.ndarray] = {}
                for cam in self.tip_cam_names:
                    _, depth = self.env.unwrapped._viewer.render_segment_depth(
                        self.env.unwrapped.cam_ids[cam]
                    )
                    depth = self._depthimg2meters(depth)
                    live_tip_imgs[cam] = depth_to_uint8(depth)
                    stored_tip_imgs[cam] = ensure_uint8(self.tip_depth_arrays[cam][step_idx])

                # Front comparison (live vs stored)
                if self.front_imgs is not None and "front" in obs:
                    live_front = ensure_uint8(obs["front"])
                    stored_front = ensure_uint8(self.front_imgs[step_idx])
                    front_pair = np.concatenate([live_front, stored_front], axis=1)
                elif "front" in obs:
                    front_img = ensure_uint8(obs["front"])
                    front_pair = np.concatenate([front_img, front_img], axis=1)
                else:
                    front_pair = None

                # Tip comparison panels
                tip_rows = []
                for cam in self.tip_cam_names:
                    live_img = live_tip_imgs[cam]
                    stored_img = stored_tip_imgs[cam]
                    pair = np.concatenate([live_img, stored_img], axis=1)
                    cv2.putText(
                        pair,
                        f"{cam} (live | stored)",
                        (5, 15),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (0, 255, 0),
                        1,
                        cv2.LINE_AA,
                    )
                    tip_rows.append(pair)

                tip_panel = np.concatenate(tip_rows, axis=0)
                if front_pair is not None:
                    display_img = np.concatenate([front_pair, tip_panel], axis=0)
                else:
                    display_img = tip_panel

                cv2.imshow("Tip Depth Checker", display_img)
                self.env.render()
                wait_time = 200 if slow else 1
                key = cv2.waitKey(wait_time) & 0xFF
                if key == ord("q"):
                    print("User requested exit (q).")
                    cv2.destroyAllWindows()
                    return

                if done and step_idx + 1 < end:
                    obs = self._reset_episode(ep)

        cv2.destroyAllWindows()
        print("Visualization complete.")
    
    def _depthimg2meters(self, depth: np.ndarray) -> np.ndarray:
        """Convert MuJoCo depth buffer [0,1] to metric distances for fair comparison."""
        model = self.env.unwrapped.model
        extent = model.stat.extent
        near = model.vis.map.znear * extent
        far = model.vis.map.zfar * extent
        return near / (1 - depth * (1 - near / far)) - near


def main():
    parser = argparse.ArgumentParser(description="Replay demos and verify saved tip-depth images.")
    parser.add_argument("--zarr_path", default="data/1014/pick_and_place_47demos_1014_pc_tip_gray.zarr", help="Path to the Zarr dataset containing tip_depth.")
    parser.add_argument("--start_episode", type=int, default=0, help="Episode index to start replaying from.")
    parser.add_argument("--num_episodes", type=int, default=None, help="Number of episodes to replay (default: all).")
    parser.add_argument("--slow", action="store_true", help="Slow down playback for manual inspection.")
    args = parser.parse_args()

    checker = TipDepthReplayChecker(args.zarr_path)
    checker.run(start_episode=args.start_episode, max_episodes=args.num_episodes, slow=args.slow)


if __name__ == "__main__":
    main()
