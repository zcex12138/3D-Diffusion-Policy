from typing import Dict, List
import torch
import numpy as np
import copy
import os
from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.common.replay_buffer import ReplayBuffer
from diffusion_policy_3d.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy_3d.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from diffusion_policy_3d.dataset.base_dataset import BaseDataset
from diffusion_policy_3d.common.normalize_util import get_image_range_normalizer

TIP_CAM_NAMES = [
    "thumb_tip_cam",
    "index_tip_cam",
    "middle_tip_cam",
    "ring_tip_cam",
    "little_tip_cam",
]

class DphandDataset(BaseDataset):
    def __init__(self,
            zarr_path, 
            horizon=1,
            pad_before=0,
            pad_after=0,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None,
            use_tip_depth=False,
            tip_cam_names: List[str] = None,
            ):
        super().__init__()
        
        # dphand环境的数据键包括：state, action, point_cloud, img, depth, full_state
        base_keys = ['state', 'front', 'wrist', 'action', 'point_cloud', 'full_state']
        self.use_tip_depth = use_tip_depth
        self.tip_cam_names: List[str] = tip_cam_names or TIP_CAM_NAMES
        if self.use_tip_depth:
            tip_keys = [f"tip_depth/{cam}" for cam in self.tip_cam_names]
            base_keys.extend(tip_keys)

        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=base_keys)
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask, 
            max_n=max_train_episodes, 
            seed=seed)

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask)
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
            )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        data = {
            'action': self.replay_buffer['action'],
            'agent_pos': self.replay_buffer['state'],  # 使用state作为agent_pos
            'point_cloud': self.replay_buffer['point_cloud'],
            'full_state': self.replay_buffer['full_state'],
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        if self.use_tip_depth:
            normalizer['tip_depth'] = get_image_range_normalizer()
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        # 使用state作为agent_pos（机器人关节状态：位置+速度）
        agent_pos = sample['state'][:,].astype(np.float32)
        # 提取点云数据
        point_cloud = sample['point_cloud'][:,].astype(np.float32)
        # 提取完整状态（包含机器人状态+物体状态）
        full_state = sample['full_state'][:,].astype(np.float32)

        data = {
            'obs': {
                'point_cloud': point_cloud, 
                'agent_pos': agent_pos,
                'full_state': full_state,
            },
            'action': sample['action'].astype(np.float32)
        }
        if self.use_tip_depth:
            tip_imgs = []
            for cam in self.tip_cam_names:
                cam_img = sample[f"tip_depth/{cam}"].astype(np.float32)  # (T, H, W, 3)
                # depth images stored as BGR gray, take first channel only
                cam_gray = cam_img[..., 0:1] / 255.0
                cam_gray = np.transpose(cam_gray, (0, 3, 1, 2))  # (T, 1, H, W)
                tip_imgs.append(cam_gray)
            tip_depth = np.concatenate(tip_imgs, axis=1)  # (T, num_cam, H, W)
            data['obs']['tip_depth'] = tip_depth.astype(np.float32)
        return data
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data


class DphandImageDataset(BaseDataset):
    def __init__(self,
            zarr_path: str,
            horizon=1,
            pad_before=0,
            pad_after=0,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None,
            ):
        super().__init__()
        
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=['state', 'front', 'wrist', 'action'])
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask, 
            max_n=max_train_episodes, 
            seed=seed)

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask)
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
            )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        data = {
            'action': self.replay_buffer['action'],
            'agent_pos': self.replay_buffer['state']
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        normalizer['front'] = get_image_range_normalizer()
        normalizer['wrist'] = get_image_range_normalizer()
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        front = np.moveaxis(sample['front'].astype(np.float32),-1,1)/255
        wrist = np.moveaxis(sample['wrist'].astype(np.float32),-1,1)/255
        data = {
            'obs': {
                'front': front, 
                'wrist': wrist,
                'agent_pos': sample['state'][:,].astype(np.float32),
            },
            'action': sample['action'].astype(np.float32)
        }
        return data
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data
