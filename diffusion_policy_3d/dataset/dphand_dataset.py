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

class DphandDataset(BaseDataset):
    def __init__(self,
            zarr_path, 
            horizon=1,
            pad_before=0,
            pad_after=0,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None,
            shape_meta=None,
            ):
        super().__init__()
        
        # Build keys from shape_meta
        self.obs_meta = shape_meta.get('obs', {}) if shape_meta is not None else {}
        keys = list(self.obs_meta.keys()) + ['action']
        self.replay_buffer = ReplayBuffer.copy_from_path(zarr_path, keys=keys)
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
            'action': self.replay_buffer['action']
            # 'agent_pos'
            # 'point_cloud'
        }
        for key, meta in self.obs_meta.items():
            if meta.get('type') in ['low_dim', 'point_cloud', 'depth'] and key in self.replay_buffer:
                data[key] = self.replay_buffer[key]

        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        normalizer['image'] = get_image_range_normalizer()
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        obs_dict = {}
        for key, meta in self.obs_meta.items():
            if meta.get('type') in ['low_dim', 'point_cloud'] and key in sample:
                obs_dict[key] = sample[key][:,].astype(np.float32)
            elif meta.get('type') in ['image'] and key in sample:
                obs_dict[key] = np.moveaxis(sample[key].astype(np.float32),-1,1)/255 # (N, H, W, 3) -> (N, 3, H, W)
            elif meta.get('type') in ['depth'] and key in sample:
                obs_dict[key] = sample[key][:, np.newaxis, :, :].astype(np.float32) # (N, H, W) -> (N, 1, H, W)
        
        return {
            'obs': obs_dict,
            'action': sample['action'].astype(np.float32)
        }
    
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

def test():
    import os
    zarr_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', '1014', 'pick_and_place_47demos_1014_pc_tactile.zarr')
    shape_meta = {
        'obs': {
            'point_cloud': {
                'shape': [512, 3],
                'type': 'point_cloud'
            },
            'agent_pos': {
                'shape': [29],
                'type': 'low_dim'
            },
            'tactile/thumb_tip_cam': {
                'shape': [84, 84],
                'type': 'depth'
            },
            'tactile/index_tip_cam': {
                'shape': [84, 84],
                'type': 'depth'
            }
        },
        'action': {
            'shape': [29],
            'type': 'low_dim'
        }
    }
    dataset = DphandDataset(zarr_path, horizon=4, pad_before=3, pad_after=0, shape_meta=shape_meta)
    print(dataset[0]['obs']['agent_pos'].shape)
    print(dataset[400]['obs']['tactile/thumb_tip_cam'].shape)

if __name__ == '__main__':
    test()