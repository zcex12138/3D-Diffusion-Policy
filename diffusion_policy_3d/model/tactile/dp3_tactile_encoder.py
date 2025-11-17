from typing import Dict
import torch
import torch.nn as nn
from termcolor import cprint

from diffusion_policy_3d.model.vision.pointnet_extractor import DP3Encoder
from diffusion_policy_3d.model.tactile.resnet import ResNetEncoder


class DP3TactileEncoder(nn.Module):
    """
    Combined encoder for point cloud, state, and tactile images.
    Uses DP3Encoder for point cloud/state and ResNetEncoder for tactile images.
    """
    def __init__(self,
                 observation_space: dict,
                 img_crop_shape=None,
                 out_channel=256,
                 state_mlp_size=(64, 64),
                 state_mlp_activation_fn=nn.ReLU,
                 pointcloud_encoder_cfg=None,
                 use_pc_color=False,
                 pointnet_type='pointnet',
                 tactile_encoder_cfg=None,
                 tactile_keys=None):
        super().__init__()
        
        # Create base encoder for point cloud and state
        self.pc_encoder = DP3Encoder(
            observation_space=observation_space,
            img_crop_shape=img_crop_shape,
            out_channel=out_channel,
            state_mlp_size=state_mlp_size,
            state_mlp_activation_fn=state_mlp_activation_fn,
            pointcloud_encoder_cfg=pointcloud_encoder_cfg,
            use_pc_color=use_pc_color,
            pointnet_type=pointnet_type,
        )
        
        self.tactile_keys = tactile_keys
        self.tactile_encoder_cfg = tactile_encoder_cfg
        assert self.tactile_encoder_cfg.in_channels == len(self.tactile_keys), \
            f"tactile_encoder in_channels must be equal to the number of tactile keys"

        self.tactile_encoder = ResNetEncoder(**self.tactile_encoder_cfg)
        
        self.pc_output_dim = self.pc_encoder.output_shape()
        # Since in_channels == len(tactile_keys), we stack images as multi-channel and encode once
        # So output_dim is just the encoder's output_dim, not multiplied by num_keys
        self.tactile_output_dim = self.tactile_encoder_cfg.output_dim
        self.n_output_channels = self.pc_output_dim + self.tactile_output_dim
    
    def _check_channel_match(self, actual_channels: int, expected_channels: int, 
                            context: str = ""):
        """Check if channel dimensions match, raise error if not"""
        if actual_channels != expected_channels:
            raise RuntimeError(
                f"Channel mismatch in tactile encoder{context}:\n"
                f"  Expected: {expected_channels} channels (from tactile_encoder_cfg.in_channels)\n"
                f"  Actual: {actual_channels} channels\n"
                f"This usually means the tactile_encoder_cfg.in_channels doesn't match the data."
            )

    
    def forward(self, observations: dict) -> torch.Tensor:
        # Encode point cloud and state
        pc_feat = self.pc_encoder(observations)  # [B*T, D_pc]
        
        # Find tactile data and normalize each image
        processed_imgs = []
        for key in self.tactile_keys:
            if key not in observations:
                raise RuntimeError(f"Missing tactile key '{key}' in observations")
            img = observations[key]
            assert len(img.shape) == 4, f"Tactile image must be in [B*T, C, H, W], but {key} got {len(img.shape)}"
            processed_imgs.append(img)
        
        # Stack along channel dimension: [B*T, 1, H, W] * N -> [B*T, N, H, W]
        stacked_tactile = torch.cat(processed_imgs, dim=1)  # [B*T, num_keys, H, W]
        
        # Check channel match
        expected_channels = self.tactile_encoder_cfg.in_channels
        self._check_channel_match(stacked_tactile.shape[1], expected_channels,
                                f" (after stacking {len(processed_imgs)} images)")
        
        # Encode tactile images
        tactile_feat = self.tactile_encoder(stacked_tactile)  # [B*T, D_tactile]
        
        # Concatenate features
        final_feat = torch.cat([pc_feat, tactile_feat], dim=-1)  # [B*T, D_pc + D_tactile]
        
        return final_feat
    
    def output_shape(self):
        return self.n_output_channels
