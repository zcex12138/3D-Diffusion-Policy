import torch
import torch.nn as nn
from torchvision import models


class ResNetEncoder(nn.Module):
    """
    Thin wrapper around torchvision ResNet backbones that exposes a simple
    feature extractor returning a fixed-dimensional vector.
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        in_channels: int = 3,
        output_dim: int = 256,
        pretrained: bool = False,
        freeze_backbone: bool = False,
        global_pool: str = "avg",
    ):
        super().__init__()

        if not hasattr(models, backbone):
            raise ValueError(f"Unsupported backbone '{backbone}'")

        resnet = getattr(models, backbone)(pretrained=pretrained)

        if in_channels != 3:
            resnet.conv1 = self._build_first_conv(resnet.conv1, in_channels, pretrained)

        # keep everything up to layer4
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-2])

        if global_pool == "avg":
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
        elif global_pool == "max":
            self.pool = nn.AdaptiveMaxPool2d((1, 1))
        else:
            raise ValueError(f"Unsupported global_pool '{global_pool}'")

        feat_dim = resnet.fc.in_features
        self.head = nn.Linear(feat_dim, output_dim) if output_dim is not None else nn.Identity()
        self.output_dim = output_dim if output_dim is not None else feat_dim

        if freeze_backbone:
            for param in self.feature_extractor.parameters():
                param.requires_grad_(False)

    def _build_first_conv(self, conv1: nn.Conv2d, in_channels: int, pretrained: bool) -> nn.Conv2d:
        new_conv = nn.Conv2d(
            in_channels,
            conv1.out_channels,
            kernel_size=conv1.kernel_size,
            stride=conv1.stride,
            padding=conv1.padding,
            bias=conv1.bias is not None,
        )
        if pretrained:
            with torch.no_grad():
                weight = conv1.weight
                if in_channels == 1:
                    weight = weight.sum(dim=1, keepdim=True)
                else:
                    weight = weight.mean(dim=1, keepdim=True).repeat(1, in_channels, 1, 1) / in_channels
                new_conv.weight.copy_(weight)
                if conv1.bias is not None:
                    new_conv.bias.copy_(conv1.bias)
        return new_conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.feature_extractor(x)
        pooled = self.pool(feats)
        pooled = torch.flatten(pooled, 1)
        return self.head(pooled)
