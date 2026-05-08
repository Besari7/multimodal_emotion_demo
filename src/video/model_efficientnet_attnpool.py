from __future__ import annotations

import torch
from torch import nn
from torchvision.models import EfficientNet_B2_Weights, efficientnet_b2


class AttentiveTemporalPooling(nn.Module):
    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.Tanh(),
            nn.Linear(in_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.attn(x).squeeze(-1)
        weights = torch.softmax(weights, dim=1)
        pooled = (x * weights.unsqueeze(-1)).sum(dim=1)
        return pooled


class VideoEmotionModel(nn.Module):
    def __init__(self, embedding_dim: int = 256, num_classes: int = 7) -> None:
        super().__init__()
        base = efficientnet_b2(weights=EfficientNet_B2_Weights.IMAGENET1K_V1)
        self.backbone = base.features
        self.backbone_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.frame_proj = nn.Linear(1408, embedding_dim)
        self.temporal_pool = AttentiveTemporalPooling(embedding_dim)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def freeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_all(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(self, frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, t, c, h, w = frames.shape
        x = frames.view(bsz * t, c, h, w)
        feat = self.backbone(x)
        feat = self.backbone_pool(feat).flatten(1)
        feat = self.frame_proj(feat)
        feat = feat.view(bsz, t, -1)
        embedding = self.temporal_pool(feat)
        embedding = self.dropout(embedding)
        logits = self.classifier(embedding)
        return embedding, logits
