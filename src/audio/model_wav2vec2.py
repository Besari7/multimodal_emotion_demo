from __future__ import annotations

import torch
from torch import nn
from transformers import Wav2Vec2Model


class Wav2Vec2EmotionModel(nn.Module):
    def __init__(self, checkpoint: str = "facebook/wav2vec2-base", embedding_dim: int = 256, num_classes: int = 7) -> None:
        super().__init__()
        self.backbone = Wav2Vec2Model.from_pretrained(checkpoint)
        hidden_size = self.backbone.config.hidden_size
        self.proj = nn.Sequential(
            nn.Linear(hidden_size, embedding_dim),
            nn.GELU(),
            nn.Dropout(0.2),
        )
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def freeze_feature_extractor_and_encoder_0_9(self) -> None:
        self.backbone.feature_extractor._freeze_parameters()
        for idx, layer in enumerate(self.backbone.encoder.layers):
            if idx <= 9:
                for param in layer.parameters():
                    param.requires_grad = False

    def unfreeze_encoder_10_11(self) -> None:
        for idx, layer in enumerate(self.backbone.encoder.layers):
            if idx >= 10:
                for param in layer.parameters():
                    param.requires_grad = True

    def unfreeze_all(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(self, input_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = self.backbone(input_values=input_values)
        pooled = outputs.last_hidden_state.mean(dim=1)
        embedding = self.proj(pooled)
        logits = self.classifier(embedding)
        return embedding, logits
