from __future__ import annotations

import torch
from torch import nn
from transformers import BertModel


class BertContextEmotionModel(nn.Module):
    def __init__(self, checkpoint: str = "bert-base-uncased", embedding_dim: int = 256, num_classes: int = 7) -> None:
        super().__init__()
        self.backbone = BertModel.from_pretrained(checkpoint)
        hidden_size = self.backbone.config.hidden_size
        self.proj = nn.Sequential(
            nn.Linear(hidden_size, embedding_dim),
            nn.GELU(),
            nn.Dropout(0.2),
        )
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def freeze_embeddings_and_encoder_0_7(self) -> None:
        for param in self.backbone.embeddings.parameters():
            param.requires_grad = False
        for idx, layer in enumerate(self.backbone.encoder.layer):
            if idx <= 7:
                for param in layer.parameters():
                    param.requires_grad = False

    def unfreeze_encoder_8_11(self) -> None:
        for idx, layer in enumerate(self.backbone.encoder.layer):
            if idx >= 8:
                for param in layer.parameters():
                    param.requires_grad = True

    def unfreeze_all(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls = outputs.last_hidden_state[:, 0, :]
        embedding = self.proj(cls)
        logits = self.classifier(embedding)
        return embedding, logits
