"""
Model definitions for HMDB51 action recognition.

Available models:
- ViT_GRU_TimeS      : Vision Transformer + GRU
- ViT_Only           : Vision Transformer with temporal average pooling
- GRU_With_FeatureExtractor : Generic feature extractor + GRU
"""

import torch
import torch.nn as nn
from timm import create_model


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance."""

    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction="none")
        p_t = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        if self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class ViT_GRU_TimeS(nn.Module):
    """Vision Transformer backbone + GRU temporal modeling."""

    def __init__(
        self,
        num_classes: int = 51,
        hidden_size: int = 512,
        num_layers: int = 2,
        dropout: float = 0.5,
        vit_name: str = "vit_small_patch16_224",
        pretrained: bool = True,
    ):
        super().__init__()
        self.vit = create_model(vit_name, pretrained=pretrained)
        # Replace classification head with a projection to 512-d
        in_features = self.vit.head.in_features
        self.vit.head = nn.Linear(in_features, 512)

        self.gru = nn.GRU(
            input_size=512,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, T, C, H, W)
        N, T, C, H, W = x.shape
        frame_features = []

        for t in range(T):
            frame = x[:, t]  # (N, C, H, W)
            feat = self.vit(frame)  # (N, 512)
            frame_features.append(feat.unsqueeze(1))

        seq = torch.cat(frame_features, dim=1)  # (N, T, 512)
        gru_out, _ = self.gru(seq)
        last = gru_out[:, -1, :]  # (N, hidden_size)
        return self.classifier(last)


class ViT_Only(nn.Module):
    """Vision Transformer with simple temporal average pooling (no RNN)."""

    def __init__(
        self,
        num_classes: int = 51,
        dropout: float = 0.5,
        vit_name: str = "vit_small_patch16_224",
        pretrained: bool = True,
    ):
        super().__init__()
        self.vit = create_model(vit_name, pretrained=pretrained)
        in_features = self.vit.head.in_features
        self.vit.head = nn.Linear(in_features, 512)

        self.classifier = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, T, C, H, W = x.shape
        frame_features = []

        for t in range(T):
            frame = x[:, t]
            feat = self.vit(frame)
            frame_features.append(feat.unsqueeze(1))

        seq = torch.cat(frame_features, dim=1)  # (N, T, 512)
        avg = seq.mean(dim=1)  # (N, 512)
        return self.classifier(avg)


class GRU_With_FeatureExtractor(nn.Module):
    """
    Generic wrapper: any timm model as frame-level feature extractor + GRU.
    Useful for ablation studies.
    """

    def __init__(
        self,
        feature_extractor: nn.Module,
        input_size: int = 384,
        hidden_size: int = 512,
        num_layers: int = 2,
        num_classes: int = 51,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.feature_extractor = feature_extractor
        # Remove original classification head
        if hasattr(self.feature_extractor, "head"):
            self.feature_extractor.head = nn.Identity()

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, T, C, H, W = x.shape
        frame_features = []

        for t in range(T):
            frame = x[:, t]
            feat = self.feature_extractor(frame)
            frame_features.append(feat.unsqueeze(1))

        seq = torch.cat(frame_features, dim=1)
        gru_out, _ = self.gru(seq)
        last = gru_out[:, -1, :]
        return self.classifier(last)


def build_model(
    model_name: str,
    num_classes: int,
    device: torch.device,
    **kwargs,
) -> nn.Module:
    """Factory helper."""
    model_name = model_name.lower()
    if model_name in ("vit_gru", "vit_gru_times"):
        model = ViT_GRU_TimeS(num_classes=num_classes, **kwargs)
    elif model_name in ("vit_only", "vit"):
        model = ViT_Only(num_classes=num_classes, **kwargs)
    elif model_name == "gru_fe":
        vit = create_model("vit_small_patch16_224", pretrained=True)
        # vit_small produces 384-d features before head
        model = GRU_With_FeatureExtractor(
            vit, input_size=384, num_classes=num_classes, **kwargs
        )
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    return model.to(device)
