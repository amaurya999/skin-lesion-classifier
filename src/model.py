import torch.nn as nn
from torchvision import models

from src import config


def build_model(freeze_backbone: bool = True, pretrained: bool = True) -> nn.Module:
    """ResNet18, optionally pretrained on ImageNet, with a new final layer for our classes."""
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # Replace the final layer (1000 ImageNet classes) with a 7-class layer.
    # A newly created layer is trainable by default.
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, config.NUM_CLASSES)
    return model


def unfreeze_last_block(model: nn.Module) -> None:
    """Make the last residual block trainable, for a second 'fine-tuning' stage."""
    for param in model.layer4.parameters():
        param.requires_grad = True


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)