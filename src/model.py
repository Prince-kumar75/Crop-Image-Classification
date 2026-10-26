"""
Model architecture definition and loader for Crop Image Classification.
"""

import os
from typing import Optional, List, Dict, Any, Tuple
import torch
import torch.nn as nn
from torchvision import models


def build_model(
    num_classes: int = 139,
    architecture: str = "resnet18",
    pretrained: bool = True,
    freeze_backbone: bool = True,
    dropout_rate: float = 0.2
) -> nn.Module:
    """
    Constructs a transfer learning model for crop classification.

    Args:
        num_classes: Number of crop plant categories
        architecture: Backbone architecture ('resnet18', 'resnet34', 'mobilenet_v2')
        pretrained: Whether to load ImageNet pre-trained weights
        freeze_backbone: If True, freezes feature extractor parameters
        dropout_rate: Dropout probability in the classification head
    """
    arch = architecture.lower()

    if arch == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, num_classes)
        )

    elif arch == "resnet34":
        weights = models.ResNet34_Weights.DEFAULT if pretrained else None
        model = models.resnet34(weights=weights)
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, num_classes)
        )

    elif arch == "mobilenet_v2":
        weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v2(weights=weights)
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, num_classes)
        )

    else:
        raise ValueError(f"Unsupported architecture: {architecture}. Supported: resnet18, resnet34, mobilenet_v2")

    return model


def save_checkpoint(
    model: nn.Module,
    save_path: str,
    class_names: List[str],
    architecture: str = "resnet18",
    epoch: Optional[int] = None,
    val_acc: Optional[float] = None,
    history: Optional[Dict[str, Any]] = None
) -> None:
    """
    Saves model weights, class metadata, and training stats to disk.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "architecture": architecture,
        "num_classes": len(class_names),
        "class_names": class_names,
        "epoch": epoch,
        "val_acc": val_acc,
        "history": history
    }
    torch.save(checkpoint, save_path)


def load_model_from_checkpoint(
    checkpoint_path: str,
    device: Optional[torch.device] = None
) -> Tuple[nn.Module, List[str], Dict[str, Any]]:
    """
    Loads model and class metadata from a checkpoint file.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    arch = checkpoint.get("architecture", "resnet18")
    class_names = checkpoint.get("class_names", [])
    num_classes = len(class_names) if class_names else checkpoint.get("num_classes", 139)

    model = build_model(
        num_classes=num_classes,
        architecture=arch,
        pretrained=False,
        freeze_backbone=False
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, class_names, checkpoint
