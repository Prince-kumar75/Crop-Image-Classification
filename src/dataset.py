"""
Dataset and DataLoader utilities for Crop Image Classification.
"""

import os
import random
from typing import Optional, Tuple, Dict, List
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from PIL import Image

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms(img_size: int = 224, augment: bool = True) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Returns train and eval transforms.
    
    Args:
        img_size: Image resolution (default: 224)
        augment: Whether to apply data augmentation for training
    """
    if augment:
        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])

    eval_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    return train_transform, eval_transform


def create_balanced_subset(dataset: datasets.ImageFolder, samples_per_class: int, seed: int = 42) -> Subset:
    """
    Creates a balanced subset of an ImageFolder dataset with a fixed number of samples per class.
    Useful for rapid CPU prototyping and testing.
    """
    random.seed(seed)
    class_indices: Dict[int, List[int]] = {}
    for idx, (_, class_idx) in enumerate(dataset.samples):
        if class_idx not in class_indices:
            class_indices[class_idx] = []
        class_indices[class_idx].append(idx)

    selected_indices: List[int] = []
    for class_idx, idx_list in class_indices.items():
        if len(idx_list) > samples_per_class:
            selected_indices.extend(random.sample(idx_list, samples_per_class))
        else:
            selected_indices.extend(idx_list)

    return Subset(dataset, selected_indices)


def get_dataloaders(
    dataset_dir: str = "RGB_224x224",
    batch_size: int = 32,
    num_workers: int = 0,
    img_size: int = 224,
    samples_per_class: Optional[int] = None,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str], Dict[str, int]]:
    """
    Creates DataLoaders for train, val, and test splits.

    Returns:
        train_loader, val_loader, test_loader, class_names, class_to_idx
    """
    train_dir = os.path.join(dataset_dir, "train")
    val_dir = os.path.join(dataset_dir, "val")
    test_dir = os.path.join(dataset_dir, "test")

    train_transform, eval_transform = get_transforms(img_size=img_size, augment=True)

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=eval_transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=eval_transform)

    class_names = train_dataset.classes
    class_to_idx = train_dataset.class_to_idx

    # If samples_per_class is specified, create balanced subsets for quicker training/eval
    if samples_per_class is not None and samples_per_class > 0:
        val_samples = max(2, samples_per_class // 4)
        test_samples = max(2, samples_per_class // 4)
        train_subset = create_balanced_subset(train_dataset, samples_per_class, seed=seed)
        val_subset = create_balanced_subset(val_dataset, val_samples, seed=seed)
        test_subset = create_balanced_subset(test_dataset, test_samples, seed=seed)

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, class_names, class_to_idx
