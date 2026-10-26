"""
Training pipeline for Crop Image Classification.
"""

import os
import time
import json
import argparse
from typing import Optional, Dict, List, Any
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from src.dataset import get_dataloaders
from src.model import build_model, save_checkpoint


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device
) -> Dict[str, float]:
    """
    Trains model for one epoch.
    """
    model.train()
    running_loss = 0.0
    correct_top1 = 0
    correct_top5 = 0
    total_samples = 0

    pbar = tqdm(loader, desc="Training Batch", leave=False)
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_samples += batch_size
        running_loss += loss.item() * batch_size

        # Top-1 Accuracy
        _, preds = torch.max(outputs, 1)
        correct_top1 += (preds == labels).sum().item()

        # Top-5 Accuracy
        _, top5_preds = outputs.topk(min(5, outputs.size(1)), dim=1, largest=True, sorted=True)
        correct_top5 += top5_preds.eq(labels.view(-1, 1).expand_as(top5_preds)).sum().item()

        current_acc = (correct_top1 / total_samples) * 100.0
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{current_acc:.1f}%"})

    epoch_loss = running_loss / max(total_samples, 1)
    top1_acc = (correct_top1 / max(total_samples, 1)) * 100.0
    top5_acc = (correct_top5 / max(total_samples, 1)) * 100.0

    return {"loss": epoch_loss, "top1_acc": top1_acc, "top5_acc": top5_acc}


def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Dict[str, float]:
    """
    Evaluates model on validation/test set.
    """
    model.eval()
    running_loss = 0.0
    correct_top1 = 0
    correct_top5 = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            batch_size = labels.size(0)
            total_samples += batch_size
            running_loss += loss.item() * batch_size

            # Top-1 Accuracy
            _, preds = torch.max(outputs, 1)
            correct_top1 += (preds == labels).sum().item()

            # Top-5 Accuracy
            _, top5_preds = outputs.topk(min(5, outputs.size(1)), dim=1, largest=True, sorted=True)
            correct_top5 += top5_preds.eq(labels.view(-1, 1).expand_as(top5_preds)).sum().item()

    epoch_loss = running_loss / max(total_samples, 1)
    top1_acc = (correct_top1 / max(total_samples, 1)) * 100.0
    top5_acc = (correct_top5 / max(total_samples, 1)) * 100.0

    return {"loss": epoch_loss, "top1_acc": top1_acc, "top5_acc": top5_acc}


def train_model(
    dataset_dir: str = "RGB_224x224",
    epochs: int = 5,
    batch_size: int = 32,
    lr: float = 1e-3,
    architecture: str = "resnet18",
    samples_per_class: Optional[int] = None,
    save_dir: str = "models",
    freeze_backbone: bool = True
) -> Dict[str, Any]:
    """
    Full training routine for the crop classifier.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--> Using compute device: {device}")

    os.makedirs(save_dir, exist_ok=True)
    best_weights_path = os.path.join(save_dir, f"crop_{architecture}_best.pth")
    last_weights_path = os.path.join(save_dir, f"crop_{architecture}_latest.pth")
    history_path = os.path.join(save_dir, "training_history.json")

    print(f"--> Preparing DataLoaders (samples_per_class={samples_per_class})...")
    train_loader, val_loader, test_loader, class_names, class_to_idx = get_dataloaders(
        dataset_dir=dataset_dir,
        batch_size=batch_size,
        num_workers=0,
        samples_per_class=samples_per_class
    )
    num_classes = len(class_names)
    print(f"--> Found {num_classes} classes. Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Save class names JSON
    with open(os.path.join(save_dir, "class_names.json"), "w") as f:
        json.dump({
            "classes": class_names,
            "class_to_idx": class_to_idx,
            "idx_to_class": {idx: cls for idx, cls in enumerate(class_names)}
        }, f, indent=2)

    print(f"--> Initializing model ({architecture}) with pre-trained weights...")
    model = build_model(
        num_classes=num_classes,
        architecture=architecture,
        pretrained=True,
        freeze_backbone=freeze_backbone
    )
    model.to(device)

    # Train only trainable parameters (e.g. classification head)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable_params, lr=lr)
    criterion = nn.CrossEntropyLoss()

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "train_top1": [],
        "train_top5": [],
        "val_loss": [],
        "val_top1": [],
        "val_top5": []
    }

    best_val_acc = -1.0
    start_time = time.time()

    print("\n" + "="*60)
    print(f" Starting Training: {epochs} Epochs | Batch Size: {batch_size} | LR: {lr}")
    print("="*60)

    for epoch in range(1, epochs + 1):
        epoch_t0 = time.time()

        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(round(train_metrics["loss"], 4))
        history["train_top1"].append(round(train_metrics["top1_acc"], 2))
        history["train_top5"].append(round(train_metrics["top5_acc"], 2))
        history["val_loss"].append(round(val_metrics["loss"], 4))
        history["val_top1"].append(round(val_metrics["top1_acc"], 2))
        history["val_top5"].append(round(val_metrics["top5_acc"], 2))

        duration = time.time() - epoch_t0
        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] ({duration:.1f}s) | "
            f"Train Loss: {train_metrics['loss']:.4f} | Train Top-1: {train_metrics['top1_acc']:.2f}% | "
            f"Val Loss: {val_metrics['loss']:.4f} | Val Top-1: {val_metrics['top1_acc']:.2f}% | "
            f"Val Top-5: {val_metrics['top5_acc']:.2f}%"
        )

        # Save checkpoint
        save_checkpoint(
            model=model,
            save_path=last_weights_path,
            class_names=class_names,
            architecture=architecture,
            epoch=epoch,
            val_acc=val_metrics["top1_acc"],
            history=history
        )

        if val_metrics["top1_acc"] > best_val_acc:
            best_val_acc = val_metrics["top1_acc"]
            save_checkpoint(
                model=model,
                save_path=best_weights_path,
                class_names=class_names,
                architecture=architecture,
                epoch=epoch,
                val_acc=val_metrics["top1_acc"],
                history=history
            )
            print(f"  --> Saved new best checkpoint! (Val Top-1: {best_val_acc:.2f}%)")

    total_time = time.time() - start_time
    print("="*60)
    print(f" Training complete in {total_time/60:.2f} minutes. Best Val Top-1: {best_val_acc:.2f}%")
    print(f" Best model weights saved to: {best_weights_path}")
    print("="*60)

    # Save history json
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    return {
        "best_val_acc": best_val_acc,
        "best_weights_path": best_weights_path,
        "history": history,
        "class_names": class_names
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Crop Image Classification Model")
    parser.add_argument("--dataset-dir", type=str, default="RGB_224x224", help="Dataset root directory")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs to train")
    parser.add_argument("--batch-size", type=int, default=32, help="DataLoader batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--architecture", type=str, default="resnet18", choices=["resnet18", "resnet34", "mobilenet_v2"])
    parser.add_argument("--samples-per-class", type=int, default=None, help="Samples per class for quick training")
    parser.add_argument("--save-dir", type=str, default="models", help="Directory to save model weights")

    args = parser.parse_args()
    train_model(
        dataset_dir=args.dataset_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        architecture=args.architecture,
        samples_per_class=args.samples_per_class,
        save_dir=args.save_dir
    )
