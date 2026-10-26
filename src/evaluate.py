"""
Evaluation script for Crop Image Classification model on test dataset.
"""

import os
import json
import argparse
from typing import Dict, Any, List
import torch
import torch.nn as nn
from tqdm import tqdm

from src.dataset import get_dataloaders
from src.model import load_model_from_checkpoint


def evaluate_test_set(
    checkpoint_path: str = "models/crop_resnet18_best.pth",
    dataset_dir: str = "RGB_224x224",
    batch_size: int = 32,
    samples_per_class: int = None,
    save_results: bool = True
) -> Dict[str, Any]:
    """
    Evaluates the model on test split and generates detailed metrics.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--> Using compute device: {device}")
    print(f"--> Loading model from checkpoint: {checkpoint_path}")

    model, class_names, checkpoint = load_model_from_checkpoint(checkpoint_path, device=device)
    model.eval()

    print(f"--> Preparing test dataset from {dataset_dir}/test...")
    _, _, test_loader, _, _ = get_dataloaders(
        dataset_dir=dataset_dir,
        batch_size=batch_size,
        num_workers=0,
        samples_per_class=samples_per_class
    )

    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct_top1 = 0
    correct_top5 = 0
    total_samples = 0

    num_classes = len(class_names)
    class_correct = [0] * num_classes
    class_total = [0] * num_classes

    print(f"--> Evaluating {len(test_loader)} test batches...")
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating Test Set"):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            batch_size_cur = labels.size(0)
            total_samples += batch_size_cur
            total_loss += loss.item() * batch_size_cur

            # Top 1
            _, preds = torch.max(outputs, 1)
            correct_top1 += (preds == labels).sum().item()

            # Top 5
            _, top5_preds = outputs.topk(min(5, outputs.size(1)), dim=1, largest=True, sorted=True)
            correct_top5 += top5_preds.eq(labels.view(-1, 1).expand_as(top5_preds)).sum().item()

            # Per-class counts
            for i in range(batch_size_cur):
                label_idx = labels[i].item()
                class_total[label_idx] += 1
                if preds[i].item() == label_idx:
                    class_correct[label_idx] += 1

    top1_acc = (correct_top1 / max(total_samples, 1)) * 100.0
    top5_acc = (correct_top5 / max(total_samples, 1)) * 100.0
    avg_loss = total_loss / max(total_samples, 1)

    # Per-class metrics
    per_class_results = []
    for idx, name in enumerate(class_names):
        total = class_total[idx]
        correct = class_correct[idx]
        acc = (correct / total * 100.0) if total > 0 else 0.0
        per_class_results.append({
            "class_name": name,
            "total": total,
            "correct": correct,
            "accuracy_pct": round(acc, 2)
        })

    # Sort classes by accuracy
    per_class_results.sort(key=lambda x: x["accuracy_pct"], reverse=True)

    results = {
        "architecture": checkpoint.get("architecture", "resnet18"),
        "total_test_samples": total_samples,
        "test_loss": round(avg_loss, 4),
        "top1_accuracy_pct": round(top1_acc, 2),
        "top5_accuracy_pct": round(top5_acc, 2),
        "top_performing_classes": per_class_results[:5],
        "lowest_performing_classes": per_class_results[-5:],
        "per_class": per_class_results
    }

    print("\n" + "="*50)
    print("             TEST EVALUATION RESULTS")
    print("="*50)
    print(f"Total Test Images: {total_samples}")
    print(f"Test Loss:         {avg_loss:.4f}")
    print(f"Top-1 Accuracy:    {top1_acc:.2f}%")
    print(f"Top-5 Accuracy:    {top5_acc:.2f}%")
    print("="*50)

    if save_results:
        out_path = os.path.join(os.path.dirname(checkpoint_path), "evaluation_results.json")
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"--> Saved evaluation metrics to: {out_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Crop Classification Model")
    parser.add_argument("--checkpoint", type=str, default="models/crop_resnet18_best.pth", help="Model checkpoint path")
    parser.add_argument("--dataset-dir", type=str, default="RGB_224x224", help="Dataset directory")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--samples-per-class", type=int, default=None, help="Optional sample limit per class")

    args = parser.parse_args()
    evaluate_test_set(
        checkpoint_path=args.checkpoint,
        dataset_dir=args.dataset_dir,
        batch_size=args.batch_size,
        samples_per_class=args.samples_per_class
    )
