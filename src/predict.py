"""
Inference and prediction module for Crop Image Classification.
"""

import os
import json
import argparse
from typing import List, Dict, Any, Union
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

from src.model import load_model_from_checkpoint
from src.dataset import IMAGENET_MEAN, IMAGENET_STD


class CropPredictor:
    """
    Production-ready inference pipeline for classifying crop plant images.
    """

    def __init__(
        self,
        checkpoint_path: str = "models/crop_resnet18_best.pth",
        device: str = None
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Model checkpoint not found at: {checkpoint_path}")

        self.model, self.class_names, self.meta = load_model_from_checkpoint(
            checkpoint_path, device=self.device
        )
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])

    def predict(
        self,
        image_input: Union[str, Image.Image],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Predicts the top-K crop classes for a given image.

        Args:
            image_input: File path (str) or PIL Image
            top_k: Number of highest probability classes to return (default: 5)

        Returns:
            List of dicts containing class_name, clean_name, probability, and rank.
        """
        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                raise FileNotFoundError(f"Image not found: {image_input}")
            img = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            img = image_input.convert("RGB")
        else:
            raise ValueError("Input must be a valid file path or PIL Image object.")

        tensor = self.transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probabilities = F.softmax(logits, dim=1).squeeze(0)

        top_k = min(top_k, len(self.class_names))
        top_probs, top_indices = torch.topk(probabilities, k=top_k)

        results = []
        for rank, (prob, idx) in enumerate(zip(top_probs, top_indices), start=1):
            raw_name = self.class_names[idx.item()]
            clean_name = raw_name.replace(" plant", "").strip()
            results.append({
                "rank": rank,
                "class_name": raw_name,
                "clean_name": clean_name,
                "probability": float(prob.item()),
                "confidence_pct": round(float(prob.item()) * 100.0, 2)
            })

        return results


def main():
    parser = argparse.ArgumentParser(description="Predict Crop Plant Class from Image")
    parser.add_argument("--image", type=str, required=True, help="Path to crop image")
    parser.add_argument("--checkpoint", type=str, default="models/crop_resnet18_best.pth", help="Checkpoint path")
    parser.add_argument("--top-k", type=int, default=5, help="Top-K predictions to display")

    args = parser.parse_args()
    predictor = CropPredictor(checkpoint_path=args.checkpoint)
    predictions = predictor.predict(args.image, top_k=args.top_k)

    print("\n" + "="*50)
    print(f" Predictions for: {os.path.basename(args.image)}")
    print("="*50)
    for p in predictions:
        bar = "=" * int(p["confidence_pct"] / 5)
        print(f"#{p['rank']:02d} | {p['clean_name']:<30} | {p['confidence_pct']:>6.2f}% | [{bar}]")
    print("="*50)


if __name__ == "__main__":
    main()
