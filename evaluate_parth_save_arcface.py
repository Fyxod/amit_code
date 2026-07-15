"""Add exact ArcFace iResNet-100 cosine metrics to saved result folders.

The script reads every ``parth_save/*/metrics.json`` folder containing the
four canonical images and updates the JSON in place.  It uses the same frozen
ArcFace checkpoint and preprocessing path as the corrected optimization.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from models.target_model import FaceRecognitionModel


IMAGE_NAMES = {
    "original": "original.png",
    "perturbed": "perturbed.png",
    "original_edit": "original_edit.png",
    "perturbed_edit": "perturbed_edit.png",
}


def image_tensor(path: Path, device: torch.device) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device)


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.cosine_similarity(a, b, dim=1).item())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("/home/interns/Desktop/face4/models/arcface/iresnet100.pth"),
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    root = args.root.resolve()
    save_root = root / "parth_save"
    device = torch.device(args.device)
    model = FaceRecognitionModel(
        model_name="arcface",
        model_path=str(args.checkpoint.resolve()),
        device=str(device),
    ).eval()

    updated = []
    for metrics_path in sorted(save_root.glob("*/metrics.json")):
        case_root = metrics_path.parent
        paths = {key: case_root / value for key, value in IMAGE_NAMES.items()}
        missing = [str(path) for path in paths.values() if not path.exists()]
        if missing:
            print(f"Skipping {case_root.name}; missing: {missing}")
            continue

        with torch.inference_mode():
            embeddings = {
                key: model(image_tensor(path, device))
                for key, path in paths.items()
            }
        arcface = {
            "original_vs_perturbed": cosine(embeddings["original"], embeddings["perturbed"]),
            "original_edit_vs_perturbed_edit": cosine(
                embeddings["original_edit"], embeddings["perturbed_edit"]
            ),
            "original_vs_original_edit": cosine(
                embeddings["original"], embeddings["original_edit"]
            ),
            "perturbed_vs_perturbed_edit": cosine(
                embeddings["perturbed"], embeddings["perturbed_edit"]
            ),
        }
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        metrics["arcface_model"] = "ArcFace iResNet-100"
        metrics["arcface_cosine_similarity"] = arcface
        metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
        updated.append({"case": case_root.name, **arcface})
        print(json.dumps(updated[-1], indent=2))

    summary_path = save_root / "arcface_metrics_summary.json"
    summary_path.write_text(json.dumps(updated, indent=2) + "\n", encoding="utf-8")
    print(f"Updated {len(updated)} saved cases; summary: {summary_path}")


if __name__ == "__main__":
    main()
