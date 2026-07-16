"""Add exact ArcFace iResNet-100 identity metrics to a FLUX sweep CSV.

The output is a discovery aid only. Low edited-output similarity never becomes
an acceptance label automatically; every candidate still requires inspection
of the canonical four-image strip.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

from evaluate_parth_save_arcface import image_tensor
from models.target_model import FaceRecognitionModel


PATH_FIELDS = (
    "original_path",
    "perturbed_path",
    "clean_edit_path",
    "perturbed_edit_path",
)


def local_path(root: Path, value: str) -> Path:
    normalized = value.replace("\\", "/")
    marker = "/home/interns/Desktop/amit_code/"
    if marker in normalized:
        normalized = normalized.split(marker, 1)[1]
    path = Path(normalized)
    return path if path.is_absolute() else root / path


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.cosine_similarity(a, b, dim=1).item())


def make_review_sheet(root: Path, rows: list[dict[str, object]], output: Path) -> None:
    eligible = [
        row
        for row in rows
        if float(row.get("input_ssim", 0.0)) >= 0.88
        and float(row["arcface_original_vs_perturbed"]) >= 0.80
        and float(row["arcface_original_vs_clean_edit"]) >= 0.60
    ][:40]
    strips: list[Image.Image] = []
    for row in eligible:
        path = local_path(root, str(row["strip_path"]))
        if path.exists():
            strips.append(Image.open(path).convert("RGB"))
    if not strips:
        return
    width = max(image.width for image in strips)
    height = sum(image.height for image in strips)
    sheet = Image.new("RGB", (width, height), "white")
    y = 0
    for strip in strips:
        sheet.paste(strip, (0, y))
        y += strip.height
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, quality=90, optimize=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("/home/interns/Desktop/face4/models/arcface/iresnet100.pth"),
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    root = args.root.resolve()
    input_csv = args.input_csv if args.input_csv.is_absolute() else root / args.input_csv
    output_csv = args.output_csv if args.output_csv.is_absolute() else root / args.output_csv
    checkpoint = args.checkpoint
    if not checkpoint.is_absolute():
        checkpoint = (root / checkpoint).resolve()
    device = torch.device(args.device)
    model = FaceRecognitionModel(
        model_name="arcface", model_path=str(checkpoint), device=str(device)
    ).eval()

    with input_csv.open(newline="", encoding="utf-8") as handle:
        source_rows = list(csv.DictReader(handle))

    cache: dict[Path, torch.Tensor] = {}

    def embedding(path: Path) -> torch.Tensor:
        path = path.resolve()
        if path not in cache:
            if not path.exists():
                raise FileNotFoundError(path)
            with torch.inference_mode():
                cache[path] = model(image_tensor(path, device)).detach().cpu()
        return cache[path]

    rows: list[dict[str, object]] = []
    for index, row in enumerate(source_rows, start=1):
        try:
            paths = {name: local_path(root, row[name]) for name in PATH_FIELDS}
            vectors = {name: embedding(path) for name, path in paths.items()}
            input_similarity = cosine(
                vectors["original_path"], vectors["perturbed_path"]
            )
            edit_similarity = cosine(
                vectors["clean_edit_path"], vectors["perturbed_edit_path"]
            )
            clean_identity = cosine(
                vectors["original_path"], vectors["clean_edit_path"]
            )
            perturbed_identity = cosine(
                vectors["perturbed_path"], vectors["perturbed_edit_path"]
            )
            scored = {
                **row,
                "arcface_model": "ArcFace iResNet-100",
                "arcface_original_vs_perturbed": input_similarity,
                "arcface_clean_edit_vs_perturbed_edit": edit_similarity,
                "arcface_original_vs_clean_edit": clean_identity,
                "arcface_perturbed_vs_perturbed_edit": perturbed_identity,
                "arcface_edit_identity_drop": clean_identity - perturbed_identity,
                "arcface_review_priority": (
                    input_similarity
                    + clean_identity
                    - edit_similarity
                    - perturbed_identity
                ),
                "arcface_status": "scored",
            }
        except Exception as error:  # keep long screens resumable and auditable
            scored = {
                **row,
                "arcface_model": "ArcFace iResNet-100",
                "arcface_original_vs_perturbed": "",
                "arcface_clean_edit_vs_perturbed_edit": "",
                "arcface_original_vs_clean_edit": "",
                "arcface_perturbed_vs_perturbed_edit": "",
                "arcface_edit_identity_drop": "",
                "arcface_review_priority": "",
                "arcface_status": f"failed: {error!r}",
            }
        rows.append(scored)
        if index % 25 == 0 or index == len(source_rows):
            write_csv(output_csv, rows)
            print(f"ArcFace scored {index}/{len(source_rows)} rows")

    successful = [row for row in rows if row["arcface_status"] == "scored"]
    ranked = sorted(
        successful,
        key=lambda row: float(row["arcface_review_priority"]),
        reverse=True,
    )
    ranked_path = output_csv.with_name(output_csv.stem + "_ranked.csv")
    write_csv(ranked_path, ranked)
    sheet_path = output_csv.with_name(output_csv.stem + "_ranked_sheet.jpg")
    make_review_sheet(root, ranked, sheet_path)
    print(
        f"Wrote {output_csv}, {ranked_path}, and {sheet_path}; "
        f"embeddings cached: {len(cache)}"
    )


if __name__ == "__main__":
    main()
