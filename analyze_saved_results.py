"""Rank saved edit pairs with correctly paired image metrics.

This is read-only with respect to experiment outputs. It writes a CSV and
contact sheet under analysis_outputs/ so incomplete PPT metrics can be filled
without rerunning the editors.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


PROMPTS = {
    "sunglasses": "Add sunglasses",
    "green_hair": "Make hair color green",
}


def load_rgb(path: Path, size: tuple[int, int] | None = None) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    if size and image.size != size:
        image = image.resize(size, Image.Resampling.LANCZOS)
    return np.asarray(image, dtype=np.float32) / 255.0


def metrics(a_path: Path, b_path: Path) -> dict[str, float]:
    a_image = Image.open(a_path).convert("RGB")
    b = load_rgb(b_path, a_image.size)
    a = np.asarray(a_image, dtype=np.float32) / 255.0
    mse = float(np.mean((a - b) ** 2))
    return {
        "ssim": float(structural_similarity(a, b, channel_axis=2, data_range=1.0)),
        "psnr": float(peak_signal_noise_ratio(a, b, data_range=1.0)),
        "mse": mse,
        "l2": float(np.sqrt(mse)),
    }


def discover(root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for face_dir in sorted((root / "output_save").glob("image_[23]")):
        face_id = face_dir.name
        original = root / "new_images" / f"{face_id}.png"
        for model in ("flux", "instruct"):
            clean_root = root / "edit_output" / model
            output_root = face_dir / f"{model}_output"
            for run_dir in sorted(output_root.glob("*_*")):
                if not run_dir.is_dir():
                    continue
                warp, iterations = run_dir.name.rsplit("_", 1)
                perturbed = face_dir / "results" / run_dir.name / "vae_latent_out.png"
                if not perturbed.exists():
                    continue
                input_metrics = metrics(original, perturbed)
                for prompt_slug, prompt in PROMPTS.items():
                    clean = clean_root / f"{face_id}_{prompt_slug}.png"
                    edited = run_dir / f"output_{prompt_slug}.png"
                    if not (clean.exists() and edited.exists()):
                        continue
                    output_metrics = metrics(clean, edited)
                    rows.append({
                        "face_id": face_id,
                        "model": model,
                        "prompt_slug": prompt_slug,
                        "prompt": prompt,
                        "warp": warp,
                        "iterations": int(iterations),
                        **{f"input_{k}": v for k, v in input_metrics.items()},
                        **{f"output_{k}": v for k, v in output_metrics.items()},
                        "original_path": original.as_posix(),
                        "perturbed_path": perturbed.as_posix(),
                        "clean_edit_path": clean.as_posix(),
                        "perturbed_edit_path": edited.as_posix(),
                    })
    return rows


def make_sheet(rows: list[dict[str, object]], output: Path, max_rows: int = 16) -> None:
    chosen = sorted(
        (r for r in rows if float(r["input_ssim"]) >= 0.88),
        key=lambda r: (float(r["output_ssim"]), -float(r["input_ssim"])),
    )[:max_rows]
    tile_w, tile_h, header_h = 230, 230, 62
    canvas = Image.new("RGB", (tile_w * 4, (tile_h + header_h) * len(chosen)), "white")
    draw = ImageDraw.Draw(canvas)
    for row_index, row in enumerate(chosen):
        y = row_index * (tile_h + header_h)
        title = (
            f"{row['face_id']} | {row['model']} | {row['prompt_slug']} | "
            f"{row['warp']}_{row['iterations']}  input SSIM={float(row['input_ssim']):.3f}  "
            f"output SSIM={float(row['output_ssim']):.3f}"
        )
        draw.text((8, y + 5), title, fill="black")
        labels = ("Original", "Perturbed", "Clean edit", "Perturbed edit")
        paths = (
            row["original_path"], row["perturbed_path"],
            row["clean_edit_path"], row["perturbed_edit_path"],
        )
        for col, (label, path) in enumerate(zip(labels, paths)):
            draw.text((col * tile_w + 8, y + 29), label, fill="black")
            image = Image.open(str(path)).convert("RGB")
            image.thumbnail((tile_w - 8, tile_h - 8), Image.Resampling.LANCZOS)
            x = col * tile_w + (tile_w - image.width) // 2
            canvas.paste(image, (x, y + header_h + (tile_h - image.height) // 2))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=92)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output-dir", type=Path, default=Path("analysis_outputs"))
    args = parser.parse_args()
    root = args.root.resolve()
    output_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    rows = discover(root)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "saved_results_metrics.csv"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    make_sheet(rows, output_dir / "top_saved_results_sheet.jpg")
    print(f"Wrote {len(rows)} paired results to {csv_path}")


if __name__ == "__main__":
    main()
