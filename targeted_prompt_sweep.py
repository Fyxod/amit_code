"""Deterministic InstructPix2Pix prompt screen over saved perturbations.

The script loads the editor once, generates a paired clean/perturbed edit with
identical settings and seed, writes image metrics incrementally, and creates a
ranked contact sheet for visual inspection. It never modifies legacy outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from instruct import InstructBackend, InstructSettings


DEFAULT_PROMPTS = [
    "Change hair color to green",
    "Change hair color to bright blue",
    "Add black sunglasses",
    "Add over-ear headphones",
    "Add a red scarf",
    "Add a black jacket",
    "Add a grey hoodie",
    "Add red lipstick",
    "Make the person smile slightly",
    "Add a small beard",
]


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def load_array(path: Path, size: tuple[int, int] | None = None) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    if size and image.size != size:
        image = image.resize(size, Image.Resampling.LANCZOS)
    return np.asarray(image, dtype=np.float32) / 255.0


def pair_metrics(a_path: Path, b_path: Path) -> dict[str, float]:
    a_image = Image.open(a_path).convert("RGB")
    a = np.asarray(a_image, dtype=np.float32) / 255.0
    b = load_array(b_path, a_image.size)
    mse = float(np.mean((a - b) ** 2))
    return {
        "ssim": float(structural_similarity(a, b, channel_axis=2, data_range=1.0)),
        "psnr": float(peak_signal_noise_ratio(a, b, data_range=1.0)),
        "mse": mse,
        "l2": float(np.sqrt(mse)),
    }


def infer_face_id(path: Path) -> str | None:
    for part in path.parts:
        match = re.fullmatch(r"image_([123])", part)
        if match:
            return f"image_{match.group(1)}"
    return None


def original_for(root: Path, face_id: str) -> Path:
    if face_id == "image_1":
        choices = [root / "new_images" / "original.jpg", root / "output_save" / "image_1" / "original.jpg"]
    else:
        choices = [root / "new_images" / f"{face_id}.png"]
    for path in choices:
        if path.exists():
            return path
    raise FileNotFoundError(f"No original image found for {face_id}: {choices}")


def discover_candidates(root: Path, candidate_roots: list[Path], min_ssim: float, max_per_face: int) -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    seen: set[Path] = set()
    for candidate_root in candidate_roots:
        resolved = candidate_root if candidate_root.is_absolute() else root / candidate_root
        for path in resolved.rglob("vae_latent_out.png"):
            path = path.resolve()
            if path in seen:
                continue
            seen.add(path)
            face_id = infer_face_id(path)
            if not face_id:
                continue
            original = original_for(root, face_id)
            input_values = pair_metrics(original, path)
            if input_values["ssim"] < min_ssim:
                continue
            candidates.append({
                "face_id": face_id,
                "candidate_path": path,
                "candidate_id": slug(str(path.relative_to(root)))[:120],
                **{f"input_{key}": value for key, value in input_values.items()},
            })

    selected: list[dict[str, object]] = []
    for face_id in sorted({str(item["face_id"]) for item in candidates}):
        group = [item for item in candidates if item["face_id"] == face_id]
        # Favor high preservation, but retain separation across perturbation
        # strengths by selecting evenly from the sorted list.
        group.sort(key=lambda item: float(item["input_ssim"]), reverse=True)
        if len(group) <= max_per_face:
            selected.extend(group)
            continue
        indices = np.linspace(0, len(group) - 1, max_per_face).round().astype(int)
        selected.extend(group[index] for index in dict.fromkeys(indices.tolist()))
    return selected


def save_strip(original: Path, perturbed: Path, clean: Path, edited: Path, output: Path, title: str) -> None:
    tile_w, tile_h, header_h = 256, 256, 54
    canvas = Image.new("RGB", (tile_w * 4, tile_h + header_h), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 6), title, fill="black")
    for index, (label, path) in enumerate(zip(
        ("Original", "Perturbed", "Original edit", "Perturbed edit"),
        (original, perturbed, clean, edited),
    )):
        draw.text((index * tile_w + 8, 28), label, fill="black")
        image = Image.open(path).convert("RGB")
        image.thumbnail((tile_w - 8, tile_h - 8), Image.Resampling.LANCZOS)
        canvas.paste(image, (index * tile_w + (tile_w - image.width) // 2, header_h + (tile_h - image.height) // 2))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=94)


def make_top_sheet(rows: list[dict[str, object]], output: Path, limit: int = 20) -> None:
    chosen = sorted(rows, key=lambda row: (float(row["output_ssim"]), -float(row["input_ssim"])))[:limit]
    if not chosen:
        return
    strips = [Image.open(str(row["strip_path"])).convert("RGB") for row in chosen]
    width = max(image.width for image in strips)
    height = sum(image.height for image in strips)
    sheet = Image.new("RGB", (width, height), "white")
    y = 0
    for image in strips:
        sheet.paste(image, (0, y))
        y += image.height
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, quality=92)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--candidate-root", action="append", type=Path, default=[])
    parser.add_argument("--output-root", type=Path, default=Path("targeted_experiments/prompt_sweep"))
    parser.add_argument("--prompts-json", type=Path)
    parser.add_argument("--min-input-ssim", type=float, default=0.88)
    parser.add_argument("--max-candidates-per-face", type=int, default=5)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--image-guidance-scale", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    root = args.root.resolve()
    output_root = args.output_root if args.output_root.is_absolute() else root / args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    candidate_roots = args.candidate_root or [Path("output_save")]
    prompts = DEFAULT_PROMPTS
    if args.prompts_json:
        prompts = json.loads(args.prompts_json.read_text(encoding="utf-8"))

    candidates = discover_candidates(root, candidate_roots, args.min_input_ssim, args.max_candidates_per_face)
    if not candidates:
        raise RuntimeError("No perturbation candidates passed discovery/SSIM filtering")

    settings = InstructSettings(
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        image_guidance_scale=args.image_guidance_scale,
        seed=args.seed,
    )
    backend = InstructBackend(torch.device(args.device), settings)
    rows: list[dict[str, object]] = []
    csv_path = output_root / "prompt_sweep_metrics.csv"
    for face_id in sorted({str(candidate["face_id"]) for candidate in candidates}):
        original = original_for(root, face_id)
        for prompt in prompts:
            prompt_slug = slug(prompt)
            clean_path = output_root / face_id / prompt_slug / "clean_edit.png"
            if not clean_path.exists():
                clean = backend.generate_edit(Image.open(original).convert("RGB"), prompt, args.seed)
                clean_path.parent.mkdir(parents=True, exist_ok=True)
                clean.save(clean_path)
            for candidate in (item for item in candidates if item["face_id"] == face_id):
                candidate_path = Path(str(candidate["candidate_path"]))
                case_root = output_root / face_id / prompt_slug / str(candidate["candidate_id"])
                edited_path = case_root / "perturbed_edit.png"
                if not edited_path.exists():
                    edited = backend.generate_edit(Image.open(candidate_path).convert("RGB"), prompt, args.seed)
                    case_root.mkdir(parents=True, exist_ok=True)
                    edited.save(edited_path)
                output_values = pair_metrics(clean_path, edited_path)
                strip_path = case_root / "comparison_strip.jpg"
                title = (
                    f"{face_id} | {prompt} | input SSIM={float(candidate['input_ssim']):.3f} | "
                    f"output SSIM={output_values['ssim']:.3f}"
                )
                save_strip(original, candidate_path, clean_path, edited_path, strip_path, title)
                row = {
                    "face_id": face_id,
                    "prompt": prompt,
                    "prompt_slug": prompt_slug,
                    "candidate_id": candidate["candidate_id"],
                    "seed": args.seed,
                    "steps": args.steps,
                    "guidance_scale": args.guidance_scale,
                    "image_guidance_scale": args.image_guidance_scale,
                    **{key: candidate[key] for key in ("input_ssim", "input_psnr", "input_mse", "input_l2")},
                    **{f"output_{key}": value for key, value in output_values.items()},
                    "original_path": str(original),
                    "perturbed_path": str(candidate_path),
                    "clean_edit_path": str(clean_path),
                    "perturbed_edit_path": str(edited_path),
                    "strip_path": str(strip_path),
                }
                rows.append(row)
                with csv_path.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)

    make_top_sheet(rows, output_root / "top_prompt_sensitivity_sheet.jpg")
    summary = {
        "num_candidates": len(candidates),
        "num_prompts": len(prompts),
        "num_pairs": len(rows),
        "best_output_ssim": min(float(row["output_ssim"]) for row in rows),
        "settings": vars(args) | {"root": str(root), "output_root": str(output_root)},
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    (output_root / "DONE.json").write_text(json.dumps({"status": "complete", **summary}, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
