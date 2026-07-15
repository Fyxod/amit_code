"""Refine the most sensitive prompt/candidate pairs across editor settings."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from PIL import Image

from instruct import InstructBackend, InstructSettings
from targeted_prompt_sweep import make_top_sheet, pair_metrics, save_strip, slug


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def select_base_cases(rows: list[dict[str, str]], limit: int, min_input_ssim: float) -> list[dict[str, str]]:
    eligible = [row for row in rows if float(row["input_ssim"]) >= min_input_ssim]
    eligible.sort(key=lambda row: (float(row["output_ssim"]), -float(row["input_ssim"])))
    selected: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    prompt_counts: dict[str, int] = {}
    for row in eligible:
        key = (row["face_id"], row["prompt"], row["candidate_id"])
        if key in seen or prompt_counts.get(row["prompt"], 0) >= 3:
            continue
        seen.add(key)
        prompt_counts[row["prompt"]] = prompt_counts.get(row["prompt"], 0) + 1
        selected.append(row)
        if len(selected) >= limit:
            break
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--top-cases", type=int, default=8)
    parser.add_argument("--min-input-ssim", type=float, default=0.84)
    parser.add_argument("--image-guidance-scales", type=float, nargs="+", default=[1.0, 1.5, 2.0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[1234, 24001])
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    input_csv = args.input_csv.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    base_cases = select_base_cases(read_rows(input_csv), args.top_cases, args.min_input_ssim)
    if not base_cases:
        raise RuntimeError("No base prompt-sweep cases passed refinement selection")

    settings = InstructSettings(
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        image_guidance_scale=args.image_guidance_scales[0],
        seed=args.seeds[0],
    )
    backend = InstructBackend(torch.device(args.device), settings)
    rows: list[dict[str, object]] = []
    csv_path = output_root / "refined_prompt_metrics.csv"

    for base in base_cases:
        face_id = base["face_id"]
        prompt = base["prompt"]
        original_path = Path(base["original_path"])
        perturbed_path = Path(base["perturbed_path"])
        for image_guidance_scale in args.image_guidance_scales:
            backend.settings.image_guidance_scale = image_guidance_scale
            for seed in args.seeds:
                setting_slug = f"igs_{image_guidance_scale:g}_seed_{seed}"
                clean_path = output_root / face_id / slug(prompt) / setting_slug / "clean_edit.png"
                if not clean_path.exists():
                    clean_path.parent.mkdir(parents=True, exist_ok=True)
                    backend.generate_edit(Image.open(original_path).convert("RGB"), prompt, seed).save(clean_path)
                case_root = clean_path.parent / base["candidate_id"]
                edited_path = case_root / "perturbed_edit.png"
                if not edited_path.exists():
                    case_root.mkdir(parents=True, exist_ok=True)
                    backend.generate_edit(Image.open(perturbed_path).convert("RGB"), prompt, seed).save(edited_path)
                values = pair_metrics(clean_path, edited_path)
                strip_path = case_root / "comparison_strip.jpg"
                save_strip(
                    original_path, perturbed_path, clean_path, edited_path, strip_path,
                    f"{face_id} | {prompt} | IGS={image_guidance_scale:g} seed={seed} | "
                    f"input SSIM={float(base['input_ssim']):.3f} | output SSIM={values['ssim']:.3f}",
                )
                row = {
                    **base,
                    "seed": seed,
                    "steps": args.steps,
                    "guidance_scale": args.guidance_scale,
                    "image_guidance_scale": image_guidance_scale,
                    **{f"output_{key}": value for key, value in values.items()},
                    "clean_edit_path": str(clean_path),
                    "perturbed_edit_path": str(edited_path),
                    "strip_path": str(strip_path),
                }
                rows.append(row)
                with csv_path.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)

    make_top_sheet(rows, output_root / "top_refined_prompt_sheet.jpg", limit=24)
    summary = {
        "num_base_cases": len(base_cases),
        "num_refined_pairs": len(rows),
        "best_output_ssim": min(float(row["output_ssim"]) for row in rows),
        "image_guidance_scales": args.image_guidance_scales,
        "seeds": args.seeds,
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_root / "DONE.json").write_text(json.dumps({"status": "complete", **summary}, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
