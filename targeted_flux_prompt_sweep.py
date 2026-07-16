"""Deterministic FLUX.2 Klein prompt screen over saved geometric perturbations.

The FLUX pipeline is loaded once. Every clean/perturbed pair uses identical
prompt, seed, steps, and guidance. Results are resumable and ranked only as a
discovery aid; visual inspection remains mandatory before curation.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from PIL import Image

from flux import FluxBackend, FluxSettings
from targeted_prompt_sweep import (
    discover_candidates,
    make_top_sheet,
    original_for,
    pair_metrics,
    save_strip,
    slug,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--candidate-root", action="append", type=Path, default=[])
    parser.add_argument("--output-root", type=Path, default=Path("targeted_experiments/flux_klein_prompt_sweep"))
    parser.add_argument("--prompts-json", type=Path, required=True)
    parser.add_argument("--min-input-ssim", type=float, default=0.84)
    parser.add_argument("--max-candidates-per-face", type=int, default=2)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    root = args.root.resolve()
    output_root = args.output_root if args.output_root.is_absolute() else root / args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    prompts_path = args.prompts_json if args.prompts_json.is_absolute() else root / args.prompts_json
    prompts = json.loads(prompts_path.read_text(encoding="utf-8"))
    candidate_roots = args.candidate_root or [Path("output_save")]
    candidates = discover_candidates(root, candidate_roots, args.min_input_ssim, args.max_candidates_per_face)
    if not candidates:
        raise RuntimeError("No perturbation candidates passed discovery/SSIM filtering")

    settings = FluxSettings(
        diffusion_steps=args.steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
    )
    backend = FluxBackend(torch.device(args.device), settings)
    rows: list[dict[str, object]] = []
    csv_path = output_root / "flux_prompt_sweep_metrics.csv"

    face_ids = sorted({str(candidate["face_id"]) for candidate in candidates})
    for face_id in face_ids:
        original = original_for(root, face_id)
        face_candidates = [item for item in candidates if item["face_id"] == face_id]
        for prompt in prompts:
            prompt_slug = slug(prompt)
            clean_path = output_root / face_id / prompt_slug / "clean_edit.png"
            if not clean_path.exists():
                clean_path.parent.mkdir(parents=True, exist_ok=True)
                backend.generate_edit(Image.open(original).convert("RGB"), prompt, args.seed).save(clean_path)
            for candidate in face_candidates:
                candidate_path = Path(str(candidate["candidate_path"]))
                case_root = output_root / face_id / prompt_slug / str(candidate["candidate_id"])
                edited_path = case_root / "perturbed_edit.png"
                if not edited_path.exists():
                    case_root.mkdir(parents=True, exist_ok=True)
                    backend.generate_edit(Image.open(candidate_path).convert("RGB"), prompt, args.seed).save(edited_path)
                output_values = pair_metrics(clean_path, edited_path)
                strip_path = case_root / "comparison_strip.jpg"
                save_strip(
                    original,
                    candidate_path,
                    clean_path,
                    edited_path,
                    strip_path,
                    f"FLUX.2 Klein 4B | {face_id} | {prompt} | steps={args.steps} "
                    f"guidance={args.guidance_scale:g} seed={args.seed} | "
                    f"input SSIM={float(candidate['input_ssim']):.3f} | output SSIM={output_values['ssim']:.3f}",
                )
                row = {
                    **candidate,
                    "model": settings.model_id,
                    "prompt": prompt,
                    "seed": args.seed,
                    "steps": args.steps,
                    "guidance_scale": args.guidance_scale,
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

    make_top_sheet(rows, output_root / "top_flux_prompt_sensitivity_sheet.jpg", limit=30)
    summary = {
        "status": "complete",
        "model": settings.model_id,
        "num_faces": len(face_ids),
        "num_candidates": len(candidates),
        "num_prompts": len(prompts),
        "num_pairs": len(rows),
        "best_output_ssim": min(float(row["output_ssim"]) for row in rows),
        "steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "seed": args.seed,
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_root / "DONE.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
