#!/usr/bin/env python
"""Replay fixed attribution-stage inputs across InstructPix2Pix seeds.

The optimization is not rerun.  This script answers whether an observed stage
effect is tied to one diffusion seed or persists when the exact same saved
input is edited under several deterministic seeds.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
from pathlib import Path

import torch
from PIL import Image

from instruct import InstructBackend, InstructSettings
from instruct_stage_attribution import (
    arcface_similarity,
    atomic_save_pil,
    json_dump,
    make_strip,
    pair_metrics_pil,
    pil_to_tensor,
)
from models import FaceRecognitionModel


STAGE_PATHS = {
    "vae_reconstruction": "vae_reconstruction.png",
    "neutral_geometry_on_original": "neutral_geometry_on_original.png",
    "delta_only": "delta_only/learned_delta_only.png",
    "geometry_only_original": "geometry_only_original/learned_geometry_on_original.png",
    "geometry_only_reconstruction": (
        "geometry_only_reconstruction/learned_geometry_on_reconstruction.png"
    ),
    "combined": "combined/learned_combined.png",
    # Combined-only sweeps still save exact component replays inside their
    # combined mode directory. Include those so seed controls can distinguish
    # the latent contribution from the geometry increment.
    "joint_delta_only": "combined/learned_delta_only.png",
    "joint_geometry_on_original": "combined/learned_geometry_on_original.png",
    "joint_geometry_on_reconstruction": (
        "combined/learned_geometry_on_reconstruction.png"
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--prompt")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1234, 24001, 34007])
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--image-guidance-scale", type=float)
    parser.add_argument(
        "--arcface-checkpoint",
        default="/home/interns/Desktop/parth_cleanup/face4/models/arcface/iresnet100.pth",
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    case_root = args.case_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    config = json.loads((case_root / "config_resolved.json").read_text(encoding="utf-8"))
    prompt = args.prompt or config["prompt"]
    image_guidance_scale = (
        args.image_guidance_scale
        if args.image_guidance_scale is not None
        else float(config.get("image_guidance_scale", 1.5))
    )
    # Replay against the exact normalized image used by the attribution run,
    # not the potentially lower-resolution source asset recorded in config.
    original_path = case_root / "original.png"
    if not original_path.exists():
        raise FileNotFoundError(f"Missing normalized baseline image: {original_path}")
    stages = {"original": original_path}
    for name, relative in STAGE_PATHS.items():
        path = case_root / relative
        if path.exists():
            if Image.open(path).size != Image.open(original_path).size:
                raise RuntimeError(
                    f"Stage image size mismatch: {path} vs {original_path}"
                )
            stages[name] = path
    if len(stages) < 2:
        raise RuntimeError(f"No attribution stage images found under {case_root}")

    resolved = {
        "case_root": str(case_root),
        "output_root": str(output_root),
        "prompt": prompt,
        "seeds": args.seeds,
        "steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "image_guidance_scale": image_guidance_scale,
        "original_path": str(original_path),
        "stages": {key: str(value) for key, value in stages.items()},
    }
    json_dump(output_root / "config_resolved.json", resolved)

    settings = InstructSettings(
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        image_guidance_scale=image_guidance_scale,
    )
    device = torch.device(args.device)
    backend = InstructBackend(device, settings)
    original = Image.open(original_path).convert("RGB")
    edit_paths: dict[tuple[int, str], Path] = {}
    for seed in args.seeds:
        seed_root = output_root / f"seed_{seed}"
        seed_root.mkdir(parents=True, exist_ok=True)
        for stage_name, stage_path in stages.items():
            stage_image = Image.open(stage_path).convert("RGB")
            edit_path = seed_root / f"{stage_name}_edit.png"
            atomic_save_pil(backend.generate_edit(stage_image, prompt, seed), edit_path)
            edit_paths[(seed, stage_name)] = edit_path
    del backend
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    identity = FaceRecognitionModel(
        model_name="arcface", model_path=args.arcface_checkpoint, device=args.device
    )
    original_tensor = pil_to_tensor(original, args.device)
    rows: list[dict[str, object]] = []
    for seed in args.seeds:
        clean_path = edit_paths[(seed, "original")]
        clean_edit = Image.open(clean_path).convert("RGB")
        clean_tensor = pil_to_tensor(clean_edit, args.device)
        for stage_name, stage_path in stages.items():
            stage_image = Image.open(stage_path).convert("RGB")
            stage_edit_path = edit_paths[(seed, stage_name)]
            stage_edit = Image.open(stage_edit_path).convert("RGB")
            input_metrics = pair_metrics_pil(original, stage_image)
            output_metrics = pair_metrics_pil(clean_edit, stage_edit)
            stage_tensor = pil_to_tensor(stage_image, args.device)
            stage_edit_tensor = pil_to_tensor(stage_edit, args.device)
            row = {
                "seed": seed,
                "stage": stage_name,
                "stage_path": str(stage_path),
                "edit_path": str(stage_edit_path),
                **{f"input_{key}": value for key, value in input_metrics.items()},
                **{f"output_{key}": value for key, value in output_metrics.items()},
                "arcface_original_vs_stage": arcface_similarity(
                    identity, original_tensor, stage_tensor
                ),
                "arcface_clean_edit_vs_stage_edit": arcface_similarity(
                    identity, clean_tensor, stage_edit_tensor
                ),
            }
            rows.append(row)
            if stage_name != "original":
                make_strip(
                    [
                        ("Original", original_path),
                        ("Stage input", stage_path),
                        ("Clean edit", clean_path),
                        ("Stage edit", stage_edit_path),
                    ],
                    output_root / f"seed_{seed}" / f"{stage_name}_comparison.jpg",
                    f"seed={seed} | {stage_name} | {prompt}",
                )
    del identity
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    with (output_root / "seed_replay_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "status": "complete",
        "num_seeds": len(args.seeds),
        "num_stages": len(stages),
        "num_rows": len(rows),
        "output_root": str(output_root),
    }
    json_dump(output_root / "summary.json", summary)
    json_dump(output_root / "DONE.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
