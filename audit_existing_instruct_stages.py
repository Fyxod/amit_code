#!/usr/bin/env python
"""Post-hoc stage audit for existing VAE-latent adversarial artifacts.

New controlled runs are the primary evidence, but several visually notable
InstructPix2Pix cases already have an exact ``decoded_prewarp.png`` and
``vae_latent_out.png`` pair.  This utility adds the missing deterministic VAE
reconstruction and edits every available stage with identical editor settings.

It never infers a latent-only stage from the combined output.  If the exact
``decoded_prewarp.png`` artifact is absent, the row is marked missing instead
of substituting an iteration snapshot that may not correspond to the restored
best state.
"""

from __future__ import annotations

import argparse
import gc
import json
import shutil
from pathlib import Path

import torch

from instruct_stage_attribution import (
    add_arcface_audit,
    generate_stage_edits,
    json_dump,
)
from utils import load_image, save_image
from vae_latent_adversarial import VAELatentOptimiser


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--candidate-dir", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--face-id", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--image-guidance-scale", type=float, default=1.5)
    parser.add_argument("--image-size", type=int, nargs=2, default=(512, 512))
    parser.add_argument("--model-id", default="timbrooks/instruct-pix2pix")
    parser.add_argument("--taesd-path", default="taesd")
    parser.add_argument(
        "--arcface-checkpoint",
        default="/home/interns/Desktop/face4/models/arcface/iresnet100.pth",
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    candidate_dir = args.candidate_dir.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    device = args.device if torch.cuda.is_available() else "cpu"
    config = vars(args) | {
        "input": str(input_path),
        "candidate_dir": str(candidate_dir),
        "output_root": str(output_root),
        "audit_type": "posthoc_existing_exact_artifacts",
    }
    json_dump(output_root / "config_resolved.json", config)
    shutil.copy2(input_path, output_root / "original_source.png")

    image = load_image(str(input_path), size=tuple(args.image_size), device=device).unsqueeze(0)
    vae = VAELatentOptimiser(args.model_id, args.taesd_path, device)
    for parameter in vae.parameters():
        parameter.requires_grad_(False)
    vae.eval()
    with torch.no_grad():
        reconstruction = vae.decode(vae.encode(image))
    save_image(reconstruction[0], str(output_root / "vae_reconstruction.png"))
    del reconstruction, image, vae
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    copied: dict[str, str | None] = {"decoded_prewarp": None, "combined": None}
    decoded_path = candidate_dir / "decoded_prewarp.png"
    if decoded_path.exists():
        target = output_root / "posthoc" / "learned_delta_only.png"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(decoded_path, target)
        copied["decoded_prewarp"] = str(decoded_path)
    combined_path = candidate_dir / "vae_latent_out.png"
    if combined_path.exists():
        target = output_root / "posthoc" / "learned_combined.png"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(combined_path, target)
        copied["combined"] = str(combined_path)
    else:
        raise FileNotFoundError(f"Missing required final editor input: {combined_path}")

    edit_result = generate_stage_edits(
        root=output_root,
        original_path=input_path,
        prompt=args.prompt,
        seed=args.seed,
        steps=args.steps,
        guidance_scale=args.guidance_scale,
        image_guidance_scale=args.image_guidance_scale,
        device=device,
    )
    add_arcface_audit(output_root, input_path, args.arcface_checkpoint, device)
    exact_delta_available = decoded_path.exists()
    summary = {
        "status": "complete",
        "face_id": args.face_id,
        "prompt": args.prompt,
        "exact_decoded_prewarp_available": exact_delta_available,
        "candidate_sources": copied,
        "num_stage_edits": len(edit_result["rows"]),
        "limitation": (
            None
            if exact_delta_available
            else "The old run did not save decoded_prewarp.png, so no latent-only claim is made for it."
        ),
    }
    json_dump(output_root / "summary.json", summary)
    json_dump(output_root / "DONE.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
