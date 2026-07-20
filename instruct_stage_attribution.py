#!/usr/bin/env python
"""Controlled attribution study for the VAE-latent + geometry pipeline.

This script isolates the stages that are entangled in
``vae_latent_adversarial.py``.  It deliberately runs one input case at a time
so that the VAE/ArcFace optimisation models can be released before loading the
InstructPix2Pix editor.  The exact same prompt, seed and editor settings are
used for every saved stage.

The four optimisation modes are:

* ``delta_only``: optimise only delta-z and decode it, with no geometry.
* ``geometry_only_original``: optimise geometry directly on the original.
* ``geometry_only_reconstruction``: optimise geometry on a deterministic
  VAE encode/decode reconstruction, with delta-z fixed to zero.
* ``combined``: optimise delta-z and geometry jointly, matching the current
  pipeline order ``geometry(decode(encode(x) + delta-z))``.

For a combined run, the learned state is replayed to save the latent-only,
geometry-only and order-control outputs.  This makes it possible to attribute
an edited-output change to VAE reconstruction, learned latent delta, learned
geometry, their composition, or the order of the two stages.
"""

from __future__ import annotations

import argparse
import copy
import csv
import gc
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter

from instruct import InstructBackend, InstructSettings
from models import FaceRecognitionModel
from utils import load_image, save_image
from vae_latent_adversarial import (
    VAELatentOptimiser,
    create_all_warps,
    create_warp,
    identity_loss,
    latent_regularisation,
    pixel_l2_loss,
    ssim_loss,
)


MODES = (
    "delta_only",
    "geometry_only_original",
    "geometry_only_reconstruction",
    "combined",
)


def slug(text: str) -> str:
    import re

    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    if tensor.ndim == 4:
        tensor = tensor[0]
    array = (
        tensor.detach().float().clamp(0, 1).cpu().permute(1, 2, 0).numpy() * 255.0
    ).round().astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def temporary_image_path(path: Path) -> Path:
    return path.with_name(f".{path.stem}.{os.getpid()}.tmp{path.suffix}")


def atomic_save_pil(image: Image.Image, path: Path, **save_kwargs: Any) -> None:
    """Write a complete image before atomically exposing the final path."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = temporary_image_path(path)
    image.save(temporary, **save_kwargs)
    os.replace(temporary, path)


def atomic_save_tensor(tensor: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = temporary_image_path(path)
    save_image(tensor, str(temporary))
    os.replace(temporary, path)


def valid_image(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except (OSError, ValueError):
        return False


def pil_to_tensor(image: Image.Image, device: str = "cpu") -> torch.Tensor:
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device)


def pair_metrics_arrays(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.shape != b.shape:
        b_image = Image.fromarray((np.clip(b, 0, 1) * 255).astype(np.uint8))
        b = np.asarray(
            b_image.resize((a.shape[1], a.shape[0]), Image.Resampling.LANCZOS),
            dtype=np.float32,
        ) / 255.0
    mse = float(np.mean((a - b) ** 2))
    mu_a = gaussian_filter(a, sigma=(1.5, 1.5, 0), mode="reflect")
    mu_b = gaussian_filter(b, sigma=(1.5, 1.5, 0), mode="reflect")
    sigma_a = gaussian_filter(a * a, sigma=(1.5, 1.5, 0), mode="reflect") - mu_a * mu_a
    sigma_b = gaussian_filter(b * b, sigma=(1.5, 1.5, 0), mode="reflect") - mu_b * mu_b
    sigma_ab = gaussian_filter(a * b, sigma=(1.5, 1.5, 0), mode="reflect") - mu_a * mu_b
    c1, c2 = 0.01**2, 0.03**2
    ssim_map = ((2 * mu_a * mu_b + c1) * (2 * sigma_ab + c2)) / (
        (mu_a * mu_a + mu_b * mu_b + c1) * (sigma_a + sigma_b + c2)
    )
    return {
        "ssim": float(np.mean(ssim_map)),
        "psnr": float("inf") if mse == 0 else float(-10.0 * np.log10(mse)),
        "mse": mse,
        "l2": float(math.sqrt(mse)),
    }


def pair_metrics_pil(a: Image.Image, b: Image.Image) -> dict[str, float]:
    return pair_metrics_arrays(
        np.asarray(a.convert("RGB"), dtype=np.float32) / 255.0,
        np.asarray(b.convert("RGB"), dtype=np.float32) / 255.0,
    )


def image_pixel_hash(image: Image.Image) -> str:
    """Hash decoded RGB pixels so deterministic duplicate edits can be reused."""

    rgb = image.convert("RGB")
    digest = hashlib.sha256()
    digest.update(f"{rgb.width}x{rgb.height}:RGB".encode("ascii"))
    digest.update(np.asarray(rgb, dtype=np.uint8).tobytes())
    return digest.hexdigest()


def arcface_similarity(model: nn.Module, a: torch.Tensor, b: torch.Tensor) -> float:
    with torch.no_grad():
        ea = model(a)
        eb = model(b)
        return float(F.cosine_similarity(ea, eb, dim=1).mean().item())


def set_geometry_mask(warps: Any, warp_type: str, image_shape: tuple[int, int]) -> None:
    """Use whole-image geometry for stage attribution.

    Face-mask detection would introduce a second, non-differentiable variable
    into the experiment.  Existing warp modules interpret a missing mask as a
    whole-image transform, which is the controlled behavior required here.
    """

    _ = (warps, warp_type, image_shape)


def geometry_modules(
    warp_type: str,
    image_size: tuple[int, int],
    grid_size: tuple[int, int],
    device: str,
    scale: float,
) -> tuple[Any, list[str]]:
    if warp_type == "all":
        warps = create_all_warps(image_size, grid_size, device, scale)
        return warps, list(warps)
    return create_warp(warp_type, image_size, grid_size, scale).to(device), [warp_type]


def apply_geometry(image: torch.Tensor, warps: Any, warp_type: str, keys: list[str]) -> torch.Tensor:
    if warp_type == "all":
        output = image
        for key in keys:
            output = warps[key](output)
        return output
    return warps(image)


def geometry_parameters(warps: Any, warp_type: str, keys: list[str]) -> list[nn.Parameter]:
    params: list[nn.Parameter] = []
    if warp_type == "all":
        for key in keys:
            params.extend([p for p in warps[key].parameters() if p.requires_grad])
    else:
        params.extend([p for p in warps.parameters() if p.requires_grad])
    return params


def geometry_state(warps: Any, warp_type: str, keys: list[str]) -> dict[str, Any]:
    if warp_type == "all":
        return {key: copy.deepcopy(warps[key].state_dict()) for key in keys}
    return {warp_type: copy.deepcopy(warps.state_dict())}


def load_geometry_state(warps: Any, warp_type: str, keys: list[str], state: dict[str, Any]) -> None:
    if warp_type == "all":
        for key in keys:
            warps[key].load_state_dict(state[key])
    else:
        warps.load_state_dict(state[warp_type])


def save_history(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with path.with_suffix(".jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def run_mode(
    *,
    mode: str,
    image: torch.Tensor,
    reconstruction: torch.Tensor,
    z_orig: torch.Tensor,
    vae: VAELatentOptimiser,
    identity_model: nn.Module,
    warp_type: str,
    image_size: tuple[int, int],
    grid_size: tuple[int, int],
    imperceptibility_scale: float,
    iterations: int,
    learning_rate: float,
    identity_weight: float,
    ssim_weight: float,
    pixel_weight: float,
    latent_reg_weight: float,
    grad_clip: float,
    device: str,
    output_dir: Path,
) -> dict[str, Any]:
    if mode not in MODES:
        raise ValueError(f"Unknown mode: {mode}")
    output_dir.mkdir(parents=True, exist_ok=True)
    warps, keys = geometry_modules(warp_type, image_size, grid_size, device, imperceptibility_scale)
    set_geometry_mask(warps, warp_type, image_size)
    delta_z = nn.Parameter(torch.zeros_like(z_orig), requires_grad=mode in {"delta_only", "combined"})
    geom_params = geometry_parameters(warps, warp_type, keys)
    for parameter in geom_params:
        parameter.requires_grad_(mode in {"geometry_only_original", "geometry_only_reconstruction", "combined"})
    parameters: list[nn.Parameter] = []
    if delta_z.requires_grad:
        parameters.append(delta_z)
    parameters.extend([parameter for parameter in geom_params if parameter.requires_grad])
    if not parameters:
        raise RuntimeError(f"Mode {mode} has no trainable parameters")
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)
    with torch.no_grad():
        original_embedding = identity_model(image).detach()

    history: list[dict[str, Any]] = []
    best_loss = float("inf")
    best_iter = -1
    best_delta = delta_z.detach().clone()
    best_warps = geometry_state(warps, warp_type, keys)
    start = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    for iteration in range(iterations):
        iter_start = time.perf_counter()
        if mode == "geometry_only_original":
            decoded = image
            output = apply_geometry(image, warps, warp_type, keys)
        elif mode == "geometry_only_reconstruction":
            decoded = reconstruction
            output = apply_geometry(reconstruction, warps, warp_type, keys)
        elif mode == "delta_only":
            decoded = vae.decode(z_orig + delta_z)
            output = decoded
        else:
            decoded = vae.decode(z_orig + delta_z)
            output = apply_geometry(decoded, warps, warp_type, keys)

        embedding = identity_model(output)
        identity_distance = 1.0 - F.cosine_similarity(original_embedding, embedding, dim=1).mean()
        latent_reg = latent_regularisation(delta_z)
        # Match the preservation decomposition of the original joint pipeline.
        decoded_ssim = ssim_loss(decoded, image)
        decoded_pixel = pixel_l2_loss(decoded, image)
        if mode in {"geometry_only_original", "geometry_only_reconstruction", "combined"}:
            geometry_ssim = ssim_loss(output, decoded)
            geometry_pixel = pixel_l2_loss(output, decoded)
        else:
            geometry_ssim = output.new_zeros(())
            geometry_pixel = output.new_zeros(())
        preservation_ssim = decoded_ssim + geometry_ssim
        preservation_pixel = decoded_pixel + geometry_pixel
        loss = (
            -identity_weight * identity_distance
            + ssim_weight * preservation_ssim
            + pixel_weight * preservation_pixel
            + latent_reg_weight * latent_reg
        )
        with torch.no_grad():
            metrics = pair_metrics_arrays(
                image[0].detach().cpu().permute(1, 2, 0).numpy(),
                output[0].detach().cpu().permute(1, 2, 0).numpy(),
            )
        row = {
            "iter": iteration,
            "mode": mode,
            "loss": float(loss.item()),
            "arcface_identity_distance": float(identity_distance.item()),
            "arcface_identity_similarity": float(1.0 - identity_distance.item()),
            "ssim_preservation_loss": float(preservation_ssim.item()),
            "pixel_preservation_loss": float(preservation_pixel.item()),
            "latent_regularisation": float(latent_reg.item()),
            "input_ssim": metrics["ssim"],
            "input_psnr": metrics["psnr"],
            "input_mse": metrics["mse"],
            "seconds_iter": time.perf_counter() - iter_start,
            "seconds_elapsed": time.perf_counter() - start,
            "peak_vram_gb": (
                float(torch.cuda.max_memory_allocated() / 1024**3) if torch.cuda.is_available() else None
            ),
        }
        # Capture the exact state that produced the logged forward values.
        # Saving after optimizer.step() would associate the row with a state
        # that has never been evaluated and was the source of ambiguity in
        # several older experiment pipelines.
        if row["loss"] < best_loss:
            best_loss = row["loss"]
            best_iter = iteration
            best_delta = delta_z.detach().clone()
            best_warps = geometry_state(warps, warp_type, keys)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_value_(parameters, grad_clip)
        optimizer.step()
        history.append(row)
        print(
            f"[{mode}] {iteration + 1:03d}/{iterations:03d} "
            f"loss={row['loss']:.5f} id_sim={row['arcface_identity_similarity']:.5f} "
            f"ssim={row['input_ssim']:.5f} {row['seconds_iter']:.2f}s"
        )

    with torch.no_grad():
        delta_z.copy_(best_delta)
    load_geometry_state(warps, warp_type, keys, best_warps)
    with torch.no_grad():
        vae_reconstruction = vae.decode(z_orig)
        delta_only = vae.decode(z_orig + delta_z)
        geometry_original = apply_geometry(image, warps, warp_type, keys)
        geometry_reconstruction = apply_geometry(vae_reconstruction, warps, warp_type, keys)
        combined = apply_geometry(delta_only, warps, warp_type, keys)
        geometry_then_vae = vae.decode(vae.encode(geometry_original))

    replay = {
        "vae_reconstruction": vae_reconstruction,
        "learned_delta_only": delta_only,
        "learned_geometry_on_original": geometry_original,
        "learned_geometry_on_reconstruction": geometry_reconstruction,
        "learned_combined": combined,
        "geometry_then_vae_reconstruction": geometry_then_vae,
    }
    # The optimizer state is needed only for local reproducibility and is
    # ignored by the repository's *.pt rule.
    torch.save(
        {"delta_z": best_delta.cpu(), "geometry": best_warps, "mode": mode},
        output_dir / "stage_state.pt",
    )
    for name, tensor in replay.items():
        atomic_save_tensor(tensor[0], output_dir / f"{name}.png")
    save_history(output_dir / "history.csv", history)
    summary = {
        "mode": mode,
        "best_iter": best_iter,
        "best_loss": best_loss,
        "iterations": len(history),
        "mean_seconds_iter": float(np.mean([row["seconds_iter"] for row in history])),
        "peak_vram_gb": max(float(row["peak_vram_gb"] or 0.0) for row in history),
        "replay_images": {name: str(output_dir / f"{name}.png") for name in replay},
    }
    json_dump(output_dir / "optimization_summary.json", summary)
    return {"summary": summary}


def make_strip(paths: list[tuple[str, Path]], output: Path, title: str) -> None:
    tile_w, tile_h, label_h, title_h = 224, 224, 28, 44
    canvas = Image.new("RGB", (tile_w * len(paths), tile_h + label_h + title_h), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), title, fill="black")
    for index, (label, path) in enumerate(paths):
        x = index * tile_w
        draw.text((x + 8, title_h + 4), label, fill="black")
        if path.exists():
            image = Image.open(path).convert("RGB")
            image.thumbnail((tile_w - 10, tile_h - 10), Image.Resampling.LANCZOS)
            canvas.paste(image, (x + (tile_w - image.width) // 2, title_h + label_h + (tile_h - image.height) // 2))
        else:
            draw.rectangle((x + 10, title_h + label_h + 10, x + tile_w - 10, title_h + label_h + tile_h - 10), outline="red", width=3)
            draw.text((x + 70, title_h + label_h + 100), "MISSING", fill="red")
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_save_pil(canvas, output, quality=95)


def unload_cuda(*objects: Any) -> None:
    for obj in objects:
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def generate_stage_edits(
    *,
    root: Path,
    original_path: Path,
    prompt: str,
    seed: int,
    steps: int,
    guidance_scale: float,
    image_guidance_scale: float,
    device: str,
) -> dict[str, Any]:
    settings = InstructSettings(
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        image_guidance_scale=image_guidance_scale,
        seed=seed,
    )
    backend = InstructBackend(torch.device(device), settings)
    stage_paths = [path for path in root.rglob("*.png") if path.name in {
        "vae_reconstruction.png",
        "neutral_geometry_on_original.png",
        "neutral_geometry_on_reconstruction.png",
        "learned_delta_only.png",
        "learned_geometry_on_original.png",
        "learned_geometry_on_reconstruction.png",
        "learned_combined.png",
        "geometry_then_vae_reconstruction.png",
    }]
    clean_edit_path = root / "clean_edit.png"
    original_image = Image.open(original_path).convert("RGB")
    if not valid_image(clean_edit_path):
        atomic_save_pil(backend.generate_edit(original_image, prompt, seed), clean_edit_path)
    clean_edit = Image.open(clean_edit_path).convert("RGB")
    edit_cache: dict[str, Path] = {
        image_pixel_hash(original_image): clean_edit_path,
    }
    rows: list[dict[str, Any]] = []
    for stage_path in sorted(stage_paths):
        relative = stage_path.relative_to(root)
        mode_name = relative.parts[0] if len(relative.parts) > 1 else "baseline"
        edit_path = stage_path.parent / f"{stage_path.stem}_edit.png"
        stage_image = Image.open(stage_path).convert("RGB")
        stage_hash = image_pixel_hash(stage_image)
        reused_from = edit_cache.get(stage_hash)
        if not valid_image(edit_path):
            if reused_from is not None and valid_image(reused_from):
                atomic_save_pil(Image.open(reused_from).convert("RGB"), edit_path)
            else:
                atomic_save_pil(backend.generate_edit(stage_image, prompt, seed), edit_path)
        edit_cache.setdefault(stage_hash, edit_path)
        input_metrics = pair_metrics_pil(original_image, stage_image)
        output_metrics = pair_metrics_pil(clean_edit, Image.open(edit_path).convert("RGB"))
        rows.append({
            "mode": mode_name,
            "stage": stage_path.stem,
            "stage_path": str(stage_path),
            "edit_path": str(edit_path),
            "edit_reused_from": str(reused_from) if reused_from is not None else "",
            **{f"input_{key}": value for key, value in input_metrics.items()},
            **{f"output_{key}": value for key, value in output_metrics.items()},
        })
        make_strip(
            [
                ("Original", original_path),
                ("Stage input", stage_path),
                ("Clean edit", clean_edit_path),
                ("Stage edit", edit_path),
            ],
            stage_path.parent / f"{stage_path.stem}_comparison.jpg",
            f"{mode_name} | {stage_path.stem} | {prompt}",
        )
    del backend
    unload_cuda()
    if rows:
        with (root / "stage_edit_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return {"clean_edit_path": str(clean_edit_path), "rows": rows}


def add_arcface_audit(
    root: Path,
    original_path: Path,
    checkpoint: str,
    device: str,
) -> None:
    metrics_path = root / "stage_edit_metrics.csv"
    if not metrics_path.exists():
        return
    with metrics_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    model = FaceRecognitionModel(
        model_name="arcface", model_path=checkpoint, device=device,
    )
    original = pil_to_tensor(Image.open(original_path).convert("RGB"), device)
    clean_edit = pil_to_tensor(Image.open(root / "clean_edit.png").convert("RGB"), device)
    tensor_cache: dict[str, torch.Tensor] = {
        str(original_path): original,
        str(root / "clean_edit.png"): clean_edit,
    }

    def tensor_for(path_value: str) -> torch.Tensor:
        if path_value not in tensor_cache:
            tensor_cache[path_value] = pil_to_tensor(
                Image.open(path_value).convert("RGB"), device,
            )
        return tensor_cache[path_value]

    for row in rows:
        stage = pil_to_tensor(Image.open(row["stage_path"]).convert("RGB"), device)
        edit = pil_to_tensor(Image.open(row["edit_path"]).convert("RGB"), device)
        tensor_cache[row["stage_path"]] = stage
        tensor_cache[row["edit_path"]] = edit
        row["arcface_original_vs_stage"] = arcface_similarity(model, original, stage)
        row["arcface_clean_edit_vs_stage_edit"] = arcface_similarity(model, clean_edit, edit)
        row["arcface_original_vs_clean_edit"] = arcface_similarity(model, original, clean_edit)
        row["arcface_stage_vs_stage_edit"] = arcface_similarity(model, stage, edit)
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    # Incremental comparisons isolate the effect added at each stage.  For
    # example, delta-edit vs combined-edit measures what learned geometry adds
    # after the same learned latent delta rather than comparing both to the
    # clean editor output.
    pair_definitions = (
        ("reconstruction_effect", "clean_edit", "vae_reconstruction"),
        ("delta_increment_after_reconstruction", "vae_reconstruction", "learned_delta_only"),
        ("geometry_on_original_effect", "clean_edit", "learned_geometry_on_original"),
        ("geometry_on_reconstruction_effect", "vae_reconstruction", "learned_geometry_on_reconstruction"),
        ("geometry_increment_after_delta", "learned_delta_only", "learned_combined"),
        ("order_effect", "learned_combined", "geometry_then_vae_reconstruction"),
    )
    incremental_rows: list[dict[str, Any]] = []
    baseline_rows = {row["stage"]: row for row in rows if row["mode"] == "baseline"}
    baseline_pairs = (
        ("reconstruction_effect", "clean_edit", "vae_reconstruction"),
        ("neutral_geometry_resampling_on_original", "clean_edit", "neutral_geometry_on_original"),
        (
            "neutral_geometry_resampling_after_reconstruction",
            "vae_reconstruction",
            "neutral_geometry_on_reconstruction",
        ),
    )
    clean_edit_path = str(root / "clean_edit.png")
    for comparison, left_name, right_name in baseline_pairs:
        left_path = clean_edit_path if left_name == "clean_edit" else baseline_rows.get(left_name, {}).get("edit_path")
        right_path = baseline_rows.get(right_name, {}).get("edit_path")
        if not left_path or not right_path:
            continue
        visual = pair_metrics_pil(
            Image.open(left_path).convert("RGB"),
            Image.open(right_path).convert("RGB"),
        )
        incremental_rows.append({
            "mode": "baseline_control",
            "comparison": comparison,
            "left_stage": left_name,
            "right_stage": right_name,
            "left_edit_path": left_path,
            "right_edit_path": right_path,
            "edit_ssim": visual["ssim"],
            "edit_psnr": visual["psnr"],
            "edit_mse": visual["mse"],
            "edit_l2": visual["l2"],
            "arcface_edit_cosine_similarity": arcface_similarity(
                model, tensor_for(left_path), tensor_for(right_path),
            ),
        })
    modes = sorted({row["mode"] for row in rows if row["mode"] != "baseline"})
    for mode in modes:
        stage_rows = {row["stage"]: row for row in rows if row["mode"] == mode}
        for comparison, left_name, right_name in pair_definitions:
            if left_name == "clean_edit":
                left_path = clean_edit_path
            else:
                left_row = stage_rows.get(left_name)
                left_path = left_row["edit_path"] if left_row else None
            right_row = stage_rows.get(right_name)
            right_path = right_row["edit_path"] if right_row else None
            if not left_path or not right_path:
                continue
            visual = pair_metrics_pil(
                Image.open(left_path).convert("RGB"),
                Image.open(right_path).convert("RGB"),
            )
            incremental_rows.append({
                "mode": mode,
                "comparison": comparison,
                "left_stage": left_name,
                "right_stage": right_name,
                "left_edit_path": left_path,
                "right_edit_path": right_path,
                "edit_ssim": visual["ssim"],
                "edit_psnr": visual["psnr"],
                "edit_mse": visual["mse"],
                "edit_l2": visual["l2"],
                "arcface_edit_cosine_similarity": arcface_similarity(
                    model, tensor_for(left_path), tensor_for(right_path),
                ),
            })
    if incremental_rows:
        with (root / "incremental_attribution_metrics.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(incremental_rows[0]))
            writer.writeheader()
            writer.writerows(incremental_rows)
    unload_cuda(model)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--face-id", default="unknown")
    parser.add_argument("--warp-type", default="bspline")
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--identity-weight", type=float, default=10.0)
    parser.add_argument("--ssim-weight", type=float, default=5.0)
    parser.add_argument("--pixel-weight", type=float, default=0.1)
    parser.add_argument("--latent-reg-weight", type=float, default=0.001)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--imperceptibility-scale", type=float, default=1.0)
    parser.add_argument("--grid-size", type=int, nargs=2, default=(8, 8))
    parser.add_argument("--image-size", type=int, nargs=2, default=(512, 512))
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--image-guidance-scale", type=float, default=1.5)
    parser.add_argument("--arcface-checkpoint", default="/home/interns/Desktop/face4/models/arcface/iresnet100.pth")
    parser.add_argument("--model-id", default="timbrooks/instruct-pix2pix")
    parser.add_argument("--taesd-path", default="taesd")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-edits", action="store_true")
    parser.add_argument("--skip-arcface-audit", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    input_path = args.input.resolve()
    image_size = tuple(args.image_size)
    device = args.device if torch.cuda.is_available() else "cpu"
    config = vars(args) | {
        "input": str(input_path),
        "output_root": str(output_root),
        "device_resolved": device,
        "pipeline_order": "geometry(decode(encode(original) + delta_z))",
    }
    json_dump(output_root / "config_resolved.json", config)
    tensor = load_image(str(input_path), size=image_size, device=device).unsqueeze(0)
    atomic_save_tensor(tensor[0], output_root / "original.png")

    vae = VAELatentOptimiser(args.model_id, args.taesd_path, device)
    for parameter in vae.parameters():
        parameter.requires_grad_(False)
    vae.eval()
    identity_model = FaceRecognitionModel(
        model_name="arcface", model_path=args.arcface_checkpoint, device=device,
    )
    with torch.no_grad():
        z_orig = vae.encode(tensor)
        reconstruction = vae.decode(z_orig)
    atomic_save_tensor(reconstruction[0], output_root / "vae_reconstruction.png")

    # A zero-parameter geometry module can still alter pixels through the
    # interpolation kernel used by grid_sample.  Save that resampling floor
    # explicitly so it is never mistaken for a learned geometry effect.
    neutral_warps, neutral_keys = geometry_modules(
        args.warp_type, image_size, tuple(args.grid_size), device,
        args.imperceptibility_scale,
    )
    set_geometry_mask(neutral_warps, args.warp_type, image_size)
    with torch.no_grad():
        neutral_original = apply_geometry(tensor, neutral_warps, args.warp_type, neutral_keys)
        neutral_reconstruction = apply_geometry(
            reconstruction, neutral_warps, args.warp_type, neutral_keys,
        )
    atomic_save_tensor(neutral_original[0], output_root / "neutral_geometry_on_original.png")
    atomic_save_tensor(
        neutral_reconstruction[0],
        output_root / "neutral_geometry_on_reconstruction.png",
    )
    del neutral_warps, neutral_original, neutral_reconstruction

    mode_summaries: list[dict[str, Any]] = []
    for mode in args.modes:
        result = run_mode(
            mode=mode,
            image=tensor,
            reconstruction=reconstruction,
            z_orig=z_orig,
            vae=vae,
            identity_model=identity_model,
            warp_type=args.warp_type,
            image_size=image_size,
            grid_size=tuple(args.grid_size),
            imperceptibility_scale=args.imperceptibility_scale,
            iterations=args.iterations,
            learning_rate=args.lr,
            identity_weight=args.identity_weight,
            ssim_weight=args.ssim_weight,
            pixel_weight=args.pixel_weight,
            latent_reg_weight=args.latent_reg_weight,
            grad_clip=args.grad_clip,
            device=device,
            output_dir=output_root / mode,
        )
        mode_summaries.append(result["summary"])
        del result

    # Explicitly drop the optimisation models before constructing the full
    # InstructPix2Pix pipeline.  This phase separation is what keeps the study
    # safe on a GPU that is concurrently hosting another intern's process.
    del identity_model, vae, tensor, reconstruction, z_orig
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    edit_result: dict[str, Any] = {"rows": []}
    if not args.skip_edits:
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
        if not args.skip_arcface_audit:
            add_arcface_audit(output_root, input_path, args.arcface_checkpoint, device)

    summary = {
        "status": "complete",
        "face_id": args.face_id,
        "prompt": args.prompt,
        "warp_type": args.warp_type,
        "modes": args.modes,
        "mode_summaries": mode_summaries,
        "num_stage_edits": len(edit_result["rows"]),
        "output_root": str(output_root),
    }
    json_dump(output_root / "summary.json", summary)
    json_dump(output_root / "DONE.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
