"""Audit whether zero-parameter geometry modules are numerically neutral.

This is a cheap CPU check.  It deliberately uses high-frequency deterministic
input so even a one-pixel shift or an ill-conditioned identity transform is
easy to detect.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from vae_latent_adversarial import WARP_TYPES, create_warp


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--grid-size", type=int, default=8)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--mean-abs-tolerance", type=float, default=1e-4)
    args = parser.parse_args()

    torch.manual_seed(7)
    image = torch.rand(1, 3, args.image_size, args.image_size)
    rows: list[dict[str, object]] = []
    for warp_type in WARP_TYPES:
        row: dict[str, object] = {"warp_type": warp_type}
        try:
            warp = create_warp(
                warp_type,
                (args.image_size, args.image_size),
                (args.grid_size, args.grid_size),
                1.0,
            ).eval()
            with torch.no_grad():
                output = warp(image)
            delta = (output - image).abs()
            row.update(
                mean_abs_delta=float(delta.mean().item()),
                max_abs_delta=float(delta.max().item()),
                neutral=bool(delta.mean().item() <= args.mean_abs_tolerance),
            )
        except Exception as exc:  # keep auditing the remaining modules
            row.update(neutral=False, error=f"{type(exc).__name__}: {exc}")
        rows.append(row)
        print(json.dumps(row, sort_keys=True))

    payload = {
        "image_size": args.image_size,
        "grid_size": args.grid_size,
        "mean_abs_tolerance": args.mean_abs_tolerance,
        "all_neutral": all(bool(row.get("neutral")) for row in rows),
        "warps": rows,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if not payload["all_neutral"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
