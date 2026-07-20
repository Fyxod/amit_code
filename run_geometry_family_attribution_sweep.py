#!/usr/bin/env python
"""Run resumable geometry-family and geometry-combination attribution tests.

This driver keeps the verified stage-attribution report untouched. New runs
are written below ``targeted_experiments/geometry_family_attribution``.
Each completed child folder contains its own exact configuration, histories,
stage images, InstructPix2Pix edits, and SSIM/PSNR/L2/ArcFace metrics.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


SINGLE_WARPS = [
    "bspline",
    "bezier",
    "mobius",
    "laplacian",
    "geodesic",
    "diffgeom",
    "delaunay",
    "rolling",
    "fft",
    "homography",
]

# B-spline is repeated in combined mode because its earlier saved result was
# entangled with delta-z and had an invalid clean-baseline resolution.
COMBINATION_WARPS = [
    "bspline",
    "bspline+delaunay",
    "bspline+rolling",
    "bspline+lens",
    "bspline+polar",
    "bspline+tps",
    "bspline+bezier+laplacian",
    "all",
]


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_and_tee(command: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        return process.wait()


def build_command(args: argparse.Namespace, output: Path, warp: str, modes: list[str]) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).with_name("instruct_stage_attribution.py")),
        "--input", str(args.input),
        "--output-root", str(output),
        "--prompt", args.prompt,
        "--face-id", args.face_id,
        "--warp-type", warp,
        "--modes", *modes,
        "--iterations", str(args.iterations),
        "--lr", str(args.lr),
        "--identity-weight", str(args.identity_weight),
        "--ssim-weight", str(args.ssim_weight),
        "--pixel-weight", str(args.pixel_weight),
        "--latent-reg-weight", str(args.latent_reg_weight),
        "--imperceptibility-scale", str(args.imperceptibility_scale),
        "--seed", str(args.seed),
        "--steps", str(args.steps),
        "--guidance-scale", str(args.guidance_scale),
        "--image-guidance-scale", str(args.image_guidance_scale),
        "--arcface-checkpoint", str(args.arcface_checkpoint),
    ]
    return command


def planned_runs(args: argparse.Namespace) -> list[tuple[str, str, list[str]]]:
    runs: list[tuple[str, str, list[str]]] = []
    if args.phase in {"singles", "all"}:
        runs.extend(
            (f"single_{slug(warp)}", warp, ["geometry_only_original", "geometry_only_reconstruction"])
            for warp in SINGLE_WARPS
        )
    if args.phase in {"combinations", "all"}:
        runs.extend(
            (f"combined_{slug(warp)}", warp, ["combined"])
            for warp in COMBINATION_WARPS
        )
    return runs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("singles", "combinations", "all"), default="all")
    parser.add_argument("--input", type=Path, default=Path("parth_save/image_2_black_jacket_igs2_seed24001/original.png"))
    parser.add_argument("--output-root", type=Path, default=Path("targeted_experiments/geometry_family_attribution/image2_black_jacket"))
    parser.add_argument("--prompt", default="Add a black jacket")
    parser.add_argument("--face-id", default="image_2")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--identity-weight", type=float, default=10.0)
    parser.add_argument("--ssim-weight", type=float, default=5.0)
    parser.add_argument("--pixel-weight", type=float, default=0.1)
    parser.add_argument("--latent-reg-weight", type=float, default=0.001)
    parser.add_argument("--imperceptibility-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=24001)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--image-guidance-scale", type=float, default=2.0)
    parser.add_argument("--arcface-checkpoint", type=Path, default=Path("/home/interns/Desktop/parth_cleanup/face4/models/arcface/iresnet100.pth"))
    parser.add_argument("--rerun-failed", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    plans = planned_runs(args)
    dump(args.output_root / "sweep_plan.json", {
        "phase": args.phase,
        "input": str(args.input),
        "prompt": args.prompt,
        "iterations": args.iterations,
        "seed": args.seed,
        "runs": [{"id": run_id, "warp": warp, "modes": modes} for run_id, warp, modes in plans],
    })
    completed: list[str] = []
    skipped: list[str] = []
    failed: list[dict[str, Any]] = []
    for index, (run_id, warp, modes) in enumerate(plans, start=1):
        output = args.output_root / run_id
        if (output / "DONE.json").exists():
            print(f"[{index}/{len(plans)}] skip completed {run_id}")
            skipped.append(run_id)
            continue
        if (output / "FAILED.json").exists() and not args.rerun_failed:
            print(f"[{index}/{len(plans)}] skip failed {run_id}; pass --rerun-failed to retry")
            skipped.append(run_id)
            continue
        output.mkdir(parents=True, exist_ok=True)
        command = build_command(args, output, warp, modes)
        dump(output / "launcher_config.json", {"run_id": run_id, "warp": warp, "modes": modes, "command": command})
        print(f"[{index}/{len(plans)}] run {run_id}: warp={warp} modes={modes}")
        code = run_and_tee(command, output / "launcher.log")
        if code == 0 and (output / "DONE.json").exists():
            completed.append(run_id)
        else:
            failure = {"run_id": run_id, "warp": warp, "exit_code": code}
            dump(output / "FAILED.json", failure)
            failed.append(failure)
        dump(args.output_root / "sweep_status.json", {
            "completed_this_invocation": completed,
            "skipped": skipped,
            "failed": failed,
            "remaining": [item[0] for item in plans if not (args.output_root / item[0] / "DONE.json").exists()],
        })
    if failed:
        raise SystemExit(f"{len(failed)} sweep runs failed; see sweep_status.json")
    print(json.dumps({"completed": completed, "skipped": skipped, "failed": failed}, indent=2))


if __name__ == "__main__":
    main()
