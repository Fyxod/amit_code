#!/usr/bin/env python
"""Run resumable longer-depth and strength controls for geometry attribution.

This follow-up is intentionally narrow. It checks whether conclusions from the
20-iteration family sweep change when B-spline is optimized longer, when its
configured scale is increased, and when the strongest alternative families are
tested at a less aggressive scale. Existing experiment folders are untouched.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


CASES = (
    {
        "id": "image2_black_jacket",
        "face_id": "image_2",
        "input": "parth_save/image_2_black_jacket_igs2_seed24001/original.png",
        "prompt": "Add a black jacket",
    },
    {
        "id": "image4_red_scarf",
        "face_id": "image_4",
        "input": "targeted_experiments/stage_attribution/image4_red_scarf_polar_25_corrected/original.png",
        "prompt": "Add a red scarf",
    },
)


CONTROLS = (
    {
        "id": "bspline_iter100_scale1",
        "warp": "bspline",
        "iterations": 100,
        "scale": 1.0,
        "modes": ("geometry_only_original", "geometry_only_reconstruction", "combined"),
    },
    {
        "id": "bspline_iter100_scale2",
        "warp": "bspline",
        "iterations": 100,
        "scale": 2.0,
        "modes": ("geometry_only_original", "geometry_only_reconstruction", "combined"),
    },
    {
        "id": "rolling_iter100_scale1",
        "warp": "rolling",
        "iterations": 100,
        "scale": 1.0,
        "modes": ("geometry_only_original", "geometry_only_reconstruction"),
    },
    {
        "id": "diffgeom_iter50_scale05",
        "warp": "diffgeom",
        "iterations": 50,
        "scale": 0.5,
        "modes": ("geometry_only_original", "geometry_only_reconstruction"),
    },
    {
        "id": "all_iter50_scale05",
        "warp": "all",
        "iterations": 50,
        "scale": 0.5,
        "modes": ("geometry_only_original", "geometry_only_reconstruction", "combined"),
    },
)


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


def build_command(args: argparse.Namespace, case: dict[str, Any], control: dict[str, Any], output: Path) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).with_name("instruct_stage_attribution.py")),
        "--input", case["input"],
        "--output-root", str(output),
        "--prompt", case["prompt"],
        "--face-id", case["face_id"],
        "--warp-type", control["warp"],
        "--modes", *control["modes"],
        "--iterations", str(control["iterations"]),
        "--imperceptibility-scale", str(control["scale"]),
        "--lr", str(args.lr),
        "--identity-weight", str(args.identity_weight),
        "--ssim-weight", str(args.ssim_weight),
        "--pixel-weight", str(args.pixel_weight),
        "--latent-reg-weight", str(args.latent_reg_weight),
        "--seed", str(args.seed),
        "--steps", str(args.steps),
        "--guidance-scale", str(args.guidance_scale),
        "--image-guidance-scale", str(args.image_guidance_scale),
        "--arcface-checkpoint", str(args.arcface_checkpoint),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("targeted_experiments/geometry_depth_strength_controls"),
    )
    parser.add_argument("--case", choices=("all", *(case["id"] for case in CASES)), default="all")
    parser.add_argument("--control", choices=("all", *(item["id"] for item in CONTROLS)), default="all")
    parser.add_argument("--seed", type=int, default=24001)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--image-guidance-scale", type=float, default=2.0)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--identity-weight", type=float, default=10.0)
    parser.add_argument("--ssim-weight", type=float, default=5.0)
    parser.add_argument("--pixel-weight", type=float, default=0.1)
    parser.add_argument("--latent-reg-weight", type=float, default=0.001)
    parser.add_argument(
        "--arcface-checkpoint",
        type=Path,
        default=Path("/home/interns/Desktop/parth_cleanup/face4/models/arcface/iresnet100.pth"),
    )
    parser.add_argument("--rerun-failed", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected_cases = [case for case in CASES if args.case in {"all", case["id"]}]
    selected_controls = [control for control in CONTROLS if args.control in {"all", control["id"]}]
    plans = [(case, control) for case in selected_cases for control in selected_controls]
    dump(args.output_root / "control_plan.json", {
        "seed": args.seed,
        "cases": selected_cases,
        "controls": selected_controls,
    })
    completed: list[str] = []
    skipped: list[str] = []
    failed: list[dict[str, Any]] = []
    for index, (case, control) in enumerate(plans, start=1):
        run_id = f"{case['id']}__{control['id']}"
        output = args.output_root / run_id
        if (output / "DONE.json").exists():
            print(f"[{index}/{len(plans)}] skip completed {run_id}")
            skipped.append(run_id)
            continue
        if (output / "FAILED.json").exists() and not args.rerun_failed:
            print(f"[{index}/{len(plans)}] skip failed {run_id}")
            skipped.append(run_id)
            continue
        command = build_command(args, case, control, output)
        dump(output / "launcher_config.json", {
            "run_id": run_id,
            "case": case,
            "control": control,
            "command": command,
        })
        print(f"[{index}/{len(plans)}] run {run_id}")
        code = run_and_tee(command, output / "launcher.log")
        if code == 0 and (output / "DONE.json").exists():
            completed.append(run_id)
        else:
            failure = {"run_id": run_id, "exit_code": code}
            dump(output / "FAILED.json", failure)
            failed.append(failure)
        dump(args.output_root / "control_status.json", {
            "completed_this_invocation": completed,
            "skipped": skipped,
            "failed": failed,
        })
    if failed:
        raise SystemExit(f"{len(failed)} controls failed; see control_status.json")
    print(json.dumps({"completed": completed, "skipped": skipped}, indent=2))


if __name__ == "__main__":
    main()
