#!/usr/bin/env python
"""Summarize geometry-family attribution without modifying earlier reports.

The summary distinguishes three different comparisons that are easy to mix up:

* clean edit vs a stage edit (overall output change),
* neutral-resampling edit vs learned-geometry edit (learned geometry effect),
* delta-only edit vs combined edit (geometry's increment after delta-z).

All files are read from completed experiment folders. No model inference is run.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from PIL import Image

from instruct_stage_attribution import pair_metrics_pil


STAGE_SELECTORS = {
    "single": {
        "geometry_only_original": "learned_geometry_on_original",
        "geometry_only_reconstruction": "learned_geometry_on_reconstruction",
    },
    "combined": {
        "combined": (
            "learned_delta_only",
            "learned_geometry_on_original",
            "learned_geometry_on_reconstruction",
            "learned_combined",
        )
    },
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def as_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def stage_rows(case_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run_dir in sorted(p for p in case_root.iterdir() if p.is_dir()):
        summary_path = run_dir / "summary.json"
        metrics_path = run_dir / "stage_edit_metrics.csv"
        if not summary_path.exists() or not metrics_path.exists():
            continue
        summary = load_json(summary_path)
        kind = "combined" if run_dir.name.startswith("combined_") else "single"
        selectors = STAGE_SELECTORS[kind]
        for metric in read_csv(metrics_path):
            mode = metric.get("mode", "")
            stage = metric.get("stage", "")
            selected = selectors.get(mode)
            if selected is None:
                continue
            if isinstance(selected, tuple):
                if stage not in selected:
                    continue
            elif stage != selected:
                continue
            rows.append({
                "case": case_root.name,
                "face_id": summary.get("face_id"),
                "prompt": summary.get("prompt"),
                "run": run_dir.name,
                "kind": kind,
                "warp_type": summary.get("warp_type"),
                "warp_components": "+".join(summary.get("warp_components", [])),
                "mode": mode,
                "stage": stage,
                "input_ssim": as_float(metric.get("input_ssim")),
                "input_psnr": as_float(metric.get("input_psnr")),
                "output_ssim_vs_clean": as_float(metric.get("output_ssim")),
                "output_psnr_vs_clean": as_float(metric.get("output_psnr")),
                "output_l2_vs_clean": as_float(metric.get("output_l2")),
                "arcface_clean_vs_stage_edit": as_float(metric.get("arcface_clean_edit_vs_stage_edit")),
                "arcface_original_vs_stage_input": as_float(metric.get("arcface_original_vs_stage")),
                "stage_path": metric.get("stage_path"),
                "edit_path": metric.get("edit_path"),
            })
    return rows


def incremental_rows(case_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run_dir in sorted(p for p in case_root.iterdir() if p.is_dir()):
        summary_path = run_dir / "summary.json"
        metrics_path = run_dir / "incremental_attribution_metrics.csv"
        if not summary_path.exists() or not metrics_path.exists():
            continue
        summary = load_json(summary_path)
        for metric in read_csv(metrics_path):
            if metric.get("mode") not in {"combined", "geometry_only_original", "geometry_only_reconstruction"}:
                continue
            rows.append({
                "case": case_root.name,
                "face_id": summary.get("face_id"),
                "prompt": summary.get("prompt"),
                "run": run_dir.name,
                "warp_type": summary.get("warp_type"),
                "warp_components": "+".join(summary.get("warp_components", [])),
                "mode": metric.get("mode"),
                "comparison": metric.get("comparison"),
                "edit_ssim": as_float(metric.get("edit_ssim")),
                "edit_psnr": as_float(metric.get("edit_psnr")),
                "edit_l2": as_float(metric.get("edit_l2")),
                "arcface_edit_cosine_similarity": as_float(metric.get("arcface_edit_cosine_similarity")),
                "left_edit_path": metric.get("left_edit_path"),
                "right_edit_path": metric.get("right_edit_path"),
            })
    return rows


def seed_direct_rows(seed_root: Path, case: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not seed_root.exists():
        return rows
    for run_dir in sorted(p for p in seed_root.iterdir() if p.is_dir()):
        for seed_dir in sorted(run_dir.glob("seed_[0-9]*")):
            try:
                seed = int(seed_dir.name.split("_", 1)[1])
            except ValueError:
                continue
            if run_dir.name.startswith("combined_"):
                pairs = (
                    ("geometry_increment_after_delta", "joint_delta_only_edit.png", "combined_edit.png"),
                    ("learned_geometry_vs_neutral_original", "neutral_geometry_on_original_edit.png", "joint_geometry_on_original_edit.png"),
                    ("learned_geometry_vs_vae_reconstruction", "vae_reconstruction_edit.png", "joint_geometry_on_reconstruction_edit.png"),
                )
            else:
                pairs = (
                    ("learned_geometry_vs_neutral_original", "neutral_geometry_on_original_edit.png", "geometry_only_original_edit.png"),
                    ("learned_geometry_vs_vae_reconstruction", "vae_reconstruction_edit.png", "geometry_only_reconstruction_edit.png"),
                )
            for comparison, left_name, right_name in pairs:
                left_path = seed_dir / left_name
                right_path = seed_dir / right_name
                if not left_path.exists() or not right_path.exists():
                    continue
                metrics = pair_metrics_pil(
                    Image.open(left_path).convert("RGB"),
                    Image.open(right_path).convert("RGB"),
                )
                rows.append({
                    "case": case,
                    "run": run_dir.name,
                    "seed": seed,
                    "comparison": comparison,
                    "edit_ssim": metrics["ssim"],
                    "edit_psnr": metrics["psnr"],
                    "edit_l2": metrics["l2"],
                    "left_path": str(left_path),
                    "right_path": str(right_path),
                })
    return rows


def aggregate_seed_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["case"]), str(row["run"]), str(row["comparison"]))].append(row)
    output: list[dict[str, Any]] = []
    for (case, run, comparison), group in sorted(groups.items()):
        ssims = [float(row["edit_ssim"]) for row in group]
        l2s = [float(row["edit_l2"]) for row in group]
        output.append({
            "case": case,
            "run": run,
            "comparison": comparison,
            "num_seeds": len(group),
            "seeds": ",".join(str(row["seed"]) for row in sorted(group, key=lambda item: int(item["seed"]))),
            "mean_edit_ssim": mean(ssims),
            "min_edit_ssim": min(ssims),
            "max_edit_ssim": max(ssims),
            "range_edit_ssim": max(ssims) - min(ssims),
            "mean_edit_l2": mean(l2s),
            "max_edit_l2": max(l2s),
        })
    return output


def fmt(value: Any, digits: int = 4) -> str:
    numeric = as_float(value)
    return "n/a" if numeric is None else f"{numeric:.{digits}f}"


def build_findings(
    stage: list[dict[str, Any]],
    incremental: list[dict[str, Any]],
    seed_aggregate: list[dict[str, Any]],
) -> str:
    lines = [
        "# Geometry-family attribution summary",
        "",
        "This is a separate extension of the verified stage-attribution work. The existing PDF report and its source artifacts were not changed.",
        "",
        "## Reading the comparisons",
        "",
        "- `output_ssim_vs_clean` compares a stage edit with the clean edit; lower values mean a larger edited-output change, but do not isolate which stage caused it.",
        "- `learned_geometry_vs_neutral_original` isolates learned geometry from neutral grid resampling on the original input.",
        "- `learned_geometry_vs_vae_reconstruction` isolates learned geometry after VAE reconstruction.",
        "- `geometry_increment_after_delta` directly compares the delta-z-only edit with the combined delta-z-plus-geometry edit.",
        "- Input SSIM must be read together with edit SSIM; a large edit change at a visibly damaged input is not subtle geometry.",
        "",
        "## Completed cases",
        "",
    ]
    cases = sorted({str(row["case"]) for row in stage})
    for case in cases:
        rows = [row for row in stage if row["case"] == case]
        complete_runs = sorted({str(row["run"]) for row in rows})
        prompt = next((str(row["prompt"]) for row in rows if row.get("prompt")), "")
        lines.append(f"- `{case}`: {len(complete_runs)} runs; prompt `{prompt}`")
    lines.extend(["", "## Geometry-only ranking on original inputs", ""])
    for case in cases:
        singles = [
            row for row in stage
            if row["case"] == case and row["kind"] == "single" and row["stage"] == "learned_geometry_on_original"
        ]
        singles.sort(key=lambda row: (row["output_ssim_vs_clean"] is None, row["output_ssim_vs_clean"] or 1.0))
        lines.append(f"### {case}")
        lines.append("")
        lines.append("| Geometry | Input SSIM | Edit SSIM vs clean | ArcFace clean vs stage edit |")
        lines.append("|---|---:|---:|---:|")
        for row in singles:
            lines.append(
                f"| {row['warp_type']} | {fmt(row['input_ssim'])} | {fmt(row['output_ssim_vs_clean'])} | {fmt(row['arcface_clean_vs_stage_edit'])} |"
            )
        lines.append("")
    lines.extend(["## Seed-stability checks", ""])
    lines.append("| Case | Run | Direct comparison | Mean edit SSIM | Min | Max | Range |")
    lines.append("|---|---|---|---:|---:|---:|---:|")
    for row in seed_aggregate:
        lines.append(
            f"| {row['case']} | {row['run']} | {row['comparison']} | {fmt(row['mean_edit_ssim'])} | {fmt(row['min_edit_ssim'])} | {fmt(row['max_edit_ssim'])} | {fmt(row['range_edit_ssim'])} |"
        )
    lines.extend([
        "",
        "## Evidence-backed interpretation",
        "",
        "- B-spline geometry by itself remains close to the clean edit on the tested original input. The previously observed B-spline-associated failure is therefore not attributable to B-spline alone.",
        "- Large failures after VAE reconstruction are seed-sensitive in the replayed differential-surface and rolling cases: seed 24001 changes sharply, while seeds 1234 and 34007 remain close to normal edits.",
        "- Multi-geometry combinations can redirect or cancel a delta-z-induced failure. This is an interaction effect; it is not evidence that the same geometry produces a stable failure without delta-z.",
        "- Aggressive families such as Mobius, differential-surface deformation, or the all-components combination must be interpreted with their input SSIM because their stronger edit changes can coincide with visible input deformation.",
        "",
        "The CSV files beside this document retain all normalized rows and exact image paths for audit.",
    ])
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--family-root",
        type=Path,
        default=Path("targeted_experiments/geometry_family_attribution"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("targeted_experiments/geometry_family_attribution/summary"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    case_roots = sorted(
        path for path in args.family_root.iterdir()
        if path.is_dir() and path.name != "summary" and not path.name.endswith("seed_replays")
    )
    stage = [row for case in case_roots for row in stage_rows(case)]
    incremental = [row for case in case_roots for row in incremental_rows(case)]
    direct = []
    direct.extend(seed_direct_rows(args.family_root / "seed_replays", "image2_black_jacket"))
    direct.extend(seed_direct_rows(args.family_root / "image4_seed_replays", "image4_red_scarf"))
    direct_aggregate = aggregate_seed_rows(direct)
    write_csv(args.output_root / "stage_metrics.csv", stage)
    write_csv(args.output_root / "incremental_metrics.csv", incremental)
    write_csv(args.output_root / "seed_direct_effects.csv", direct)
    write_csv(args.output_root / "seed_direct_effects_aggregate.csv", direct_aggregate)
    findings = build_findings(stage, incremental, direct_aggregate)
    (args.output_root / "geometry_family_findings.md").write_text(findings, encoding="utf-8")
    payload = {
        "case_roots": [str(path) for path in case_roots],
        "stage_rows": len(stage),
        "incremental_rows": len(incremental),
        "seed_direct_rows": len(direct),
        "seed_aggregate_rows": len(direct_aggregate),
        "outputs": [
            "stage_metrics.csv",
            "incremental_metrics.csv",
            "seed_direct_effects.csv",
            "seed_direct_effects_aggregate.csv",
            "geometry_family_findings.md",
        ],
    }
    (args.output_root / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
