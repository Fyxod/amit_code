#!/usr/bin/env python
"""Normalize and summarize the longer geometry depth/strength controls."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def number(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def fmt(value: Any) -> str:
    parsed = number(value)
    return "n/a" if parsed is None else f"{parsed:.4f}"


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


def collect(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    stage_rows: list[dict[str, Any]] = []
    incremental_rows: list[dict[str, Any]] = []
    for run_dir in sorted(path for path in root.iterdir() if path.is_dir() and path.name != "summary"):
        if "__" not in run_dir.name:
            continue
        case, control = run_dir.name.split("__", 1)
        summary_path = run_dir / "summary.json"
        launcher_path = run_dir / "launcher_config.json"
        if not summary_path.exists() or not launcher_path.exists():
            continue
        summary = load_json(summary_path)
        launcher = load_json(launcher_path)
        control_config = launcher.get("control", {})
        wanted = {
            ("geometry_only_original", "learned_geometry_on_original"),
            ("geometry_only_reconstruction", "learned_geometry_on_reconstruction"),
            ("combined", "learned_delta_only"),
            ("combined", "learned_geometry_on_original"),
            ("combined", "learned_geometry_on_reconstruction"),
            ("combined", "learned_combined"),
        }
        for row in read_csv(run_dir / "stage_edit_metrics.csv"):
            if (row.get("mode", ""), row.get("stage", "")) not in wanted:
                continue
            stage_rows.append({
                "case": case,
                "control": control,
                "warp_type": summary.get("warp_type"),
                "warp_components": "+".join(summary.get("warp_components", [])),
                "configured_iterations": control_config.get("iterations"),
                "configured_scale": control_config.get("scale"),
                "mode": row.get("mode"),
                "stage": row.get("stage"),
                "input_ssim": number(row.get("input_ssim")),
                "input_psnr": number(row.get("input_psnr")),
                "output_ssim_vs_clean": number(row.get("output_ssim")),
                "output_psnr_vs_clean": number(row.get("output_psnr")),
                "output_l2_vs_clean": number(row.get("output_l2")),
                "arcface_clean_vs_stage_edit": number(row.get("arcface_clean_edit_vs_stage_edit")),
                "stage_path": row.get("stage_path"),
                "edit_path": row.get("edit_path"),
            })
        for row in read_csv(run_dir / "incremental_attribution_metrics.csv"):
            if row.get("comparison") not in {
                "geometry_on_original_effect",
                "geometry_on_reconstruction_effect",
                "delta_increment_after_reconstruction",
                "geometry_increment_after_delta",
            }:
                continue
            incremental_rows.append({
                "case": case,
                "control": control,
                "warp_type": summary.get("warp_type"),
                "configured_iterations": control_config.get("iterations"),
                "configured_scale": control_config.get("scale"),
                "mode": row.get("mode"),
                "comparison": row.get("comparison"),
                "edit_ssim": number(row.get("edit_ssim")),
                "edit_psnr": number(row.get("edit_psnr")),
                "edit_l2": number(row.get("edit_l2")),
                "arcface_edit_cosine_similarity": number(row.get("arcface_edit_cosine_similarity")),
                "left_edit_path": row.get("left_edit_path"),
                "right_edit_path": row.get("right_edit_path"),
            })
    return stage_rows, incremental_rows


def markdown(stage_rows: list[dict[str, Any]], incremental_rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Geometry depth and strength controls",
        "",
        "These controls extend the 20-iteration family sweep without changing its artifacts.",
        "",
    ]
    for case in sorted({str(row["case"]) for row in stage_rows}):
        lines.extend([
            f"## {case}",
            "",
            "| Control | Mode | Stage | Input SSIM | Edit SSIM vs clean | ArcFace clean vs stage edit |",
            "|---|---|---|---:|---:|---:|",
        ])
        rows = [row for row in stage_rows if row["case"] == case]
        for row in rows:
            lines.append(
                f"| {row['control']} | {row['mode']} | {row['stage']} | {fmt(row['input_ssim'])} | "
                f"{fmt(row['output_ssim_vs_clean'])} | {fmt(row['arcface_clean_vs_stage_edit'])} |"
            )
        lines.append("")
    lines.extend([
        "## Direct stage increments",
        "",
        "| Case | Control | Comparison | Edit SSIM | ArcFace edit similarity |",
        "|---|---|---|---:|---:|",
    ])
    for row in incremental_rows:
        lines.append(
            f"| {row['case']} | {row['control']} | {row['comparison']} | {fmt(row['edit_ssim'])} | "
            f"{fmt(row['arcface_edit_cosine_similarity'])} |"
        )
    lines.extend([
        "",
        "Interpret edit changes together with input SSIM. Controls with severe input damage are not evidence of an imperceptible geometric effect.",
        "",
    ])
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("targeted_experiments/geometry_depth_strength_controls"),
    )
    parser.add_argument("--output-root", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output_root or args.results_root / "summary"
    output.mkdir(parents=True, exist_ok=True)
    stage_rows, incremental_rows = collect(args.results_root)
    write_csv(output / "stage_metrics.csv", stage_rows)
    write_csv(output / "incremental_metrics.csv", incremental_rows)
    (output / "findings.md").write_text(markdown(stage_rows, incremental_rows), encoding="utf-8")
    payload = {
        "completed_runs": len({(row["case"], row["control"]) for row in stage_rows}),
        "stage_rows": len(stage_rows),
        "incremental_rows": len(incremental_rows),
    }
    (output / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
