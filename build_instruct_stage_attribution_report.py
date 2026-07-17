#!/usr/bin/env python
"""Build the final InstructPix2Pix pipeline-stage attribution PDF.

The report is generated only from saved artifacts.  It does not rerun an
optimizer, editor, or identity model.  Run directories without DONE.json and
the required metric CSVs are listed as incomplete rather than silently used.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


STAGE_LABELS = {
    "vae_reconstruction": "VAE reconstruction",
    "learned_delta_only": "Learned delta-z only",
    "learned_geometry_on_original": "Geometry on original",
    "learned_geometry_on_reconstruction": "Geometry on reconstruction",
    "learned_combined": "Combined delta-z + geometry",
    "geometry_then_vae_reconstruction": "Geometry then VAE reconstruction",
}

COMPARISON_LABELS = {
    "reconstruction_effect": "Original edit -> VAE reconstruction edit",
    "delta_increment_after_reconstruction": "VAE reconstruction edit -> delta-z edit",
    "geometry_on_original_effect": "Original edit -> geometry(original) edit",
    "geometry_on_reconstruction_effect": "VAE reconstruction edit -> geometry(reconstruction) edit",
    "geometry_increment_after_delta": "Delta-z edit -> combined edit",
    "order_effect": "Combined edit -> geometry-then-VAE edit",
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key, value in list(row.items()):
            try:
                row[key] = float(value)
            except (TypeError, ValueError):
                pass
    return rows


def discover_runs(results_root: Path) -> tuple[list[dict[str, Any]], list[str]]:
    runs: list[dict[str, Any]] = []
    incomplete: list[str] = []
    for done_path in sorted(results_root.rglob("DONE.json")):
        root = done_path.parent
        stage_csv = root / "stage_edit_metrics.csv"
        incremental_csv = root / "incremental_attribution_metrics.csv"
        config_path = root / "config_resolved.json"
        if not stage_csv.exists() or not config_path.exists():
            incomplete.append(str(root))
            continue
        config = read_json(config_path)
        summary = read_json(done_path)
        runs.append({
            "root": root,
            "id": str(root.relative_to(results_root)),
            "config": config,
            "summary": summary,
            "stages": read_csv(stage_csv),
            "incremental": read_csv(incremental_csv) if incremental_csv.exists() else [],
        })
    return runs, incomplete


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def plot_incremental(runs: list[dict[str, Any]], assets: Path) -> list[Path]:
    assets.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    rows = [row | {"run_id": run["id"]} for run in runs for row in run["incremental"]]
    if not rows:
        return paths
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["comparison"])].append(row)
    labels = list(grouped)
    visual = [np.mean([1.0 - float(row["edit_ssim"]) for row in grouped[key]]) for key in labels]
    identity = [
        np.mean([1.0 - float(row["arcface_edit_cosine_similarity"]) for row in grouped[key]])
        for key in labels
    ]
    x = np.arange(len(labels))
    width = 0.38
    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    ax.bar(x - width / 2, visual, width, label="1 - edit SSIM")
    ax.bar(x + width / 2, identity, width, label="1 - ArcFace edit similarity")
    ax.set_xticks(x, [COMPARISON_LABELS.get(label, label) for label in labels], rotation=24, ha="right")
    ax.set_ylabel("Mean dissimilarity")
    ax.set_title("Incremental edited-output change by pipeline stage")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path = assets / "incremental_stage_effects.png"
    fig.savefig(path, dpi=190)
    plt.close(fig)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    for run in runs:
        if not run["incremental"]:
            continue
        run_labels = [str(row["comparison"]) for row in run["incremental"]]
        values = [1.0 - float(row["edit_ssim"]) for row in run["incremental"]]
        ax.scatter(run_labels, values, label=run["id"], alpha=0.8)
    ax.set_xticks(labels, [COMPARISON_LABELS.get(label, label) for label in labels], rotation=24, ha="right")
    ax.set_ylabel("1 - edit SSIM")
    ax.set_title("Per-run incremental output disruption")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    path = assets / "per_run_incremental_output_change.png"
    fig.savefig(path, dpi=190)
    plt.close(fig)
    paths.append(path)
    return paths


def plot_histories(runs: list[dict[str, Any]], assets: Path) -> list[Path]:
    paths: list[Path] = []
    for run in runs:
        histories: list[tuple[str, list[dict[str, Any]]]] = []
        for history_path in sorted(run["root"].glob("*/history.csv")):
            histories.append((history_path.parent.name, read_csv(history_path)))
        if not histories:
            continue
        fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.7))
        for mode, rows in histories:
            iters = [float(row["iter"]) for row in rows]
            axes[0].plot(iters, [float(row["loss"]) for row in rows], label=mode)
            axes[1].plot(iters, [float(row["arcface_identity_similarity"]) for row in rows], label=mode)
            axes[2].plot(iters, [float(row["input_ssim"]) for row in rows], label=mode)
        for ax, title, ylabel in zip(
            axes,
            ("Optimization loss", "Input ArcFace similarity", "Input SSIM"),
            ("Loss", "Cosine similarity", "SSIM"),
        ):
            ax.set_title(title)
            ax.set_xlabel("Iteration")
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.25)
        axes[0].legend(fontsize=7)
        fig.suptitle(run["id"])
        fig.tight_layout()
        path = assets / f"history_{run['id'].replace('/', '_')}.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)
    return paths


def scaled_image(path: Path, max_width: float, max_height: float) -> Image:
    with PILImage.open(path) as image:
        width, height = image.size
    scale = min(max_width / width, max_height / height)
    return Image(str(path), width=width * scale, height=height * scale)


def table(data: list[list[Any]], widths: list[float] | None = None, font_size: int = 7) -> Table:
    converted = [[Paragraph(str(cell), ParagraphStyle("cell", fontName="Helvetica", fontSize=font_size, leading=font_size + 2)) for cell in row] for row in data]
    result = Table(converted, colWidths=widths, repeatRows=1)
    result.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17375E")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#A7B6C7")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F4F7FA")]),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return result


def footer(canvas: Any, doc: Any) -> None:
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#66788A"))
    canvas.drawString(14 * mm, 8 * mm, "InstructPix2Pix pipeline-stage attribution")
    canvas.drawRightString(283 * mm, 8 * mm, f"Page {doc.page}")
    canvas.restoreState()


def aggregate_findings(runs: list[dict[str, Any]]) -> list[str]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        for row in run["incremental"]:
            grouped[str(row["comparison"])].append(row)
    findings: list[str] = []
    if not grouped:
        return ["Incremental attribution metrics were not available."]
    ranked = sorted(
        grouped,
        key=lambda key: np.mean([1.0 - float(row["edit_ssim"]) for row in grouped[key]]),
        reverse=True,
    )
    for key in ranked:
        mean_ssim = np.mean([float(row["edit_ssim"]) for row in grouped[key]])
        mean_arc = np.mean([float(row["arcface_edit_cosine_similarity"]) for row in grouped[key]])
        findings.append(
            f"{COMPARISON_LABELS.get(key, key)}: mean edit SSIM {mean_ssim:.3f}; "
            f"mean ArcFace cosine similarity {mean_arc:.3f} across {len(grouped[key])} comparisons."
        )
    return findings


def build_report(results_root: Path, output_root: Path) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    assets = output_root / "assets"
    runs, incomplete = discover_runs(results_root)
    if not runs:
        raise RuntimeError(f"No complete stage-attribution runs found under {results_root}")
    graph_paths = plot_incremental(runs, assets) + plot_histories(runs, assets)
    all_stages = [row | {"run_id": run["id"]} for run in runs for row in run["stages"]]
    all_incremental = [row | {"run_id": run["id"]} for run in runs for row in run["incremental"]]
    save_csv(output_root / "all_stage_metrics.csv", all_stages)
    save_csv(output_root / "all_incremental_metrics.csv", all_incremental)

    styles = getSampleStyleSheet()
    title = ParagraphStyle("title", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=25, leading=29, textColor=colors.HexColor("#17375E"), alignment=TA_CENTER)
    h1 = ParagraphStyle("h1", parent=styles["Heading1"], fontName="Helvetica-Bold", fontSize=17, leading=21, textColor=colors.HexColor("#17375E"), spaceBefore=8, spaceAfter=7)
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=12, leading=15, textColor=colors.HexColor("#2E5D87"), spaceBefore=6, spaceAfter=5)
    body = ParagraphStyle("body", parent=styles["BodyText"], fontName="Helvetica", fontSize=9, leading=13, alignment=TA_LEFT, spaceAfter=5)
    small = ParagraphStyle("small", parent=body, fontSize=7.5, leading=10)

    pdf_path = output_root / "instruct_pipeline_stage_attribution_report.pdf"
    doc = SimpleDocTemplate(
        str(pdf_path), pagesize=landscape(A4), rightMargin=13 * mm,
        leftMargin=13 * mm, topMargin=13 * mm, bottomMargin=14 * mm,
        title="InstructPix2Pix Pipeline-Stage Attribution",
    )
    story: list[Any] = []
    story += [Spacer(1, 22 * mm), Paragraph("InstructPix2Pix Pipeline-Stage Attribution", title), Spacer(1, 5 * mm), Paragraph("Controlled analysis of VAE reconstruction, learned latent delta-z, geometric perturbation, their composition, and operation order", ParagraphStyle("subtitle", parent=body, fontSize=12, leading=16, alignment=TA_CENTER, textColor=colors.HexColor("#40566B"))), Spacer(1, 11 * mm)]
    story.append(table([
        ["Completed run folders", "Stage/edit rows", "Incremental comparisons", "Incomplete folders"],
        [len(runs), len(all_stages), len(all_incremental), len(incomplete)],
    ], [55 * mm] * 4, 9))
    story += [Spacer(1, 9 * mm), Paragraph("Evidence rule", h2), Paragraph("Every edited stage uses the same prompt, seed, diffusion steps, text guidance, and image guidance within its case. Claims are based on saved image artifacts plus SSIM, PSNR, L2, and exact ArcFace iResNet-100 comparisons. Missing stages are not inferred from other files.", body), PageBreak()]

    story += [Paragraph("1. Pipeline and ablation design", h1), Paragraph("The current pipeline encodes the original input with the frozen InstructPix2Pix VAE, adds a trainable latent delta-z, decodes to image space, and then applies trainable geometry. ArcFace identity distance and preservation terms optimize delta-z and/or geometry depending on the ablation mode. InstructPix2Pix is loaded only after the optimization models have been released.", body)]
    story.append(table([
        ["Ablation", "Trainable part", "Editor input"],
        ["Stock", "None", "Original image"],
        ["VAE reconstruction", "None", "decode(encode(original))"],
        ["Delta-z only", "delta-z", "decode(encode(original) + delta-z)"],
        ["Geometry only - original", "Geometry", "geometry(original)"],
        ["Geometry only - reconstruction", "Geometry", "geometry(decode(encode(original)))"],
        ["Combined", "delta-z + geometry", "geometry(decode(encode(original) + delta-z))"],
        ["Order control", "Replay", "decode(encode(geometry(original)))"],
    ], [54 * mm, 55 * mm, 145 * mm], 8))
    story += [Spacer(1, 5 * mm), Paragraph("Optimization modes are independent. Combined-run replays use the exact best state that produced the logged loss, captured before the optimizer step. This avoids associating metrics with an unevaluated post-step state.", body), Paragraph("2. Run inventory", h1)]
    inventory = [["Run", "Face", "Prompt", "Warp", "Iterations", "Seed", "IGS"]]
    for run in runs:
        cfg = run["config"]
        inventory.append([
            run["id"], cfg.get("face_id", run["summary"].get("face_id", "")),
            cfg.get("prompt", ""), cfg.get("warp_type", ""), cfg.get("iterations", ""),
            cfg.get("seed", ""), cfg.get("image_guidance_scale", ""),
        ])
    story.append(table(inventory, [52 * mm, 20 * mm, 72 * mm, 28 * mm, 24 * mm, 22 * mm, 20 * mm], 7))
    story.append(PageBreak())

    story += [Paragraph("3. Aggregate stage attribution", h1)]
    for finding in aggregate_findings(runs):
        story.append(Paragraph(f"- {finding}", body))
    for path in graph_paths[:2]:
        story += [Spacer(1, 3 * mm), scaled_image(path, 250 * mm, 79 * mm)]
    story.append(PageBreak())

    for index, run in enumerate(runs, start=1):
        cfg = run["config"]
        story += [Paragraph(f"4.{index} {run['id']}", h1), Paragraph(f"Face: {cfg.get('face_id', run['summary'].get('face_id', ''))} | Prompt: {cfg.get('prompt', '')} | Warp: {cfg.get('warp_type', '')} | Iterations: {cfg.get('iterations', '')} | Seed: {cfg.get('seed', '')} | Image guidance: {cfg.get('image_guidance_scale', '')}", body)]
        if run["incremental"]:
            data = [["Mode", "Incremental comparison", "Edit SSIM", "Edit PSNR", "Edit L2", "ArcFace similarity"]]
            for row in run["incremental"]:
                data.append([
                    row["mode"], COMPARISON_LABELS.get(str(row["comparison"]), row["comparison"]),
                    f"{float(row['edit_ssim']):.3f}", f"{float(row['edit_psnr']):.2f}",
                    f"{float(row['edit_l2']):.3f}", f"{float(row['arcface_edit_cosine_similarity']):.3f}",
                ])
            story.append(table(data, [38 * mm, 98 * mm, 24 * mm, 24 * mm, 22 * mm, 29 * mm], 6.5))
        history_graph = assets / f"history_{run['id'].replace('/', '_')}.png"
        if history_graph.exists():
            story += [Spacer(1, 4 * mm), scaled_image(history_graph, 245 * mm, 72 * mm)]
        story.append(PageBreak())

        strips = sorted(run["root"].rglob("*_comparison.jpg"))
        for strip_index in range(0, len(strips), 2):
            story.append(Paragraph(f"{run['id']} - saved stage comparisons", h2))
            for strip in strips[strip_index:strip_index + 2]:
                story.append(Paragraph(strip.stem.replace("_comparison", "").replace("_", " "), small))
                story.append(scaled_image(strip, 250 * mm, 75 * mm))
                story.append(Spacer(1, 2 * mm))
            story.append(PageBreak())

    story += [Paragraph("5. Factual limitations and artifact provenance", h1), Paragraph("The report distinguishes independent optimization modes from post-hoc replay controls. A replay shows the effect of a component state learned jointly; it does not claim that the same component would be learned by an independent optimizer. ArcFace is an identity diagnostic, while SSIM/PSNR/L2 quantify image similarity and do not alone establish semantic edit success or failure. All model conclusions should be read together with the saved image strips.", body), Paragraph(f"Source root: {results_root}", small), Paragraph(f"Incomplete folders excluded: {len(incomplete)}", small)]
    if incomplete:
        for path in incomplete:
            story.append(Paragraph(f"- {path}", small))
    doc.build(story, onFirstPage=footer, onLaterPages=footer)

    markdown = [
        "# InstructPix2Pix Pipeline-Stage Attribution",
        "",
        f"Completed run folders: {len(runs)}",
        f"Stage/edit rows: {len(all_stages)}",
        f"Incremental comparisons: {len(all_incremental)}",
        "",
        "## Aggregate findings",
        "",
        *[f"- {item}" for item in aggregate_findings(runs)],
        "",
        "## Artifacts",
        "",
        f"- PDF: `{pdf_path.name}`",
        "- `all_stage_metrics.csv`",
        "- `all_incremental_metrics.csv`",
    ]
    (output_root / "report.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    summary = {
        "completed_runs": len(runs),
        "stage_rows": len(all_stages),
        "incremental_rows": len(all_incremental),
        "incomplete": incomplete,
        "graphs": [str(path) for path in graph_paths],
        "pdf": str(pdf_path),
    }
    (output_root / "report_data_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return pdf_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path("targeted_experiments/stage_attribution"))
    parser.add_argument("--output-root", type=Path, default=Path("targeted_experiments/stage_attribution_report"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pdf = build_report(args.results_root.resolve(), args.output_root.resolve())
    print(pdf)


if __name__ == "__main__":
    main()
