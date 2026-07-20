#!/usr/bin/env python
"""Build the verified InstructPix2Pix stage-attribution report.

The builder reads saved artifacts only. It deliberately uses an explicit
manifest of valid runs so superseded pilots cannot leak into the analysis.
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
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


PRIMARY_RUNS = [
    {
        "id": "image1_blue_mask_tps_20_corrected_v2",
        "label": "Image 1 - add a blue medical mask",
        "verdict": "The clean baseline is already an imperfect mask edit. Delta-z changes texture and face coverage, while learned TPS adds little beyond the delta-z replay.",
    },
    {
        "id": "image2_black_jacket_lens_30",
        "label": "Image 2 - add a black jacket",
        "verdict": "The clearest transition: latent delta-z produces a seed-sensitive hood-like collapse after 20-30 iterations. Geometry on the original is negligible.",
    },
    {
        "id": "image4_red_scarf_polar_25_corrected",
        "label": "Image 4 - add a red scarf",
        "verdict": "The requested scarf remains visible. Delta-z mainly changes texture and color; corrected polar geometry modestly changes accessory placement.",
    },
]

EXCLUDED_RUNS = [
    {
        "id": "pilot3_image1_blue_mask_bspline_10",
        "reason": "Excluded: clean edit used a 256x256 source while ablation stages used a 512x512 normalized source.",
    },
    {
        "id": "image1_blue_mask_tps_20_corrected",
        "reason": "Excluded: superseded by corrected_v2 after the same clean-baseline resolution mismatch was fixed.",
    },
    {
        "id": "image4_red_scarf_polar_25",
        "reason": "Excluded: zero-state polar warp was not an identity transform; replaced by the corrected polar run.",
    },
]

SEED_REPLAYS = {
    "Image 2 - black jacket": "seed_replay_image2_black_jacket",
    "Image 4 - red scarf": "seed_replay_image4_red_scarf",
}

ITERATION_ROOT = "delta_iteration_sweep_image2_jacket"
LOSS_ROOT = "loss_controls_image2_jacket"

COMPARISON_LABELS = {
    "reconstruction_effect": "VAE reconstruction",
    "neutral_geometry_resampling_on_original": "Neutral resampling on original",
    "neutral_geometry_resampling_after_reconstruction": "Neutral resampling after VAE",
    "delta_increment_after_reconstruction": "Latent delta-z after VAE",
    "geometry_on_original_effect": "Learned geometry on original",
    "geometry_on_reconstruction_effect": "Learned geometry after VAE",
    "geometry_increment_after_delta": "Geometry after latent delta-z",
    "order_effect": "Operation-order change",
}

ATTRIBUTION_MODES = {
    "reconstruction_effect": "baseline_control",
    "delta_increment_after_reconstruction": "delta_only",
    "geometry_on_original_effect": "geometry_only_original",
    "geometry_on_reconstruction_effect": "geometry_only_reconstruction",
    "geometry_increment_after_delta": "combined",
    "order_effect": "combined",
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


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def load_primary(results_root: Path) -> list[dict[str, Any]]:
    runs = []
    for item in PRIMARY_RUNS:
        root = require(results_root / item["id"])
        require(root / "DONE.json")
        run = dict(item)
        run.update(
            root=root,
            config=read_json(require(root / "config_resolved.json")),
            stages=read_csv(require(root / "stage_edit_metrics.csv")),
            incremental=read_csv(require(root / "incremental_attribution_metrics.csv")),
        )
        runs.append(run)
    return runs


def select_increment(run: dict[str, Any], comparison: str) -> dict[str, Any] | None:
    wanted = ATTRIBUTION_MODES[comparison]
    rows = [r for r in run["incremental"] if r.get("comparison") == comparison]
    exact = [r for r in rows if r.get("mode") == wanted]
    return exact[0] if exact else (rows[0] if rows else None)


def stage_row(root: Path, stage: str, mode: str = "delta_only") -> dict[str, Any]:
    rows = read_csv(require(root / "stage_edit_metrics.csv"))
    matches = [r for r in rows if r.get("stage") == stage and r.get("mode") == mode]
    if not matches:
        raise RuntimeError(f"Missing stage={stage} mode={mode} in {root}")
    return matches[0]


def iteration_rows(results_root: Path) -> list[dict[str, Any]]:
    roots = [(1, results_root / ITERATION_ROOT / "iter_1"), (5, results_root / ITERATION_ROOT / "iter_5"), (10, results_root / ITERATION_ROOT / "iter_10"), (20, results_root / ITERATION_ROOT / "iter_20"), (30, results_root / "image2_black_jacket_lens_30")]
    out = []
    for iterations, root in roots:
        delta = stage_row(root, "learned_delta_only")
        history = read_csv(require(root / "delta_only" / "history.csv"))
        inc = read_csv(require(root / "incremental_attribution_metrics.csv"))
        delta_inc = next(r for r in inc if r.get("comparison") == "delta_increment_after_reconstruction" and r.get("mode") == "delta_only")
        out.append({
            "iterations": iterations,
            "final_loss": float(history[-1]["loss"]),
            "input_ssim": float(delta["input_ssim"]),
            "input_arcface": float(delta["arcface_original_vs_stage"]),
            "clean_vs_delta_edit_ssim": float(delta["output_ssim"]),
            "clean_vs_delta_edit_arcface": float(delta["arcface_clean_edit_vs_stage_edit"]),
            "delta_increment_ssim": float(delta_inc["edit_ssim"]),
            "delta_increment_arcface": float(delta_inc["arcface_edit_cosine_similarity"]),
            "root": str(root),
        })
    return out


def loss_rows(results_root: Path) -> list[dict[str, Any]]:
    roots = [
        ("Default", results_root / "image2_black_jacket_lens_30"),
        ("Identity term disabled", results_root / LOSS_ROOT / "no_identity"),
        ("Larger identity and preservation weights", results_root / LOSS_ROOT / "strong_preservation"),
    ]
    out = []
    for label, root in roots:
        cfg = read_json(require(root / "config_resolved.json"))
        delta = stage_row(root, "learned_delta_only")
        history = read_csv(require(root / "delta_only" / "history.csv"))
        out.append({
            "label": label,
            "identity_weight": float(cfg["identity_weight"]),
            "ssim_weight": float(cfg["ssim_weight"]),
            "pixel_weight": float(cfg["pixel_weight"]),
            "final_loss": float(history[-1]["loss"]),
            "input_ssim": float(delta["input_ssim"]),
            "input_arcface": float(delta["arcface_original_vs_stage"]),
            "output_ssim": float(delta["output_ssim"]),
            "output_arcface": float(delta["arcface_clean_edit_vs_stage_edit"]),
            "root": str(root),
        })
    return out


def seed_rows(results_root: Path) -> dict[str, list[dict[str, Any]]]:
    result = {}
    for label, folder in SEED_REPLAYS.items():
        result[label] = read_csv(require(results_root / folder / "seed_replay_metrics.csv"))
    return result


def make_graphs(primary: list[dict[str, Any]], iterations: list[dict[str, Any]], losses: list[dict[str, Any]], seeds: dict[str, list[dict[str, Any]]], assets: Path) -> list[Path]:
    assets.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 12, "axes.labelsize": 9})

    comparisons = ["reconstruction_effect", "delta_increment_after_reconstruction", "geometry_on_original_effect", "geometry_on_reconstruction_effect", "geometry_increment_after_delta"]
    x = np.arange(len(primary))
    width = 0.15
    fig, ax = plt.subplots(figsize=(11.8, 5.6))
    for idx, comparison in enumerate(comparisons):
        values = []
        for run in primary:
            row = select_increment(run, comparison)
            values.append(1.0 - float(row["edit_ssim"]) if row else np.nan)
        ax.bar(x + (idx - 2) * width, values, width, label=COMPARISON_LABELS[comparison])
    ax.set_xticks(x, [r["label"] for r in primary])
    ax.set_ylabel("Edited-output disruption (1 - SSIM)")
    ax.set_title("Incremental attribution by pipeline stage")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    path = assets / "incremental_stage_attribution.png"
    fig.savefig(path, dpi=190)
    plt.close(fig)
    paths.append(path)

    xs = [r["iterations"] for r in iterations]
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.3))
    axes[0].plot(xs, [r["input_ssim"] for r in iterations], "o-", label="Input SSIM")
    axes[0].plot(xs, [r["input_arcface"] for r in iterations], "o-", label="Input ArcFace")
    axes[0].set_title("Input preservation vs optimization depth")
    axes[1].plot(xs, [r["clean_vs_delta_edit_ssim"] for r in iterations], "o-", label="Clean vs delta edit SSIM")
    axes[1].plot(xs, [r["clean_vs_delta_edit_arcface"] for r in iterations], "o-", label="Clean vs delta edit ArcFace")
    axes[1].set_title("Edited-output similarity vs optimization depth")
    for ax in axes:
        ax.set_xlabel("Iterations")
        ax.set_ylabel("Similarity")
        ax.set_ylim(-0.65, 1.05)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.tight_layout()
    path = assets / "iteration_sweep.png"
    fig.savefig(path, dpi=190)
    plt.close(fig)
    paths.append(path)

    labels = [r["label"] for r in losses]
    xx = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(11.8, 4.7))
    values = [r["input_ssim"] for r in losses]
    values2 = [r["output_ssim"] for r in losses]
    values3 = [r["input_arcface"] for r in losses]
    w = 0.25
    ax.bar(xx - w, values, w, label="Input SSIM")
    ax.bar(xx, values2, w, label="Clean vs delta edit SSIM")
    ax.bar(xx + w, values3, w, label="Input ArcFace")
    ax.set_xticks(xx, labels)
    ax.set_ylim(-0.65, 1.05)
    ax.set_ylabel("Similarity")
    ax.set_title("Loss-term controls at 30 iterations")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    path = assets / "loss_controls.png"
    fig.savefig(path, dpi=190)
    plt.close(fig)
    paths.append(path)

    for label, rows in seeds.items():
        stages = ["vae_reconstruction", "delta_only", "geometry_only_original", "geometry_only_reconstruction", "combined"]
        seed_values = sorted({int(r["seed"]) for r in rows})
        fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.7))
        for stage in stages:
            stage_rows = [r for r in rows if r.get("stage") == stage]
            by_seed = {int(r["seed"]): r for r in stage_rows}
            axes[0].plot(seed_values, [float(by_seed[s]["output_ssim"]) for s in seed_values], "o-", label=stage.replace("_", " "))
            axes[1].plot(seed_values, [float(by_seed[s]["arcface_clean_edit_vs_stage_edit"]) for s in seed_values], "o-", label=stage.replace("_", " "))
        axes[0].set_title("Edited-output SSIM across seeds")
        axes[1].set_title("Edited-output ArcFace across seeds")
        for ax in axes:
            ax.set_xlabel("Seed")
            ax.set_ylabel("Similarity")
            ax.set_ylim(-0.3, 1.05)
            ax.grid(alpha=0.25)
        axes[0].legend(fontsize=7, ncol=2)
        fig.suptitle(label)
        fig.tight_layout()
        path = assets / f"seed_replay_{label.lower().replace(' ', '_').replace('-', '')}.png"
        fig.savefig(path, dpi=190)
        plt.close(fig)
        paths.append(path)
    return paths


def scaled_image(path: Path, max_width: float, max_height: float) -> Image:
    with PILImage.open(path) as im:
        width, height = im.size
    scale = min(max_width / width, max_height / height)
    return Image(str(path), width=width * scale, height=height * scale)


def para_style(name: str, size: float, leading: float, color: str = "#20354A", bold: bool = False, align: int = TA_LEFT, before: float = 0, after: float = 4) -> ParagraphStyle:
    return ParagraphStyle(name, fontName="Helvetica-Bold" if bold else "Helvetica", fontSize=size, leading=leading, textColor=colors.HexColor(color), alignment=align, spaceBefore=before, spaceAfter=after)


def styled_table(data: list[list[Any]], widths: list[float], font_size: float = 7.2) -> Table:
    cell = para_style("cell", font_size, font_size + 2, color="#1D2A36", after=0)
    rows = [[Paragraph(str(value), cell) for value in row] for row in data]
    result = Table(rows, colWidths=widths, repeatRows=1)
    result.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17375E")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#A9B7C5")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F2F6F9")]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
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
    canvas.drawString(13 * mm, 7 * mm, "InstructPix2Pix pipeline-stage attribution")
    canvas.drawRightString(284 * mm, 7 * mm, f"Page {doc.page}")
    canvas.restoreState()


def fmt(value: Any, digits: int = 3) -> str:
    try:
        number = float(value)
        return "inf" if math.isinf(number) else f"{number:.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def representative_strip(root: Path, mode: str, stem: str) -> Path:
    return require(root / mode / f"{stem}_comparison.jpg")


def build_report(results_root: Path, output_root: Path) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    assets = output_root / "assets"
    primary = load_primary(results_root)
    iterations = iteration_rows(results_root)
    losses = loss_rows(results_root)
    seeds = seed_rows(results_root)
    graphs = make_graphs(primary, iterations, losses, seeds, assets)

    attribution_rows: list[dict[str, Any]] = []
    for run in primary:
        for comparison in ATTRIBUTION_MODES:
            row = select_increment(run, comparison)
            if row:
                attribution_rows.append({"case": run["label"], **row})
    write_csv(output_root / "valid_primary_attribution_metrics.csv", attribution_rows)
    write_csv(output_root / "iteration_sweep.csv", iterations)
    write_csv(output_root / "loss_controls.csv", losses)
    write_csv(output_root / "seed_replay_metrics.csv", [dict(case=label, **row) for label, rows in seeds.items() for row in rows])

    title = para_style("title", 25, 29, color="#17375E", bold=True, align=TA_CENTER)
    subtitle = para_style("subtitle", 12, 16, color="#40566B", align=TA_CENTER)
    h1 = para_style("h1", 17, 21, color="#17375E", bold=True, before=6, after=7)
    h2 = para_style("h2", 12, 15, color="#2E5D87", bold=True, before=4, after=5)
    body = para_style("body", 9, 13, color="#253746", after=5)
    small = para_style("small", 7.4, 9.5, color="#40566B", after=3)
    callout = para_style("callout", 10, 14, color="#17375E", bold=True, after=5)

    pdf_path = output_root / "instruct_pipeline_stage_attribution_report.pdf"
    doc = SimpleDocTemplate(str(pdf_path), pagesize=landscape(A4), leftMargin=13 * mm, rightMargin=13 * mm, topMargin=12 * mm, bottomMargin=13 * mm, title="InstructPix2Pix Pipeline-Stage Attribution")
    story: list[Any] = []

    story += [Spacer(1, 19 * mm), Paragraph("InstructPix2Pix Pipeline-Stage Attribution", title), Spacer(1, 4 * mm), Paragraph("Controlled ablations of VAE reconstruction, latent delta-z, geometric perturbation, composition order, seeds, optimization depth, and loss terms", subtitle), Spacer(1, 10 * mm)]
    story.append(styled_table([
        ["Valid primary cases", "Editor seeds replayed", "Iteration depths", "Loss settings"],
        [len(primary), sum(len({int(r['seed']) for r in rows}) for rows in seeds.values()), "1, 5, 10, 20, 30", len(losses)],
    ], [62 * mm] * 4, 9))
    story += [Spacer(1, 9 * mm), Paragraph("Central attribution", h2), Paragraph("Across the corrected runs, the largest learned effect comes from latent delta-z. VAE encode/decode reconstruction creates a smaller baseline shift. Learned geometry on the original is usually minor, while geometry after reconstruction can be amplified by diffusion seed sensitivity. The extreme black-jacket collapse appears only after deeper latent optimization and is not a geometry-only result.", callout), Paragraph("All findings below use saved images and exact within-case prompt, seed, diffusion-step, guidance, SSIM, PSNR, L2, and ArcFace iResNet-100 records. Visual review takes precedence when a face-identity embedding conflicts with the visible result.", body), PageBreak()]

    story += [Paragraph("1. Controlled pipeline", h1), Paragraph("The stock branch edits the normalized original image. The ablation branches separately replay: VAE encode/decode reconstruction; optimized latent delta-z; learned geometry on the original; learned geometry on the reconstruction; the current combined order geometry(decode(encode(x) + delta-z)); and the order control decode(encode(geometry(x))). The editor settings are held fixed within a comparison.", body)]
    story.append(styled_table([
        ["Stage", "Editor input", "What it isolates"],
        ["Stock", "x", "Reference edit"],
        ["VAE reconstruction", "decode(encode(x))", "Encode/decode reconstruction effect"],
        ["Latent delta-z", "decode(encode(x) + delta-z)", "Optimized latent contribution"],
        ["Geometry on original", "geometry(x)", "Learned geometry without VAE reconstruction"],
        ["Geometry after VAE", "geometry(decode(encode(x)))", "Geometry acting on reconstructed pixels"],
        ["Combined", "geometry(decode(encode(x) + delta-z))", "Current latent plus geometry path"],
        ["Order control", "decode(encode(geometry(x)))", "Sensitivity to operation order"],
    ], [49 * mm, 93 * mm, 114 * mm], 8))
    story += [Spacer(1, 6 * mm), Paragraph("Validity controls", h2), Paragraph("The normalized 512x512 original saved inside each case is used for both stock editing and stage comparison. Neutral warps were audited at 512x512. Polar and TPS coordinate bugs were fixed before the corrected runs. A neutral grid-sample round trip can still change saved pixels by one 8-bit level and can therefore cause a small diffusion-output change; this is reported as a resampling floor, not as learned geometry.", body), PageBreak()]

    story += [Paragraph("2. Primary attribution summary", h1), scaled_image(assets / "incremental_stage_attribution.png", 258 * mm, 118 * mm), Spacer(1, 3 * mm)]
    summary_table = [["Case", "VAE recon SSIM", "Delta increment SSIM", "Geometry-original SSIM", "Geometry-after-VAE SSIM", "Geometry-after-delta SSIM"]]
    for run in primary:
        vals = [select_increment(run, key) for key in ["reconstruction_effect", "delta_increment_after_reconstruction", "geometry_on_original_effect", "geometry_on_reconstruction_effect", "geometry_increment_after_delta"]]
        summary_table.append([run["label"], *[fmt(v["edit_ssim"]) if v else "missing" for v in vals]])
    story.append(styled_table(summary_table, [68 * mm, 35 * mm, 38 * mm, 41 * mm, 41 * mm, 41 * mm], 7.2))
    story.append(PageBreak())

    for idx, run in enumerate(primary, start=1):
        cfg = run["config"]
        root = run["root"]
        story += [Paragraph(f"3.{idx} {run['label']}", h1), Paragraph(f"Warp: {cfg.get('warp_type')} | iterations: {cfg.get('iterations')} | seed: {cfg.get('seed')} | image guidance: {cfg.get('image_guidance_scale')}", body), Paragraph(run["verdict"], callout)]
        delta_strip = representative_strip(root, "delta_only", "learned_delta_only")
        story.append(KeepTogether([Paragraph("Latent delta-z replay", h2), scaled_image(delta_strip, 258 * mm, 72 * mm)]))
        geometry_mode = "geometry_only_reconstruction" if run["id"].startswith("image2") else "geometry_only_original"
        geometry_stem = "learned_geometry_on_reconstruction" if geometry_mode.endswith("reconstruction") else "learned_geometry_on_original"
        geometry_strip = representative_strip(root, geometry_mode, geometry_stem)
        story.append(KeepTogether([Paragraph("Independent geometry replay", h2), scaled_image(geometry_strip, 258 * mm, 72 * mm)]))
        story.append(PageBreak())

        rows = [["Increment", "Mode", "Edit SSIM", "Edit PSNR", "Edit L2", "ArcFace edit similarity"]]
        for comparison in ATTRIBUTION_MODES:
            row = select_increment(run, comparison)
            if row:
                rows.append([COMPARISON_LABELS[comparison], row["mode"], fmt(row["edit_ssim"]), fmt(row["edit_psnr"], 2), fmt(row["edit_l2"]), fmt(row["arcface_edit_cosine_similarity"])])
        story += [Paragraph(f"3.{idx}.1 Incremental metrics", h2), styled_table(rows, [73 * mm, 43 * mm, 29 * mm, 29 * mm, 26 * mm, 45 * mm], 7.2)]
        combined = root / "combined" / "learned_combined_comparison.jpg"
        if combined.exists():
            story += [Spacer(1, 4 * mm), KeepTogether([Paragraph("Combined latent delta-z plus geometry", h2), scaled_image(combined, 258 * mm, 82 * mm)])]
        story.append(PageBreak())

    story += [Paragraph("4. Optimization-depth control", h1), Paragraph("Image 2, black-jacket prompt, seed 24001, delta-z only. The 20-iteration point is where the saved edit changes abruptly from a recognizable jacket edit to a hood-like collapse. Input identity similarity has already degraded strongly by 10 iterations; input SSIM continues to decline more gradually.", body), scaled_image(assets / "iteration_sweep.png", 258 * mm, 93 * mm)]
    iter_table = [["Iterations", "Final loss", "Input SSIM", "Input ArcFace", "Clean vs delta edit SSIM", "Clean vs delta edit ArcFace"]]
    for row in iterations:
        iter_table.append([row["iterations"], fmt(row["final_loss"]), fmt(row["input_ssim"]), fmt(row["input_arcface"]), fmt(row["clean_vs_delta_edit_ssim"]), fmt(row["clean_vs_delta_edit_arcface"])])
    story.append(styled_table(iter_table, [32 * mm, 38 * mm, 38 * mm, 39 * mm, 54 * mm, 54 * mm], 7.5))
    story.append(PageBreak())

    story += [Paragraph("4.1 Visual transition across iteration depth", h1)]
    for it in (5, 10, 20, 30):
        root = results_root / ITERATION_ROOT / f"iter_{it}" if it < 30 else results_root / "image2_black_jacket_lens_30"
        story.append(KeepTogether([Paragraph(f"{it} iterations", h2), scaled_image(representative_strip(root, "delta_only", "learned_delta_only"), 258 * mm, 48 * mm)]))
    story.append(PageBreak())

    story += [Paragraph("5. Loss-term controls", h1), Paragraph("The optimization loss includes a negative identity-distance term, so a larger identity weight rewards identity disruption rather than preservation. The no-identity control removes that driver. The third control increases both the identity-disruption weight and the SSIM/pixel penalties; its high identity weight still dominates and produces collapse.", body), scaled_image(assets / "loss_controls.png", 258 * mm, 91 * mm)]
    loss_table = [["Setting", "Identity w.", "SSIM w.", "Pixel w.", "Input SSIM", "Input ArcFace", "Edit SSIM", "Edit ArcFace"]]
    for row in losses:
        loss_table.append([row["label"], fmt(row["identity_weight"], 1), fmt(row["ssim_weight"], 1), fmt(row["pixel_weight"], 1), fmt(row["input_ssim"]), fmt(row["input_arcface"]), fmt(row["output_ssim"]), fmt(row["output_arcface"])])
    story.append(styled_table(loss_table, [62 * mm, 27 * mm, 27 * mm, 27 * mm, 29 * mm, 31 * mm, 29 * mm, 31 * mm], 7.1))
    story += [Spacer(1, 5 * mm), Paragraph("Result", callout), Paragraph("With identity weight set to zero, the 30-iteration output remains close to the VAE-reconstruction edit (clean-vs-stage edit SSIM 0.891; ArcFace 0.955). This isolates the identity-disruption objective as the cause of the large latent change in the default run.", body), PageBreak()]

    story += [Paragraph("5.1 Loss-control image comparison", h1)]
    for label, folder in [("Identity term disabled", "no_identity"), ("Larger identity and preservation weights", "strong_preservation")]:
        root = results_root / LOSS_ROOT / folder
        story.append(KeepTogether([Paragraph(label, h2), scaled_image(representative_strip(root, "delta_only", "learned_delta_only"), 258 * mm, 75 * mm)]))
    story.append(PageBreak())

    story += [Paragraph("6. Seed sensitivity", h1), Paragraph("Saved stage inputs were replayed with three diffusion seeds without rerunning optimization. Image 2 shows a consistent delta-z effect, but the extreme hood collapse is concentrated at seed 24001. Image 4 keeps the requested scarf across all seeds; its low or negative ArcFace values conflict with the visible face and are treated as an embedding failure under global recoloring, not as visual proof of identity loss.", body)]
    story += [KeepTogether([Paragraph("Image 2 - black jacket", h2), scaled_image(assets / "seed_replay_image_2__black_jacket.png", 258 * mm, 82 * mm)]), KeepTogether([Paragraph("Image 4 - red scarf", h2), scaled_image(assets / "seed_replay_image_4__red_scarf.png", 258 * mm, 82 * mm)]), PageBreak()]

    for label, folder in SEED_REPLAYS.items():
        story += [Paragraph(f"6.1 Representative seed replays - {label}", h1)]
        for seed in (1234, 24001, 34007):
            strip = results_root / folder / f"seed_{seed}" / "delta_only_comparison.jpg"
            story.append(KeepTogether([Paragraph(f"Seed {seed} - latent delta-z", h2), scaled_image(require(strip), 258 * mm, 52 * mm)]))
        story.append(PageBreak())

    story += [Paragraph("7. Correctness caveats and excluded runs", h1), Paragraph("The report does not average every DONE folder. It uses an explicit valid-run manifest and records why superseded results are excluded.", body)]
    excluded_table = [["Excluded folder", "Reason"]] + [[item["id"], item["reason"]] for item in EXCLUDED_RUNS]
    story.append(styled_table(excluded_table, [78 * mm, 178 * mm], 7.7))
    story += [Spacer(1, 7 * mm), Paragraph("Neutral-warp audit", h2), Paragraph("After correction, all 13 audited warps passed at 512x512. TPS and polar zero states now produce identity-coordinate grids to floating-point tolerance. Saving a neutral grid-sampled image to 8-bit PNG can still create one-level pixel changes, so neutral edit differences are retained as resampling controls.", body), Paragraph("Metric caveat", h2), Paragraph("ArcFace iResNet-100 is informative for ordinary face crops but can become unreliable when the edit recolors or occludes most of the face. Negative ArcFace similarity is therefore not called an identity failure unless it agrees with the visible images. SSIM and PSNR are also similarity diagnostics, not semantic success labels.", body), PageBreak()]

    story += [Paragraph("8. Evidence-backed conclusions", h1)]
    conclusions = [
        "VAE reconstruction alone causes a measurable but usually moderate edit shift.",
        "Latent delta-z is the dominant optimized source of the observed InstructPix2Pix changes in the corrected cases.",
        "Learned geometry on the original is usually close to the neutral-resampling control; it does not explain the black-jacket collapse.",
        "Geometry after reconstruction can be amplified by editor sensitivity, but its effect is more seed-dependent and smaller than delta-z in the strongest case.",
        "The large black-jacket collapse begins between 10 and 20 optimization iterations and is most extreme at seed 24001.",
        "Removing the identity-disruption term prevents the large latent collapse at 30 iterations.",
        "The strongest output disruptions coincide with visible input degradation, so they are not evidence of an imperceptible geometric attack.",
        "The red-scarf and blue-mask prompts remain broadly successful; their metric changes are not convincing edit failures.",
    ]
    for idx, finding in enumerate(conclusions, start=1):
        story.append(Paragraph(f"{idx}. {finding}", callout if idx in (2, 3, 5, 6) else body))
    story += [Spacer(1, 7 * mm), Paragraph("Final attribution", h2), Paragraph("The observed InstructPix2Pix edit changes are primarily produced by optimized latent delta-z, with smaller baseline effects from VAE reconstruction and context-dependent amplification from reconstruction plus resampling or geometry. The current evidence does not support learned geometry alone as the main cause.", callout), Paragraph(f"Artifacts collected from: {results_root}", small)]

    doc.build(story, onFirstPage=footer, onLaterPages=footer)

    markdown = [
        "# InstructPix2Pix Pipeline-Stage Attribution",
        "",
        "## Final attribution",
        "",
        "The observed edit changes are primarily produced by optimized latent delta-z. VAE reconstruction creates a smaller baseline shift. Learned geometry on the original is usually minor; geometry after reconstruction is seed-sensitive and can be amplified by the editor.",
        "",
        "## Corrected primary cases",
        "",
        *[f"- **{run['label']}**: {run['verdict']}" for run in primary],
        "",
        "## Controls",
        "",
        "- Optimization depths: 1, 5, 10, 20, 30 iterations.",
        "- Diffusion seeds: 1234, 24001, 34007 for two cases.",
        "- Loss controls: default, identity term disabled, and larger identity plus preservation weights.",
        "- Neutral geometry and operation-order controls are included.",
        "",
        "## Excluded superseded runs",
        "",
        *[f"- `{item['id']}`: {item['reason']}" for item in EXCLUDED_RUNS],
        "",
        "## Files",
        "",
        "- `instruct_pipeline_stage_attribution_report.pdf`",
        "- `valid_primary_attribution_metrics.csv`",
        "- `iteration_sweep.csv`",
        "- `loss_controls.csv`",
        "- `seed_replay_metrics.csv`",
    ]
    (output_root / "report.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    data_summary = {
        "primary_valid_runs": [r["id"] for r in primary],
        "excluded_runs": EXCLUDED_RUNS,
        "iteration_depths": [r["iterations"] for r in iterations],
        "loss_controls": [r["label"] for r in losses],
        "seed_replays": {label: sorted({int(r["seed"]) for r in rows}) for label, rows in seeds.items()},
        "graphs": [str(path) for path in graphs],
        "pdf": str(pdf_path),
        "conclusion": "Latent delta-z is the dominant optimized source; learned geometry alone is not the main cause in the corrected runs.",
    }
    (output_root / "report_data_summary.json").write_text(json.dumps(data_summary, indent=2), encoding="utf-8")
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
