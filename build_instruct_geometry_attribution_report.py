#!/usr/bin/env python
"""Build the extended InstructPix2Pix attribution report from saved artifacts.

The verified stage-attribution PDF is preserved verbatim. This builder creates
a geometry-family appendix, validates the appendix, and merges both documents
into a new extended report. No optimization or image editing is performed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image as PILImage
from pypdf import PdfReader, PdfWriter
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle
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


SINGLE_METHODS = [
    "bspline",
    "bezier",
    "delaunay",
    "rolling",
    "geodesic",
    "fft",
    "homography",
    "diffgeom",
    "mobius",
]

CASE_LABELS = {
    "image2_black_jacket": "Image 2 - add a black jacket",
    "image4_red_scarf": "Image 4 - add a red scarf",
}


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key, value in list(row.items()):
            try:
                row[key] = float(value)
            except (TypeError, ValueError):
                pass
    return rows


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


def require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def number(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def fmt(value: Any, digits: int = 3) -> str:
    parsed = number(value)
    return "n/a" if parsed is None else f"{parsed:.{digits}f}"


def style(
    name: str,
    size: float,
    leading: float,
    color: str = "#20354A",
    bold: bool = False,
    align: int = TA_LEFT,
    before: float = 0,
    after: float = 4,
) -> ParagraphStyle:
    return ParagraphStyle(
        name,
        fontName="Helvetica-Bold" if bold else "Helvetica",
        fontSize=size,
        leading=leading,
        textColor=colors.HexColor(color),
        alignment=align,
        spaceBefore=before,
        spaceAfter=after,
    )


CELL = style("cell", 7.0, 9.0, color="#1D2A36", after=0)


def table(data: list[list[Any]], widths: list[float], font_size: float = 7.0) -> Table:
    cell = style(f"cell-{font_size}", font_size, font_size + 2, color="#1D2A36", after=0)
    header = style(
        f"header-{font_size}",
        font_size,
        font_size + 2,
        color="#FFFFFF",
        bold=True,
        after=0,
    )
    rows = [
        [Paragraph(str(value), header if row_index == 0 else cell) for value in row]
        for row_index, row in enumerate(data)
    ]
    result = Table(rows, colWidths=widths, repeatRows=1)
    result.setStyle(
        TableStyle(
            [
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
            ]
        )
    )
    return result


def scaled(path: Path, max_width: float, max_height: float) -> Image:
    with PILImage.open(path) as image:
        width, height = image.size
    factor = min(max_width / width, max_height / height)
    return Image(str(path), width=width * factor, height=height * factor)


def footer(canvas: Any, doc: Any) -> None:
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#66788A"))
    canvas.drawString(13 * mm, 7 * mm, "Geometry-family attribution appendix")
    canvas.drawRightString(284 * mm, 7 * mm, f"Appendix page {doc.page}")
    canvas.restoreState()


def select_family_single_rows(rows: list[dict[str, Any]], case: str) -> list[dict[str, Any]]:
    selected = [
        row
        for row in rows
        if row.get("case") == case
        and row.get("kind") == "single"
        and row.get("mode") == "geometry_only_original"
        and row.get("stage") == "learned_geometry_on_original"
    ]
    return sorted(selected, key=lambda row: str(row.get("warp_type")))


def make_family_graph(stage_rows: list[dict[str, Any]], output: Path) -> Path:
    methods = sorted(
        {
            str(row["warp_type"])
            for row in stage_rows
            if row.get("kind") == "single"
            and row.get("mode") == "geometry_only_original"
            and row.get("stage") == "learned_geometry_on_original"
        }
    )
    fig, axes = plt.subplots(2, 1, figsize=(11.8, 7.2), sharex=True)
    x = np.arange(len(methods))
    width = 0.36
    for offset, case in [(-width / 2, "image2_black_jacket"), (width / 2, "image4_red_scarf")]:
        rows = {str(row["warp_type"]): row for row in select_family_single_rows(stage_rows, case)}
        input_values = [number(rows.get(method, {}).get("input_ssim")) or np.nan for method in methods]
        output_values = [number(rows.get(method, {}).get("output_ssim_vs_clean")) or np.nan for method in methods]
        label = CASE_LABELS[case]
        axes[0].bar(x + offset, input_values, width, label=label)
        axes[1].bar(x + offset, output_values, width, label=label)
    axes[0].set_title("Input similarity after learned geometry")
    axes[0].set_ylabel("Original vs stage input SSIM")
    axes[1].set_title("Edited-output similarity after learned geometry")
    axes[1].set_ylabel("Clean edit vs stage edit SSIM")
    axes[1].set_xticks(x, methods, rotation=30, ha="right")
    for axis in axes:
        axis.set_ylim(0.6, 1.01)
        axis.grid(axis="y", alpha=0.25)
        axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=190)
    plt.close(fig)
    return output


def make_seed_graph(rows: list[dict[str, Any]], output: Path) -> Path:
    wanted = {
        "combined_bspline",
        "combined_bspline_tps",
        "combined_all",
        "single_bspline",
        "single_diffgeom",
        "single_rolling",
    }
    comparisons = {"geometry_increment_after_delta", "learned_geometry_vs_vae_reconstruction"}
    selected = [
        row
        for row in rows
        if row.get("run") in wanted and row.get("comparison") in comparisons
    ]
    keys = sorted({(str(row["case"]), str(row["run"]), str(row["comparison"])) for row in selected})
    fig, ax = plt.subplots(figsize=(11.8, 6.2))
    labels: list[str] = []
    means: list[float] = []
    low: list[float] = []
    high: list[float] = []
    colors_list: list[str] = []
    for case, run, comparison in keys:
        row = next(item for item in selected if item["case"] == case and item["run"] == run and item["comparison"] == comparison)
        mean = float(row["mean_edit_ssim"])
        labels.append(f"{case.replace('image', 'img')}\n{run.replace('combined_', '').replace('single_', '')}\n{comparison.replace('geometry_increment_after_delta', 'geom after delta').replace('learned_geometry_vs_vae_reconstruction', 'geom after VAE')}")
        means.append(mean)
        low.append(mean - float(row["min_edit_ssim"]))
        high.append(float(row["max_edit_ssim"]) - mean)
        colors_list.append("#2E75B6" if case == "image2_black_jacket" else "#70AD47")
    x = np.arange(len(labels))
    ax.bar(x, means, color=colors_list, alpha=0.86)
    ax.errorbar(x, means, yerr=np.asarray([low, high]), fmt="none", ecolor="#253746", capsize=4)
    ax.set_xticks(x, labels, rotation=35, ha="right", fontsize=7)
    ax.set_ylim(0.4, 1.02)
    ax.set_ylabel("Edited-output SSIM across three seeds")
    ax.set_title("Cross-seed range of selected geometry effects")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=190)
    plt.close(fig)
    return output


def direct_depth_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if row.get("mode") in {"geometry_only_original", "geometry_only_reconstruction"}
        and row.get("comparison")
        in {"geometry_on_original_effect", "geometry_on_reconstruction_effect"}
        and (
            (row.get("mode") == "geometry_only_original" and row.get("comparison") == "geometry_on_original_effect")
            or (
                row.get("mode") == "geometry_only_reconstruction"
                and row.get("comparison") == "geometry_on_reconstruction_effect"
            )
        )
    ]


def make_depth_graph(stage_rows: list[dict[str, Any]], output: Path) -> Path:
    selected = [
        row
        for row in stage_rows
        if row.get("mode") in {"geometry_only_original", "geometry_only_reconstruction"}
        and row.get("stage") in {"learned_geometry_on_original", "learned_geometry_on_reconstruction"}
        and (
            (row.get("mode") == "geometry_only_original" and row.get("stage") == "learned_geometry_on_original")
            or (
                row.get("mode") == "geometry_only_reconstruction"
                and row.get("stage") == "learned_geometry_on_reconstruction"
            )
        )
    ]
    fig, ax = plt.subplots(figsize=(11.8, 6.0))
    markers = {"geometry_only_original": "o", "geometry_only_reconstruction": "s"}
    colors_map = {"image2_black_jacket": "#2E75B6", "image4_red_scarf": "#70AD47"}
    for row in selected:
        x = float(row["input_ssim"])
        y = 1.0 - float(row["output_ssim_vs_clean"])
        ax.scatter(x, y, s=58, marker=markers[str(row["mode"])], color=colors_map[str(row["case"])], alpha=0.85)
        ax.annotate(str(row["control"]).replace("_iter", "\niter"), (x, y), xytext=(4, 4), textcoords="offset points", fontsize=6.5)
    ax.set_xlabel("Input SSIM to original (higher is closer)")
    ax.set_ylabel("Edited-output disruption (1 - SSIM)")
    ax.set_title("Longer depth and strength controls")
    ax.grid(alpha=0.25)
    ax.set_xlim(0.70, 1.005)
    ax.set_ylim(-0.01, 0.60)
    legend_rows = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#2E75B6", markersize=8, label="Image 2"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#70AD47", markersize=8, label="Image 4"),
        plt.Line2D([0], [0], marker="o", color="#253746", linestyle="none", markersize=7, label="Geometry on original"),
        plt.Line2D([0], [0], marker="s", color="#253746", linestyle="none", markersize=7, label="Geometry after VAE"),
    ]
    ax.legend(handles=legend_rows, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(output, dpi=190)
    plt.close(fig)
    return output


def collect_replay_rows(replay_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not replay_root.exists():
        return rows
    for csv_path in sorted(replay_root.glob("*/seed_replay_metrics.csv")):
        control = csv_path.parent.name
        for row in read_csv(csv_path):
            if row.get("stage") not in {"geometry_only_reconstruction", "geometry_only_original", "joint_delta_only", "combined"}:
                continue
            rows.append({"control": control, **row})
    return rows


def make_replay_graph(rows: list[dict[str, Any]], output: Path) -> Path | None:
    selected = [row for row in rows if row.get("stage") == "geometry_only_reconstruction"]
    if not selected:
        return None
    fig, ax = plt.subplots(figsize=(11.8, 5.1))
    for control in sorted({str(row["control"]) for row in selected}):
        control_rows = sorted((row for row in selected if row["control"] == control), key=lambda row: int(row["seed"]))
        ax.plot(
            [int(row["seed"]) for row in control_rows],
            [float(row["output_ssim"]) for row in control_rows],
            "o-",
            label=control.replace("image2_black_jacket__", ""),
        )
    ax.set_ylim(0.35, 1.02)
    ax.set_xlabel("Diffusion seed")
    ax.set_ylabel("Clean edit vs geometry-after-VAE edit SSIM")
    ax.set_title("100-iteration learned geometry replayed across seeds")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=190)
    plt.close(fig)
    return output


def merge_pdfs(base: Path, appendix: Path, output: Path) -> None:
    writer = PdfWriter()
    for source in (base, appendix):
        reader = PdfReader(str(source))
        for page in reader.pages:
            writer.add_page(page)
    with output.open("wb") as handle:
        writer.write(handle)


def build(args: argparse.Namespace) -> Path:
    output = args.output_root.resolve()
    assets = output / "assets"
    output.mkdir(parents=True, exist_ok=True)
    assets.mkdir(parents=True, exist_ok=True)

    family_summary = require(args.family_root / "summary")
    depth_summary = require(args.depth_root / "summary")
    family_stage = read_csv(require(family_summary / "stage_metrics.csv"))
    family_incremental = read_csv(require(family_summary / "incremental_metrics.csv"))
    seed_aggregate = read_csv(require(family_summary / "seed_direct_effects_aggregate.csv"))
    depth_stage = read_csv(require(depth_summary / "stage_metrics.csv"))
    depth_incremental = read_csv(require(depth_summary / "incremental_metrics.csv"))
    replay_rows = collect_replay_rows(args.depth_root / "seed_replays_100iter")

    family_graph = make_family_graph(family_stage, assets / "single_geometry_family_effects.png")
    seed_graph = make_seed_graph(seed_aggregate, assets / "geometry_seed_stability.png")
    depth_graph = make_depth_graph(depth_stage, assets / "geometry_depth_strength.png")
    replay_graph = make_replay_graph(replay_rows, assets / "geometry_100iter_seed_replay.png")

    write_csv(output / "geometry_family_stage_metrics.csv", family_stage)
    write_csv(output / "geometry_family_incremental_metrics.csv", family_incremental)
    write_csv(output / "geometry_seed_effects_aggregate.csv", seed_aggregate)
    write_csv(output / "geometry_depth_stage_metrics.csv", depth_stage)
    write_csv(output / "geometry_depth_incremental_metrics.csv", depth_incremental)
    write_csv(output / "geometry_100iter_seed_replay_metrics.csv", replay_rows)

    title = style("title", 24, 28, color="#17375E", bold=True, align=TA_CENTER)
    subtitle = style("subtitle", 12, 16, color="#40566B", align=TA_CENTER)
    h1 = style("h1", 17, 21, color="#17375E", bold=True, before=6, after=7)
    h2 = style("h2", 12, 15, color="#2E5D87", bold=True, before=4, after=5)
    body = style("body", 9, 13, color="#253746", after=5)
    callout = style("callout", 10, 14, color="#17375E", bold=True, after=5)
    small = style("small", 7.3, 9.4, color="#40566B", after=3)

    appendix_path = output / "geometry_family_attribution_appendix.pdf"
    doc = SimpleDocTemplate(
        str(appendix_path),
        pagesize=landscape(A4),
        leftMargin=13 * mm,
        rightMargin=13 * mm,
        topMargin=12 * mm,
        bottomMargin=13 * mm,
        title="Geometry-family attribution appendix",
    )
    story: list[Any] = [
        Spacer(1, 18 * mm),
        Paragraph("Geometry-Family Attribution Extension", title),
        Spacer(1, 4 * mm),
        Paragraph(
            "Independent geometry replays, cross-seed controls, and longer depth/strength controls for the InstructPix2Pix pipeline",
            subtitle,
        ),
        Spacer(1, 10 * mm),
        table(
            [
                ["20-iteration family runs", "Cases", "Cross-seed replay seeds", "Long controls"],
                [36, 2, "1234, 24001, 34007", 10],
            ],
            [62 * mm] * 4,
            9,
        ),
        Spacer(1, 8 * mm),
        Paragraph("Extension finding", h2),
        Paragraph(
            "The broad family sweep confirms that geometry can change edited outputs, but the largest single-family changes usually coincide with visible input deformation. B-spline is mild on the original image. At the fixed seed, longer B-spline or rolling optimization after VAE reconstruction can cross an editor discontinuity and produce a severe jacket collapse; the cross-seed controls determine whether that behavior is repeatable.",
            callout,
        ),
        Paragraph(
            "The Image 4 red-scarf clean baseline does not form a convincing scarf and is retained only as a sensitivity control, not as an attack-success case.",
            body,
        ),
        PageBreak(),
    ]

    story += [
        Paragraph("A1. Single-family geometry effects", h1),
        Paragraph(
            "Each point uses the independently optimized geometry-only-on-original branch. Lower edited-output SSIM means a larger pixel-level edit change, but it must be read together with input SSIM and the saved images.",
            body,
        ),
        scaled(family_graph, 258 * mm, 132 * mm),
        PageBreak(),
    ]

    for case in ("image2_black_jacket", "image4_red_scarf"):
        rows = select_family_single_rows(family_stage, case)
        data = [["Family", "Input SSIM", "Edit SSIM vs clean", "Edit ArcFace vs clean", "Input PSNR"]]
        for row in rows:
            data.append(
                [
                    row["warp_type"],
                    fmt(row["input_ssim"]),
                    fmt(row["output_ssim_vs_clean"]),
                    fmt(row["arcface_clean_vs_stage_edit"]),
                    fmt(row["input_psnr"], 2),
                ]
            )
        story += [
            Paragraph(f"A2. {CASE_LABELS[case]}", h1),
            table(data, [55 * mm, 42 * mm, 50 * mm, 54 * mm, 42 * mm], 7.4),
            Spacer(1, 5 * mm),
        ]
        bspline_strip = args.family_root / case / "single_bspline" / "geometry_only_original" / "learned_geometry_on_original_comparison.jpg"
        diff_strip = args.family_root / case / "single_diffgeom" / "geometry_only_original" / "learned_geometry_on_original_comparison.jpg"
        story += [
            KeepTogether([Paragraph("B-spline geometry-only replay", h2), scaled(require(bspline_strip), 258 * mm, 62 * mm)]),
            KeepTogether([Paragraph("Differential-geometry replay", h2), scaled(require(diff_strip), 258 * mm, 62 * mm)]),
            PageBreak(),
        ]

    story += [
        Paragraph("A3. Cross-seed stability", h1),
        Paragraph(
            "Bars show mean edited-output SSIM and error bars show the observed min-to-max range across three deterministic diffusion seeds. A wide range indicates editor-seed sensitivity rather than a stable transformation effect.",
            body,
        ),
        scaled(seed_graph, 258 * mm, 135 * mm),
        PageBreak(),
    ]

    selected_seed = [
        row
        for row in seed_aggregate
        if row.get("case") == "image2_black_jacket"
        and row.get("run") in {"combined_bspline", "combined_bspline_tps", "single_bspline", "single_diffgeom", "single_rolling"}
        and row.get("comparison") in {"geometry_increment_after_delta", "learned_geometry_vs_vae_reconstruction"}
    ]
    seed_table = [["Run", "Comparison", "Mean SSIM", "Min", "Max", "Range"]]
    for row in selected_seed:
        seed_table.append(
            [
                row["run"],
                str(row["comparison"]).replace("_", " "),
                fmt(row["mean_edit_ssim"]),
                fmt(row["min_edit_ssim"]),
                fmt(row["max_edit_ssim"]),
                fmt(row["range_edit_ssim"]),
            ]
        )
    story += [
        Paragraph("A3.1 Image 2 cross-seed metrics", h1),
        table(seed_table, [52 * mm, 76 * mm, 31 * mm, 28 * mm, 28 * mm, 30 * mm], 7.2),
        Spacer(1, 6 * mm),
        Paragraph(
            "At 20 iterations, B-spline on the reconstructed input is close to its neutral baseline for seeds 1234 and 34007 but changes sharply at seed 24001. Differential geometry is stronger across seeds, but it also produces visible facial deformation in the stage input.",
            callout,
        ),
        PageBreak(),
    ]

    story += [
        Paragraph("A4. Longer depth and strength controls", h1),
        Paragraph(
            "B-spline and rolling were extended to 100 iterations. Differential geometry and the all-family combination were tested at reduced amplitude. The plot separates input similarity from edited-output disruption and distinguishes geometry applied to the original from geometry applied after VAE reconstruction.",
            body,
        ),
        scaled(depth_graph, 258 * mm, 133 * mm),
        PageBreak(),
    ]

    depth_direct = direct_depth_rows(depth_incremental)
    depth_table = [["Case", "Control", "Branch", "Edit SSIM", "ArcFace edit similarity"]]
    for row in depth_direct:
        depth_table.append(
            [
                str(row["case"]).replace("_", " "),
                row["control"],
                str(row["mode"]).replace("geometry_only_", ""),
                fmt(row["edit_ssim"]),
                fmt(row["arcface_edit_cosine_similarity"]),
            ]
        )
    story += [
        Paragraph("A4.1 Direct long-control metrics", h1),
        table(depth_table, [57 * mm, 62 * mm, 40 * mm, 35 * mm, 47 * mm], 6.9),
        Spacer(1, 5 * mm),
        Paragraph(
            "The 100-iteration B-spline branch remains visually close to the clean edit when applied directly to the original. After VAE reconstruction, scale 1 produces a hood-like jacket at seed 24001 and scale 2 produces a stronger full-jacket collapse. Rolling shows the same stage interaction at this seed.",
            callout,
        ),
        PageBreak(),
    ]

    representative = [
        (
            "B-spline, 100 iterations, scale 1 - geometry on original",
            args.depth_root / "image2_black_jacket__bspline_iter100_scale1" / "geometry_only_original" / "learned_geometry_on_original_comparison.jpg",
        ),
        (
            "B-spline, 100 iterations, scale 1 - geometry after VAE",
            args.depth_root / "image2_black_jacket__bspline_iter100_scale1" / "geometry_only_reconstruction" / "learned_geometry_on_reconstruction_comparison.jpg",
        ),
        (
            "B-spline, 100 iterations, scale 2 - geometry after VAE",
            args.depth_root / "image2_black_jacket__bspline_iter100_scale2" / "geometry_only_reconstruction" / "learned_geometry_on_reconstruction_comparison.jpg",
        ),
        (
            "Rolling, 100 iterations - geometry after VAE",
            args.depth_root / "image2_black_jacket__rolling_iter100_scale1" / "geometry_only_reconstruction" / "learned_geometry_on_reconstruction_comparison.jpg",
        ),
    ]
    story += [Paragraph("A4.2 Image 2 long-control images", h1)]
    for label, path in representative:
        story.append(KeepTogether([Paragraph(label, h2), scaled(require(path), 258 * mm, 50 * mm)]))
    story.append(PageBreak())

    combined_representative = [
        (
            "Latent delta-z only within the B-spline joint run",
            args.depth_root / "image2_black_jacket__bspline_iter100_scale1" / "combined" / "learned_delta_only_comparison.jpg",
        ),
        (
            "Latent delta-z plus B-spline",
            args.depth_root / "image2_black_jacket__bspline_iter100_scale1" / "combined" / "learned_combined_comparison.jpg",
        ),
    ]
    story += [
        Paragraph("A4.3 Combined-path attribution", h1),
        Paragraph(
            "The joint run preserves the earlier attribution: delta-z already produces the severe collapse, and adding B-spline changes the saved edit only slightly (geometry-after-delta SSIM 0.991 at scale 1 and 0.990 at scale 2).",
            callout,
        ),
    ]
    for label, path in combined_representative:
        story.append(KeepTogether([Paragraph(label, h2), scaled(require(path), 258 * mm, 62 * mm)]))
    story.append(PageBreak())

    if replay_graph is not None:
        story += [
            Paragraph("A5. 100-iteration geometry seed replay", h1),
            Paragraph(
                "The exact saved long-control inputs were re-edited at three seeds without rerunning optimization. B-spline scale 2 produces output SSIM 0.934, 0.430, and 0.935 at seeds 1234, 24001, and 34007. Rolling produces 0.931, 0.426, and 0.927. The collapse is therefore concentrated at seed 24001 rather than stable across seeds.",
                body,
            ),
            scaled(replay_graph, 258 * mm, 82 * mm),
        ]
        replay_table = [["Control", "Seed", "Stage", "Input SSIM", "Output SSIM", "Edit ArcFace"]]
        for row in replay_rows:
            if row.get("stage") != "geometry_only_reconstruction":
                continue
            replay_table.append(
                [
                    str(row["control"]).replace("image2_black_jacket__", ""),
                    int(row["seed"]),
                    "geometry after VAE",
                    fmt(row["input_ssim"]),
                    fmt(row["output_ssim"]),
                    fmt(row["arcface_clean_edit_vs_stage_edit"]),
                ]
            )
        story += [Spacer(1, 3 * mm), table(replay_table, [61 * mm, 23 * mm, 48 * mm, 34 * mm, 36 * mm, 43 * mm], 6.4), PageBreak()]
        story += [
            Paragraph("A5.1 Exact B-spline scale-2 input across seeds", h1),
            Paragraph(
                "The stage input is identical in all three rows. Only the diffusion seed changes. The saved strips visually confirm that the severe jacket collapse occurs at seed 24001 and not at the other two seeds.",
                callout,
            ),
        ]
        replay_case = args.depth_root / "seed_replays_100iter" / "image2_black_jacket__bspline_iter100_scale2"
        for seed in (1234, 24001, 34007):
            replay_strip = replay_case / f"seed_{seed}" / "geometry_only_reconstruction_comparison.jpg"
            story.append(KeepTogether([Paragraph(f"Seed {seed}", h2), scaled(require(replay_strip), 258 * mm, 39 * mm)]))
        story.append(PageBreak())

    story += [
        Paragraph("A6. Updated attribution", h1),
        Paragraph("1. The original report's central finding is unchanged: optimized latent delta-z is the dominant learned source in the current combined pipeline.", callout),
        Paragraph("2. B-spline is mild when optimized and replayed directly on the original image, including at 100 iterations and doubled configured scale.", body),
        Paragraph("3. Differential geometry, Mobius, and homography can move edited outputs more strongly, but the saved inputs are visibly deformed; these are not subtle geometry-only effects.", body),
        Paragraph("4. VAE reconstruction can place a mild learned warp near an InstructPix2Pix decision boundary. Long B-spline and rolling controls collapse the jacket edit at seed 24001, while the exact same inputs produce normal edits at seeds 1234 and 34007.", callout),
        Paragraph("5. The geometry-after-delta increment remains small in the long B-spline joint controls. The joint-path collapse is already present in the delta-z-only replay.", body),
        Paragraph("6. Image 4 confirms geometric sensitivity but cannot establish edit failure because its clean red-scarf baseline is semantically weak.", body),
        Paragraph("7. SSIM and ArcFace values are used as diagnostics and are interpreted together with the saved image strips; neither metric alone determines semantic edit success.", body),
        Spacer(1, 6 * mm),
        Paragraph(
            "Final conclusion: the observed InstructPix2Pix changes are primarily latent-delta effects, with a separate seed-sensitive interaction between VAE reconstruction and learned geometry. The evidence does not support geometry alone as a stable, subtle cause of the strongest failures.",
            callout,
        ),
    ]

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    merged = output / "instruct_pipeline_stage_attribution_report_extended.pdf"
    merge_pdfs(require(args.base_report), appendix_path, merged)

    payload = {
        "base_report": str(args.base_report.resolve()),
        "appendix": str(appendix_path),
        "extended_report": str(merged),
        "family_stage_rows": len(family_stage),
        "family_incremental_rows": len(family_incremental),
        "depth_stage_rows": len(depth_stage),
        "depth_incremental_rows": len(depth_incremental),
        "long_seed_replay_rows": len(replay_rows),
        "base_pages": len(PdfReader(str(args.base_report)).pages),
        "appendix_pages": len(PdfReader(str(appendix_path)).pages),
        "extended_pages": len(PdfReader(str(merged)).pages),
        "conclusion": "Latent delta-z remains dominant; geometry-after-VAE has a separate seed-sensitive interaction.",
    }
    (output / "report_data_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (output / "report.md").write_text(
        "# InstructPix2Pix Pipeline-Stage Attribution - Extended\n\n"
        "The verified stage-attribution report is preserved as the first section. A geometry-family appendix adds the broad family sweep, three-seed controls, longer depth/strength controls, and representative strips.\n\n"
        "## Updated conclusion\n\n"
        "Optimized latent delta-z remains the dominant learned source in the combined pipeline. Learned geometry can interact with VAE reconstruction and a diffusion seed to trigger abrupt edit changes, but the strongest geometry-only changes are either seed-specific or accompanied by visible input deformation.\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2))
    return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-report",
        type=Path,
        default=Path("targeted_experiments/stage_attribution_report/instruct_pipeline_stage_attribution_report.pdf"),
    )
    parser.add_argument(
        "--family-root",
        type=Path,
        default=Path("targeted_experiments/geometry_family_attribution"),
    )
    parser.add_argument(
        "--depth-root",
        type=Path,
        default=Path("targeted_experiments/geometry_depth_strength_controls"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("targeted_experiments/stage_attribution_report_extended"),
    )
    return parser.parse_args()


def main() -> None:
    build(parse_args())


if __name__ == "__main__":
    main()
