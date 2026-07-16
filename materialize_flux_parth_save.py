"""Materialize visually approved FLUX cases into ``parth_save/flux``.

The manifest identifies exact CSV rows. The canonical four images, comparison
strip, normalized metrics, source paths, hashes, and reproducible editor
settings are copied into one self-contained folder per approved case.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

from PIL import Image


IMAGE_FIELDS = {
    "original_path": "original.png",
    "perturbed_path": "perturbed.png",
    "clean_edit_path": "original_edit.png",
    "perturbed_edit_path": "perturbed_edit.png",
    "strip_path": "comparison_strip.jpg",
}


def local_path(root: Path, value: str) -> Path:
    normalized = value.replace("\\", "/")
    marker = "/home/interns/Desktop/amit_code/"
    if marker in normalized:
        normalized = normalized.split(marker, 1)[1]
    path = Path(normalized)
    return path if path.is_absolute() else root / path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scalar(value: str) -> object:
    try:
        number = float(value)
        return int(number) if number.is_integer() else number
    except (TypeError, ValueError):
        return value


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def find_row(rows: list[dict[str, str]], selection: dict[str, object]) -> dict[str, str]:
    matches = [
        row
        for row in rows
        if row["face_id"] == selection["face_id"]
        and row["prompt"] == selection["prompt"]
        and str(row["seed"]) == str(selection["seed"])
        and row["candidate_id"] == selection["candidate_id"]
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one CSV row for {selection['folder']}, found {len(matches)}"
        )
    return matches[0]


def make_sheet(case_roots: list[Path], output: Path) -> None:
    strips = [Image.open(case / "comparison_strip.jpg").convert("RGB") for case in case_roots]
    if not strips:
        return
    width = max(image.width for image in strips)
    height = sum(image.height for image in strips)
    sheet = Image.new("RGB", (width, height), "white")
    y = 0
    for image in strips:
        sheet.paste(image, (0, y))
        y += image.height
    sheet.save(output, quality=94)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("parth_save/flux/selection_manifest.json"),
    )
    args = parser.parse_args()
    root = args.root.resolve()
    manifest_path = args.manifest if args.manifest.is_absolute() else root / args.manifest
    selections = json.loads(manifest_path.read_text(encoding="utf-8"))
    output_root = manifest_path.parent
    output_root.mkdir(parents=True, exist_ok=True)

    csv_cache: dict[Path, list[dict[str, str]]] = {}
    case_roots: list[Path] = []
    for selection in selections:
        csv_path = local_path(root, str(selection["csv"]))
        rows = csv_cache.setdefault(csv_path, load_rows(csv_path))
        row = find_row(rows, selection)
        case_root = output_root / str(selection["folder"])
        case_root.mkdir(parents=True, exist_ok=True)
        copied: dict[str, dict[str, str]] = {}
        for field, destination_name in IMAGE_FIELDS.items():
            source = local_path(root, row[field])
            if not source.exists():
                raise FileNotFoundError(f"Missing {field} for {selection['folder']}: {source}")
            destination = case_root / destination_name
            shutil.copy2(source, destination)
            copied[field] = {
                "source": source.relative_to(root).as_posix(),
                "saved": destination_name,
                "sha256": sha256(destination),
            }

        metrics = {
            "face_id": row["face_id"],
            "prompt": row["prompt"],
            "candidate_id": row["candidate_id"],
            "model": row["model"],
            "seed": scalar(row["seed"]),
            "steps": scalar(row["steps"]),
            "guidance_scale": scalar(row["guidance_scale"]),
            "input_ssim": scalar(row["input_ssim"]),
            "input_psnr": scalar(row["input_psnr"]),
            "input_mse": scalar(row["input_mse"]),
            "input_l2": scalar(row["input_l2"]),
            "output_ssim": scalar(row["output_ssim"]),
            "output_psnr": scalar(row["output_psnr"]),
            "output_mse": scalar(row["output_mse"]),
            "output_l2": scalar(row["output_l2"]),
            "source_csv": csv_path.relative_to(root).as_posix(),
            "observation": selection["observation"],
            "visual_verdict": selection.get("visual_verdict", "presentable"),
            "perturbation_reproduction": selection.get("perturbation_reproduction", {}),
            "files": copied,
        }
        (case_root / "metrics.json").write_text(
            json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
        )
        case_roots.append(case_root)

    make_sheet(case_roots, output_root / "all_flux_examples_sheet.jpg")
    summary = {
        "model": "black-forest-labs/FLUX.2-klein-4B",
        "num_cases": len(case_roots),
        "cases": [case.name for case in case_roots],
        "manifest": manifest_path.name,
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
