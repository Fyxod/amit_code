"""Materialize curated prompt-sensitivity cases from result CSVs."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path


PATH_FIELDS = {
    "original.png": "original_path",
    "perturbed.png": "perturbed_path",
    "original_edit.png": "clean_edit_path",
    "perturbed_edit.png": "perturbed_edit_path",
    "comparison_strip.jpg": "strip_path",
}


def local_path(root: Path, raw: str) -> Path:
    path = Path(raw)
    if path.exists():
        return path
    linux_prefix = "/home/interns/Desktop/amit_code/"
    normalized = raw.replace("\\", "/")
    if normalized.startswith(linux_prefix):
        path = root / normalized[len(linux_prefix):]
    if not path.exists():
        raise FileNotFoundError(f"Missing source artifact: {raw} (resolved {path})")
    return path


def match_row(rows: list[dict[str, str]], selection: dict[str, str]) -> dict[str, str]:
    keys = ("face_id", "prompt", "image_guidance_scale", "seed", "candidate_id")
    matches = [
        row for row in rows
        if all(str(row.get(key)) == str(selection[key]) for key in keys)
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one row for {selection['folder']}, found {len(matches)}")
    return matches[0]


def numeric(value: str):
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--manifest", type=Path, default=Path("parth_save/selection_manifest.json"))
    args = parser.parse_args()

    root = args.root.resolve()
    manifest_path = args.manifest if args.manifest.is_absolute() else root / args.manifest
    selections = json.loads(manifest_path.read_text(encoding="utf-8"))
    cache: dict[Path, list[dict[str, str]]] = {}

    for selection in selections:
        csv_path = root / selection["csv"]
        if csv_path not in cache:
            with csv_path.open(newline="", encoding="utf-8") as handle:
                cache[csv_path] = list(csv.DictReader(handle))
        row = match_row(cache[csv_path], selection)
        case_root = root / "parth_save" / selection["folder"]
        case_root.mkdir(parents=True, exist_ok=True)
        for destination, field in PATH_FIELDS.items():
            shutil.copy2(local_path(root, row[field]), case_root / destination)

        metrics_path = case_root / "metrics.json"
        existing_metrics = {}
        if metrics_path.exists():
            existing_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        metrics = {
            **existing_metrics,
            **{key: numeric(value) for key, value in row.items() if not key.endswith("_path")},
        }
        metrics.update({
            "editor": "InstructPix2Pix",
            "optimization_identity_model": "ArcFace iResNet-100",
            "source_csv": selection["csv"],
            "observation": selection["observation"],
        })
        metrics_path.write_text(
            json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
        )
        print(f"Saved {selection['folder']}")


if __name__ == "__main__":
    main()
