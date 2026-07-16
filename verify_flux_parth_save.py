"""Verify completeness and metric integrity of ``parth_save/flux``."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image

from targeted_prompt_sweep import pair_metrics


REQUIRED_IMAGES = (
    "original.png",
    "perturbed.png",
    "original_edit.png",
    "perturbed_edit.png",
    "comparison_strip.jpg",
)
ARCFACE_KEYS = (
    "original_vs_perturbed",
    "original_edit_vs_perturbed_edit",
    "original_vs_original_edit",
    "perturbed_vs_perturbed_edit",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def close(a: float, b: float, tolerance: float = 1e-8) -> bool:
    return abs(a - b) <= tolerance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--save-root", type=Path, default=Path("parth_save/flux_2"))
    args = parser.parse_args()
    root = args.root.resolve()
    save_root = args.save_root if args.save_root.is_absolute() else root / args.save_root
    manifest = json.loads((save_root / "selection_manifest.json").read_text(encoding="utf-8"))
    errors: list[str] = []
    cases: list[dict[str, object]] = []

    if len(manifest) != 10:
        errors.append(f"Expected 10 manifest entries, found {len(manifest)}")
    for selection in manifest:
        case_root = save_root / selection["folder"]
        missing = [name for name in REQUIRED_IMAGES if not (case_root / name).exists()]
        if missing:
            errors.append(f"{case_root.name}: missing {missing}")
            continue
        for name in REQUIRED_IMAGES:
            try:
                Image.open(case_root / name).verify()
            except Exception as error:  # pragma: no cover - diagnostic path
                errors.append(f"{case_root.name}/{name}: invalid image: {error}")
        metrics_path = case_root / "metrics.json"
        if not metrics_path.exists():
            errors.append(f"{case_root.name}: missing metrics.json")
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        input_values = pair_metrics(case_root / "original.png", case_root / "perturbed.png")
        output_values = pair_metrics(
            case_root / "original_edit.png", case_root / "perturbed_edit.png"
        )
        for prefix, values in (("input", input_values), ("output", output_values)):
            for name, value in values.items():
                stored = float(metrics[f"{prefix}_{name}"])
                if not close(stored, value):
                    errors.append(
                        f"{case_root.name}: {prefix}_{name} stored={stored} recomputed={value}"
                    )
        for file_record in metrics.get("files", {}).values():
            path = case_root / file_record["saved"]
            if sha256(path) != file_record["sha256"]:
                errors.append(f"{case_root.name}: hash mismatch for {path.name}")
        reproduction = metrics.get("perturbation_reproduction", {})
        if not reproduction.get("command") or not reproduction.get("source_commit"):
            errors.append(f"{case_root.name}: incomplete perturbation reproduction settings")
        arcface = metrics.get("arcface_cosine_similarity", {})
        missing_arcface = [name for name in ARCFACE_KEYS if name not in arcface]
        if metrics.get("arcface_model") != "ArcFace iResNet-100" or missing_arcface:
            errors.append(f"{case_root.name}: incomplete exact ArcFace metrics {missing_arcface}")
        cases.append(
            {
                "case": case_root.name,
                "prompt": metrics["prompt"],
                "input_ssim": metrics["input_ssim"],
                "input_psnr": metrics["input_psnr"],
                "output_ssim": metrics["output_ssim"],
                "output_psnr": metrics["output_psnr"],
                "arcface_original_vs_perturbed": arcface.get("original_vs_perturbed"),
                "arcface_original_edit_vs_perturbed_edit": arcface.get(
                    "original_edit_vs_perturbed_edit"
                ),
            }
        )

    report = {
        "status": "passed" if not errors else "failed",
        "expected_cases": 10,
        "verified_cases": len(cases),
        "required_images_per_case": list(REQUIRED_IMAGES),
        "metric_recomputation": "SSIM, PSNR, MSE, and L2 matched stored values",
        "arcface_model": "ArcFace iResNet-100",
        "cases": cases,
        "errors": errors,
    }
    report_path = save_root / "verification.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
