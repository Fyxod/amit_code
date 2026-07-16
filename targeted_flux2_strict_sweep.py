"""Strict FLUX.2 Klein screen over explicit high-strength geometry artifacts.

This is a discovery stage. It deliberately does not label a case successful
from metrics alone. A candidate is only eligible for ``parth_save/flux_2``
after direct inspection confirms identity collapse, severe semantic failure,
or major whole-image disruption.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from PIL import Image

from flux import FluxBackend, FluxSettings
from targeted_prompt_sweep import make_top_sheet, original_for, pair_metrics, save_strip, slug


def resolve(root: Path, path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else root / value


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def append_failure(path: Path, record: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument(
        "--candidate-manifest",
        type=Path,
        default=Path("configs/flux2_strict_candidates.json"),
    )
    parser.add_argument(
        "--prompts-json", type=Path, default=Path("configs/flux2_strict_prompts.json")
    )
    parser.add_argument(
        "--settings-json",
        type=Path,
        default=Path("configs/flux2_strict_editor_settings.json"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("targeted_experiments/flux2_strict_sweep"),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[1234, 24001, 7777])
    parser.add_argument("--min-input-ssim", type=float, default=0.88)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    root = args.root.resolve()
    output_root = resolve(root, args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    candidates = json.loads(resolve(root, args.candidate_manifest).read_text(encoding="utf-8"))
    prompts = json.loads(resolve(root, args.prompts_json).read_text(encoding="utf-8"))
    editor_settings = json.loads(resolve(root, args.settings_json).read_text(encoding="utf-8"))

    eligible: list[dict[str, object]] = []
    for candidate in candidates:
        path = resolve(root, candidate["path"]).resolve()
        original = original_for(root, candidate["face_id"])
        metrics = pair_metrics(original, path)
        if metrics["ssim"] < args.min_input_ssim:
            print(
                f"REJECT input {candidate['face_id']} {path}: "
                f"SSIM {metrics['ssim']:.4f} < {args.min_input_ssim:.4f}"
            )
            continue
        eligible.append(
            {
                **candidate,
                "candidate_path": path,
                "candidate_id": slug(str(path.relative_to(root)))[:120],
                **{f"input_{name}": value for name, value in metrics.items()},
            }
        )
    if not eligible:
        raise RuntimeError("No explicit strict candidate passed the input SSIM gate")

    first = editor_settings[0]
    backend = FluxBackend(
        torch.device(args.device),
        FluxSettings(
            diffusion_steps=int(first["steps"]),
            guidance_scale=float(first["guidance_scale"]),
            seed=args.seeds[0],
        ),
    )
    csv_path = output_root / "flux2_strict_metrics.csv"
    rows: list[dict[str, object]] = []
    if csv_path.exists():
        with csv_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    completed = {
        (
            str(row["setting_name"]),
            str(row["face_id"]),
            str(row["prompt"]),
            int(row["seed"]),
            str(row["candidate_id"]),
        )
        for row in rows
    }
    failures_path = output_root / "failures.jsonl"
    attempted = 0

    for setting in editor_settings:
        setting_name = str(setting["name"])
        steps = int(setting["steps"])
        guidance_scale = float(setting["guidance_scale"])
        backend.settings.diffusion_steps = steps
        backend.settings.guidance_scale = guidance_scale
        for face_id in sorted({str(item["face_id"]) for item in eligible}):
            original = original_for(root, face_id)
            face_candidates = [item for item in eligible if item["face_id"] == face_id]
            for prompt in prompts:
                prompt_slug = slug(prompt)
                for seed in args.seeds:
                    clean_root = output_root / setting_name / face_id / prompt_slug / f"seed_{seed}"
                    clean_path = clean_root / "clean_edit.png"
                    if not clean_path.exists():
                        try:
                            clean_root.mkdir(parents=True, exist_ok=True)
                            backend.generate_edit(
                                Image.open(original).convert("RGB"), prompt, seed
                            ).save(clean_path)
                        except Exception as error:  # pragma: no cover - GPU diagnostic path
                            append_failure(
                                failures_path,
                                {
                                    "stage": "clean_edit",
                                    "setting_name": setting_name,
                                    "face_id": face_id,
                                    "prompt": prompt,
                                    "seed": seed,
                                    "error": repr(error),
                                },
                            )
                            continue
                    for candidate in face_candidates:
                        key = (
                            setting_name,
                            face_id,
                            prompt,
                            seed,
                            str(candidate["candidate_id"]),
                        )
                        if key in completed:
                            continue
                        attempted += 1
                        case_root = clean_root / str(candidate["candidate_id"])
                        perturbed_edit_path = case_root / "perturbed_edit.png"
                        try:
                            if not perturbed_edit_path.exists():
                                case_root.mkdir(parents=True, exist_ok=True)
                                backend.generate_edit(
                                    Image.open(candidate["candidate_path"]).convert("RGB"),
                                    prompt,
                                    seed,
                                ).save(perturbed_edit_path)
                            output_metrics = pair_metrics(clean_path, perturbed_edit_path)
                            strip_path = case_root / "comparison_strip.jpg"
                            save_strip(
                                original,
                                candidate["candidate_path"],
                                clean_path,
                                perturbed_edit_path,
                                strip_path,
                                f"FLUX.2 Klein 4B | {setting_name} | {face_id} | {prompt} | "
                                f"seed={seed} | input SSIM={float(candidate['input_ssim']):.3f} | "
                                f"output SSIM={output_metrics['ssim']:.3f}",
                            )
                            row = {
                                "setting_name": setting_name,
                                "face_id": face_id,
                                "candidate_path": str(candidate["candidate_path"]),
                                "candidate_id": candidate["candidate_id"],
                                "perturbation_type": candidate["perturbation_type"],
                                "perturbation_iterations": candidate["iterations"],
                                "perturbation_source_commit": candidate["source_commit"],
                                "input_ssim": candidate["input_ssim"],
                                "input_psnr": candidate["input_psnr"],
                                "input_mse": candidate["input_mse"],
                                "input_l2": candidate["input_l2"],
                                "model": backend.settings.model_id,
                                "prompt": prompt,
                                "seed": seed,
                                "steps": steps,
                                "guidance_scale": guidance_scale,
                                **{
                                    f"output_{name}": value
                                    for name, value in output_metrics.items()
                                },
                                "discovery_disruption_score": (
                                    1.0 - output_metrics["ssim"]
                                )
                                * float(candidate["input_ssim"]),
                                "strict_visual_status": "unreviewed",
                                "original_path": str(original),
                                "perturbed_path": str(candidate["candidate_path"]),
                                "clean_edit_path": str(clean_path),
                                "perturbed_edit_path": str(perturbed_edit_path),
                                "strip_path": str(strip_path),
                            }
                            rows.append(row)
                            completed.add(key)
                            write_csv(csv_path, rows)
                            print(
                                f"[{len(rows)}] {setting_name} {face_id} seed={seed} "
                                f"{prompt!r} {candidate['perturbation_type']}_100 "
                                f"input={float(candidate['input_ssim']):.3f} "
                                f"output={output_metrics['ssim']:.3f}"
                            )
                        except Exception as error:  # pragma: no cover - GPU diagnostic path
                            append_failure(
                                failures_path,
                                {
                                    "stage": "perturbed_edit",
                                    "setting_name": setting_name,
                                    "face_id": face_id,
                                    "prompt": prompt,
                                    "seed": seed,
                                    "candidate_id": candidate["candidate_id"],
                                    "error": repr(error),
                                },
                            )

    if not rows:
        raise RuntimeError("Strict sweep produced no completed rows")
    make_top_sheet(rows, output_root / "top_flux2_strict_sheet.jpg", limit=80)
    review_rows = sorted(
        rows,
        key=lambda row: float(row["discovery_disruption_score"]),
        reverse=True,
    )[:100]
    review_csv = output_root / "strict_candidates_for_visual_review.csv"
    write_csv(review_csv, review_rows)
    summary = {
        "status": "complete",
        "model": backend.settings.model_id,
        "acceptance_rule": (
            "Visual identity collapse, severe semantic failure, or major whole-image "
            "disruption required; minor edit/style variation is rejected."
        ),
        "num_candidates": len(eligible),
        "num_prompts": len(prompts),
        "num_settings": len(editor_settings),
        "seeds": args.seeds,
        "num_pairs": len(rows),
        "new_pairs_this_run": attempted,
        "best_output_ssim": min(float(row["output_ssim"]) for row in rows),
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_root / "DONE.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
