#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${1:-/home/interns/Desktop/amit_code}"
PYTHON="${PYTHON:-/home/interns/Desktop/clean/.venv-linux-gpu/bin/python}"

cd "$ROOT"

"$PYTHON" targeted_flux_refinement_sweep.py \
  --root "$ROOT" \
  --candidate-root output_save/image_1/results \
  --candidate-root output_save/image_2/results \
  --candidate-root targeted_experiments/new_face_round2_perturbations \
  --output-root targeted_experiments/flux_klein_refinement_sweep \
  --prompts-json configs/flux_klein_refinement_prompt_bank.json \
  --face-ids image_1 image_2 image_5 image_7 image_8 \
  --seeds 1234 24001 7777 \
  --min-input-ssim 0.84 \
  --max-candidates-per-face 2 \
  --steps 10 \
  --guidance-scale 1.0 \
  --device cuda

git add targeted_flux_refinement_sweep.py configs/flux_klein_refinement_prompt_bank.json \
  run_flux_klein_refinement_sweep.sh targeted_experiments/flux_klein_refinement_sweep
git commit -m "Add FLUX.2 Klein refinement sweep results" || true
git pull --rebase origin main
git push origin main
