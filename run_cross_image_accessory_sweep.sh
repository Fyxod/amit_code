#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${1:-/home/interns/Desktop/amit_code}"
PYTHON="${PYTHON:-/home/interns/Desktop/clean/.venv-linux-gpu/bin/python}"

cd "$ROOT"

"$PYTHON" targeted_prompt_sweep.py \
  --root "$ROOT" \
  --candidate-root targeted_experiments/corrected_perturbations \
  --candidate-root targeted_experiments/tightened_perturbations \
  --output-root targeted_experiments/prompt_sweep_cross_image_accessories \
  --prompts-json configs/cross_image_accessory_prompts.json \
  --min-input-ssim 0.84 \
  --max-candidates-per-face 6 \
  --steps 20 \
  --guidance-scale 7.5 \
  --image-guidance-scale 1.5 \
  --seed 1234 \
  --device cuda

git add configs/cross_image_accessory_prompts.json run_cross_image_accessory_sweep.sh \
  targeted_experiments/prompt_sweep_cross_image_accessories
git commit -m "Add cross-image accessory prompt sweep results" || true
git pull --rebase origin main
git push origin main
