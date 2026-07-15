#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-/home/interns/Desktop/amit_code}"
PYTHON="${PYTHON:-/home/interns/Desktop/clean/.venv-linux-gpu/bin/python}"

cd "$ROOT"

"$PYTHON" targeted_prompt_sweep.py \
  --root "$ROOT" \
  --candidate-root targeted_experiments/corrected_perturbations \
  --candidate-root targeted_experiments/tightened_perturbations \
  --output-root targeted_experiments/prompt_sweep_hair_localized \
  --prompts-json configs/hair_localized_prompts.json \
  --min-input-ssim 0.84 \
  --max-candidates-per-face 8 \
  --steps 20 \
  --guidance-scale 7.5 \
  --image-guidance-scale 1.5 \
  --seed 1234 \
  --device cuda

"$PYTHON" refine_prompt_sweep.py \
  --input-csv targeted_experiments/prompt_sweep_hair_localized/prompt_sweep_metrics.csv \
  --output-root targeted_experiments/prompt_refinement_hair_localized \
  --top-cases 12 \
  --min-input-ssim 0.84 \
  --image-guidance-scales 1.0 1.5 2.0 \
  --seeds 1234 24001 7777 \
  --steps 20 \
  --guidance-scale 7.5 \
  --device cuda

git add configs/hair_localized_prompts.json run_hair_localized_discovery.sh \
  targeted_experiments/prompt_sweep_hair_localized \
  targeted_experiments/prompt_refinement_hair_localized
git commit -m "Add black-hair and localized prompt discovery results" || true
git push origin main
