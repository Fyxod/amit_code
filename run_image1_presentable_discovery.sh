#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-/home/interns/Desktop/amit_code}"
PYTHON="${PYTHON:-/home/interns/Desktop/clean/.venv-linux-gpu/bin/python}"

cd "$ROOT"

"$PYTHON" targeted_prompt_sweep.py \
  --root "$ROOT" \
  --candidate-root output_save/image_1/results \
  --output-root targeted_experiments/prompt_sweep_image1_presentable \
  --prompts-json configs/image1_presentable_prompts.json \
  --min-input-ssim 0.90 \
  --max-candidates-per-face 7 \
  --steps 20 \
  --guidance-scale 7.5 \
  --image-guidance-scale 1.5 \
  --seed 1234 \
  --device cuda

"$PYTHON" refine_prompt_sweep.py \
  --input-csv targeted_experiments/prompt_sweep_image1_presentable/prompt_sweep_metrics.csv \
  --output-root targeted_experiments/prompt_refinement_image1_presentable \
  --top-cases 15 \
  --min-input-ssim 0.90 \
  --image-guidance-scales 1.0 1.5 2.0 \
  --seeds 1234 24001 7777 \
  --steps 20 \
  --guidance-scale 7.5 \
  --device cuda

git add configs/image1_presentable_prompts.json run_image1_presentable_discovery.sh \
  targeted_experiments/prompt_sweep_image1_presentable \
  targeted_experiments/prompt_refinement_image1_presentable
git commit -m "Add Image 1 presentable prompt discovery results" || true
git pull --rebase origin main
git push origin main
