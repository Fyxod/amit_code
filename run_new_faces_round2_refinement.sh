#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${1:-/home/interns/Desktop/amit_code}"
PYTHON="${PYTHON:-/home/interns/Desktop/clean/.venv-linux-gpu/bin/python}"

cd "$ROOT"

"$PYTHON" refine_prompt_sweep.py \
  --input-csv targeted_experiments/prompt_sweep_new_faces_round2/prompt_sweep_metrics.csv \
  --output-root targeted_experiments/prompt_refinement_new_faces_round2 \
  --prompts-json configs/new_face_round2_refine_prompts.json \
  --top-cases 45 \
  --min-input-ssim 0.84 \
  --image-guidance-scales 0.6 0.8 1.0 1.2 1.5 2.0 \
  --seeds 1234 24001 7777 \
  --steps 20 \
  --guidance-scale 7.5 \
  --device cuda

git add configs/new_face_round2_refine_prompts.json run_new_faces_round2_refinement.sh \
  targeted_experiments/prompt_refinement_new_faces_round2
git commit -m "Add second new-face prompt refinement results" || true
git pull --rebase origin main
git push origin main
