#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${1:-/home/interns/Desktop/amit_code}"
PYTHON="${PYTHON:-/home/interns/Desktop/clean/.venv-linux-gpu/bin/python}"
ARCFACE_CHECKPOINT="${ARCFACE_CHECKPOINT:-/home/interns/Desktop/face4/models/arcface/iresnet100.pth}"

cd "$ROOT"

"$PYTHON" targeted_flux2_strict_sweep.py \
  --root "$ROOT" \
  --candidate-manifest configs/flux2_strict_candidates.json \
  --prompts-json configs/flux2_strict_prompts.json \
  --settings-json configs/flux2_strict_editor_settings.json \
  --output-root targeted_experiments/flux2_strict_sweep \
  --seeds 1234 24001 7777 \
  --min-input-ssim 0.88 \
  --device cuda

"$PYTHON" rank_flux_candidates_arcface.py \
  --root "$ROOT" \
  --input-csv targeted_experiments/flux2_strict_sweep/flux2_strict_metrics.csv \
  --output-csv targeted_experiments/flux2_strict_sweep/flux2_strict_arcface.csv \
  --checkpoint "$ARCFACE_CHECKPOINT" \
  --device cuda

git add targeted_flux2_strict_sweep.py configs/flux2_strict_candidates.json \
  configs/flux2_strict_prompts.json configs/flux2_strict_editor_settings.json \
  rank_flux_candidates_arcface.py run_flux2_strict_sweep.sh \
  targeted_experiments/flux2_strict_sweep
git commit -m "Add strict FLUX.2 Klein identity-disruption sweep results" || true
git pull --rebase origin main
git push origin main
