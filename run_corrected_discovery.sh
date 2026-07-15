#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${ROOT:-/home/interns/Desktop/amit_code}"
PYTHON="${PYTHON:-python}"
OUT="$ROOT/targeted_experiments/corrected_perturbations"

cd "$ROOT"
mkdir -p "$OUT" logs

export PYTHONPATH="/home/interns/Desktop/face4${PYTHONPATH:+:$PYTHONPATH}"

# These runs deliberately vary the geometry family while holding the corrected
# identity-disruption objective and preservation weights fixed.
for face in image_2 image_3; do
  for warp in bspline tps polar lens; do
    run_dir="$OUT/$face/${warp}_30"
    if [[ -f "$run_dir/vae_latent_out.png" ]]; then
      echo "SKIP existing $face/$warp"
      continue
    fi
    "$PYTHON" vae_latent_adversarial.py \
      --input "new_images/${face}.png" \
      --warp-type "$warp" \
      --iterations 30 \
      --lr 0.005 \
      --identity-weight 2.0 \
      --ssim-weight 20.0 \
      --pixel-weight 2.0 \
      --latent-reg-weight 0.01 \
      --imperceptibility-scale 0.75 \
      --save-interval 15 \
      --device cuda \
      --target-model arcface \
      --arcface-checkpoint /home/interns/Desktop/face4/models/arcface/iresnet100.pth \
      --seed 24001 \
      --save-dir "$run_dir" \
      --whole-image \
      --disable-landmarks \
      --verbose
  done
done

"$PYTHON" targeted_prompt_sweep.py \
  --root "$ROOT" \
  --candidate-root targeted_experiments/corrected_perturbations \
  --output-root targeted_experiments/prompt_sweep_corrected \
  --min-input-ssim 0.84 \
  --max-candidates-per-face 4 \
  --steps 20 \
  --guidance-scale 7.5 \
  --image-guidance-scale 1.5 \
  --seed 1234 \
  --device cuda

"$PYTHON" refine_prompt_sweep.py \
  --input-csv targeted_experiments/prompt_sweep_corrected/prompt_sweep_metrics.csv \
  --output-root targeted_experiments/prompt_refinement_corrected \
  --top-cases 8 \
  --min-input-ssim 0.84 \
  --image-guidance-scales 1.0 1.5 2.0 \
  --seeds 1234 24001 \
  --steps 20 \
  --guidance-scale 7.5 \
  --device cuda

git add targeted_experiments analysis_outputs || true
if ! git diff --cached --quiet; then
  git commit -m "Add corrected VAE geometry discovery results"
fi
git push origin main
