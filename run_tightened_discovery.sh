#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${ROOT:-/home/interns/Desktop/amit_code}"
PYTHON="${PYTHON:-/home/interns/Desktop/clean/.venv-linux-gpu/bin/python}"
OUT="$ROOT/targeted_experiments/tightened_perturbations"
export PYTHONPATH="/home/interns/Desktop/face4${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT"
mkdir -p "$OUT"

run_profile() {
  local profile="$1" identity="$2" ssim="$3" pixel="$4" scale="$5" lr="$6"
  for face in image_2 image_3; do
    for warp in bspline polar lens; do
      local run_dir="$OUT/$face/$profile/${warp}_25"
      if [[ -f "$run_dir/vae_latent_out.png" ]]; then
        echo "SKIP existing $face/$profile/$warp"
        continue
      fi
      "$PYTHON" vae_latent_adversarial.py \
        --input "new_images/${face}.png" \
        --warp-type "$warp" \
        --iterations 25 \
        --lr "$lr" \
        --identity-weight "$identity" \
        --ssim-weight "$ssim" \
        --pixel-weight "$pixel" \
        --latent-reg-weight 0.02 \
        --imperceptibility-scale "$scale" \
        --save-interval 12 \
        --device cuda \
        --target-model arcface \
        --arcface-checkpoint /home/interns/Desktop/face4/models/arcface/iresnet100.pth \
        --seed 25001 \
        --save-dir "$run_dir" \
        --whole-image \
        --disable-landmarks \
        --verbose
    done
  done
}

run_profile preserve40 1.0 40.0 5.0 0.50 0.003
run_profile preserve60 0.6 60.0 8.0 0.35 0.002

"$PYTHON" targeted_prompt_sweep.py \
  --root "$ROOT" \
  --candidate-root targeted_experiments/tightened_perturbations \
  --output-root targeted_experiments/prompt_sweep_tightened \
  --prompts-json configs/tightened_prompts.json \
  --min-input-ssim 0.88 \
  --max-candidates-per-face 6 \
  --steps 20 \
  --guidance-scale 7.5 \
  --image-guidance-scale 1.5 \
  --seed 1234 \
  --device cuda

"$PYTHON" refine_prompt_sweep.py \
  --input-csv targeted_experiments/prompt_sweep_tightened/prompt_sweep_metrics.csv \
  --output-root targeted_experiments/prompt_refinement_tightened \
  --top-cases 10 \
  --min-input-ssim 0.88 \
  --image-guidance-scales 1.0 1.5 2.0 \
  --seeds 1234 24001 7777 \
  --steps 20 \
  --guidance-scale 7.5 \
  --device cuda

git add targeted_experiments
if ! git diff --cached --quiet; then
  git commit -m "Add tightened ArcFace prompt discovery results"
fi
git push origin main
