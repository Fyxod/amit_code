#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${1:-/home/interns/Desktop/amit_code}"
MAT_ROOT="${MAT_ROOT:-/home/interns/Desktop/mat}"
PYTHON="${PYTHON:-/home/interns/Desktop/clean/.venv-linux-gpu/bin/python}"
OUT="$ROOT/targeted_experiments/new_face_round2_perturbations"

cd "$ROOT"
mkdir -p new_images "$OUT"

for id in 003 005 007 008; do
  short="${id#00}"
  cp "$MAT_ROOT/data/face_${id}/instruct_512.png" "new_images/image_${short}.png"
done

export PYTHONPATH="/home/interns/Desktop/face4${PYTHONPATH:+:$PYTHONPATH}"

for face in image_3 image_5 image_7 image_8; do
  for warp in bspline polar lens; do
    run_dir="$OUT/$face/${warp}_30"
    if [[ -f "$run_dir/vae_latent_out.png" ]]; then
      echo "SKIP existing $face/$warp"
      continue
    fi
    "$PYTHON" vae_latent_adversarial.py \
      --input "new_images/${face}.png" \
      --warp-type "$warp" \
      --iterations 30 \
      --lr 0.006 \
      --identity-weight 2.0 \
      --ssim-weight 18.0 \
      --pixel-weight 1.5 \
      --latent-reg-weight 0.008 \
      --imperceptibility-scale 0.80 \
      --save-interval 30 \
      --device cuda \
      --target-model arcface \
      --arcface-checkpoint /home/interns/Desktop/face4/models/arcface/iresnet100.pth \
      --seed 31001 \
      --save-dir "$run_dir" \
      --whole-image \
      --disable-landmarks \
      --verbose
  done
done

"$PYTHON" targeted_prompt_sweep.py \
  --root "$ROOT" \
  --candidate-root targeted_experiments/new_face_round2_perturbations \
  --output-root targeted_experiments/prompt_sweep_new_faces_round2 \
  --prompts-json configs/new_face_round2_prompt_bank.json \
  --min-input-ssim 0.82 \
  --max-candidates-per-face 3 \
  --steps 20 \
  --guidance-scale 7.5 \
  --image-guidance-scale 1.5 \
  --seed 1234 \
  --device cuda

git add configs/new_face_round2_prompt_bank.json run_new_faces_round2_discovery.sh \
  new_images/image_3.png new_images/image_5.png new_images/image_7.png new_images/image_8.png \
  targeted_experiments/new_face_round2_perturbations \
  targeted_experiments/prompt_sweep_new_faces_round2
git commit -m "Add second new-face perturbation discovery results" || true
git pull --rebase origin main
git push origin main
