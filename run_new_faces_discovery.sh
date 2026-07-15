#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${1:-/home/interns/Desktop/amit_code}"
MAT_ROOT="${MAT_ROOT:-/home/interns/Desktop/mat}"
PYTHON="${PYTHON:-/home/interns/Desktop/clean/.venv-linux-gpu/bin/python}"
OUT="$ROOT/targeted_experiments/new_face_perturbations"

cd "$ROOT"
mkdir -p new_images "$OUT"

cp "$MAT_ROOT/data/face_004/instruct_512.png" new_images/image_4.png
cp "$MAT_ROOT/data/face_006/instruct_512.png" new_images/image_6.png

export PYTHONPATH="/home/interns/Desktop/face4${PYTHONPATH:+:$PYTHONPATH}"

for face in image_4 image_6; do
  for warp in bspline polar lens; do
    run_dir="$OUT/$face/${warp}_25"
    if [[ -f "$run_dir/vae_latent_out.png" ]]; then
      echo "SKIP existing $face/$warp"
      continue
    fi
    "$PYTHON" vae_latent_adversarial.py \
      --input "new_images/${face}.png" \
      --warp-type "$warp" \
      --iterations 25 \
      --lr 0.005 \
      --identity-weight 2.0 \
      --ssim-weight 20.0 \
      --pixel-weight 2.0 \
      --latent-reg-weight 0.01 \
      --imperceptibility-scale 0.75 \
      --save-interval 25 \
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
  --candidate-root targeted_experiments/new_face_perturbations \
  --output-root targeted_experiments/prompt_sweep_new_faces \
  --prompts-json configs/new_face_prompt_bank.json \
  --min-input-ssim 0.84 \
  --max-candidates-per-face 3 \
  --steps 20 \
  --guidance-scale 7.5 \
  --image-guidance-scale 1.5 \
  --seed 1234 \
  --device cuda

git add targeted_prompt_sweep.py configs/new_face_prompt_bank.json \
  run_new_faces_discovery.sh new_images/image_4.png new_images/image_6.png \
  targeted_experiments/new_face_perturbations \
  targeted_experiments/prompt_sweep_new_faces
git commit -m "Add new-face geometric prompt discovery results" || true
git pull --rebase origin main
git push origin main
