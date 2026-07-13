#!/bin/bash
# Complete pipeline script for all combinations
# Images: image_2.png, image_3.png
# Warps: bspline, all
# Iterations: 10, 15, 20, 50, 75, 100
# Prompts: "Add sunglasses", "Make hair color green"

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Define arrays
IMAGES=("image_2.png" "image_3.png")
WARP_TYPES=("bspline" "all")
ITERATIONS=(10 15 20 50 75 100)
PROMPTS=("Add sunglasses" "Make hair color green")

echo "=========================================="
echo "Starting complete pipeline execution"
echo "=========================================="

for IMAGE in "${IMAGES[@]}"; do
    IMAGE_NAME="${IMAGE%.png}"
    echo ""
    echo "=========================================="
    echo "Processing image: $IMAGE"
    echo "=========================================="
    
    for WARP in "${WARP_TYPES[@]}"; do
        echo ""
        echo "--- Warp type: $WARP ---"
        
        for ITER in "${ITERATIONS[@]}"; do
            echo ""
            echo "=== Iterations: $ITER ==="
            
            # Define paths
            VAE_OUTPUT_DIR="output_save/${IMAGE_NAME}/results/${WARP}_${ITER}"
            FLUX_OUTPUT_DIR="output_save/${IMAGE_NAME}/flux_output/${WARP}_${ITER}"
            INSTRUCT_OUTPUT_DIR="output_save/${IMAGE_NAME}/instruct_output/${WARP}_${ITER}"
            
            # Create directories
            mkdir -p "$VAE_OUTPUT_DIR"
            mkdir -p "$FLUX_OUTPUT_DIR"
            mkdir -p "$INSTRUCT_OUTPUT_DIR"
            
            # Run VAE latent adversarial optimization
            echo "Running VAE latent optimization..."
            python vae_latent_adversarial.py \
                -i "new_images/${IMAGE}" \
                --warp-type "$WARP" \
                --iterations "$ITER" \
                --verbose \
                --device cuda \
                --save-dir "$VAE_OUTPUT_DIR"
            
            # Copy vae_latent_out.png to parent directory for reference
            cp "${VAE_OUTPUT_DIR}/vae_latent_out.png" "output_save/${IMAGE_NAME}/"
            
            # Run FLUX with both prompts
            echo "Running FLUX with 'Add sunglasses'..."
            /home/interns/Desktop/clean/.venv-linux-gpu/bin/python flux.py \
                -i "${VAE_OUTPUT_DIR}/vae_latent_out.png" \
                -o "${FLUX_OUTPUT_DIR}/output_sunglasses.png" \
                -p "Add sunglasses"
            
            echo "Running FLUX with 'Make hair color green'..."
            /home/interns/Desktop/clean/.venv-linux-gpu/bin/python flux.py \
                -i "${VAE_OUTPUT_DIR}/vae_latent_out.png" \
                -o "${FLUX_OUTPUT_DIR}/output_green_hair.png" \
                -p "Make hair color green"
            
            # Run InstructPix2Pix with both prompts
            echo "Running InstructPix2Pix with 'Add sunglasses'..."
            python instruct.py \
                -i "${VAE_OUTPUT_DIR}/vae_latent_out.png" \
                -o "${INSTRUCT_OUTPUT_DIR}/output_sunglasses.png" \
                -p "Add sunglasses"
            
            echo "Running InstructPix2Pix with 'Make hair color green'..."
            python instruct.py \
                -i "${VAE_OUTPUT_DIR}/vae_latent_out.png" \
                -o "${INSTRUCT_OUTPUT_DIR}/output_green_hair.png" \
                -p "Make hair color green"
            
            echo "=== Completed: ${WARP}_${ITER} ==="
        done
        
        echo "--- Completed warp: $WARP ---"
    done
    
    echo "=========================================="
    echo "Completed image: $IMAGE"
    echo "=========================================="
done

echo ""
echo "=========================================="
echo "All pipeline executions completed!"
echo "=========================================="
