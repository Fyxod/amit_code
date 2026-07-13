#!/bin/bash
# Complete pipeline script for all combinations
# Images: image_2.png, image_3.png
# Warps: bspline, all
# Iterations: 10, 15, 20, 50, 75, 100
# Prompts: "Add sunglasses", "Make hair color green"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Define arrays
IMAGES=("image_2.png" "image_3.png")
WARP_TYPES=("bspline" "all")
ITERATIONS=(10 15 20 50 75 100)

echo "=========================================="
echo "Starting complete pipeline execution"
echo "=========================================="
echo "Script is resumable - will skip completed combinations"
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
            
            # Check if this combination is already completed
            if [[ -f "${FLUX_OUTPUT_DIR}/output_sunglasses.png" && \
                  -f "${FLUX_OUTPUT_DIR}/output_green_hair.png" && \
                  -f "${INSTRUCT_OUTPUT_DIR}/output_sunglasses.png" && \
                  -f "${INSTRUCT_OUTPUT_DIR}/output_green_hair.png" && \
                  -f "${VAE_OUTPUT_DIR}/vae_latent_out.png" ]]; then
                echo "SKIP: ${IMAGE_NAME}/${WARP}_${ITER} already completed"
                continue
            fi
            
            echo "Processing: ${IMAGE_NAME}/${WARP}_${ITER}"
            
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
                --save-dir "$VAE_OUTPUT_DIR" || {
                    echo "ERROR: VAE optimization failed for ${IMAGE_NAME}/${WARP}_${ITER}"
                    echo "Continuing to next combination..."
                    continue
                }
            
            # Copy vae_latent_out.png to parent directory for reference
            cp "${VAE_OUTPUT_DIR}/vae_latent_out.png" "output_save/${IMAGE_NAME}/"
            
            # Run FLUX with both prompts
            echo "Running FLUX with 'Add sunglasses'..."
            /home/interns/Desktop/clean/.venv-linux-gpu/bin/python flux.py \
                -i "${VAE_OUTPUT_DIR}/vae_latent_out.png" \
                -o "${FLUX_OUTPUT_DIR}/output_sunglasses.png" \
                -p "Add sunglasses" || {
                    echo "ERROR: FLUX sunglasses failed for ${IMAGE_NAME}/${WARP}_${ITER}"
                    echo "Continuing to next combination..."
                    continue
                }
            
            echo "Running FLUX with 'Make hair color green'..."
            /home/interns/Desktop/clean/.venv-linux-gpu/bin/python flux.py \
                -i "${VAE_OUTPUT_DIR}/vae_latent_out.png" \
                -o "${FLUX_OUTPUT_DIR}/output_green_hair.png" \
                -p "Make hair color green" || {
                    echo "ERROR: FLUX green hair failed for ${IMAGE_NAME}/${WARP}_${ITER}"
                    echo "Continuing to next combination..."
                    continue
                }
            
            # Run InstructPix2Pix with both prompts
            echo "Running InstructPix2Pix with 'Add sunglasses'..."
            python instruct.py \
                -i "${VAE_OUTPUT_DIR}/vae_latent_out.png" \
                -o "${INSTRUCT_OUTPUT_DIR}/output_sunglasses.png" \
                -p "Add sunglasses" || {
                    echo "ERROR: Instruct sunglasses failed for ${IMAGE_NAME}/${WARP}_${ITER}"
                    echo "Continuing to next combination..."
                    continue
                }
            
            echo "Running InstructPix2Pix with 'Make hair color green'..."
            python instruct.py \
                -i "${VAE_OUTPUT_DIR}/vae_latent_out.png" \
                -o "${INSTRUCT_OUTPUT_DIR}/output_green_hair.png" \
                -p "Make hair color green" || {
                    echo "ERROR: Instruct green hair failed for ${IMAGE_NAME}/${WARP}_${ITER}"
                    echo "Continuing to next combination..."
                    continue
                }
            
            echo "=== Completed: ${IMAGE_NAME}/${WARP}_${ITER} ==="
            
            # Git commit only (no push)
            echo "Committing results..."
            git add -A
            if git diff --cached --quiet; then
                echo "No changes to commit"
            else
                git commit -m "Add results for ${IMAGE_NAME} ${WARP}_${ITER} iterations
- VAE: ${WARP} warp with ${ITER} iterations
- FLUX: Add sunglasses, Make hair color green
- InstructPix2Pix: Add sunglasses, Make hair color green"
            fi
            
            echo "=== Committed: ${IMAGE_NAME}/${WARP}_${ITER} ==="
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
