#!/bin/bash
# Edit all images in new_images/ with FLUX and InstructPix2Pix

INPUT_DIR="new_images"
OUTPUT_DIR="edit_output"

# Create output directory structure
mkdir -p "$OUTPUT_DIR/flux"
mkdir -p "$OUTPUT_DIR/instruct"

# Prompts
PROMPT1="Add sunglasses"
PROMPT2="Change hair color to green"

# Get all image files
IMAGES=$(ls "$INPUT_DIR"/*.png "$INPUT_DIR"/*.jpg 2>/dev/null)

echo "=========================================="
echo "Starting image editing pipeline"
echo "=========================================="
echo "Input directory: $INPUT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Prompts: '$PROMPT1', '$PROMPT2'"
echo "=========================================="

for img in $IMAGES; do
    img_name=$(basename "$img" | sed 's/\.[^.]*$//')
    echo ""
    echo "=== Processing: $img_name ==="
    
    # FLUX - Add sunglasses
    echo "Running FLUX with '$PROMPT1'..."
    /home/interns/Desktop/clean/.venv-linux-gpu/bin/python flux.py \
        -i "$img" \
        -o "$OUTPUT_DIR/flux/${img_name}_sunglasses.png" \
        -p "$PROMPT1"
    
    # FLUX - Change hair color
    echo "Running FLUX with '$PROMPT2'..."
    /home/interns/Desktop/clean/.venv-linux-gpu/bin/python flux.py \
        -i "$img" \
        -o "$OUTPUT_DIR/flux/${img_name}_green_hair.png" \
        -p "$PROMPT2"
    
    # InstructPix2Pix - Add sunglasses
    echo "Running InstructPix2Pix with '$PROMPT1'..."
    python instruct.py \
        -i "$img" \
        -o "$OUTPUT_DIR/instruct/${img_name}_sunglasses.png" \
        -p "$PROMPT1"
    
    # InstructPix2Pix - Change hair color
    echo "Running InstructPix2Pix with '$PROMPT2'..."
    python instruct.py \
        -i "$img" \
        -o "$OUTPUT_DIR/instruct/${img_name}_green_hair.png" \
        -p "$PROMPT2"
    
    echo "=== Completed: $img_name ==="
done

echo ""
echo "=========================================="
echo "All edits completed!"
echo "FLUX outputs: $OUTPUT_DIR/flux/"
echo "InstructPix2Pix outputs: $OUTPUT_DIR/instruct/"
echo "=========================================="
