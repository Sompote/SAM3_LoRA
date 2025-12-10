#!/bin/bash
# Compare LoRA vs Base model on 5 validation images

OUTPUT_DIR="comparison_outputs"
mkdir -p "$OUTPUT_DIR"

# 5 randomly selected validation images
IMAGES=(
  "img_0034_jpg.rf.4070bd337ac7ca2868490ac3968a8d91.jpg"
  "img_0001_jpg.rf.ca0b564fc4064d14246f4db4563041e1.jpg"
  "img_0080_jpg.rf.851383d982ff0f6e94dd4fc222cd77d3.jpg"
  "img_0070_jpg.rf.5763253f32c578319a52091ee2dc662b.jpg"
  "img_0060_jpg.rf.566e934d48f13f2fe049f0d06a5810ed.jpg"
)

VAL_DIR="/workspace/data3/valid"
DATA_DIR="/workspace/data3/valid"

echo "==============================================================="
echo "Visual Comparison: LoRA vs Base Model (with Ground Truth)"
echo "==============================================================="
echo ""
echo "Generating predictions for ${#IMAGES[@]} validation images..."
echo "Output directory: $OUTPUT_DIR/"
echo ""

for img in "${IMAGES[@]}"; do
  img_path="$VAL_DIR/$img"
  basename="${img%.*}"
  output_path="$OUTPUT_DIR/comparison_$img"

  echo "Processing: $img"
  python3 compare_lora_base.py \
    --image "$img_path" \
    --data-dir "$DATA_DIR" \
    --config configs/full_lora_config.yaml \
    --weights outputs/sam3_lora_full/best_lora_weights.pt \
    --output "$output_path" \
    --threshold 0.5 \
    --resolution 1008 2>&1 | grep -E "Saved|Ground Truth:|LoRA:|Base:|Prompt:|Error" || true

  echo ""
done

echo "==============================================================="
echo "Comparison complete!"
echo "==============================================================="
echo "Check the outputs in: $OUTPUT_DIR/"
echo ""
ls -lh "$OUTPUT_DIR/"

