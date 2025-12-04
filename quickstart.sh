#!/bin/bash
# Quick Start Script for SAM3 LoRA Training

set -e

echo "======================================"
echo "SAM3 LoRA Training - Quick Start"
echo "======================================"

# Check Python version
echo -e "\n[1/6] Checking Python version..."
python --version

# Install dependencies
echo -e "\n[2/6] Installing dependencies..."
pip install -r requirements.txt

# Check HuggingFace login
echo -e "\n[3/6] Checking HuggingFace authentication..."
if huggingface-cli whoami > /dev/null 2>&1; then
    echo "✓ Already logged in to HuggingFace"
else
    echo "Please login to HuggingFace (required for SAM3 access):"
    huggingface-cli login
fi

# Create data directory structure
echo -e "\n[4/6] Creating data directory structure..."
python prepare_data.py create --output_dir data

# Run example usage
echo -e "\n[5/6] Running example usage (this may take a few minutes)..."
echo "This will download the SAM3 model (~3GB) and test LoRA application."
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    python example_usage.py
else
    echo "Skipped example usage"
fi

# Print next steps
echo -e "\n[6/6] Setup complete!"
echo "======================================"
echo "Next Steps:"
echo "======================================"
echo ""
echo "1. Prepare your data:"
echo "   python prepare_data.py coco --coco_json path/to/annotations.json --images_dir path/to/images --output_dir data"
echo "   OR"
echo "   python prepare_data.py yolo --yolo_dir path/to/yolo --classes 'cat,dog,car' --output_dir data"
echo ""
echo "2. Validate your data:"
echo "   python prepare_data.py validate --data_dir data --split train"
echo ""
echo "3. Configure training:"
echo "   Edit configs/base_config.yaml or use configs/full_lora_config.yaml"
echo ""
echo "4. Start training:"
echo "   python train_sam3_lora.py --config configs/base_config.yaml"
echo ""
echo "5. Run inference:"
echo "   python inference.py --model_name facebook/sam3 --lora_weights outputs/sam3_lora/best_model/lora_weights.pt --image test.jpg --text_prompt 'your object'"
echo ""
echo "See README.md for detailed documentation!"
echo "======================================"
