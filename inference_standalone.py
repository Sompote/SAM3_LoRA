#!/usr/bin/env python3
"""
Inference script for Standalone SAM3 LoRA.

This script loads a trained LoRA checkpoint and runs inference using the
SimpleSegmentationModel. It is designed to verify that the trained LoRA
adapters can be successfully loaded and used.
"""

import argparse
import os
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from sam3_lora import LoRAConfig, inject_lora_into_model, load_lora_state_dict
from sam3_lora.model import SimpleSegmentationModel

def load_image(image_path, size=(256, 256)):
    """Load and preprocess image."""
    if not os.path.exists(image_path):
        # Create dummy image if not found
        print(f"Image {image_path} not found, creating dummy image.")
        img = np.random.randint(0, 255, (size[0], size[1], 3), dtype=np.uint8)
        return torch.tensor(img).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    
    img = Image.open(image_path).convert('RGB')
    img = img.resize(size)
    # Convert to tensor (1, 3, H, W) - though SimpleModel takes (B, Seq, Dim) usually?
    # Wait, SimpleSegmentationModel forward takes (batch, seq, d_model)
    # But SimpleLoRATrainer used: x = torch.randn(batch_size, 10, 256)
    # So the model expects embeddings, not raw images.
    
    # For this demo, we will generate random embeddings as input 
    # because SimpleSegmentationModel doesn't have a real image encoder.
    print("Note: SimpleSegmentationModel expects embeddings. Generating random input.")
    return torch.randn(1, 10, 256) 

def main():
    parser = argparse.ArgumentParser(description="SAM3 LoRA Standalone Inference")
    parser.add_argument("--config", type=str, default="configs/sam3_lora_standalone.yaml",
                        help="Path to config YAML file")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained LoRA checkpoint (e.g., checkpoints/best.pt)")
    parser.add_argument("--image", type=str, default="test_image.jpg",
                        help="Path to input image (unused for SimpleModel but good for API compat)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run on")
    
    args = parser.parse_args()
    
    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. Load Configuration
    print(f"Loading config from {args.config}")
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # 2. Create Model
    print("Creating SimpleSegmentationModel...")
    model = SimpleSegmentationModel(d_model=256, nhead=8, dim_feedforward=1024)
    
    # 3. Setup LoRA Config
    print("Setting up LoRA configuration...")
    lora_config = LoRAConfig(
        rank=config["lora"]["rank"],
        alpha=config["lora"]["alpha"],
        dropout=config["lora"].get("dropout", 0.0),
        target_modules=config["lora"].get("target_modules", ["q_proj", "k_proj", "v_proj", "out_proj"])
    )

    # 4. Inject LoRA
    print("Injecting LoRA adapters...")
    model = inject_lora_into_model(model, lora_config, verbose=True)
    
    # 5. Load Checkpoint
    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    if "lora_state_dict" in checkpoint:
        load_lora_state_dict(model, checkpoint["lora_state_dict"])
        print("✓ LoRA weights loaded successfully")
    else:
        print("✗ Error: Checkpoint does not contain 'lora_state_dict'")
        return

    model.to(device)
    model.eval()

    # 6. Run Inference
    print("\nRunning inference...")
    input_tensor = load_image(args.image).to(device)
    
    with torch.no_grad():
        output = model(input_tensor)
    
    print(f"Output shape: {output.shape}")
    print(f"Output value mean: {output.mean().item():.4f}")
    print("Inference successful!")
    
    # Since it's a dummy model, we can't visualize much, but we can show we ran it.
    print("\nNote: This is a standalone demo model. For real SAM3 inference,")
    print("ensure you use the full SAM3 model and weights.")

if __name__ == "__main__":
    main()
