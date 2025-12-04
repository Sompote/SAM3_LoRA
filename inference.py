"""
SAM3 LoRA Inference Script
Run inference with a LoRA fine-tuned SAM3 model.
"""

import argparse
from pathlib import Path
from typing import Optional, List
import json

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from transformers import Sam3Model, Sam3Processor

from lora_layers import apply_lora_to_model, load_lora_weights, LoRAConfig


def load_model_with_lora(
    base_model_name: str,
    lora_weights_path: str,
    lora_config_path: Optional[str] = None,
    device: str = "cuda",
):
    """
    Load SAM3 model with LoRA weights.

    Args:
        base_model_name: Base SAM3 model name/path
        lora_weights_path: Path to LoRA weights
        lora_config_path: Path to LoRA config (optional)
        device: Device to load model on

    Returns:
        Model and processor
    """
    print(f"Loading base model: {base_model_name}")
    model = Sam3Model.from_pretrained(base_model_name)
    processor = Sam3Processor.from_pretrained(base_model_name)

    # Load LoRA config if provided
    if lora_config_path:
        import yaml
        with open(lora_config_path, "r") as f:
            config = yaml.safe_load(f)
        lora_config = LoRAConfig(**config["lora"])
    else:
        # Use default config
        lora_config = LoRAConfig()

    print("Applying LoRA to model...")
    model = apply_lora_to_model(model, lora_config)

    print(f"Loading LoRA weights from: {lora_weights_path}")
    load_lora_weights(model, lora_weights_path)

    model.to(device)
    model.eval()

    return model, processor


def segment_image(
    model: Sam3Model,
    processor: Sam3Processor,
    image: Image.Image,
    text_prompt: Optional[str] = None,
    bboxes: Optional[List[List[int]]] = None,
    device: str = "cuda",
):
    """
    Segment an image using SAM3 with prompts.

    Args:
        model: SAM3 model with LoRA
        processor: SAM3 processor
        image: Input PIL image
        text_prompt: Text prompt (e.g., "yellow school bus")
        bboxes: List of bounding boxes [[x1, y1, x2, y2], ...]
        device: Device

    Returns:
        Predicted masks
    """
    # Prepare inputs
    inputs = processor(
        images=image,
        text=text_prompt if text_prompt else None,
        boxes=bboxes if bboxes else None,
        return_tensors="pt",
    )

    # Move to device
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
              for k, v in inputs.items()}

    # Run inference
    with torch.no_grad():
        outputs = model(**inputs)

    # Get masks
    pred_masks = torch.sigmoid(outputs.pred_masks)
    pred_masks = pred_masks.cpu().numpy()

    return pred_masks


def visualize_results(
    image: Image.Image,
    masks: np.ndarray,
    text_prompt: Optional[str] = None,
    save_path: Optional[str] = None,
):
    """
    Visualize segmentation results.

    Args:
        image: Input image
        masks: Predicted masks [N, H, W]
        text_prompt: Text prompt used
        save_path: Path to save visualization
    """
    fig, axes = plt.subplots(1, min(4, len(masks) + 1), figsize=(15, 5))
    if len(masks) == 0:
        axes = [axes]

    # Show original image
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # Show masks
    for idx, mask in enumerate(masks[:3]):
        if idx + 1 < len(axes):
            axes[idx + 1].imshow(image)
            axes[idx + 1].imshow(mask[0], alpha=0.5, cmap="jet")
            axes[idx + 1].set_title(f"Mask {idx + 1}")
            axes[idx + 1].axis("off")

    if text_prompt:
        fig.suptitle(f'Prompt: "{text_prompt}"', fontsize=14)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved visualization to: {save_path}")
    else:
        plt.show()

    plt.close()


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(
        description="Run inference with LoRA fine-tuned SAM3"
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="facebook/sam3",
        help="Base SAM3 model name",
    )
    parser.add_argument(
        "--lora_weights",
        type=str,
        required=True,
        help="Path to LoRA weights file",
    )
    parser.add_argument(
        "--lora_config",
        type=str,
        help="Path to LoRA config file (optional)",
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to input image",
    )
    parser.add_argument(
        "--text_prompt",
        type=str,
        help="Text prompt for segmentation (e.g., 'yellow school bus')",
    )
    parser.add_argument(
        "--bboxes",
        type=str,
        help="Bounding boxes as JSON string: '[[x1,y1,x2,y2], ...]'",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to save output visualization",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on",
    )

    args = parser.parse_args()

    # Load model
    model, processor = load_model_with_lora(
        base_model_name=args.model_name,
        lora_weights_path=args.lora_weights,
        lora_config_path=args.lora_config,
        device=args.device,
    )

    # Load image
    print(f"Loading image: {args.image}")
    image = Image.open(args.image).convert("RGB")

    # Parse bboxes if provided
    bboxes = None
    if args.bboxes:
        bboxes = json.loads(args.bboxes)
        print(f"Using bounding boxes: {bboxes}")

    # Run segmentation
    print("Running segmentation...")
    masks = segment_image(
        model=model,
        processor=processor,
        image=image,
        text_prompt=args.text_prompt,
        bboxes=bboxes,
        device=args.device,
    )

    print(f"Generated {len(masks)} masks")

    # Visualize results
    visualize_results(
        image=image,
        masks=masks,
        text_prompt=args.text_prompt,
        save_path=args.output,
    )

    print("Inference completed!")


if __name__ == "__main__":
    main()
