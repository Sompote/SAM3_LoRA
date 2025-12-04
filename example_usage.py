"""
Example Usage of SAM3 LoRA Training

This script demonstrates how to use the SAM3 LoRA training code programmatically.
"""

import torch
from transformers import Sam3Model, Sam3Processor

from lora_layers import (
    LoRAConfig,
    apply_lora_to_model,
    count_parameters,
    save_lora_weights,
    load_lora_weights,
)


def example_1_basic_lora_application():
    """
    Example 1: Apply LoRA to SAM3 model with default settings.
    """
    print("=" * 60)
    print("Example 1: Basic LoRA Application")
    print("=" * 60)

    # Load base model
    print("\nLoading SAM3 model...")
    model = Sam3Model.from_pretrained("facebook/sam3")

    # Create LoRA config with default settings
    lora_config = LoRAConfig(
        rank=8,
        alpha=16,
        dropout=0.0,
    )

    # Apply LoRA
    print("\nApplying LoRA...")
    model = apply_lora_to_model(model, lora_config)

    # Check parameters
    stats = count_parameters(model)
    print(f"\nModel Statistics:")
    print(f"  Total parameters: {stats['total_parameters']:,}")
    print(f"  Trainable parameters: {stats['trainable_parameters']:,}")
    print(f"  Trainable: {stats['trainable_percentage']:.2f}%")


def example_2_selective_lora():
    """
    Example 2: Apply LoRA selectively to specific components.
    """
    print("\n" + "=" * 60)
    print("Example 2: Selective LoRA Application")
    print("=" * 60)

    # Load model
    model = Sam3Model.from_pretrained("facebook/sam3")

    # Create LoRA config for decoder only
    lora_config = LoRAConfig(
        rank=8,
        alpha=16,
        # Apply only to DETR decoder
        apply_to_vision_encoder=False,
        apply_to_text_encoder=False,
        apply_to_geometry_encoder=False,
        apply_to_detr_encoder=False,
        apply_to_detr_decoder=True,  # Only this one
        apply_to_mask_decoder=False,
    )

    # Apply LoRA
    model = apply_lora_to_model(model, lora_config)

    # Check parameters
    stats = count_parameters(model)
    print(f"\nModel Statistics (Decoder Only):")
    print(f"  Trainable: {stats['trainable_percentage']:.2f}%")
    print(f"  This is much more efficient!")


def example_3_aggressive_lora():
    """
    Example 3: Apply LoRA aggressively to all components.
    """
    print("\n" + "=" * 60)
    print("Example 3: Aggressive LoRA Application")
    print("=" * 60)

    model = Sam3Model.from_pretrained("facebook/sam3")

    # Higher rank, more modules, all components
    lora_config = LoRAConfig(
        rank=16,
        alpha=32,
        dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
        # Apply to everything
        apply_to_vision_encoder=True,
        apply_to_text_encoder=True,
        apply_to_geometry_encoder=True,
        apply_to_detr_encoder=True,
        apply_to_detr_decoder=True,
        apply_to_mask_decoder=True,
    )

    model = apply_lora_to_model(model, lora_config)

    stats = count_parameters(model)
    print(f"\nModel Statistics (Full LoRA):")
    print(f"  Trainable: {stats['trainable_percentage']:.2f}%")
    print(f"  Maximum adaptation capacity!")


def example_4_save_and_load():
    """
    Example 4: Save and load LoRA weights.
    """
    print("\n" + "=" * 60)
    print("Example 4: Save and Load LoRA Weights")
    print("=" * 60)

    # Create model with LoRA
    model = Sam3Model.from_pretrained("facebook/sam3")
    lora_config = LoRAConfig(rank=8, alpha=16)
    model = apply_lora_to_model(model, lora_config)

    # Save LoRA weights (only a few MB!)
    save_path = "lora_weights_example.pt"
    print(f"\nSaving LoRA weights to: {save_path}")
    save_lora_weights(model, save_path)

    # Load into a fresh model
    print("\nLoading into fresh model...")
    fresh_model = Sam3Model.from_pretrained("facebook/sam3")
    fresh_model = apply_lora_to_model(fresh_model, lora_config)
    load_lora_weights(fresh_model, save_path)

    print("Successfully loaded LoRA weights!")


def example_5_inference():
    """
    Example 5: Run inference with LoRA model.
    """
    print("\n" + "=" * 60)
    print("Example 5: Inference with LoRA Model")
    print("=" * 60)

    # Load model with LoRA
    model = Sam3Model.from_pretrained("facebook/sam3")
    processor = Sam3Processor.from_pretrained("facebook/sam3")

    lora_config = LoRAConfig(rank=8, alpha=16)
    model = apply_lora_to_model(model, lora_config)

    # In practice, you would load trained weights here:
    # load_lora_weights(model, "path/to/trained_weights.pt")

    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print(f"\nModel ready for inference on {device}")
    print("You can now use it with:")
    print("  - Text prompts: 'yellow school bus'")
    print("  - Bounding boxes: [[x1, y1, x2, y2]]")
    print("  - Or both!")

    # Example input preparation
    from PIL import Image
    import numpy as np

    # Create dummy image
    dummy_image = Image.fromarray(
        np.random.randint(0, 255, (1008, 1008, 3), dtype=np.uint8)
    )

    inputs = processor(
        images=dummy_image,
        text="example object",
        return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Run inference
    with torch.no_grad():
        outputs = model(**inputs)

    print(f"\nInference output shape: {outputs.pred_masks.shape}")
    print("Success!")


def example_6_config_comparison():
    """
    Example 6: Compare different LoRA configurations.
    """
    print("\n" + "=" * 60)
    print("Example 6: Configuration Comparison")
    print("=" * 60)

    configs = {
        "Minimal (r=4, decoder only)": LoRAConfig(
            rank=4,
            alpha=8,
            apply_to_vision_encoder=False,
            apply_to_text_encoder=False,
            apply_to_detr_encoder=False,
            apply_to_detr_decoder=True,
        ),
        "Balanced (r=8, encoders + decoder)": LoRAConfig(
            rank=8,
            alpha=16,
            apply_to_vision_encoder=True,
            apply_to_detr_encoder=True,
            apply_to_detr_decoder=True,
        ),
        "Full (r=16, all components)": LoRAConfig(
            rank=16,
            alpha=32,
            apply_to_vision_encoder=True,
            apply_to_text_encoder=True,
            apply_to_geometry_encoder=True,
            apply_to_detr_encoder=True,
            apply_to_detr_decoder=True,
            apply_to_mask_decoder=True,
        ),
    }

    print("\nConfiguration Comparison:")
    print("-" * 60)

    for name, config in configs.items():
        model = Sam3Model.from_pretrained("facebook/sam3")
        model = apply_lora_to_model(model, config)
        stats = count_parameters(model)

        print(f"\n{name}:")
        print(f"  Trainable params: {stats['trainable_parameters']:,}")
        print(f"  Percentage: {stats['trainable_percentage']:.3f}%")


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("SAM3 LoRA Examples")
    print("=" * 60)

    # Run examples
    try:
        example_1_basic_lora_application()
        example_2_selective_lora()
        example_3_aggressive_lora()
        example_4_save_and_load()
        example_5_inference()
        example_6_config_comparison()

    except Exception as e:
        print(f"\nError running examples: {e}")
        print("Make sure you have:")
        print("  1. Installed all requirements: pip install -r requirements.txt")
        print("  2. Logged in to HuggingFace: huggingface-cli login")
        print("  3. Sufficient GPU memory (or run on CPU)")

    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
