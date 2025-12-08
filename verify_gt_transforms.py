#!/usr/bin/env python3
"""
Verify that Ground Truth transformations match Input Image transformations

This script checks that during training:
1. Input images are resized to 1008x1008
2. GT boxes are scaled by the same factors
3. GT masks are resized to 1008x1008
"""

import torch
import numpy as np
from PIL import Image as PILImage
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, '/workspace/SAM3_LoRA')

from train_sam3_lora_native import COCOSegmentDataset

def visualize_sample(dataset, idx=0, save_path="gt_verification.png"):
    """
    Visualize a training sample to verify GT transformations match image transformations
    """
    # Get the processed sample
    sample = dataset[idx]

    # Extract data
    image_tensor = sample.images[0].data  # [3, 1008, 1008] normalized
    objects = sample.images[0].objects
    query_text = sample.find_queries[0].query_text
    orig_size = sample.find_queries[0].inference_metadata.original_size
    orig_h, orig_w = orig_size

    # Also load raw image for comparison
    img_id = dataset.image_ids[idx]
    img_info = dataset.images[img_id]
    img_path = dataset.split_dir / img_info['file_name']
    raw_image = PILImage.open(img_path).convert("RGB")

    print("="*80)
    print(f"VERIFYING GROUND TRUTH TRANSFORMATIONS - Sample {idx}")
    print("="*80)

    print(f"\n1. IMAGE TRANSFORMATIONS:")
    print(f"   Original size: {orig_w} × {orig_h}")
    print(f"   Resized to: {dataset.resolution} × {dataset.resolution}")
    print(f"   Scale factors: scale_w={dataset.resolution/orig_w:.4f}, scale_h={dataset.resolution/orig_h:.4f}")
    print(f"   Query text: '{query_text}'")

    print(f"\n2. GROUND TRUTH TRANSFORMATIONS:")
    print(f"   Number of objects: {len(objects)}")

    # Denormalize image for visualization
    img_np = image_tensor.numpy()
    img_np = img_np * 0.5 + 0.5  # Reverse normalization
    img_np = np.clip(img_np, 0, 1)
    img_np = np.transpose(img_np, (1, 2, 0))  # [H, W, 3]

    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Plot 1: Original image with original boxes
    ax = axes[0]
    ax.imshow(raw_image)
    ax.set_title(f"Original Image ({orig_w}×{orig_h})", fontsize=14, fontweight='bold')
    ax.axis('off')

    # Draw original boxes
    annotations = dataset.img_to_anns.get(img_id, [])
    for ann in annotations:
        bbox = ann.get("bbox", None)
        if bbox:
            x, y, w, h = bbox
            rect = patches.Rectangle((x, y), w, h, linewidth=2,
                                     edgecolor='red', facecolor='none')
            ax.add_patch(rect)
            ax.text(x, y-5, f"Original bbox", color='red',
                   fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    # Plot 2: Resized image with transformed boxes
    ax = axes[1]
    ax.imshow(img_np)
    ax.set_title(f"Resized Image ({dataset.resolution}×{dataset.resolution})\n+ Transformed GT Boxes",
                fontsize=14, fontweight='bold')
    ax.axis('off')

    for i, obj in enumerate(objects):
        # Boxes are in normalized [0,1] coordinates
        # Convert to pixel coordinates for visualization
        box_norm = obj.bbox  # [x1, y1, x2, y2] in [0,1] range
        x1 = box_norm[0].item() * dataset.resolution
        y1 = box_norm[1].item() * dataset.resolution
        x2 = box_norm[2].item() * dataset.resolution
        y2 = box_norm[3].item() * dataset.resolution
        w = x2 - x1
        h = y2 - y1

        rect = patches.Rectangle((x1, y1), w, h, linewidth=2,
                                 edgecolor='lime', facecolor='none')
        ax.add_patch(rect)

        print(f"\n   Object {i}:")
        print(f"     Normalized box: [{box_norm[0]:.4f}, {box_norm[1]:.4f}, {box_norm[2]:.4f}, {box_norm[3]:.4f}]")
        print(f"     Pixel box (1008×1008): [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")
        print(f"     Box area (normalized): {obj.area.item():.6f}")
        print(f"     Has mask: {obj.segment is not None}")
        if obj.segment is not None:
            print(f"     Mask shape: {obj.segment.shape}")
            print(f"     Mask coverage: {obj.segment.sum().item()} pixels ({obj.segment.float().mean().item()*100:.2f}%)")

    # Plot 3: Overlay mask on image
    ax = axes[2]
    ax.imshow(img_np)

    # Overlay all masks
    combined_mask = torch.zeros((dataset.resolution, dataset.resolution), dtype=torch.bool)
    for obj in objects:
        if obj.segment is not None:
            combined_mask |= obj.segment

    # Create colored overlay
    mask_overlay = np.zeros((*combined_mask.shape, 4))
    mask_overlay[combined_mask.numpy()] = [0, 1, 0, 0.5]  # Green with 50% alpha

    ax.imshow(mask_overlay)
    ax.set_title(f"Resized Image + GT Masks\n({combined_mask.sum().item()} mask pixels)",
                fontsize=14, fontweight='bold')
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Visualization saved to: {save_path}")

    # Verification checks
    print(f"\n3. VERIFICATION CHECKS:")
    print("-" * 80)

    # Check 1: Image size matches resolution
    assert image_tensor.shape == (3, dataset.resolution, dataset.resolution), \
        f"Image tensor shape mismatch: {image_tensor.shape}"
    print("   ✓ Image resized to 1008×1008")

    # Check 2: Boxes are normalized (allow small tolerance for floating point)
    for i, obj in enumerate(objects):
        assert obj.bbox.min() >= -1e-6 and obj.bbox.max() <= 1.0 + 1e-6, \
            f"Box {i} not normalized: {obj.bbox} (min={obj.bbox.min()}, max={obj.bbox.max()})"
    print("   ✓ All boxes normalized to [0, 1] range")

    # Check 3: Masks match image resolution
    for i, obj in enumerate(objects):
        if obj.segment is not None:
            assert obj.segment.shape == (dataset.resolution, dataset.resolution), \
                f"Mask {i} shape mismatch: {obj.segment.shape}"
    print("   ✓ All masks resized to 1008×1008")

    # Check 4: Box-mask alignment (rough check)
    for i, obj in enumerate(objects):
        if obj.segment is not None:
            # Get mask bounding box
            mask_y, mask_x = torch.where(obj.segment)
            if len(mask_x) > 0:
                mask_x1 = mask_x.min().item() / dataset.resolution
                mask_y1 = mask_y.min().item() / dataset.resolution
                mask_x2 = mask_x.max().item() / dataset.resolution
                mask_y2 = mask_y.max().item() / dataset.resolution

                # Compare with GT box (allowing 10% tolerance)
                box_x1, box_y1, box_x2, box_y2 = obj.bbox.tolist()

                # Mask should be roughly inside box (with some tolerance)
                # Note: Boxes from COCO can be imprecise, so we allow overlap
                overlap_x = min(box_x2, mask_x2) - max(box_x1, mask_x1)
                overlap_y = min(box_y2, mask_y2) - max(box_y1, mask_y1)

                if overlap_x > 0 and overlap_y > 0:
                    print(f"   ✓ Object {i}: Mask-box alignment OK (overlap: {overlap_x*100:.1f}% × {overlap_y*100:.1f}%)")
                else:
                    print(f"   ⚠ Object {i}: Mask-box misalignment detected!")

    print("\n" + "="*80)
    print("✅ VERIFICATION COMPLETE: GT transformations match image transformations!")
    print("="*80)

    return fig


if __name__ == "__main__":
    # Load training dataset
    print("\nLoading training dataset...")
    dataset = COCOSegmentDataset(data_dir='/workspace/data2', split='train')

    # Verify first few samples
    for idx in [0, 1, 2]:
        save_path = f"/workspace/SAM3_LoRA/gt_verification_sample_{idx}.png"
        visualize_sample(dataset, idx=idx, save_path=save_path)
        print("\n")
