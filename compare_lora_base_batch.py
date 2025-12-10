#!/usr/bin/env python3
"""
Compare LoRA vs Base model on multiple images in a single visualization
"""

import argparse
import os
import json
import torch
import numpy as np
from PIL import Image as PILImage
import matplotlib.pyplot as plt
import yaml
import pycocotools.mask as mask_utils
from pathlib import Path

# SAM3 imports
from sam3.model_builder import build_sam3_image_model
from sam3.train.data.sam3_image_dataset import (
    Datapoint,
    Image as SAMImage,
    FindQueryLoaded,
    InferenceMetadata
)
from sam3.train.data.collator import collate_fn_api
from sam3.model.utils.misc import copy_data_to_device
from sam3.train.transforms.basic_for_api import (
    ComposeAPI,
    RandomResizeAPI,
    ToTensorAPI,
    NormalizeAPI,
)

# LoRA imports
from lora_layers import LoRAConfig, apply_lora_to_model, load_lora_weights


def load_lora_model(config_path, weights_path, device='cuda'):
    """Load SAM3 model with LoRA weights"""
    print("Loading LoRA model...")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    model = build_sam3_image_model(
        device=device,
        compile=False,
        load_from_HF=True,
        bpe_path="sam3/assets/bpe_simple_vocab_16e6.txt.gz",
        eval_mode=True
    )

    lora_cfg = config["lora"]
    lora_config = LoRAConfig(
        rank=lora_cfg["rank"],
        alpha=lora_cfg["alpha"],
        dropout=0.0,
        target_modules=lora_cfg["target_modules"],
        apply_to_vision_encoder=lora_cfg["apply_to_vision_encoder"],
        apply_to_text_encoder=lora_cfg["apply_to_text_encoder"],
        apply_to_geometry_encoder=lora_cfg["apply_to_geometry_encoder"],
        apply_to_detr_encoder=lora_cfg["apply_to_detr_encoder"],
        apply_to_detr_decoder=lora_cfg["apply_to_detr_decoder"],
        apply_to_mask_decoder=lora_cfg["apply_to_mask_decoder"],
    )
    model = apply_lora_to_model(model, lora_config)
    load_lora_weights(model, weights_path)
    model.to(device)
    model.eval()

    return model


def load_base_model(device='cuda'):
    """Load base SAM3 model without LoRA"""
    print("Loading base model...")

    model = build_sam3_image_model(
        device=device,
        compile=False,
        load_from_HF=True,
        bpe_path="sam3/assets/bpe_simple_vocab_16e6.txt.gz",
        eval_mode=True
    )
    model.to(device)
    model.eval()

    return model


def create_datapoint(pil_image, prompt):
    """Create SAM3 datapoint"""
    w, h = pil_image.size

    sam_image = SAMImage(
        data=pil_image,
        objects=[],
        size=[h, w]
    )

    query = FindQueryLoaded(
        query_text=prompt,
        image_id=0,
        object_ids_output=[],
        is_exhaustive=True,
        query_processing_order=0,
        inference_metadata=InferenceMetadata(
            coco_image_id=0,
            original_image_id=0,
            original_category_id=1,
            original_size=[w, h],
            object_id=0,
            frame_index=0,
        )
    )

    return Datapoint(
        find_queries=[query],
        images=[sam_image]
    )


@torch.no_grad()
def predict(model, image_path, prompt, resolution=1008, threshold=0.5, device='cuda'):
    """Run inference on image"""
    pil_image = PILImage.open(image_path).convert("RGB")
    datapoint = create_datapoint(pil_image, prompt)

    transform = ComposeAPI(
        transforms=[
            RandomResizeAPI(
                sizes=resolution,
                max_size=resolution,
                square=True,
                consistent_transform=False
            ),
            ToTensorAPI(),
            NormalizeAPI(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    datapoint = transform(datapoint)

    batch = collate_fn_api([datapoint], dict_key="input")["input"]
    batch = copy_data_to_device(batch, device, non_blocking=True)

    outputs = model(batch)
    last_output = outputs[-1]

    pred_logits = last_output['pred_logits']
    pred_boxes = last_output['pred_boxes']
    pred_masks = last_output.get('pred_masks', None)

    scores = pred_logits.sigmoid()[0, :, :].max(dim=-1)[0]
    keep = scores > threshold
    num_keep = keep.sum().item()

    if num_keep == 0:
        return pil_image, None, 0

    if pred_masks is not None:
        import torch.nn.functional as F
        masks_small = pred_masks[0, keep].sigmoid() > 0.5
        orig_h, orig_w = pil_image.size[1], pil_image.size[0]
        masks_resized = F.interpolate(
            masks_small.unsqueeze(0).float(),
            size=(orig_h, orig_w),
            mode='bilinear',
            align_corners=False
        ).squeeze(0) > 0.5
        masks_np = masks_resized.cpu().numpy()
    else:
        masks_np = None

    return pil_image, masks_np, num_keep


def load_ground_truth(image_path, data_dir):
    """Load ground truth annotations"""
    image_path = Path(image_path)

    ann_file = Path(data_dir) / "_annotations.coco.json"
    if not ann_file.exists():
        return [], None

    with open(ann_file, 'r') as f:
        coco_data = json.load(f)

    image_name = image_path.name
    image_info = None
    for img in coco_data['images']:
        if img['file_name'] == image_name:
            image_info = img
            break

    if image_info is None:
        return [], None

    categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
    annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] == image_info['id']]

    gt_masks = []
    orig_h, orig_w = image_info['height'], image_info['width']
    prompt = None

    for ann in annotations:
        if prompt is None and ann['category_id'] in categories:
            prompt = categories[ann['category_id']]

        segmentation = ann.get('segmentation', None)
        if segmentation:
            try:
                if isinstance(segmentation, dict):
                    mask_np = mask_utils.decode(segmentation)
                elif isinstance(segmentation, list):
                    rles = mask_utils.frPyObjects(segmentation, orig_h, orig_w)
                    rle = mask_utils.merge(rles)
                    mask_np = mask_utils.decode(rle)
                else:
                    continue

                gt_masks.append(mask_np)
            except Exception as e:
                print(f"Error processing annotation: {e}")

    return gt_masks, prompt


def create_combined_visualization(image_results, output_path):
    """Create a single large visualization with all images"""
    num_images = len(image_results)

    # Create figure: 3 columns (GT, LoRA, Base) x N rows (images)
    fig, axes = plt.subplots(num_images, 3, figsize=(18, 6 * num_images))

    # Handle single image case
    if num_images == 1:
        axes = axes.reshape(1, -1)

    for idx, result in enumerate(image_results):
        image_name = result['image_name']
        pil_image = result['pil_image']
        gt_masks = result['gt_masks']
        lora_masks = result['lora_masks']
        lora_count = result['lora_count']
        base_masks = result['base_masks']
        base_count = result['base_count']
        prompt = result['prompt']

        # Ground Truth
        axes[idx, 0].imshow(pil_image)
        if len(gt_masks) > 0:
            overlay = np.zeros((pil_image.size[1], pil_image.size[0], 4))
            for mask in gt_masks:
                overlay[mask > 0] = [0, 1, 0, 0.5]  # Green
            axes[idx, 0].imshow(overlay)
        axes[idx, 0].set_title(f'GT ({len(gt_masks)})', fontsize=12, fontweight='bold')
        axes[idx, 0].axis('off')

        # Add image name on the left
        axes[idx, 0].text(-0.1, 0.5, image_name,
                         transform=axes[idx, 0].transAxes,
                         fontsize=10, rotation=90,
                         verticalalignment='center',
                         horizontalalignment='right')

        # LoRA predictions
        axes[idx, 1].imshow(pil_image)
        if lora_count > 0 and lora_masks is not None:
            overlay = np.zeros((pil_image.size[1], pil_image.size[0], 4))
            for mask in lora_masks:
                overlay[mask] = [1, 0, 0, 0.5]  # Red
            axes[idx, 1].imshow(overlay)
        axes[idx, 1].set_title(f'LoRA ({lora_count})', fontsize=12, fontweight='bold')
        axes[idx, 1].axis('off')

        # Base predictions
        axes[idx, 2].imshow(pil_image)
        if base_count > 0 and base_masks is not None:
            overlay = np.zeros((pil_image.size[1], pil_image.size[0], 4))
            for mask in base_masks:
                overlay[mask] = [0, 0, 1, 0.5]  # Blue
            axes[idx, 2].imshow(overlay)
        axes[idx, 2].set_title(f'Base ({base_count})', fontsize=12, fontweight='bold')
        axes[idx, 2].axis('off')

    plt.suptitle(f'Model Comparison - Prompt: "{image_results[0]["prompt"]}"',
                 fontsize=16, fontweight='bold', y=0.995)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()

    print(f"\nSaved combined visualization to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Compare LoRA vs Base on multiple images')
    parser.add_argument('--images', type=str, nargs='+', required=True, help='Paths to input images')
    parser.add_argument('--data-dir', type=str, required=True, help='Data directory with COCO annotations')
    parser.add_argument('--config', type=str, default='configs/full_lora_config.yaml',
                        help='Path to LoRA config')
    parser.add_argument('--weights', type=str, default='outputs/sam3_lora_full/best_lora_weights.pt',
                        help='Path to LoRA weights')
    parser.add_argument('--output', type=str, required=True, help='Output combined image path')
    parser.add_argument('--threshold', type=float, default=0.5, help='Detection threshold')
    parser.add_argument('--resolution', type=int, default=1008, help='Input resolution')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Processing {len(args.images)} images...")

    # Load models once
    lora_model = load_lora_model(args.config, args.weights, device)
    base_model = load_base_model(device)

    # Process all images
    results = []

    for img_idx, image_path in enumerate(args.images, 1):
        print(f"\n[{img_idx}/{len(args.images)}] Processing: {os.path.basename(image_path)}")

        # Load ground truth
        gt_masks, prompt = load_ground_truth(image_path, args.data_dir)
        if prompt is None:
            prompt = "pothole"

        print(f"  GT: {len(gt_masks)} masks | Prompt: {prompt}")

        # Run LoRA model
        pil_image, lora_masks, lora_count = predict(
            lora_model, image_path, prompt, args.resolution, args.threshold, device
        )
        print(f"  LoRA: {lora_count} detections")

        # Run Base model
        _, base_masks, base_count = predict(
            base_model, image_path, prompt, args.resolution, args.threshold, device
        )
        print(f"  Base: {base_count} detections")

        results.append({
            'image_name': os.path.basename(image_path),
            'pil_image': pil_image,
            'gt_masks': gt_masks,
            'lora_masks': lora_masks,
            'lora_count': lora_count,
            'base_masks': base_masks,
            'base_count': base_count,
            'prompt': prompt
        })

    # Create combined visualization
    print("\nCreating combined visualization...")
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
    create_combined_visualization(results, args.output)

    # Print summary
    print("\n" + "="*60)
    print("Summary:")
    for result in results:
        print(f"{result['image_name']:50s} | GT:{len(result['gt_masks'])} LoRA:{result['lora_count']} Base:{result['base_count']}")
    print("="*60)


if __name__ == "__main__":
    main()
