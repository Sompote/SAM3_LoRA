#!/usr/bin/env python3
"""
Compare LoRA-trained model vs Base SAM3 model on single images
"""

import argparse
import os
import json
import torch
import numpy as np
from PIL import Image as PILImage
import matplotlib.pyplot as plt
import matplotlib.patches as patches
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

    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Build base model
    model = build_sam3_image_model(
        device=device,
        compile=False,
        load_from_HF=True,
        bpe_path="sam3/assets/bpe_simple_vocab_16e6.txt.gz",
        eval_mode=True
    )

    # Apply LoRA
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

    # Load weights
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
    # Load image
    pil_image = PILImage.open(image_path).convert("RGB")

    # Create datapoint
    datapoint = create_datapoint(pil_image, prompt)

    # Apply transforms
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

    # Collate and move to device
    batch = collate_fn_api([datapoint], dict_key="input")["input"]
    batch = copy_data_to_device(batch, device, non_blocking=True)

    # Forward pass
    outputs = model(batch)
    last_output = outputs[-1]

    pred_logits = last_output['pred_logits']
    pred_boxes = last_output['pred_boxes']
    pred_masks = last_output.get('pred_masks', None)

    # Get scores
    scores = pred_logits.sigmoid()[0, :, :].max(dim=-1)[0]
    keep = scores > threshold
    num_keep = keep.sum().item()

    if num_keep == 0:
        return pil_image, None, None, None, 0

    # Process boxes (convert cxcywh to xyxy)
    boxes_cxcywh = pred_boxes[0, keep]
    cx, cy, w_box, h_box = boxes_cxcywh.unbind(-1)

    orig_w, orig_h = pil_image.size
    x1 = (cx - w_box / 2) * orig_w
    y1 = (cy - h_box / 2) * orig_h
    x2 = (cx + w_box / 2) * orig_w
    y2 = (cy + h_box / 2) * orig_h

    boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=-1).cpu().numpy()

    # Process masks
    if pred_masks is not None:
        import torch.nn.functional as F
        masks_small = pred_masks[0, keep].sigmoid() > 0.5
        masks_resized = F.interpolate(
            masks_small.unsqueeze(0).float(),
            size=(orig_h, orig_w),
            mode='bilinear',
            align_corners=False
        ).squeeze(0) > 0.5
        masks_np = masks_resized.cpu().numpy()
    else:
        masks_np = None

    scores_np = scores[keep].cpu().numpy()

    return pil_image, boxes_xyxy, scores_np, masks_np, num_keep


def load_ground_truth(image_path, data_dir):
    """Load ground truth annotations"""
    image_path = Path(image_path)

    # Load annotations
    ann_file = Path(data_dir) / "_annotations.coco.json"
    if not ann_file.exists():
        return [], None

    with open(ann_file, 'r') as f:
        coco_data = json.load(f)

    # Find image
    image_name = image_path.name
    image_info = None
    for img in coco_data['images']:
        if img['file_name'] == image_name:
            image_info = img
            break

    if image_info is None:
        return [], None

    # Get category name (use as prompt)
    categories = {cat['id']: cat['name'] for cat in coco_data['categories']}

    # Get annotations
    annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] == image_info['id']]

    # Process masks
    gt_masks = []
    orig_h, orig_w = image_info['height'], image_info['width']
    prompt = None

    for ann in annotations:
        # Get category name for prompt
        if prompt is None and ann['category_id'] in categories:
            prompt = categories[ann['category_id']]

        # Get segmentation
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


def visualize_comparison(image_path, lora_results, base_results, gt_masks, output_path, prompt):
    """Create side-by-side comparison visualization with ground truth"""
    pil_image, lora_boxes, lora_scores, lora_masks, lora_count = lora_results
    _, base_boxes, base_scores, base_masks, base_count = base_results

    # Create figure with 3 panels
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))

    # Ground Truth
    axes[0].imshow(pil_image)
    if len(gt_masks) > 0:
        overlay = np.zeros((pil_image.size[1], pil_image.size[0], 4))
        for mask in gt_masks:
            overlay[mask > 0] = [0, 1, 0, 0.5]  # Green
        axes[0].imshow(overlay)
    axes[0].set_title(f'Ground Truth ({len(gt_masks)} masks)', fontsize=14, fontweight='bold')
    axes[0].axis('off')

    # LoRA predictions
    axes[1].imshow(pil_image)
    if lora_count > 0 and lora_masks is not None:
        overlay = np.zeros((pil_image.size[1], pil_image.size[0], 4))
        for mask in lora_masks:
            overlay[mask] = [1, 0, 0, 0.5]  # Red
        axes[1].imshow(overlay)
    axes[1].set_title(f'LoRA Model ({lora_count} detections)', fontsize=14, fontweight='bold')
    axes[1].axis('off')

    # Base predictions
    axes[2].imshow(pil_image)
    if base_count > 0 and base_masks is not None:
        overlay = np.zeros((pil_image.size[1], pil_image.size[0], 4))
        for mask in base_masks:
            overlay[mask] = [0, 0, 1, 0.5]  # Blue
        axes[2].imshow(overlay)
    axes[2].set_title(f'Base Model ({base_count} detections)', fontsize=14, fontweight='bold')
    axes[2].axis('off')

    # Overall title
    plt.suptitle(f'Prompt: "{prompt}" | Image: {os.path.basename(image_path)}',
                 fontsize=16, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()

    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Compare LoRA vs Base SAM3 model')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--data-dir', type=str, required=True, help='Data directory with COCO annotations')
    parser.add_argument('--config', type=str, default='configs/full_lora_config.yaml',
                        help='Path to LoRA config')
    parser.add_argument('--weights', type=str, default='outputs/sam3_lora_full/best_lora_weights.pt',
                        help='Path to LoRA weights')
    parser.add_argument('--output', type=str, required=True, help='Output comparison image path')
    parser.add_argument('--threshold', type=float, default=0.5, help='Detection threshold')
    parser.add_argument('--resolution', type=int, default=1008, help='Input resolution')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load ground truth and get prompt from category
    print("Loading ground truth...")
    gt_masks, prompt = load_ground_truth(args.image, args.data_dir)

    if prompt is None:
        print(f"Warning: No ground truth found, using default prompt 'pothole'")
        prompt = "pothole"

    print(f"Device: {device}")
    print(f"Image: {args.image}")
    print(f"Prompt: {prompt}")
    print(f"Ground truth masks: {len(gt_masks)}")

    # Load models
    lora_model = load_lora_model(args.config, args.weights, device)
    base_model = load_base_model(device)

    # Run predictions
    print("\nRunning LoRA model...")
    lora_results = predict(lora_model, args.image, prompt,
                          args.resolution, args.threshold, device)

    print("Running Base model...")
    base_results = predict(base_model, args.image, prompt,
                          args.resolution, args.threshold, device)

    # Create visualization
    print("\nCreating comparison...")
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
    visualize_comparison(args.image, lora_results, base_results, gt_masks, args.output, prompt)

    print(f"\nGround Truth: {len(gt_masks)} masks")
    print(f"LoRA: {lora_results[4]} detections")
    print(f"Base: {base_results[4]} detections")
    print("\nDone!")


if __name__ == "__main__":
    main()
