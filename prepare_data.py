"""
Data Preparation Utility for SAM3 LoRA Training

This script helps convert various annotation formats to SAM3 training format.
Supports COCO format, Pascal VOC, and custom formats.
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Any
import shutil

import numpy as np
from PIL import Image
from tqdm import tqdm


def create_dataset_structure(output_dir: str):
    """Create the required directory structure."""
    output_path = Path(output_dir)

    # Create directories
    for split in ["train", "val"]:
        (output_path / split / "images").mkdir(parents=True, exist_ok=True)
        (output_path / split / "annotations").mkdir(parents=True, exist_ok=True)

    print(f"Created dataset structure at: {output_dir}")


def convert_coco_to_sam3(
    coco_json_path: str,
    images_dir: str,
    output_dir: str,
    split: str = "train",
):
    """
    Convert COCO format annotations to SAM3 format.

    COCO format:
    {
        "images": [{"id": 1, "file_name": "img.jpg", ...}],
        "annotations": [{"image_id": 1, "category_id": 1, "bbox": [x,y,w,h], ...}],
        "categories": [{"id": 1, "name": "cat"}, ...]
    }
    """
    print(f"Converting COCO annotations from: {coco_json_path}")

    # Load COCO annotations
    with open(coco_json_path, "r") as f:
        coco_data = json.load(f)

    # Create category mapping
    categories = {cat["id"]: cat["name"] for cat in coco_data["categories"]}

    # Group annotations by image
    image_to_anns = {}
    for ann in coco_data["annotations"]:
        image_id = ann["image_id"]
        if image_id not in image_to_anns:
            image_to_anns[image_id] = []
        image_to_anns[image_id].append(ann)

    # Create image mapping
    images = {img["id"]: img for img in coco_data["images"]}

    # Process each image
    output_path = Path(output_dir) / split
    processed = 0

    for image_id, image_info in tqdm(images.items(), desc="Converting"):
        if image_id not in image_to_anns:
            continue

        # Copy image
        src_image_path = Path(images_dir) / image_info["file_name"]
        if not src_image_path.exists():
            print(f"Warning: Image not found: {src_image_path}")
            continue

        dst_image_path = output_path / "images" / image_info["file_name"]
        shutil.copy2(src_image_path, dst_image_path)

        # Convert annotations
        anns = image_to_anns[image_id]
        bboxes = []
        masks = []
        category_names = []

        for ann in anns:
            # Convert COCO bbox [x, y, width, height] to [x1, y1, x2, y2]
            x, y, w, h = ann["bbox"]
            bboxes.append([int(x), int(y), int(x + w), int(y + h)])

            # Get category name
            cat_name = categories.get(ann["category_id"], "object")
            category_names.append(cat_name)

            # TODO: Convert segmentation to mask if available
            # For now, we'll leave masks empty and use bboxes

        # Create SAM3 annotation
        sam3_annotation = {
            "text_prompt": ", ".join(set(category_names)),  # e.g., "cat, dog"
            "bboxes": bboxes,
            "masks": masks,  # Empty for now, can be generated from segmentation
        }

        # Save annotation
        annotation_path = output_path / "annotations" / f"{Path(image_info['file_name']).stem}.json"
        with open(annotation_path, "w") as f:
            json.dump(sam3_annotation, f, indent=2)

        processed += 1

    print(f"Converted {processed} images to SAM3 format")


def convert_yolo_to_sam3(
    yolo_data_dir: str,
    output_dir: str,
    class_names: List[str],
    split: str = "train",
):
    """
    Convert YOLO format to SAM3 format.

    YOLO format:
    - Images in: {yolo_data_dir}/images/{split}/
    - Labels in: {yolo_data_dir}/labels/{split}/
    - Each label file: class_id x_center y_center width height (normalized)
    """
    print(f"Converting YOLO annotations from: {yolo_data_dir}")

    yolo_path = Path(yolo_data_dir)
    images_dir = yolo_path / "images" / split
    labels_dir = yolo_path / "labels" / split

    if not images_dir.exists():
        raise ValueError(f"Images directory not found: {images_dir}")
    if not labels_dir.exists():
        raise ValueError(f"Labels directory not found: {labels_dir}")

    output_path = Path(output_dir) / split

    # Process each image
    image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
    processed = 0

    for image_file in tqdm(image_files, desc="Converting"):
        # Load image to get dimensions
        image = Image.open(image_file)
        img_width, img_height = image.size

        # Copy image
        dst_image_path = output_path / "images" / image_file.name
        shutil.copy2(image_file, dst_image_path)

        # Load YOLO label
        label_file = labels_dir / f"{image_file.stem}.txt"
        if not label_file.exists():
            print(f"Warning: Label not found for {image_file.name}")
            continue

        bboxes = []
        category_names = []

        with open(label_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue

                class_id = int(parts[0])
                x_center, y_center, width, height = map(float, parts[1:5])

                # Convert normalized YOLO to absolute coordinates
                x_center *= img_width
                y_center *= img_height
                width *= img_width
                height *= img_height

                # Convert to [x1, y1, x2, y2]
                x1 = int(x_center - width / 2)
                y1 = int(y_center - height / 2)
                x2 = int(x_center + width / 2)
                y2 = int(y_center + height / 2)

                bboxes.append([x1, y1, x2, y2])
                category_names.append(class_names[class_id] if class_id < len(class_names) else f"class_{class_id}")

        # Create SAM3 annotation
        sam3_annotation = {
            "text_prompt": ", ".join(set(category_names)),
            "bboxes": bboxes,
            "masks": [],
        }

        # Save annotation
        annotation_path = output_path / "annotations" / f"{image_file.stem}.json"
        with open(annotation_path, "w") as f:
            json.dump(sam3_annotation, f, indent=2)

        processed += 1

    print(f"Converted {processed} images to SAM3 format")


def validate_dataset(data_dir: str, split: str = "train"):
    """
    Validate SAM3 dataset format.

    Checks:
    - Image and annotation files match
    - Annotations are valid JSON
    - Required fields are present
    """
    print(f"Validating {split} dataset...")

    data_path = Path(data_dir) / split
    images_dir = data_path / "images"
    annotations_dir = data_path / "annotations"

    image_files = sorted(list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")))
    errors = []
    warnings = []

    for image_file in tqdm(image_files, desc="Validating"):
        # Check annotation exists
        annotation_file = annotations_dir / f"{image_file.stem}.json"
        if not annotation_file.exists():
            errors.append(f"Missing annotation for: {image_file.name}")
            continue

        # Load and validate annotation
        try:
            with open(annotation_file, "r") as f:
                annotation = json.load(f)

            # Check required fields
            if "text_prompt" not in annotation and "bboxes" not in annotation:
                warnings.append(f"No prompts in: {image_file.name}")

            # Validate bboxes format
            if "bboxes" in annotation:
                for bbox in annotation["bboxes"]:
                    if len(bbox) != 4:
                        errors.append(f"Invalid bbox format in: {image_file.name}")

        except json.JSONDecodeError:
            errors.append(f"Invalid JSON in: {annotation_file.name}")
        except Exception as e:
            errors.append(f"Error processing {image_file.name}: {str(e)}")

    # Print results
    print(f"\nValidation Results:")
    print(f"  Total images: {len(image_files)}")
    print(f"  Errors: {len(errors)}")
    print(f"  Warnings: {len(warnings)}")

    if errors:
        print("\nErrors:")
        for error in errors[:10]:
            print(f"  - {error}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")

    if warnings:
        print("\nWarnings:")
        for warning in warnings[:10]:
            print(f"  - {warning}")
        if len(warnings) > 10:
            print(f"  ... and {len(warnings) - 10} more")

    return len(errors) == 0


def main():
    parser = argparse.ArgumentParser(
        description="Prepare data for SAM3 LoRA training"
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Create structure command
    create_parser = subparsers.add_parser("create", help="Create dataset structure")
    create_parser.add_argument("--output_dir", required=True, help="Output directory")

    # Convert COCO command
    coco_parser = subparsers.add_parser("coco", help="Convert COCO format")
    coco_parser.add_argument("--coco_json", required=True, help="COCO JSON file")
    coco_parser.add_argument("--images_dir", required=True, help="Images directory")
    coco_parser.add_argument("--output_dir", required=True, help="Output directory")
    coco_parser.add_argument("--split", default="train", help="Split name (train/val)")

    # Convert YOLO command
    yolo_parser = subparsers.add_parser("yolo", help="Convert YOLO format")
    yolo_parser.add_argument("--yolo_dir", required=True, help="YOLO data directory")
    yolo_parser.add_argument("--output_dir", required=True, help="Output directory")
    yolo_parser.add_argument("--classes", required=True, help="Comma-separated class names")
    yolo_parser.add_argument("--split", default="train", help="Split name (train/val)")

    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate dataset")
    validate_parser.add_argument("--data_dir", required=True, help="Dataset directory")
    validate_parser.add_argument("--split", default="train", help="Split name (train/val)")

    args = parser.parse_args()

    if args.command == "create":
        create_dataset_structure(args.output_dir)

    elif args.command == "coco":
        create_dataset_structure(args.output_dir)
        convert_coco_to_sam3(
            args.coco_json,
            args.images_dir,
            args.output_dir,
            args.split,
        )

    elif args.command == "yolo":
        create_dataset_structure(args.output_dir)
        class_names = [name.strip() for name in args.classes.split(",")]
        convert_yolo_to_sam3(
            args.yolo_dir,
            args.output_dir,
            class_names,
            args.split,
        )

    elif args.command == "validate":
        is_valid = validate_dataset(args.data_dir, args.split)
        if is_valid:
            print("\n✓ Dataset is valid!")
        else:
            print("\n✗ Dataset has errors!")
            exit(1)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
