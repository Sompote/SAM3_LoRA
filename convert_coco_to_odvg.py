#!/usr/bin/env python3
"""
Convert a COCO/Roboflow export into ODVG jsonl (training data_format: odvg).

Roboflow has no ODVG export, so this bridges the gap: point it at a split that
already has `_annotations.coco.json` and it writes `_annotations.odvg.jsonl`
alongside, preserving segmentation masks.

Where the text comes from (same resolution order as the COCO trainer):

    annotation["caption"|"description"|"text"]   -> region phrase (per SEGMENT)
    category["description"|"caption"]            -> region phrase (per CLASS)
    category["name"]                             -> region phrase (fallback)

    image["caption"|"description"]               -> grounding.caption
    otherwise the distinct region phrases joined  -> grounding.caption

A `_descriptions.json` sidecar in the split directory is honoured too.

`tokens_positive` is emitted whenever a phrase occurs verbatim in the caption,
so the phrase↔caption link ODVG defines is preserved rather than invented.

Examples:
    # one split
    python3 convert_coco_to_odvg.py --split-dir data/crack/train

    # every split under a dataset root
    python3 convert_coco_to_odvg.py --data-dir data/crack

    # boxes only, drop masks
    python3 convert_coco_to_odvg.py --data-dir data/crack --no-masks
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from dataset_formats import (
    ANNOTATION_TEXT_KEYS,
    CATEGORY_TEXT_KEYS,
    IMAGE_TEXT_KEYS,
    first_text,
    load_descriptions_sidecar,
)

DEFAULT_SPLITS = ("train", "valid", "test")


def find_token_spans(caption, phrase):
    """
    Character spans of `phrase` inside `caption`, ODVG's tokens_positive.

    Returns [] when the phrase does not occur verbatim - better to omit the
    field than to emit a span that points at the wrong words.
    """
    if not caption or not phrase:
        return []
    spans = []
    haystack = caption.lower()
    needle = phrase.lower()
    start = haystack.find(needle)
    while start != -1:
        spans.append([start, start + len(needle)])
        start = haystack.find(needle, start + 1)
    return spans


def convert_split(split_dir, keep_masks=True, output_name="_annotations.odvg.jsonl",
                  skip_empty=False):
    split_dir = Path(split_dir)
    ann_file = split_dir / "_annotations.coco.json"
    if not ann_file.exists():
        raise FileNotFoundError(f"COCO annotation file not found: {ann_file}")

    with open(ann_file, "r") as f:
        coco = json.load(f)

    images = {img["id"]: img for img in coco["images"]}
    category_defs = {cat["id"]: cat for cat in coco.get("categories", [])}
    anns_by_image = defaultdict(list)
    for ann in coco.get("annotations", []):
        anns_by_image[ann["image_id"]].append(ann)

    desc_by_annotation, desc_by_image, desc_by_category = load_descriptions_sidecar(split_dir)

    records = []
    n_regions = 0
    n_masks = 0
    n_spans = 0

    for image_id in sorted(images):
        img = images[image_id]
        annotations = anns_by_image.get(image_id, [])
        if skip_empty and not annotations:
            continue

        regions = []
        for ann in annotations:
            bbox = ann.get("bbox")
            if bbox is None or len(bbox) != 4:
                continue

            category_id = ann.get("category_id", 0)
            category_def = category_defs.get(category_id, {})
            class_name = category_def.get("name", "object")

            phrase = (
                desc_by_annotation.get(str(ann.get("id")))
                or first_text(ann, ANNOTATION_TEXT_KEYS)
                or desc_by_category.get(class_name.lower())
                or first_text(category_def, CATEGORY_TEXT_KEYS)
                or class_name
            )

            # COCO xywh -> ODVG absolute xyxy
            x, y, w, h = bbox
            region = {
                "phrase": phrase,
                "bbox": [x, y, x + w, y + h],
            }
            if keep_masks and ann.get("segmentation"):
                region["segmentation"] = ann["segmentation"]
                n_masks += 1
            regions.append(region)
            n_regions += 1

        file_name = img.get("file_name", "")
        caption = (
            desc_by_image.get(file_name)
            or desc_by_image.get(str(image_id))
            or first_text(img, IMAGE_TEXT_KEYS)
        )
        if not caption:
            # No authored caption: join the distinct phrases so the whole-image
            # query still says something true about the image.
            seen = []
            for region in regions:
                if region["phrase"] not in seen:
                    seen.append(region["phrase"])
            caption = ", ".join(seen)

        for region in regions:
            spans = find_token_spans(caption, region["phrase"])
            if spans:
                region["tokens_positive"] = spans
                n_spans += 1

        records.append({
            "filename": file_name,
            "height": img.get("height"),
            "width": img.get("width"),
            "grounding": {"caption": caption, "regions": regions},
        })

    out_path = split_dir / output_name
    with open(out_path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    print(f"{split_dir.name}: {len(records)} images, {n_regions} regions "
          f"({n_masks} with masks, {n_spans} with tokens_positive) -> {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Convert COCO/Roboflow exports to ODVG jsonl",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--data-dir", help="Dataset root containing train/valid/test")
    group.add_argument("--split-dir", help="A single split directory to convert")
    parser.add_argument("--splits", nargs="+", default=list(DEFAULT_SPLITS),
                        help=f"Splits to convert under --data-dir (default: {' '.join(DEFAULT_SPLITS)})")
    parser.add_argument("--no-masks", action="store_true",
                        help="Drop segmentation, emit boxes only")
    parser.add_argument("--skip-empty", action="store_true",
                        help="Omit images with no annotations instead of "
                             "emitting them as caption-only negatives")
    parser.add_argument("--output-name", default="_annotations.odvg.jsonl",
                        help="Output filename written into each split directory")
    args = parser.parse_args()

    if args.split_dir:
        convert_split(args.split_dir, keep_masks=not args.no_masks,
                      output_name=args.output_name, skip_empty=args.skip_empty)
        return

    data_dir = Path(args.data_dir)
    converted = 0
    for split in args.splits:
        split_dir = data_dir / split
        if not (split_dir / "_annotations.coco.json").exists():
            print(f"Skipping {split}: no _annotations.coco.json")
            continue
        convert_split(split_dir, keep_masks=not args.no_masks,
                      output_name=args.output_name, skip_empty=args.skip_empty)
        converted += 1
    if converted == 0:
        raise SystemExit(f"No convertible splits found under {data_dir}")


if __name__ == "__main__":
    main()
