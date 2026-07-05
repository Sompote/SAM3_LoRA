#!/usr/bin/env python3
"""
Convert a RefCOCO-format dataset (e.g. RefCOCO/RefCOCO+/RefCOCOg, RRSIS-D)
into the Roboflow-style COCO layout used by the trainers in this repo.

RefCOCO format:
    <refcoco_dir>/
    ├── instances.json     # COCO-style images/annotations/categories
    └── refs(unc).p        # pickle: referring expressions, each pointing at
                           # one annotation (ann_id) with a train/val/test split

Output (what train_sam3_lora_native.py expects):
    <output_dir>/
    ├── train/
    │   ├── _annotations.coco.json
    │   └── *.jpg
    ├── valid/
    └── test/

Two conversion modes:

--mode ref (default) — true referring segmentation. Each referring expression
    becomes its own "category" whose name is the expression text, annotated
    with exactly the ONE instance it refers to. The trainers use category
    names verbatim as text prompts, so the model is trained to segment only
    the referred instance for that sentence. Other instances of the same
    class count as negatives for the query, which is exactly the referring
    semantics. The num_negatives mechanism then samples other expressions as
    hard negatives automatically.

--mode category — plain open-vocabulary detection. Keeps the original COCO
    categories and ALL annotations of the images covered by the refs; the
    referring expressions are ignored. Use this if you only need class-level
    prompts (e.g. 'airplane') instead of instance-level expressions.

Example (RRSIS-D):
    python3 convert_refcoco_to_coco.py \
        --refcoco-dir RRSIS-D/rrsisd \
        --images-dir RRSIS-D/images/rrsisd/JPEGImages \
        --output-dir data/rrsisd_coco \
        --mode ref
"""

import argparse
import json
import pickle
import shutil
from collections import defaultdict
from pathlib import Path

# RefCOCO split names -> this repo's folder names
SPLIT_MAP = {
    "train": "train",
    "val": "valid",
    "validation": "valid",
    "test": "test",
    "testA": "test",
    "testB": "test",
}


def load_refs(refcoco_dir: Path, refs_file=None):
    if refs_file:
        refs_path = refcoco_dir / refs_file
    else:
        candidates = sorted(refcoco_dir.glob("refs*.p"))
        if not candidates:
            raise FileNotFoundError(f"No refs*.p pickle found in {refcoco_dir}")
        refs_path = candidates[0]
        if len(candidates) > 1:
            print(f"Multiple refs files found, using {refs_path.name} "
                  f"(override with --refs-file)")
    print(f"Loading referring expressions from {refs_path}")
    with open(refs_path, "rb") as f:
        return pickle.load(f)


def convert(refcoco_dir, images_dir, output_dir, mode="ref",
            sentences="first", refs_file=None, symlink=False):
    refcoco_dir = Path(refcoco_dir)
    images_dir = Path(images_dir)
    output_dir = Path(output_dir)

    with open(refcoco_dir / "instances.json", "r") as f:
        instances = json.load(f)
    images_by_id = {img["id"]: img for img in instances["images"]}
    anns_by_id = {ann["id"]: ann for ann in instances["annotations"]}
    anns_by_image = defaultdict(list)
    for ann in instances["annotations"]:
        anns_by_image[ann["image_id"]].append(ann)

    refs = load_refs(refcoco_dir, refs_file)
    print(f"Loaded {len(refs)} refs, {len(images_by_id)} images, "
          f"{len(anns_by_id)} annotations")

    # Group refs by output split
    refs_by_split = defaultdict(list)
    skipped_split = 0
    for ref in refs:
        split = SPLIT_MAP.get(ref.get("split"))
        if split is None:
            skipped_split += 1
            continue
        refs_by_split[split].append(ref)
    if skipped_split:
        print(f"Warning: skipped {skipped_split} refs with unknown split names")

    for split, split_refs in sorted(refs_by_split.items()):
        split_dir = output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)

        coco = {"images": [], "annotations": [], "categories": []}
        image_ids_added = set()
        skipped_missing = 0

        def add_image(image_id):
            """Register the image entry once and copy/symlink the file."""
            img = images_by_id[image_id]
            file_name = Path(img["file_name"]).name
            if image_id not in image_ids_added:
                src = images_dir / file_name
                if not src.exists():
                    return None
                entry = dict(img)
                entry["file_name"] = file_name
                coco["images"].append(entry)
                image_ids_added.add(image_id)
                dst = split_dir / file_name
                if not dst.exists():
                    if symlink:
                        dst.symlink_to(src.resolve())
                    else:
                        shutil.copy2(src, dst)
            return file_name

        if mode == "ref":
            next_cat_id = 0
            next_ann_id = 0
            for ref in split_refs:
                ann = anns_by_id.get(ref["ann_id"])
                if ann is None or ref["image_id"] not in images_by_id:
                    skipped_missing += 1
                    continue
                if add_image(ref["image_id"]) is None:
                    skipped_missing += 1
                    continue
                sents = ref.get("sentences", [])
                if sentences == "first":
                    sents = sents[:1]
                for sent in sents:
                    text = sent["sent"].strip().lower()
                    if not text:
                        continue
                    coco["categories"].append({"id": next_cat_id, "name": text})
                    new_ann = dict(ann)
                    new_ann["id"] = next_ann_id
                    new_ann["category_id"] = next_cat_id
                    coco["annotations"].append(new_ann)
                    next_cat_id += 1
                    next_ann_id += 1
        elif mode == "category":
            coco["categories"] = instances["categories"]
            next_ann_id = 0
            split_image_ids = {r["image_id"] for r in split_refs}
            for image_id in sorted(split_image_ids):
                if image_id not in images_by_id or add_image(image_id) is None:
                    skipped_missing += 1
                    continue
                for ann in anns_by_image[image_id]:
                    new_ann = dict(ann)
                    new_ann["id"] = next_ann_id
                    coco["annotations"].append(new_ann)
                    next_ann_id += 1
        else:
            raise ValueError(f"Unknown mode: {mode}")

        with open(split_dir / "_annotations.coco.json", "w") as f:
            json.dump(coco, f)

        print(f"{split}: {len(coco['images'])} images, "
              f"{len(coco['annotations'])} annotations, "
              f"{len(coco['categories'])} categories"
              + (f" ({skipped_missing} refs/images skipped: missing file or ann)"
                 if skipped_missing else ""))

    print(f"\nDone. Point the training config at: {output_dir}")
    print('    training:\n      data_dir: "%s"' % output_dir)
    if mode == "ref":
        print("\nNote (ref mode): each expression is a category, so the "
              "category count equals the ref/sentence count — this is "
              "expected. Keep num_negatives enabled: negatives are sampled "
              "from other expressions, which is exactly the hard-negative "
              "supervision referring segmentation needs.")


def main():
    parser = argparse.ArgumentParser(
        description="Convert RefCOCO-format datasets (RefCOCO, RRSIS-D, ...) "
                    "to the COCO layout used by this repo's trainers")
    parser.add_argument("--refcoco-dir", required=True,
                        help="Directory containing instances.json and refs(*).p")
    parser.add_argument("--images-dir", required=True,
                        help="Directory containing the actual image files "
                             "(e.g. RRSIS-D/images/rrsisd/JPEGImages)")
    parser.add_argument("--output-dir", required=True,
                        help="Output root; train/valid/test folders are created inside")
    parser.add_argument("--mode", choices=["ref", "category"], default="ref",
                        help="ref: expression-level prompts (default); "
                             "category: original class-level prompts")
    parser.add_argument("--sentences", choices=["first", "all"], default="first",
                        help="Use only the first sentence per ref, or all of them "
                             "(ref mode only; default: first)")
    parser.add_argument("--refs-file", default=None,
                        help="Refs pickle filename, e.g. 'refs(unc).p' "
                             "(default: auto-detect)")
    parser.add_argument("--symlink", action="store_true",
                        help="Symlink images instead of copying (saves disk)")
    args = parser.parse_args()

    convert(args.refcoco_dir, args.images_dir, args.output_dir,
            mode=args.mode, sentences=args.sentences,
            refs_file=args.refs_file, symlink=args.symlink)


if __name__ == "__main__":
    main()
