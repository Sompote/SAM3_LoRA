#!/usr/bin/env python3
"""
Text-prompt dataset formats for SAM3-LoRA training.

SAM3 is prompted with text, and until now this repo always built that text from
the COCO *category name*. Two formats here let a dataset supply a real
description instead — per segment, per image, or per class.

--------------------------------------------------------------------------
Format 1: "coco" (default) — COCO/Roboflow layout with descriptions
--------------------------------------------------------------------------
Same `<data_dir>/<split>/_annotations.coco.json` you already use. The prompt
text for each annotation is resolved in this order (first hit wins):

    annotation["caption"|"description"|"text"]     per SEGMENT
    image["caption"|"description"]                 per IMAGE
    category["description"|"caption"]              per CLASS  (RF100-VL style)
    category["name"]                               class name (old behavior)

Descriptions may also live in a sidecar `_descriptions.json` next to the
annotations, so a Roboflow re-export never clobbers them:

    {
      "by_annotation": {"12": "the hairline crack at the slab edge"},
      "by_image":      {"pier_07.jpg": "cracks on a concrete bridge pier"},
      "by_category":   {"crack": "a long thin fracture in the concrete surface"}
    }

--------------------------------------------------------------------------
Format 2: "odvg" — Grounding-DINO ODVG jsonl
--------------------------------------------------------------------------
`<data_dir>/<split>/_annotations.odvg.jsonl`, one JSON object per line:

    {"filename": "pier_07.jpg", "height": 693, "width": 1024,
     "grounding": {
       "caption": "a long diagonal crack under the ledge and rust below it",
       "regions": [
         {"phrase": "a long diagonal crack under the ledge",
          "bbox": [20, 215, 985, 665],
          "tokens_positive": [[0, 37]],
          "segmentation": [[x1, y1, x2, y2, ...]]},
         {"phrase": "rust", "bbox": [...], "segmentation": {"counts": ..., "size": ...}}
       ]}}

Detection mode is also accepted:

    {"filename": "...", "height": h, "width": w,
     "detection": {"instances": [{"bbox": [x1,y1,x2,y2], "label": 0,
                                  "category": "crack", "mask": {...}}]}}

`bbox` is ODVG's absolute xyxy. Masks are optional (`segmentation` or `mask`,
polygon list or RLE dict); without them the sample trains boxes only.

--------------------------------------------------------------------------
Negative prompts
--------------------------------------------------------------------------
The existing two-tier hard-negative scheme is preserved. With class-name
prompts the pool is tiny, so every absent class is ALWAYS added as a
must-return-nothing query. With free-text descriptions the pool is as large as
the dataset, so absent texts are SAMPLED (`num_cross_negatives` per image)
instead; the switch happens automatically at `max_always_negatives`.
"""

import json
import random
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image as PILImage
from torchvision.transforms import v2
import pycocotools.mask as mask_utils

from sam3.train.data.sam3_image_dataset import (
    Datapoint, Image, Object, FindQueryLoaded, InferenceMetadata,
)

DEFAULT_RESOLUTION = 1008

# Keys searched for a description, in priority order, at each level.
ANNOTATION_TEXT_KEYS = ("caption", "description", "text", "phrase", "expression")
IMAGE_TEXT_KEYS = ("caption", "description", "text")
CATEGORY_TEXT_KEYS = ("description", "caption", "instructions", "prompt")

# Generic out-of-domain concepts used as additional hard negatives, so the
# model learns to NOT fire on unrelated prompts (e.g. 'car' on a crack set).
DEFAULT_GENERIC_NEGATIVES = [
    "car", "person", "dog", "cat", "tree", "building", "sky", "road",
    "chair", "table", "bottle", "phone", "bicycle", "bird", "flower",
    "water", "grass", "cloud", "window", "door",
]

# Above this many distinct prompt texts, "always add every absent prompt as a
# negative" would explode the query count per image, so we sample instead.
MAX_ALWAYS_NEGATIVES = 50

PROMPT_SOURCES = ("auto", "caption", "image_caption", "category_description",
                  "category_name")


def build_transform():
    """Image normalization shared by every format."""
    return v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


def first_text(container, keys):
    """Return the first non-empty string found under `keys`, else None."""
    if not isinstance(container, dict):
        return None
    for key in keys:
        value = container.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def load_descriptions_sidecar(split_dir, filename="_descriptions.json"):
    """
    Load the optional `_descriptions.json` sidecar.

    Returns (by_annotation, by_image, by_category) dicts, empty when absent.
    Keys are normalized to str / lowercase so lookups are forgiving.
    """
    path = Path(split_dir) / filename
    if not path.exists():
        return {}, {}, {}

    with open(path, "r") as f:
        data = json.load(f)

    by_annotation = {str(k): v for k, v in (data.get("by_annotation") or {}).items()}
    by_image = {str(k): v for k, v in (data.get("by_image") or {}).items()}
    by_category = {str(k).lower(): v for k, v in (data.get("by_category") or {}).items()}
    print(f"Loaded descriptions sidecar {path.name}: "
          f"{len(by_annotation)} per-annotation, {len(by_image)} per-image, "
          f"{len(by_category)} per-category")
    return by_annotation, by_image, by_category


def decode_segmentation(segmentation, orig_h, orig_w, resolution):
    """
    Decode a COCO polygon list or RLE dict into a boolean mask at `resolution`.

    Returns None when the annotation carries no usable mask.
    """
    if not segmentation:
        return None
    try:
        if isinstance(segmentation, dict):
            # RLE: {"counts": ..., "size": [h, w]}. Uncompressed RLE stores
            # counts as a list, which pycocotools needs re-encoded first.
            rle = segmentation
            if isinstance(rle.get("counts"), list):
                rle = mask_utils.frPyObjects(rle, rle["size"][0], rle["size"][1])
            mask_np = mask_utils.decode(rle)
        elif isinstance(segmentation, list):
            # Polygon: [[x1, y1, x2, y2, ...], ...]
            rles = mask_utils.frPyObjects(segmentation, orig_h, orig_w)
            rle = mask_utils.merge(rles)
            mask_np = mask_utils.decode(rle)
        else:
            print(f"Warning: unknown segmentation format: {type(segmentation)}")
            return None

        if mask_np.ndim == 3:
            mask_np = mask_np[..., 0]

        mask_t = torch.from_numpy(mask_np).float().unsqueeze(0).unsqueeze(0)
        mask_t = torch.nn.functional.interpolate(
            mask_t, size=(resolution, resolution), mode="nearest"
        )
        return mask_t.squeeze() > 0.5
    except Exception as e:
        print(f"Warning: error decoding mask: {e}")
        return None


def make_query(query_text, object_ids, img_id, orig_h, orig_w, is_negative=False):
    """Build one FindQueryLoaded. Negatives carry category_id -1 and no targets."""
    return FindQueryLoaded(
        query_text=query_text,
        image_id=0,
        object_ids_output=list(object_ids),
        is_exhaustive=True,
        query_processing_order=0,
        inference_metadata=InferenceMetadata(
            coco_image_id=img_id,
            original_image_id=img_id,
            original_category_id=-1 if is_negative else 0,
            original_size=(orig_h, orig_w),
            object_id=-1,
            frame_index=-1,
        ),
    )


def filter_generic_negatives(generic_negatives, prompt_texts):
    """
    Drop generic negatives sharing a word with any positive prompt.

    Training 'water' to return nothing contradicts a 'water ingress' class, and
    the same trap appears with descriptions ('road' vs 'crack in the road').
    """
    prompt_words = {w for text in prompt_texts for w in text.lower().split()}
    return [n for n in generic_negatives
            if not set(n.lower().split()) & prompt_words]


def build_negative_queries(present_texts, prompt_corpus, generic_negatives,
                           num_negatives, num_cross_negatives,
                           img_id, orig_h, orig_w,
                           max_always_negatives=MAX_ALWAYS_NEGATIVES,
                           rng=random):
    """
    Build the must-return-nothing queries for one image.

    Tier 1 (in-domain): prompts from the dataset that this image does not
        contain. A small corpus (class names) contributes ALL of them, because
        confusable classes need negative supervision on every image. A large
        corpus (free-text descriptions) contributes `num_cross_negatives`
        sampled entries, otherwise a single image would produce thousands of
        queries.
    Tier 2 (out-of-domain): `num_negatives` generic concepts sampled per image.
    """
    queries = []
    present = {t.lower() for t in present_texts}

    absent = sorted({t.lower() for t in prompt_corpus} - present)
    if len(prompt_corpus) <= max_always_negatives:
        chosen = absent
    elif num_cross_negatives > 0 and absent:
        chosen = rng.sample(absent, min(num_cross_negatives, len(absent)))
    else:
        chosen = []
    for text in chosen:
        queries.append(make_query(text, [], img_id, orig_h, orig_w, is_negative=True))

    generic_pool = sorted({n.lower() for n in generic_negatives} - present)
    if num_negatives > 0 and generic_pool:
        for text in rng.sample(generic_pool, min(num_negatives, len(generic_pool))):
            queries.append(make_query(text, [], img_id, orig_h, orig_w, is_negative=True))

    return queries


def xyxy_to_normalized_cxcywh(x1, y1, x2, y2, orig_w, orig_h):
    """ODVG absolute xyxy -> SAM3's normalized CxCyWH."""
    w = max(x2 - x1, 0.0)
    h = max(y2 - y1, 0.0)
    return torch.tensor([
        (x1 + w / 2.0) / orig_w,
        (y1 + h / 2.0) / orig_h,
        w / orig_w,
        h / orig_h,
    ], dtype=torch.float32)


class ODVGDataset(torch.utils.data.Dataset):
    """
    Grounding-DINO ODVG jsonl dataset (format 2).

    Every region carries its own `phrase`, so descriptions are per SEGMENT.
    Regions sharing a phrase within one image are grouped into a single query
    with several targets, rather than competing single-target queries.

    When `use_full_caption` is set and the line has `grounding.caption`, an
    extra whole-image query is emitted targeting every region — training the
    sentence and its constituent phrases from one annotation.
    """

    def __init__(self, data_dir, split="train", num_negatives=2,
                 generic_negatives=None, num_cross_negatives=2,
                 use_full_caption=True, annotations_file="_annotations.odvg.jsonl",
                 resolution=DEFAULT_RESOLUTION):
        self.data_dir = Path(data_dir)
        self.split = split
        self.split_dir = self.data_dir / split
        self.num_negatives = num_negatives
        self.num_cross_negatives = num_cross_negatives
        self.use_full_caption = use_full_caption
        self.resolution = resolution

        ann_file = self.split_dir / annotations_file
        if not ann_file.exists():
            raise FileNotFoundError(f"ODVG annotation file not found: {ann_file}")

        self.records = []
        with open(ann_file, "r") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    self.records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise ValueError(f"{ann_file}:{line_no} is not valid JSON: {e}")

        if not self.records:
            raise ValueError(f"No records found in {ann_file}")

        # image_ids are positional; ODVG has no id field.
        self.image_ids = list(range(len(self.records)))

        # Corpus of every phrase in the split -> the in-domain negative pool.
        self.prompt_corpus = sorted({
            p.lower() for rec in self.records for p in self._record_phrases(rec)
        })
        self.generic_negatives = filter_generic_negatives(
            list(generic_negatives if generic_negatives is not None
                 else DEFAULT_GENERIC_NEGATIVES),
            self.prompt_corpus,
        )

        n_regions = sum(len(self._record_regions(rec)) for rec in self.records)
        print(f"Loaded ODVG dataset: {split} split")
        print(f"  Images: {len(self.records)}")
        print(f"  Regions: {n_regions}")
        print(f"  Distinct phrases: {len(self.prompt_corpus)}")
        if len(self.prompt_corpus) > MAX_ALWAYS_NEGATIVES:
            print(f"  Phrase corpus > {MAX_ALWAYS_NEGATIVES}: sampling "
                  f"{self.num_cross_negatives} cross-phrase negatives per image")

        self.transform = build_transform()

    @staticmethod
    def _record_regions(record):
        """Normalize grounding.regions / detection.instances into one list."""
        grounding = record.get("grounding") or {}
        regions = grounding.get("regions")
        if regions:
            return regions
        detection = record.get("detection") or {}
        return detection.get("instances") or []

    @classmethod
    def _record_phrases(cls, record):
        out = []
        for region in cls._record_regions(record):
            phrase = first_text(region, ("phrase", "caption", "category", "text"))
            if phrase:
                out.append(phrase)
        return out

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        img_id = idx

        img_path = self.split_dir / record["filename"]
        pil_image = PILImage.open(img_path).convert("RGB")
        orig_w, orig_h = pil_image.size
        # Trust the image on disk over the record's declared size.
        orig_w = record.get("width", orig_w) or orig_w
        orig_h = record.get("height", orig_h) or orig_h

        pil_image = pil_image.resize((self.resolution, self.resolution),
                                     PILImage.BILINEAR)
        image_tensor = self.transform(pil_image)

        objects = []
        phrase_to_object_ids = defaultdict(list)

        for i, region in enumerate(self._record_regions(record)):
            bbox = region.get("bbox")
            if bbox is None or len(bbox) != 4:
                continue
            phrase = first_text(region, ("phrase", "caption", "category", "text"))
            if not phrase:
                continue

            box_tensor = xyxy_to_normalized_cxcywh(*bbox, orig_w=orig_w, orig_h=orig_h)
            segment = decode_segmentation(
                region.get("segmentation") or region.get("mask"),
                orig_h, orig_w, self.resolution,
            )

            obj = Object(
                bbox=box_tensor,
                area=(box_tensor[2] * box_tensor[3]).item(),
                object_id=len(objects),
                segment=segment,
            )
            objects.append(obj)
            phrase_to_object_ids[phrase.lower()].append(obj.object_id)

        image_obj = Image(
            data=image_tensor,
            objects=objects,
            size=(self.resolution, self.resolution),
        )

        queries = [
            make_query(phrase, obj_ids, img_id, orig_h, orig_w)
            for phrase, obj_ids in phrase_to_object_ids.items()
        ]

        # Whole-image sentence covering every region it grounds.
        caption = first_text(record.get("grounding") or {}, IMAGE_TEXT_KEYS)
        if self.use_full_caption and caption and objects:
            if caption.lower() not in phrase_to_object_ids:
                queries.append(make_query(
                    caption, [obj.object_id for obj in objects],
                    img_id, orig_h, orig_w,
                ))

        if not queries:
            queries.append(make_query("object", [], img_id, orig_h, orig_w))

        present_texts = set(phrase_to_object_ids)
        if caption:
            present_texts.add(caption.lower())
        queries.extend(build_negative_queries(
            present_texts=present_texts,
            prompt_corpus=self.prompt_corpus,
            generic_negatives=self.generic_negatives,
            num_negatives=self.num_negatives,
            num_cross_negatives=self.num_cross_negatives,
            img_id=img_id, orig_h=orig_h, orig_w=orig_w,
        ))

        return Datapoint(
            find_queries=queries,
            images=[image_obj],
            raw_images=[pil_image],
        )
