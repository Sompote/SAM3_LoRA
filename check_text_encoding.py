#!/usr/bin/env python3
"""
Check text encoding during training to identify any issues
"""

import torch
import sys
sys.path.insert(0, '/workspace/SAM3_LoRA')

from train_sam3_lora_native import COCOSegmentDataset
from sam3.model_builder import build_sam3_image_model
from sam3.train.data.collator import collate_fn_api

print("="*80)
print("CHECKING TEXT ENCODING DURING TRAINING")
print("="*80)

# Load dataset
print("\n1. Loading dataset...")
dataset = COCOSegmentDataset(data_dir='/workspace/data2', split='train')

# Check prompts from dataset
print("\n2. CHECKING DATASET PROMPTS:")
print("-" * 80)
prompt_counts = {}
for i in range(min(10, len(dataset))):
    sample = dataset[i]
    query_text = sample.find_queries[0].query_text
    prompt_counts[query_text] = prompt_counts.get(query_text, 0) + 1

    # Get category info
    img_id = dataset.image_ids[i]
    anns = dataset.img_to_anns.get(img_id, [])
    cat_ids = [ann['category_id'] for ann in anns]
    cat_names = [dataset.categories[cid] for cid in cat_ids]

    print(f"Sample {i}:")
    print(f"  Query text: '{query_text}'")
    print(f"  Query text type: {type(query_text)}")
    print(f"  Query text bytes: {query_text.encode('utf-8')}")
    print(f"  Categories in image: {cat_names}")
    print(f"  Number of objects: {len(sample.images[0].objects)}")
    print()

print("\n3. PROMPT DISTRIBUTION (first 10 samples):")
print("-" * 80)
for prompt, count in sorted(prompt_counts.items()):
    print(f"  '{prompt}': {count} images")

# Build model to check text encoder
print("\n4. CHECKING TEXT ENCODER:")
print("-" * 80)
print("Building SAM3 model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = build_sam3_image_model(
    device=device.type,
    compile=False,
    load_from_HF=True,
    bpe_path="sam3/assets/bpe_simple_vocab_16e6.txt.gz",
    eval_mode=False
)

# Test text encoding
test_prompts = ["cracks", "crack", "CRACKS", "joint", "marking", "location"]

print("\nTesting text encoding for different prompts:")
print("-" * 80)

for prompt in test_prompts:
    # The model's text encoder tokenizes the text
    print(f"\nPrompt: '{prompt}'")
    print(f"  Length: {len(prompt)} chars")
    print(f"  ASCII: {[ord(c) for c in prompt]}")
    print(f"  Lowercase: '{prompt.lower()}'")

    # Try to access text encoder if available
    if hasattr(model, 'backbone') and hasattr(model.backbone, 'text_backbone'):
        print(f"  ✓ Model has text_backbone")
        # The text encoding happens inside the model forward pass
        # We can't easily test it without a full forward pass
    else:
        print(f"  ✗ Cannot directly access text encoder")

# Test with actual batch
print("\n\n5. TESTING TEXT ENCODING WITH ACTUAL BATCH:")
print("-" * 80)

def collate_fn(batch):
    return collate_fn_api(batch, dict_key="input", with_seg_masks=True)

# Get a small batch
samples = [dataset[i] for i in range(2)]
batch_dict = collate_fn(samples)
input_batch = batch_dict["input"]

print(f"\nBatch structure:")
print(f"  Type: {type(input_batch)}")
print(f"  Has find_queries: {hasattr(input_batch, 'find_queries')}")

if hasattr(input_batch, 'find_queries'):
    print(f"  Number of queries: {len(input_batch.find_queries)}")
    for i, query in enumerate(input_batch.find_queries):
        print(f"\n  Query {i}:")
        if hasattr(query, 'query_text'):
            print(f"    query_text: '{query.query_text}'")
            print(f"    Type: {type(query.query_text)}")
        if hasattr(query, 'query_text_encoded'):
            print(f"    query_text_encoded: {query.query_text_encoded.shape if torch.is_tensor(query.query_text_encoded) else type(query.query_text_encoded)}")

# Check categories
print("\n\n6. CHECKING CATEGORY MAPPING:")
print("-" * 80)
print(f"Categories in dataset: {dataset.categories}")
print()

# Check for issues
issues = []

# Issue 1: Multiple category IDs map to same name
cat_name_to_ids = {}
for cid, cname in dataset.categories.items():
    if cname not in cat_name_to_ids:
        cat_name_to_ids[cname] = []
    cat_name_to_ids[cname].append(cid)

print("Category ID mapping:")
for cname, cids in cat_name_to_ids.items():
    print(f"  '{cname}': IDs {cids}")
    if len(cids) > 1:
        issues.append(f"⚠ Multiple category IDs ({cids}) map to same name '{cname}'")

# Issue 2: Check for uppercase/lowercase inconsistency
for cname in cat_name_to_ids.keys():
    if cname.upper() == cname:
        issues.append(f"⚠ Category '{cname}' is ALL UPPERCASE (dataset uses lowercase prompts)")
    elif cname != cname.lower():
        issues.append(f"⚠ Category '{cname}' has mixed case (dataset uses lowercase prompts)")

print("\n\n7. POTENTIAL ISSUES DETECTED:")
print("="*80)
if issues:
    for issue in issues:
        print(issue)
    print()
    print("These issues could cause training instability:")
    print("  - Model tries to learn 'cracks' but categories say 'CRACKS'")
    print("  - Multiple category IDs for same object type confuses the model")
    print("  - Inconsistent capitalization causes text embedding mismatches")
else:
    print("✓ No obvious text encoding issues detected!")

print("\n" + "="*80)
print("TEXT ENCODING CHECK COMPLETE")
print("="*80)
