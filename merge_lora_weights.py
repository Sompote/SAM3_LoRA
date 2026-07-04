#!/usr/bin/env python3
"""
Merge trained LoRA weights into the base SAM3 model.

Produces a single full-model checkpoint that loads into the ORIGINAL SAM3
architecture — no LoRA code, config, or separate adapter file needed at
inference time:

    from sam3.model_builder import build_sam3_image_model
    model = build_sam3_image_model(load_from_HF=True, eval_mode=True, ...)
    model.load_state_dict(torch.load("sam3_merged.pt", map_location="cpu"))

The merged model is numerically equivalent to base + LoRA (W' = W + B^T A^T *
alpha/rank), so predictions are identical to loading the adapter on top.

Trade-off: the merged checkpoint is full model size (~3 GB) instead of a
10-50 MB adapter, and the adaptation is baked in — you can no longer swap
adapters on one shared base model.

Usage:
    python3 merge_lora_weights.py \
        --config configs/full_lora_config.yaml \
        --lora-weights outputs/sam3_lora_full/best_lora_weights.pt \
        --output sam3_merged.pt

--config must be the SAME config used for training: the adapter file stores
only the lora_A/lora_B tensors, so the model must be rebuilt with the same
LoRA structure before the weights can be folded in.
"""

import argparse
import os

import torch
import torch.nn as nn
import yaml

from lora_layers import (
    LoRAConfig,
    LoRALinear,
    MultiheadAttentionLoRA,
    apply_lora_to_model,
    load_lora_weights,
)


def _get_parent(model: nn.Module, module_name: str):
    """Return (parent_module, attribute_name) for a dotted module path."""
    *parent_path, attr_name = module_name.split(".")
    parent = model
    for p in parent_path:
        parent = getattr(parent, p)
    return parent, attr_name


def merge_lora_linears(model: nn.Module) -> int:
    """Fold every LoRALinear's low-rank delta into its frozen base Linear and
    put the plain nn.Linear back in place. Returns the number of merged layers."""
    to_merge = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, LoRALinear)
    ]

    with torch.no_grad():
        for name, module in to_merge:
            # forward is: original(x) + (x @ A @ B) * scaling, so the weight
            # delta in nn.Linear layout (out, in) is (A @ B)^T * scaling
            delta = (module.lora.lora_A @ module.lora.lora_B).T * module.lora.scaling
            base = module.original_layer
            base.weight += delta.to(dtype=base.weight.dtype, device=base.weight.device)

            parent, attr_name = _get_parent(model, name)
            setattr(parent, attr_name, base)

    return len(to_merge)


def restore_multihead_attention(model: nn.Module) -> int:
    """Convert MultiheadAttentionLoRA modules (separate q/k/v/out projections)
    back to stock nn.MultiheadAttention (fused in_proj). Must run AFTER
    merge_lora_linears so the projections are plain nn.Linear with merged
    weights. Returns the number of restored modules."""
    to_restore = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, MultiheadAttentionLoRA)
    ]

    with torch.no_grad():
        for name, module in to_restore:
            for proj in (module.q_proj, module.k_proj, module.v_proj, module.out_proj):
                if not isinstance(proj, nn.Linear):
                    raise RuntimeError(
                        f"{name} still contains un-merged LoRA projections; "
                        "merge_lora_linears must run first"
                    )

            has_bias = module.q_proj.bias is not None
            mha = nn.MultiheadAttention(
                embed_dim=module.embed_dim,
                num_heads=module.num_heads,
                dropout=module.dropout,
                bias=has_bias,
                batch_first=module.batch_first,
            )
            mha.in_proj_weight.copy_(
                torch.cat(
                    [module.q_proj.weight, module.k_proj.weight, module.v_proj.weight],
                    dim=0,
                )
            )
            mha.out_proj.weight.copy_(module.out_proj.weight)
            if has_bias:
                mha.in_proj_bias.copy_(
                    torch.cat(
                        [module.q_proj.bias, module.k_proj.bias, module.v_proj.bias],
                        dim=0,
                    )
                )
                mha.out_proj.bias.copy_(module.out_proj.bias)

            parent, attr_name = _get_parent(model, name)
            setattr(parent, attr_name, mha)

    return len(to_restore)


def main():
    # Deferred: importing sam3 pulls in GPU-only dependencies (e.g. triton),
    # while the merge helpers above are pure torch and usable anywhere.
    from sam3.model_builder import build_sam3_image_model

    parser = argparse.ArgumentParser(
        description="Merge LoRA weights into the base SAM3 model"
    )
    parser.add_argument(
        "--config", required=True,
        help="Training config YAML (must match the one used to train the adapter)",
    )
    parser.add_argument(
        "--lora-weights", default=None,
        help="Path to LoRA weights .pt (default: <output_dir>/best_lora_weights.pt from config)",
    )
    parser.add_argument(
        "--output", default="sam3_merged.pt",
        help="Where to save the merged full-model state dict (default: sam3_merged.pt)",
    )
    parser.add_argument(
        "--bpe-path", default="sam3/assets/bpe_simple_vocab_16e6.txt.gz",
        help="BPE vocab path for the SAM3 text tokenizer",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    weights_path = args.lora_weights
    if weights_path is None:
        output_dir = config.get("output", {}).get("output_dir", "outputs/sam3_lora_full")
        weights_path = os.path.join(output_dir, "best_lora_weights.pt")
        print(f"Auto-detected LoRA weights: {weights_path}")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"LoRA weights not found: {weights_path}")

    # Merging is pure tensor arithmetic — CPU is fine and needs no VRAM.
    print("Building base SAM3 model...")
    model = build_sam3_image_model(
        device="cpu",
        compile=False,
        load_from_HF=True,
        bpe_path=args.bpe_path,
        eval_mode=True,
    )
    reference_keys = set(model.state_dict().keys())

    print("Applying LoRA structure from config...")
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

    print(f"Loading LoRA weights from {weights_path}...")
    load_lora_weights(model, weights_path)

    print("Merging LoRA deltas into base weights...")
    n_linear = merge_lora_linears(model)
    print(f"  Merged {n_linear} LoRA linear layers")
    n_mha = restore_multihead_attention(model)
    print(f"  Restored {n_mha} MultiheadAttention modules")

    # The merged model must be architecturally identical to stock SAM3.
    merged_keys = set(model.state_dict().keys())
    if merged_keys != reference_keys:
        missing = sorted(reference_keys - merged_keys)[:10]
        extra = sorted(merged_keys - reference_keys)[:10]
        raise RuntimeError(
            "Merged state dict does not match the original SAM3 architecture.\n"
            f"  Missing keys (first 10): {missing}\n"
            f"  Unexpected keys (first 10): {extra}"
        )
    print("Verified: merged state dict matches the original SAM3 architecture")

    torch.save(model.state_dict(), args.output)
    size_gb = os.path.getsize(args.output) / 1024**3
    print(f"\nSaved merged model to {args.output} ({size_gb:.2f} GB)")
    print("\nLoad it WITHOUT any LoRA code:")
    print("    from sam3.model_builder import build_sam3_image_model")
    print("    model = build_sam3_image_model(load_from_HF=True, eval_mode=True, ...)")
    print(f"    model.load_state_dict(torch.load({args.output!r}, map_location='cpu'))")


if __name__ == "__main__":
    main()
