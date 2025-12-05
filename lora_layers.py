"""
LoRA (Low-Rank Adaptation) implementation for SAM3 model fine-tuning.
Supports selective application to different transformer components.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Set
import math


class LoRALayer(nn.Module):
    """
    LoRA layer that replaces a linear layer with low-rank adaptation.

    Args:
        in_features: Input dimension
        out_features: Output dimension
        rank: Rank of the low-rank matrices (r in the paper)
        alpha: Scaling factor (typically set to rank)
        dropout: Dropout probability for LoRA weights
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: int = 16,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # LoRA low-rank matrices
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))

        # Dropout
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        # Initialize weights
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply LoRA transformation: x @ (A @ B) * scaling
        """
        # x shape: (..., in_features)
        lora_out = self.dropout(x) @ self.lora_A @ self.lora_B
        return lora_out * self.scaling


class LoRALinear(nn.Module):
    """
    Linear layer with LoRA adaptation.
    Combines the original frozen linear layer with a LoRA layer.
    """

    def __init__(
        self,
        original_layer: nn.Linear,
        rank: int = 8,
        alpha: int = 16,
        dropout: float = 0.0,
    ):
        super().__init__()

        # Freeze the original layer
        self.original_layer = original_layer
        for param in self.original_layer.parameters():
            param.requires_grad = False

        # Create LoRA layer
        self.lora = LoRALayer(
            in_features=original_layer.in_features,
            out_features=original_layer.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: original output + LoRA output
        """
        return self.original_layer(x) + self.lora(x)


class LoRAConfig:
    """
    Configuration for LoRA application to SAM3 model.

    Args:
        rank: Rank of LoRA matrices
        alpha: Scaling factor
        dropout: Dropout probability
        target_modules: Which modules to apply LoRA to
        apply_to_vision_encoder: Whether to apply LoRA to vision encoder
        apply_to_text_encoder: Whether to apply LoRA to text encoder
        apply_to_geometry_encoder: Whether to apply LoRA to geometry encoder
        apply_to_detr_encoder: Whether to apply LoRA to DETR encoder
        apply_to_detr_decoder: Whether to apply LoRA to DETR decoder
        apply_to_mask_decoder: Whether to apply LoRA to mask decoder
    """

    def __init__(
        self,
        rank: int = 8,
        alpha: int = 16,
        dropout: float = 0.0,
        target_modules: Optional[List[str]] = None,
        # Component-level control
        apply_to_vision_encoder: bool = True,
        apply_to_text_encoder: bool = True,
        apply_to_geometry_encoder: bool = False,
        apply_to_detr_encoder: bool = True,
        apply_to_detr_decoder: bool = True,
        apply_to_mask_decoder: bool = False,
    ):
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout

        # Default target modules: typically Q, K, V projections
        if target_modules is None:
            target_modules = ["q_proj", "k_proj", "v_proj", "out_proj"]
        self.target_modules = set(target_modules)

        # Component flags
        self.apply_to_vision_encoder = apply_to_vision_encoder
        self.apply_to_text_encoder = apply_to_text_encoder
        self.apply_to_geometry_encoder = apply_to_geometry_encoder
        self.apply_to_detr_encoder = apply_to_detr_encoder
        self.apply_to_detr_decoder = apply_to_detr_decoder
        self.apply_to_mask_decoder = apply_to_mask_decoder

    def to_dict(self) -> Dict:
        """Convert config to dictionary."""
        return {
            "rank": self.rank,
            "alpha": self.alpha,
            "dropout": self.dropout,
            "target_modules": list(self.target_modules),
            "apply_to_vision_encoder": self.apply_to_vision_encoder,
            "apply_to_text_encoder": self.apply_to_text_encoder,
            "apply_to_geometry_encoder": self.apply_to_geometry_encoder,
            "apply_to_detr_encoder": self.apply_to_detr_encoder,
            "apply_to_detr_decoder": self.apply_to_detr_decoder,
            "apply_to_mask_decoder": self.apply_to_mask_decoder,
        }


def apply_lora_to_model(model: nn.Module, config: LoRAConfig) -> nn.Module:
    """
    Apply LoRA to specified modules in the SAM3 model.

    Args:
        model: SAM3 model to apply LoRA to
        config: LoRA configuration

    Returns:
        Model with LoRA applied
    """

    def should_apply_lora(module_name: str) -> bool:
        """Determine if LoRA should be applied to this module."""

        # Check component-level flags
        if ("vision_encoder" in module_name or "vision_backbone" in module_name) and not config.apply_to_vision_encoder:
            return False
        if ("text_encoder" in module_name or "language_backbone" in module_name) and not config.apply_to_text_encoder:
            return False
        if "geometry_encoder" in module_name and not config.apply_to_geometry_encoder:
            return False
        if ("detr_encoder" in module_name or "transformer.encoder" in module_name) and not config.apply_to_detr_encoder:
            return False
        if ("detr_decoder" in module_name or "transformer.decoder" in module_name) and not config.apply_to_detr_decoder:
            return False
        if "mask_decoder" in module_name and not config.apply_to_mask_decoder:
            return False

        # Check if module name matches target modules
        module_basename = module_name.split('.')[-1]
        
        # Skip out_proj to avoid breaking nn.MultiheadAttention which accesses .weight directly
        if module_basename == "out_proj":
            return False
            
        return module_basename in config.target_modules

    # Track replacements
    lora_modules_applied = []

    # Recursively replace linear layers
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and should_apply_lora(name):
            # Get parent module and attribute name
            *parent_path, attr_name = name.split('.')
            parent = model
            for p in parent_path:
                parent = getattr(parent, p)

            # Replace with LoRA linear
            lora_linear = LoRALinear(
                module,
                rank=config.rank,
                alpha=config.alpha,
                dropout=config.dropout,
            )
            setattr(parent, attr_name, lora_linear)
            lora_modules_applied.append(name)

    print(f"Applied LoRA to {len(lora_modules_applied)} modules:")
    for module_name in lora_modules_applied[:10]:  # Show first 10
        print(f"  - {module_name}")
    if len(lora_modules_applied) > 10:
        print(f"  ... and {len(lora_modules_applied) - 10} more")

    return model


def get_lora_parameters(model: nn.Module) -> List[nn.Parameter]:
    """
    Get all LoRA parameters from the model.

    Args:
        model: Model with LoRA layers

    Returns:
        List of LoRA parameters
    """
    lora_params = []
    for module in model.modules():
        if isinstance(module, LoRALayer):
            lora_params.extend([module.lora_A, module.lora_B])
    return lora_params


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count total and trainable parameters in the model.

    Returns:
        Dictionary with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "trainable_percentage": 100 * trainable_params / total_params if total_params > 0 else 0,
    }


def save_lora_weights(model: nn.Module, save_path: str):
    """
    Save only LoRA weights (not the full model).

    Args:
        model: Model with LoRA layers
        save_path: Path to save LoRA weights
    """
    lora_state_dict = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALayer):
            lora_state_dict[f"{name}.lora_A"] = module.lora_A
            lora_state_dict[f"{name}.lora_B"] = module.lora_B

    torch.save(lora_state_dict, save_path)
    print(f"Saved LoRA weights to {save_path}")


def load_lora_weights(model: nn.Module, load_path: str):
    """
    Load LoRA weights into a model.

    Args:
        model: Model with LoRA layers
        load_path: Path to LoRA weights
    """
    lora_state_dict = torch.load(load_path)
    model.load_state_dict(lora_state_dict, strict=False)
    print(f"Loaded LoRA weights from {load_path}")
