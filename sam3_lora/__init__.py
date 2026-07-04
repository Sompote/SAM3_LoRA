"""
SAM3 LoRA - Standalone LoRA Fine-tuning for Segment Anything Model 3

This is a standalone implementation that doesn't require SAM3 installation.
"""

__version__ = "0.2.0"

from .lora import LoRALayer, LinearWithLoRA, LoRAConfig, inject_lora_into_model
from .data import LoRASAM3Dataset, create_dataloaders

__all__ = [
    "LoRALayer",
    "LinearWithLoRA",
    "LoRAConfig",
    "inject_lora_into_model",
    "LoRASAM3Dataset",
    "create_dataloaders",
]
