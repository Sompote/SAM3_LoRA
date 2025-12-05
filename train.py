#!/usr/bin/env python3
"""
Main training script for SAM3 LoRA fine-tuning.

This script loads a pretrained SAM3 model, injects LoRA adapters,
and fine-tunes it on a custom dataset.

Usage:
    python train.py --config configs/lora_config_example.yaml
"""

import argparse
import logging
import os
import sys

import torch
import torch.nn as nn
import yaml

# Add SAM3 to path
# sys.path.insert(0, '/workspace/sam3')  # SAM3 is now local

from sam3.model_builder import build_sam3_image_model

from src.lora.lora_utils import LoRAConfig
from src.data.dataset import create_dataloaders
from src.train.train_lora import LoRATrainer


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def create_optimizer(model: nn.Module, config: dict) -> torch.optim.Optimizer:
    """Create optimizer for LoRA parameters only."""
    # Get only LoRA parameters (trainable)
    lora_params = [p for p in model.parameters() if p.requires_grad]

    optimizer_name = config["training"]["optimizer"].lower()
    lr = float(config["training"]["learning_rate"])
    weight_decay = float(config["training"]["weight_decay"])

    if optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(
            lora_params,
            lr=lr,
            weight_decay=weight_decay,
            betas=config["training"].get("betas", [0.9, 0.999]),
        )
    elif optimizer_name == "adam":
        optimizer = torch.optim.Adam(
            lora_params,
            lr=lr,
            weight_decay=weight_decay,
            betas=config["training"].get("betas", [0.9, 0.999]),
        )
    elif optimizer_name == "sgd":
        optimizer = torch.optim.SGD(
            lora_params,
            lr=lr,
            weight_decay=weight_decay,
            momentum=config["training"].get("momentum", 0.9),
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    return optimizer


def create_scheduler(optimizer: torch.optim.Optimizer, config: dict, num_training_steps: int):
    """Create learning rate scheduler."""
    scheduler_name = config["training"].get("scheduler", "cosine")

    if scheduler_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_training_steps
        )
    elif scheduler_name == "linear":
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0, end_factor=0.0, total_iters=num_training_steps
        )
    elif scheduler_name == "inverse_sqrt":
        # Custom inverse sqrt scheduler (similar to SAM3)
        warmup_steps = config["training"].get("warmup_steps", 100)

        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            return max(0.0, float(warmup_steps) ** 0.5 / float(step) ** 0.5)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        scheduler = None

    return scheduler


def main():
    parser = argparse.ArgumentParser(description="SAM3 LoRA Fine-tuning")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config YAML file",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda or cpu)",
    )
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    logging.info(f"Loaded config from {args.config}")
    logging.info(f"Config: {config}")

    # Set device
    device = args.device if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")

    # Build SAM3 model
    logging.info("Building SAM3 model...")
    model = build_sam3_image_model(
        bpe_path=config["paths"]["bpe_path"],
        device="cpu",  # Load on CPU first, will move to GPU after LoRA injection
        eval_mode=False,
        enable_segmentation=config["training"].get("enable_segmentation", False),
    )

    # Load pretrained checkpoint if specified
    if "sam3_checkpoint" in config["paths"] and config["paths"]["sam3_checkpoint"]:
        checkpoint_path = config["paths"]["sam3_checkpoint"]
        if os.path.exists(checkpoint_path):
            logging.info(f"Loading SAM3 checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            model.load_state_dict(checkpoint["model"], strict=False)
        else:
            logging.warning(f"Checkpoint not found: {checkpoint_path}")

    # Create LoRA config
    lora_config = LoRAConfig(
        rank=config["lora"]["rank"],
        alpha=config["lora"]["alpha"],
        dropout=config["lora"]["dropout"],
        target_modules=config["lora"]["target_modules"],
    )

    # Create dataloaders
    logging.info("Creating dataloaders...")
    train_loader, val_loader = create_dataloaders(
        train_img_folder=config["dataset"]["train_img_folder"],
        train_ann_file=config["dataset"]["train_ann_file"],
        val_img_folder=config["dataset"]["val_img_folder"],
        val_ann_file=config["dataset"]["val_ann_file"],
        batch_size=config["training"]["batch_size"],
        num_workers=config["dataset"]["num_workers"],
        pin_memory=config["dataset"]["pin_memory"],
        max_ann_per_img=config["dataset"]["max_ann_per_img"],
    )

    logging.info(f"Train dataset size: {len(train_loader.dataset)}")
    logging.info(f"Val dataset size: {len(val_loader.dataset)}")

    # Create optimizer
    logging.info("Creating optimizer (LoRA will be injected in trainer)...")
    # Note: optimizer will be created after LoRA injection in the trainer

    # Calculate total training steps
    num_training_steps = (
        len(train_loader) * config["training"]["max_epochs"]
    ) // config["training"]["gradient_accumulation_steps"]

    # Create trainer (this will inject LoRA and create optimizer)
    logging.info("Creating trainer...")

    # For now, create a dummy optimizer that will be replaced
    dummy_optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["training"]["learning_rate"]))

    trainer = LoRATrainer(
        model=model,
        lora_config=lora_config,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=dummy_optimizer,  # Will be recreated with LoRA params
        device=device,
        max_epochs=config["training"]["max_epochs"],
        val_epoch_freq=config["training"]["val_epoch_freq"],
        log_dir=config["logging"]["log_dir"],
        checkpoint_dir=config["checkpoint"]["save_dir"],
        save_freq=config["checkpoint"]["save_freq"],
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        max_grad_norm=config["training"]["max_grad_norm"],
        use_amp=config["training"]["use_amp"],
        amp_dtype=config["training"]["amp_dtype"],
    )

    # Recreate optimizer with LoRA parameters
    trainer.optimizer = create_optimizer(trainer.model, config)

    # Create scheduler
    trainer.scheduler = create_scheduler(trainer.optimizer, config, num_training_steps)

    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Start training
    logging.info("Starting training...")
    trainer.train()


if __name__ == "__main__":
    main()
