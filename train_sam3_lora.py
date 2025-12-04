"""
SAM3 LoRA Training Script
Train SAM3 model with LoRA for efficient fine-tuning on custom segmentation tasks.
"""

import argparse
import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
from transformers import (
    Sam3Model,
    Sam3Processor,
    get_scheduler,
)
from tqdm import tqdm
import numpy as np

from lora_layers import (
    LoRAConfig,
    apply_lora_to_model,
    get_lora_parameters,
    count_parameters,
    save_lora_weights,
)


class TrainingConfig:
    """Training configuration loaded from YAML and CLI arguments."""

    def __init__(self, config_dict: Dict[str, Any]):
        self.config = config_dict

        # Model settings
        self.model_name = config_dict["model"]["name"]
        self.cache_dir = config_dict["model"].get("cache_dir")

        # LoRA settings
        lora_cfg = config_dict["lora"]
        self.lora_config = LoRAConfig(
            rank=lora_cfg["rank"],
            alpha=lora_cfg["alpha"],
            dropout=lora_cfg["dropout"],
            target_modules=lora_cfg["target_modules"],
            apply_to_vision_encoder=lora_cfg["apply_to_vision_encoder"],
            apply_to_text_encoder=lora_cfg["apply_to_text_encoder"],
            apply_to_geometry_encoder=lora_cfg["apply_to_geometry_encoder"],
            apply_to_detr_encoder=lora_cfg["apply_to_detr_encoder"],
            apply_to_detr_decoder=lora_cfg["apply_to_detr_decoder"],
            apply_to_mask_decoder=lora_cfg["apply_to_mask_decoder"],
        )

        # Training settings
        train_cfg = config_dict["training"]
        self.train_data_path = train_cfg["train_data_path"]
        self.val_data_path = train_cfg["val_data_path"]
        self.batch_size = train_cfg["batch_size"]
        self.num_workers = train_cfg["num_workers"]
        self.learning_rate = train_cfg["learning_rate"]
        self.weight_decay = train_cfg["weight_decay"]
        self.adam_beta1 = train_cfg["adam_beta1"]
        self.adam_beta2 = train_cfg["adam_beta2"]
        self.adam_epsilon = train_cfg["adam_epsilon"]
        self.max_grad_norm = train_cfg["max_grad_norm"]
        self.num_epochs = train_cfg["num_epochs"]
        self.warmup_steps = train_cfg["warmup_steps"]
        self.lr_scheduler = train_cfg["lr_scheduler"]
        self.logging_steps = train_cfg["logging_steps"]
        self.eval_steps = train_cfg["eval_steps"]
        self.save_steps = train_cfg["save_steps"]
        self.save_total_limit = train_cfg["save_total_limit"]
        self.mixed_precision = train_cfg["mixed_precision"]
        self.seed = train_cfg["seed"]
        self.gradient_accumulation_steps = train_cfg["gradient_accumulation_steps"]

        # Output settings
        output_cfg = config_dict["output"]
        self.output_dir = output_cfg["output_dir"]
        self.logging_dir = output_cfg["logging_dir"]
        self.save_lora_only = output_cfg["save_lora_only"]
        self.push_to_hub = output_cfg["push_to_hub"]
        self.hub_model_id = output_cfg.get("hub_model_id")

        # Evaluation settings
        eval_cfg = config_dict["evaluation"]
        self.metric = eval_cfg["metric"]
        self.save_predictions = eval_cfg["save_predictions"]
        self.compute_metrics_during_training = eval_cfg["compute_metrics_during_training"]

        # Hardware settings
        hw_cfg = config_dict["hardware"]
        self.device = hw_cfg["device"]
        self.dataloader_pin_memory = hw_cfg["dataloader_pin_memory"]
        self.use_compile = hw_cfg["use_compile"]

    @classmethod
    def from_yaml(cls, yaml_path: str):
        """Load configuration from YAML file."""
        with open(yaml_path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls(config_dict)

    def update_from_cli_args(self, args: argparse.Namespace):
        """Update configuration with CLI arguments."""
        # CLI arguments override YAML settings
        for key, value in vars(args).items():
            if value is not None and hasattr(self, key):
                setattr(self, key, value)

    def save(self, save_path: str):
        """Save configuration to file."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as f:
            yaml.dump(self.config, f, default_flow_style=False)


class SAM3Dataset(torch.utils.data.Dataset):
    """
    Custom dataset for SAM3 training.

    Expected directory structure:
    data/
        train/
            images/
                image1.jpg
                image2.jpg
            annotations/
                image1.json
                image2.json
        val/
            images/
            annotations/

    Annotation format (JSON):
    {
        "text_prompt": "yellow school bus",
        "masks": [...],  # Binary masks
        "bboxes": [...],  # Bounding boxes [x1, y1, x2, y2]
    }
    """

    def __init__(self, data_path: str, processor: Sam3Processor):
        self.data_path = Path(data_path)
        self.processor = processor

        # Load image paths
        self.image_dir = self.data_path / "images"
        self.annotation_dir = self.data_path / "annotations"

        self.image_files = sorted(list(self.image_dir.glob("*.jpg")) +
                                  list(self.image_dir.glob("*.png")))

        print(f"Loaded {len(self.image_files)} images from {data_path}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load image
        image_path = self.image_files[idx]
        from PIL import Image
        image = Image.open(image_path).convert("RGB")

        # Load annotation
        annotation_path = self.annotation_dir / f"{image_path.stem}.json"
        with open(annotation_path, "r") as f:
            annotation = json.load(f)

        # Process inputs
        text_prompt = annotation.get("text_prompt", "")
        bboxes = annotation.get("bboxes", [])

        # Use processor to prepare inputs
        inputs = self.processor(
            images=image,
            text=text_prompt if text_prompt else None,
            boxes=bboxes if bboxes else None,
            return_tensors="pt",
        )

        # Add ground truth masks
        if "masks" in annotation:
            inputs["ground_truth_masks"] = torch.tensor(annotation["masks"])

        # Remove batch dimension (will be added by DataLoader)
        inputs = {k: v.squeeze(0) if isinstance(v, torch.Tensor) else v
                  for k, v in inputs.items()}

        return inputs


def compute_iou(pred_masks: torch.Tensor, gt_masks: torch.Tensor) -> float:
    """
    Compute Intersection over Union (IoU) metric.

    Args:
        pred_masks: Predicted masks [B, H, W]
        gt_masks: Ground truth masks [B, H, W]

    Returns:
        Mean IoU score
    """
    pred_masks = (pred_masks > 0.5).float()
    gt_masks = (gt_masks > 0.5).float()

    intersection = (pred_masks * gt_masks).sum(dim=(1, 2))
    union = ((pred_masks + gt_masks) > 0).float().sum(dim=(1, 2))

    iou = intersection / (union + 1e-7)
    return iou.mean().item()


class SAM3Trainer:
    """Trainer class for SAM3 with LoRA."""

    def __init__(self, config: TrainingConfig):
        self.config = config

        # Set seed for reproducibility
        self.set_seed(config.seed)

        # Setup device
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Load model and processor
        print(f"Loading model: {config.model_name}")
        self.model = Sam3Model.from_pretrained(
            config.model_name,
            cache_dir=config.cache_dir,
        )
        self.processor = Sam3Processor.from_pretrained(
            config.model_name,
            cache_dir=config.cache_dir,
        )

        # Apply LoRA
        print("\nApplying LoRA to model...")
        self.model = apply_lora_to_model(self.model, config.lora_config)

        # Print parameter statistics
        param_stats = count_parameters(self.model)
        print(f"\nParameter Statistics:")
        print(f"  Total parameters: {param_stats['total_parameters']:,}")
        print(f"  Trainable parameters: {param_stats['trainable_parameters']:,}")
        print(f"  Trainable percentage: {param_stats['trainable_percentage']:.2f}%")

        # Move model to device
        self.model.to(self.device)

        # Setup optimizer
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = AdamW(
            trainable_params,
            lr=config.learning_rate,
            betas=(config.adam_beta1, config.adam_beta2),
            eps=config.adam_epsilon,
            weight_decay=config.weight_decay,
        )

        # Setup mixed precision
        self.scaler = None
        if config.mixed_precision == "fp16":
            self.scaler = torch.cuda.amp.GradScaler()

        # Training state
        self.global_step = 0
        self.best_metric = 0.0

        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)
        os.makedirs(config.logging_dir, exist_ok=True)

    def set_seed(self, seed: int):
        """Set random seed for reproducibility."""
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

    def create_dataloaders(self):
        """Create training and validation dataloaders."""
        train_dataset = SAM3Dataset(self.config.train_data_path, self.processor)
        val_dataset = SAM3Dataset(self.config.val_data_path, self.processor)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.config.dataloader_pin_memory,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.dataloader_pin_memory,
        )

        return train_loader, val_loader

    def setup_scheduler(self, num_training_steps: int):
        """Setup learning rate scheduler."""
        self.scheduler = get_scheduler(
            name=self.config.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=num_training_steps,
        )

    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """
        Perform a single training step.

        Returns:
            Loss value
        """
        # Move batch to device
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        # Extract ground truth masks
        gt_masks = batch.pop("ground_truth_masks", None)

        # Forward pass
        with torch.cuda.amp.autocast(enabled=self.config.mixed_precision == "fp16"):
            outputs = self.model(**batch)

            # Compute loss (custom loss for segmentation)
            if gt_masks is not None:
                pred_masks = outputs.pred_masks
                loss = nn.functional.binary_cross_entropy_with_logits(
                    pred_masks, gt_masks
                )
            else:
                # Use default loss if available
                loss = outputs.loss if hasattr(outputs, "loss") else None
                if loss is None:
                    raise ValueError("No loss computed. Ensure ground_truth_masks are provided.")

        # Backward pass
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        return loss.item()

    def train(self):
        """Main training loop."""
        print("\n" + "="*50)
        print("Starting Training")
        print("="*50)

        # Create dataloaders
        train_loader, val_loader = self.create_dataloaders()

        # Calculate total training steps
        num_training_steps = (
            len(train_loader) * self.config.num_epochs //
            self.config.gradient_accumulation_steps
        )

        # Setup scheduler
        self.setup_scheduler(num_training_steps)

        # Training loop
        self.model.train()
        accumulation_loss = 0.0

        for epoch in range(self.config.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")
            progress_bar = tqdm(train_loader, desc="Training")

            for step, batch in enumerate(progress_bar):
                loss = self.train_step(batch)
                accumulation_loss += loss

                # Gradient accumulation
                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    if self.config.max_grad_norm > 0:
                        if self.scaler is not None:
                            self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config.max_grad_norm
                        )

                    # Optimizer step
                    if self.scaler is not None:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()

                    self.scheduler.step()
                    self.optimizer.zero_grad()

                    self.global_step += 1

                    # Logging
                    if self.global_step % self.config.logging_steps == 0:
                        avg_loss = accumulation_loss / self.config.gradient_accumulation_steps
                        progress_bar.set_postfix({
                            "loss": f"{avg_loss:.4f}",
                            "lr": f"{self.scheduler.get_last_lr()[0]:.2e}"
                        })
                        accumulation_loss = 0.0

                    # Evaluation
                    if self.global_step % self.config.eval_steps == 0:
                        metrics = self.evaluate(val_loader)
                        print(f"\nStep {self.global_step} - Validation IoU: {metrics['iou']:.4f}")

                        # Save best model
                        if metrics['iou'] > self.best_metric:
                            self.best_metric = metrics['iou']
                            self.save_checkpoint(f"best_model")

                        self.model.train()

                    # Save checkpoint
                    if self.global_step % self.config.save_steps == 0:
                        self.save_checkpoint(f"checkpoint-{self.global_step}")

        # Final save
        self.save_checkpoint("final_model")
        print("\nTraining completed!")

    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate the model on validation set.

        Returns:
            Dictionary of metrics
        """
        self.model.eval()

        total_iou = 0.0
        num_batches = 0

        for batch in tqdm(val_loader, desc="Evaluating"):
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            gt_masks = batch.pop("ground_truth_masks", None)

            outputs = self.model(**batch)

            if gt_masks is not None:
                pred_masks = torch.sigmoid(outputs.pred_masks)
                iou = compute_iou(pred_masks, gt_masks)
                total_iou += iou
                num_batches += 1

        avg_iou = total_iou / num_batches if num_batches > 0 else 0.0

        return {"iou": avg_iou}

    def save_checkpoint(self, checkpoint_name: str):
        """Save model checkpoint."""
        checkpoint_dir = Path(self.config.output_dir) / checkpoint_name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        if self.config.save_lora_only:
            # Save only LoRA weights
            lora_path = checkpoint_dir / "lora_weights.pt"
            save_lora_weights(self.model, str(lora_path))
        else:
            # Save full model
            self.model.save_pretrained(str(checkpoint_dir))
            self.processor.save_pretrained(str(checkpoint_dir))

        # Save config
        config_path = checkpoint_dir / "training_config.yaml"
        self.config.save(str(config_path))

        print(f"Saved checkpoint to {checkpoint_dir}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train SAM3 with LoRA for efficient fine-tuning"
    )

    # Config file
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file",
    )

    # Override common parameters via CLI
    parser.add_argument("--output_dir", type=str, help="Output directory")
    parser.add_argument("--learning_rate", type=float, help="Learning rate")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--num_epochs", type=int, help="Number of epochs")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--lora_rank", type=int, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, help="LoRA alpha")
    parser.add_argument("--train_data_path", type=str, help="Training data path")
    parser.add_argument("--val_data_path", type=str, help="Validation data path")

    return parser.parse_args()


def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()

    # Load configuration from YAML
    config = TrainingConfig.from_yaml(args.config)

    # Override with CLI arguments
    config.update_from_cli_args(args)

    # Create trainer
    trainer = SAM3Trainer(config)

    # Start training
    trainer.train()


if __name__ == "__main__":
    main()
