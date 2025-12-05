"""Simplified standalone trainer for LoRA."""

import os
import time
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..lora.lora_utils import (
    LoRAConfig,
    inject_lora_into_model,
    get_lora_state_dict,
    load_lora_state_dict,
)
from ..utils.training_utils import print_trainable_parameters


class SimpleLoRATrainer:
    """
    Simplified standalone trainer for LoRA.

    This trainer doesn't require SAM3 and works with any PyTorch model.

    Args:
        model: PyTorch model to train
        lora_config: LoRA configuration
        train_loader: Training data loader
        val_loader: Validation data loader (optional)
        optimizer: Optimizer
        device: Device to use
        max_epochs: Maximum number of epochs
        save_dir: Directory to save checkpoints
    """

    def __init__(
        self,
        model: nn.Module,
        lora_config: LoRAConfig,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: str = "cuda",
        max_epochs: int = 10,
        save_dir: str = "./checkpoints",
        inject_lora: bool = True,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.max_epochs = max_epochs
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        # Inject LoRA
        if inject_lora:
            print("Injecting LoRA into model...")
            self.model = inject_lora_into_model(model, lora_config, verbose=True)
        else:
            print("Skipping LoRA injection (assumed already injected)...")
            self.model = model
            
        self.model = self.model.to(self.device)

        print("\nModel statistics:")
        print_trainable_parameters(self.model)

        # Data loaders
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Optimizer
        if optimizer is None:
            # Create default optimizer for LoRA parameters
            lora_params = [p for p in self.model.parameters() if p.requires_grad]
            self.optimizer = torch.optim.AdamW(lora_params, lr=1e-4, weight_decay=0.01)
        else:
            self.optimizer = optimizer

        # Training state
        self.epoch = 0
        self.best_loss = float("inf")

    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = 0

        progress = tqdm(self.train_loader, desc=f"Epoch {self.epoch+1}/{self.max_epochs}")

        for batch in progress:
            # Move to device
            if isinstance(batch, dict):
                # Handle dict batches
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(self.device)
            elif isinstance(batch, (list, tuple)):
                # Handle tuple/list batches
                batch = [b.to(self.device) if isinstance(b, torch.Tensor) else b for b in batch]
            else:
                # Handle tensor batch
                batch = batch.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            loss = self.compute_loss(batch)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Track loss
            total_loss += loss.item()
            num_batches += 1

            progress.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / num_batches
        return avg_loss

    def compute_loss(self, batch) -> torch.Tensor:
        """
        Compute loss. Override this method for custom loss computation.

        Default: Assumes model returns loss or uses dummy loss.
        """
        # Try to get loss from model output
        output = self.model(batch)

        if isinstance(output, dict) and "loss" in output:
            return output["loss"]
        elif isinstance(output, torch.Tensor):
            # Dummy loss for demonstration
            return output.mean()
        else:
            raise NotImplementedError(
                "Override compute_loss() method for custom loss computation"
            )

    def validate(self) -> float:
        """Validate the model."""
        if self.val_loader is None:
            return 0.0

        self.model.eval()
        total_loss = 0
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                # Move to device
                if isinstance(batch, dict):
                    for key in batch:
                        if isinstance(batch[key], torch.Tensor):
                            batch[key] = batch[key].to(self.device)

                # Forward pass
                loss = self.compute_loss(batch)

                # Track loss
                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches
        return avg_loss

    def train(self):
        """Main training loop."""
        print(f"\nStarting training for {self.max_epochs} epochs...")
        print("=" * 60)

        for epoch in range(self.max_epochs):
            self.epoch = epoch

            # Train
            train_loss = self.train_epoch()
            print(f"Epoch {epoch+1}/{self.max_epochs} - Train Loss: {train_loss:.4f}")

            # Validate
            if self.val_loader is not None:
                val_loss = self.validate()
                print(f"Epoch {epoch+1}/{self.max_epochs} - Val Loss: {val_loss:.4f}")

                # Save best model
                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    self.save_checkpoint("best")
            else:
                # Save based on train loss if no validation
                if train_loss < self.best_loss:
                    self.best_loss = train_loss
                    self.save_checkpoint("best")

            # Save last checkpoint (overwriting previous)
            self.save_checkpoint("last")

        print("\n" + "=" * 60)
        print("Training complete!")
        print(f"Best loss: {self.best_loss:.4f}")
        print(f"Checkpoints saved to: {self.save_dir}")
        print("=" * 60)

    def save_checkpoint(self, name: str):
        """Save LoRA checkpoint."""
        checkpoint_path = os.path.join(self.save_dir, f"{name}.pt")

        lora_state_dict = get_lora_state_dict(self.model)

        checkpoint = {
            "epoch": self.epoch,
            "lora_state_dict": lora_state_dict,
            "optimizer": self.optimizer.state_dict(),
            "best_loss": self.best_loss,
        }

        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load LoRA checkpoint."""
        print(f"Loading checkpoint from {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Load LoRA weights
        load_lora_state_dict(self.model, checkpoint["lora_state_dict"])

        # Load optimizer
        self.optimizer.load_state_dict(checkpoint["optimizer"])

        # Load training state
        self.epoch = checkpoint["epoch"]
        self.best_loss = checkpoint.get("best_loss", float("inf"))

        print(f"Loaded checkpoint from epoch {self.epoch}")
