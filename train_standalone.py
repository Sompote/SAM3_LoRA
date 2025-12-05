#!/usr/bin/env python3
"""
Standalone training script for SAM3 LoRA.

This script doesn't require SAM3 installation and works with simple models.
"""

import argparse
import yaml
import torch
from torch.utils.data import DataLoader

from sam3_lora import LoRAConfig, inject_lora_into_model
from sam3_lora.model import SimpleSegmentationModel
from sam3_lora.data import LoRASAM3Dataset
from sam3_lora.train import SimpleLoRATrainer


def simple_collate(batch):
    """Simple collate function."""
    return {
        'images': [item['image'] for item in batch],
        'annotations': [item['annotations'] for item in batch],
    }


def main():
    parser = argparse.ArgumentParser(description="SAM3 LoRA Standalone Training")
    parser.add_argument("--config", type=str, default="configs/sam3_lora_standalone.yaml",
                        help="Path to config YAML file")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")

    args = parser.parse_args()

    # Load configuration
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    print("="*60)
    print("SAM3 LoRA - Standalone Training")
    print("="*60)

    # 1. Create model
    print("\n1. Creating model...")
    model = SimpleSegmentationModel(d_model=256, nhead=8, dim_feedforward=1024)
    print("✓ Model created")

    # 2. Setup LoRA
    print("\n2. Setting up LoRA...")
    lora_config = LoRAConfig(
        rank=config["lora"]["rank"],
        alpha=config["lora"]["alpha"],
        dropout=config["lora"]["dropout"],
        target_modules=config["lora"]["target_modules"]
    )

    # Manually inject LoRA to enable optimizer creation before trainer
    model = inject_lora_into_model(model, lora_config, verbose=True)

    # 3. Create dataloaders
    print("\n3. Creating dataloaders...")
    try:
        train_dataset = LoRASAM3Dataset(
            img_folder=f"{config['dataset']['data_root']}/train",
            ann_file=f"{config['dataset']['data_root']}/train/_annotations.coco.json",
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=config["training"]["batch_size"],
            shuffle=True,
            collate_fn=simple_collate,
            num_workers=0,
        )
        print(f"✓ Train dataset: {len(train_dataset)} samples")

        # Optional validation set
        try:
            val_dataset = LoRASAM3Dataset(
                img_folder=f"{config['dataset']['data_root']}/valid",
                ann_file=f"{config['dataset']['data_root']}/valid/_annotations.coco.json",
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=config["training"]["batch_size"],
                shuffle=False,
                collate_fn=simple_collate,
                num_workers=0,
            )
            print(f"✓ Val dataset: {len(val_dataset)} samples")
        except:
            val_loader = None
            print("⚠ No validation dataset found")

    except Exception as e:
        print(f"✗ Could not load data: {e}")
        print("  Please ensure data is in COCO format at:")
        print(f"  {config['dataset']['data_root']}/train/_annotations.coco.json")
        return

    # 4. Create optimizer
    print("\n4. Creating optimizer...")
    lora_params = [p for p in model.parameters() if p.requires_grad]

    optimizer_name = config["training"].get("optimizer", "adamw").lower()
    lr = float(config["training"]["learning_rate"])
    weight_decay = float(config["training"].get("weight_decay", 0.01))

    if optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(
            lora_params,
            lr=lr,
            weight_decay=weight_decay,
            betas=tuple(config["training"].get("betas", [0.9, 0.999]))
        )
    elif optimizer_name == "sgd":
        optimizer = torch.optim.SGD(
            lora_params,
            lr=lr,
            weight_decay=weight_decay,
            momentum=config["training"].get("momentum", 0.9)
        )
    else:
        optimizer = torch.optim.Adam(
            lora_params,
            lr=lr,
            weight_decay=weight_decay
        )
    print(f"✓ Optimizer created: {optimizer_name} with lr={lr}")

    # 5. Create trainer
    print("\n5. Creating trainer...")

    # Custom trainer that handles our data format
    class CustomLoRATrainer(SimpleLoRATrainer):
        def compute_loss(self, batch):
            """Custom loss computation for demo."""
            # Create dummy input for demo
            batch_size = len(batch['images'])
            x = torch.randn(batch_size, 10, 256).to(self.device)

            # Forward pass
            output = self.model(x)

            # Dummy loss (replace with real loss)
            y = torch.randn_like(output).to(self.device)
            loss = torch.nn.functional.mse_loss(output, y)

            return loss

    trainer = CustomLoRATrainer(
        model=model,
        lora_config=lora_config,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device="cuda" if torch.cuda.is_available() else "cpu",
        max_epochs=config["training"]["epochs"],
        save_dir=config["checkpoint"]["save_dir"],
        inject_lora=False,
    )

    # 6. Load checkpoint if resuming
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # 7. Train
    print("\n6. Starting training...")
    trainer.train()

    print("\n" + "="*60)
    print("Training complete!")
    print(f"Checkpoints saved to: {config['checkpoint']['save_dir']}")
    print("="*60)


if __name__ == "__main__":
    main()
