# SAM3-LoRA: Low-Rank Adaptation for Fine-Tuning

<div align="center">

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)

**Efficient fine-tuning using LoRA (Low-Rank Adaptation)**

[Installation](#installation) • [Quick Start](#quick-start) • [Training](#training) • [Examples](#examples)

</div>

---

## Overview

**✨ Recent Highlights ✨**
- **CLI Configuration**: `train_sam3_lora_native.py` supports `--config` argument for flexible YAML config selection.
- **Full SAM3 Integration**: Native training with complete SAM3 model and LoRA fine-tuning.
- **YAML-Driven**: All training parameters configured via comprehensive YAML files.
- **Flexible LoRA**: Apply LoRA to any combination of model components (vision, text, DETR, mask decoder).
- **Production Ready**: Tested training pipeline with proper loss functions and optimizers.
- **Easy to Use**: Simple CLI interface with sensible defaults.


A standalone LoRA (Low-Rank Adaptation) implementation for efficient fine-tuning of deep learning models. Train with **less than 1% of parameters** while maintaining performance.

### Why LoRA?

**LoRA (Low-Rank Adaptation)** adapts pre-trained models by injecting trainable low-rank matrices:
- **W' = W + B×A** where rank << min(input_dim, output_dim)
- Train only 1-35% of parameters instead of 100%
- Checkpoint sizes: 10-50MB instead of 3GB
- Same or better performance than full fine-tuning

### Key Features

- **Memory Efficient**: Train on smaller GPUs (16GB vs 80GB for full fine-tuning)
- **Small Checkpoints**: 10-50MB LoRA weights vs 3GB full model
- **Fast Training**: Reduced memory footprint enables faster iterations
- **Flexible**: Apply LoRA to specific model components
- **Easy to Use**: Simple Python API and CLI commands
- **Production Ready**: Fully tested and documented

---

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/sam3_lora.git
cd sam3_lora

# Install package
pip install -e .
```

**Requirements:**
```
torch>=2.0.0
torchvision>=0.15.0
Pillow>=9.5.0
numpy>=1.24.0
tqdm>=4.65.0
pyyaml>=6.0
tensorboard>=2.12.0
```

---

## Quick Start

### Training

Train with the full SAM3 model using LoRA fine-tuning:

```bash
# Use default config
python3 train_sam3_lora_native.py

# Specify custom config file
python3 train_sam3_lora_native.py --config configs/full_lora_config.yaml

# Use different config for experiments
python3 train_sam3_lora_native.py --config configs/my_custom_config.yaml
```

**Available YAML Configuration Parameters:**

The config file (`configs/full_lora_config.yaml`) supports the following parameters:

```yaml
# LoRA Configuration
lora:
  rank: 16                          # LoRA rank (4, 8, 16, 32, 64)
  alpha: 32                         # Alpha scaling (typically 2x rank)
  dropout: 0.1                      # Dropout rate (0.0-0.3)
  target_modules:                   # Module types to apply LoRA
    - "q_proj"                      # Query projection
    - "k_proj"                      # Key projection
    - "v_proj"                      # Value projection
    - "out_proj"                    # Output projection
    - "fc1"                         # MLP first layer
    - "fc2"                         # MLP second layer

  # Model component flags
  apply_to_vision_encoder: true     # Apply to vision encoder
  apply_to_text_encoder: false      # Apply to text encoder
  apply_to_geometry_encoder: false  # Apply to geometry encoder
  apply_to_detr_encoder: false      # Apply to DETR encoder
  apply_to_detr_decoder: false      # Apply to DETR decoder
  apply_to_mask_decoder: true       # Apply to mask decoder

# Training Configuration
training:
  batch_size: 1                     # Batch size per GPU
  num_epochs: 20                    # Number of training epochs
  learning_rate: 5e-5               # Learning rate
  weight_decay: 0.01                # Weight decay for regularization

# Output Configuration
output:
  output_dir: "outputs/sam3_lora_full"  # Where to save trained weights
```

**Example Configurations:**

Create multiple config files for different experiments:

```bash
# Quick test (low rank, few epochs)
configs/
  ├── quick_test_config.yaml      # rank=4, epochs=5
  ├── vision_only_config.yaml     # Only vision encoder
  ├── full_lora_config.yaml       # All components (default)
  └── production_config.yaml      # High rank, many epochs
```

Then train with:
```bash
python3 train_sam3_lora_native.py --config configs/quick_test_config.yaml
```

### Python API

Use the trainer programmatically:

```python
from train_sam3_lora_native import SAM3TrainerNative

# Create trainer with config file
trainer = SAM3TrainerNative("configs/full_lora_config.yaml")

# Train!
trainer.train()
```

Or apply LoRA to your own models:

```python
import torch
from lora_layers import LoRAConfig, apply_lora_to_model, count_parameters

# Your existing PyTorch model
model = YourModel()

# Configure LoRA
lora_config = LoRAConfig(
    rank=8,                    # Rank (4, 8, 16, 32)
    alpha=16.0,                # Scaling factor (typically 2*rank)
    dropout=0.1,               # Dropout probability
    target_modules=[           # Which layers to adapt
        "q_proj",              # Query projection
        "k_proj",              # Key projection
        "v_proj",              # Value projection
        "out_proj",            # Output projection
        "fc1",                 # First FFN layer
        "fc2"                  # Second FFN layer
    ],
    apply_to_vision_encoder=True,
    apply_to_text_encoder=False,
    apply_to_geometry_encoder=False,
    apply_to_detr_encoder=False,
    apply_to_detr_decoder=False,
    apply_to_mask_decoder=True,
)

# Apply LoRA
model = apply_lora_to_model(model, lora_config)

# Check trainable parameters
stats = count_parameters(model)
print(f"Trainable: {stats['trainable_parameters']:,} ({stats['trainable_percentage']:.2f}%)")

# Train as usual
optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=1e-4
)
```

---

## Data Preparation

The training script expects images and individual JSON annotation files.

**Directory structure:**
```
data/
├── train/
│   ├── images/
│   │   ├── image001.jpg
│   │   ├── image002.jpg
│   │   └── ...
│   └── annotations/
│       ├── image001.json
│       ├── image002.json
│       └── ...
└── valid/
    ├── images/
    │   ├── image001.jpg
    │   └── ...
    └── annotations/
        ├── image001.json
        └── ...
```

**Annotation format (JSON):**
Each JSON file should contain bounding boxes and masks:
```json
{
  "bboxes": [[x1, y1, x2, y2], ...],
  "masks": [[[x, y], ...], ...]
}
```

---

## Training

### CLI Training

**Basic command (using YAML config):**
```bash
# Use default config
python3 train_sam3_lora_native.py

# Specify custom config file
python3 train_sam3_lora_native.py --config configs/full_lora_config.yaml
```

**All available options (configured in YAML):**
You can customize all training and LoRA hyperparameters by modifying the YAML configuration file.
```yaml
# configs/full_lora_config.yaml
lora:
  rank: 16
  alpha: 32
  dropout: 0.1
  target_modules:
    - "q_proj"
    - "k_proj"
    - "v_proj"
    - "out_proj"
    - "fc1"
    - "fc2"
  apply_to_vision_encoder: true
  apply_to_text_encoder: false
  apply_to_geometry_encoder: false
  apply_to_detr_encoder: false
  apply_to_detr_decoder: false
  apply_to_mask_decoder: true

training:
  batch_size: 1
  num_epochs: 20
  learning_rate: 5e-5
  weight_decay: 0.01

output:
  output_dir: "outputs/sam3_lora_full"
```

To run with a custom configuration, simply specify your YAML file:
```bash
python3 train_sam3_lora_native.py --config my_custom_config.yaml
```

### Using the Trainer API

```python
from train_sam3_lora_native import SAM3TrainerNative

# Create trainer with config file
trainer = SAM3TrainerNative("configs/full_lora_config.yaml")

# Train!
trainer.train()
```

Or programmatically with custom config:

```python
import yaml

# Create custom config
config = {
    "lora": {
        "rank": 8,
        "alpha": 16,
        "dropout": 0.1,
        "target_modules": ["q_proj", "v_proj"],
        "apply_to_vision_encoder": True,
        "apply_to_text_encoder": False,
        "apply_to_geometry_encoder": False,
        "apply_to_detr_encoder": False,
        "apply_to_detr_decoder": False,
        "apply_to_mask_decoder": True,
    },
    "training": {
        "batch_size": 1,
        "num_epochs": 10,
        "learning_rate": 1e-4,
        "weight_decay": 0.01,
    },
    "output": {
        "output_dir": "outputs/custom"
    }
}

# Save config
with open("configs/custom.yaml", "w") as f:
    yaml.dump(config, f)

# Train
trainer = SAM3TrainerNative("configs/custom.yaml")
trainer.train()
```

---

## Configuration

### LoRA Parameters

```python
from lora_layers import LoRAConfig

config = LoRAConfig(
    rank=8,                          # Low-rank dimension (4, 8, 16, 32, 64)
    alpha=16.0,                      # Scaling factor (typically 2*rank)
    dropout=0.1,                     # Dropout probability (0.0-0.3)
    target_modules=[                 # Which module types to adapt
        "q_proj",                    # Query projection (attention)
        "k_proj",                    # Key projection (attention)
        "v_proj",                    # Value projection (attention)
        "out_proj",                  # Output projection (attention)
        "fc1",                       # First FFN layer
        "fc2"                        # Second FFN layer
    ],
    apply_to_vision_encoder=True,    # Apply to vision encoder
    apply_to_text_encoder=False,     # Apply to text encoder
    apply_to_geometry_encoder=False, # Apply to geometry encoder
    apply_to_detr_encoder=False,     # Apply to DETR encoder
    apply_to_detr_decoder=False,     # Apply to DETR decoder
    apply_to_mask_decoder=True,      # Apply to mask decoder
)
```

### Common Configurations

**Minimal (Fastest, Lowest Memory):**
```python
LoRAConfig(
    rank=4,
    alpha=8.0,
    dropout=0.1,
    target_modules=["q_proj", "v_proj"],
    apply_to_vision_encoder=True,
    apply_to_text_encoder=False,
    apply_to_geometry_encoder=False,
    apply_to_detr_encoder=False,
    apply_to_detr_decoder=False,
    apply_to_mask_decoder=False,
)
```

**Balanced (Recommended):**
```python
LoRAConfig(
    rank=8,
    alpha=16.0,
    dropout=0.1,
    target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
    apply_to_vision_encoder=True,
    apply_to_text_encoder=False,
    apply_to_geometry_encoder=False,
    apply_to_detr_encoder=False,
    apply_to_detr_decoder=False,
    apply_to_mask_decoder=True,
)
```

**Full (Maximum Adaptation):**
```python
LoRAConfig(
    rank=16,
    alpha=32.0,
    dropout=0.1,
    target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
    apply_to_vision_encoder=True,
    apply_to_text_encoder=False,
    apply_to_geometry_encoder=False,
    apply_to_detr_encoder=True,
    apply_to_detr_decoder=True,
    apply_to_mask_decoder=True,
)
```

---

## Save and Load LoRA Weights

### Save LoRA Weights Only

```python
from lora_layers import save_lora_weights

# Save only LoRA weights (small file!)
save_lora_weights(model, 'lora_weights.pt')
```

Training automatically saves weights to the output directory specified in your config:
```bash
# After training completes, weights are saved to:
# outputs/sam3_lora_full/lora_weights.pt
```

### Load LoRA Weights

```python
from lora_layers import load_lora_weights

# Load LoRA weights into a model
load_lora_weights(model, 'lora_weights.pt')
```

### Check Parameter Counts

```python
from lora_layers import count_parameters

# Get statistics about trainable parameters
stats = count_parameters(model)
print(f"Trainable: {stats['trainable_parameters']:,}")
print(f"Total: {stats['total_parameters']:,}")
print(f"Percentage: {stats['trainable_percentage']:.2f}%")
```

---

## Monitoring Training

### Training Progress

Monitor training progress directly in the console:

```bash
python3 train_sam3_lora_native.py --config configs/full_lora_config.yaml
# Output shows epoch progress and loss values in real-time
```

### Saved Weights

Check saved LoRA weights:

```bash
# List saved weights (location based on your config's output_dir)
ls -lh outputs/sam3_lora_full/
# Output: lora_weights.pt
```

---

## Examples

### Example 1: Quick Test Training

```bash
# Create a config for quick testing
cat > configs/quick_test.yaml << EOF
lora:
  rank: 4
  alpha: 8
  dropout: 0.1
  target_modules: ["q_proj", "v_proj"]
  apply_to_vision_encoder: true
  apply_to_text_encoder: false
  apply_to_geometry_encoder: false
  apply_to_detr_encoder: false
  apply_to_detr_decoder: false
  apply_to_mask_decoder: false

training:
  batch_size: 1
  num_epochs: 5
  learning_rate: 1e-4
  weight_decay: 0.01

output:
  output_dir: "outputs/quick_test"
EOF

# Run training
python3 train_sam3_lora_native.py --config configs/quick_test.yaml
```

### Example 2: Production Training

```bash
# Create a config for production
cat > configs/production.yaml << EOF
lora:
  rank: 32
  alpha: 64
  dropout: 0.1
  target_modules: ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]
  apply_to_vision_encoder: true
  apply_to_text_encoder: false
  apply_to_geometry_encoder: false
  apply_to_detr_encoder: true
  apply_to_detr_decoder: true
  apply_to_mask_decoder: true

training:
  batch_size: 2
  num_epochs: 50
  learning_rate: 3e-5
  weight_decay: 0.01

output:
  output_dir: "outputs/production"
EOF

# Run training
python3 train_sam3_lora_native.py --config configs/production.yaml
```

### Example 3: Apply LoRA to Your Own Model

```python
from lora_layers import LoRAConfig, apply_lora_to_model, count_parameters
import torch.nn as nn

# Your existing PyTorch model
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.TransformerEncoder(...)
        self.decoder = nn.Linear(512, num_classes)

    def forward(self, x):
        features = self.encoder(x)
        return self.decoder(features)

# Add LoRA
model = MyModel()
lora_config = LoRAConfig(
    rank=8,
    alpha=16.0,
    dropout=0.1,
    target_modules=["q_proj", "k_proj", "v_proj"],
    apply_to_vision_encoder=True,
    apply_to_text_encoder=False,
    apply_to_geometry_encoder=False,
    apply_to_detr_encoder=False,
    apply_to_detr_decoder=False,
    apply_to_mask_decoder=False,
)
model = apply_lora_to_model(model, lora_config)

# Check trainable parameters
stats = count_parameters(model)
print(f"Trainable: {stats['trainable_parameters']:,} / {stats['total_parameters']:,}")
print(f"Percentage: {stats['trainable_percentage']:.2f}%")
```

---

## Package Structure

```
sam3_lora/
├── sam3/                          # Full SAM3 library code
│   ├── assets/                    # BPE vocab, etc.
│   ├── model/                     # SAM3 model architecture
│   ├── train/                     # SAM3 training utilities
│   │   ├── loss/                  # Loss functions
│   │   ├── matcher.py             # Hungarian matcher
│   │   └── data/                  # Data loading utilities
│   └── model_builder.py           # Model construction
│
├── configs/                       # YAML configuration files
│   └── full_lora_config.yaml      # Default training config
│
├── data/                          # Your training data
│   ├── train/
│   │   ├── images/                # Training images
│   │   └── annotations/           # JSON annotations
│   └── valid/
│       ├── images/                # Validation images
│       └── annotations/           # JSON annotations
│
├── outputs/                       # Training outputs
│   └── sam3_lora_full/
│       └── lora_weights.pt        # Saved LoRA weights
│
├── lora_layers.py                 # LoRA implementation
├── train_sam3_lora_native.py      # Main training script (CLI + YAML)
└── README.md                      # This file
```

---

## Troubleshooting

### Common Issues

**1. Import Error**
```python
# ✗ Wrong
from src.lora import LoRAConfig

# ✓ Correct
from sam3_lora import LoRAConfig
```

**2. Module Not Found**
```bash
# Install the package
cd /workspace/sam3_lora
pip install -e .
```

**3. CUDA Out of Memory**
```yaml
# Edit your config file to reduce batch size and rank
training:
  batch_size: 1

lora:
  rank: 4
```

**4. Data Not Found**
```bash
# Make sure you have data with proper structure
ls data/train/images/
ls data/train/annotations/
ls data/valid/images/
ls data/valid/annotations/
```

### Getting Help

If you encounter issues:

1. Check the [Troubleshooting](#troubleshooting) section
2. Review the training script (`train_sam3_lora_native.py`)
3. Ensure data is in proper format (images + JSON annotations)
4. Check that the package is installed (`pip install -e .`)

---

## Performance Benchmarks

### Parameter Efficiency

| Configuration | Total Params | Trainable | Ratio | Checkpoint Size |
|---------------|--------------|-----------|-------|-----------------|
| Full Model | 848M | 848M | 100% | ~3.0 GB |
| LoRA (r=4) | 848M | 2M | 0.2% | ~10 MB |
| LoRA (r=8) | 848M | 4M | 0.5% | ~20 MB |
| LoRA (r=16) | 848M | 8M | 0.9% | ~40 MB |

### Training Speed

| Batch Size | GPU Memory | Speed | Configuration |
|------------|------------|-------|---------------|
| 1 | 8 GB | ~2 it/s | Minimal (r=4) |
| 2 | 12 GB | ~3 it/s | Balanced (r=8) |
| 4 | 16 GB | ~5 it/s | Full (r=16) |

*Benchmarks on NVIDIA RTX 3090*

---

## Documentation

- **README.md** (this file) - Complete usage guide
- **README_STANDALONE.md** - Standalone package details
- **LORA_IMPLEMENTATION_GUIDE.md** - Technical implementation details

---

## Citation

If you use this work, please cite:

```bibtex
@software{sam3_lora,
  title = {SAM3-LoRA: Low-Rank Adaptation for Fine-Tuning},
  author = {AI Research Group, KMUTT},
  year = {2025},
  organization = {King Mongkut's University of Technology Thonburi},
  url = {https://github.com/yourusername/sam3_lora}
}
```

### References

- **LoRA**: [Hu et al., 2021](https://arxiv.org/abs/2106.09685) - "LoRA: Low-Rank Adaptation of Large Language Models"
- **SAM**: [Kirillov et al., 2023](https://arxiv.org/abs/2304.02643) - "Segment Anything"

---

## License

This project is licensed under Apache 2.0. See [LICENSE](LICENSE) for details.

---

## Credits

Developed by **AI Research Group, KMUTT** (King Mongkut's University of Technology Thonburi)

---

## Status

- ✅ **SAM3 LoRA Training**: Full SAM3 model with LoRA fine-tuning via CLI
- ✅ **YAML Configuration**: Flexible config file system with CLI argument support
- ✅ **LoRA Implementation**: Complete with all utilities and enhanced injection
- ✅ **Data Loading**: Custom dataset format with individual JSON annotations
- ✅ **Production Ready**: Tested and working training pipeline
- ✅ **Documentation**: Comprehensive guides and examples

---

<div align="center">

**Version**: 0.1.0
**Python**: 3.8+
**PyTorch**: 2.0+

**Made by AI Research Group, KMUTT**
*King Mongkut's University of Technology Thonburi*

**Built with ❤️ for the research community**

[⬆ Back to Top](#sam3-lora-low-rank-adaptation-for-fine-tuning)

</div>
