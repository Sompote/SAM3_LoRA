# SAM3-LoRA Training Guide
## AI Research Group - KMUTT

---

## 📖 Complete Training Workflow

### Step-by-Step Training Process

```
┌─────────────────────────────────────────────────────────────┐
│                    SAM3-LoRA Training Pipeline              │
└─────────────────────────────────────────────────────────────┘

1. SETUP
   ├── Install dependencies
   ├── Login to HuggingFace
   └── Verify GPU availability

2. DATA PREPARATION
   ├── Organize dataset structure
   ├── Convert annotations (COCO/YOLO → SAM3)
   └── Validate dataset

3. CONFIGURATION
   ├── Choose LoRA strategy (minimal/balanced/full)
   ├── Set training parameters
   └── Configure hardware settings

4. TRAINING
   ├── Initialize model with LoRA
   ├── Train on custom data
   └── Save checkpoints

5. EVALUATION
   ├── Validate on test set
   ├── Compute IoU metrics
   └── Export results

6. DEPLOYMENT
   ├── Load trained LoRA weights
   ├── Run inference
   └── Visualize results
```

---

## 🎯 Quick Reference Commands

### Installation & Setup

```bash
# Clone and setup
git clone <repository-url>
cd sam3-lora
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Login to HuggingFace
huggingface-cli login
```

### Data Preparation

```bash
# Create directory structure
python prepare_data.py create --output_dir data

# Convert COCO format
python prepare_data.py coco \
  --coco_json annotations.json \
  --images_dir images/ \
  --output_dir data \
  --split train

# Convert YOLO format
python prepare_data.py yolo \
  --yolo_dir dataset/ \
  --classes "person,car,dog" \
  --output_dir data \
  --split train

# Validate dataset
python prepare_data.py validate \
  --data_dir data \
  --split train
```

### Training

```bash
# Minimal training (fastest)
python train_sam3_lora.py \
  --config configs/minimal_lora_config.yaml

# Balanced training (recommended)
python train_sam3_lora.py \
  --config configs/base_config.yaml

# Full training (maximum capacity)
python train_sam3_lora.py \
  --config configs/full_lora_config.yaml

# Custom training with CLI overrides
python train_sam3_lora.py \
  --config configs/base_config.yaml \
  --batch_size 8 \
  --learning_rate 2e-4 \
  --num_epochs 20 \
  --lora_rank 16
```

### Inference

```bash
# Text prompt
python inference.py \
  --lora_weights outputs/sam3_lora/best_model/lora_weights.pt \
  --image test.jpg \
  --text_prompt "yellow school bus" \
  --output result.png

# Bounding box prompt
python inference.py \
  --lora_weights outputs/sam3_lora/best_model/lora_weights.pt \
  --image test.jpg \
  --bboxes '[[100, 150, 400, 350]]' \
  --output result.png

# Combined prompts
python inference.py \
  --lora_weights outputs/sam3_lora/best_model/lora_weights.pt \
  --image test.jpg \
  --text_prompt "red car" \
  --bboxes '[[100, 150, 400, 350]]' \
  --output result.png
```

---

## 🔧 Configuration Comparison

### LoRA Strategy Selection

| Strategy | Parameters | Memory | Speed | Use Case |
|----------|-----------|--------|-------|----------|
| **Minimal** | 0.06% (500K) | ~10GB | Fast | Quick adaptation, limited data |
| **Balanced** | 0.47% (4M) | ~18GB | Medium | General fine-tuning |
| **Full** | 1.77% (15M) | ~32GB | Slow | Complex tasks, large datasets |

### Component Selection Guide

```yaml
# Minimal: Decoder only
apply_to_vision_encoder: false
apply_to_text_encoder: false
apply_to_detr_encoder: false
apply_to_detr_decoder: true    # Only this

# Balanced: Vision + Reasoning
apply_to_vision_encoder: true
apply_to_text_encoder: false
apply_to_detr_encoder: true
apply_to_detr_decoder: true

# Full: All components
apply_to_vision_encoder: true
apply_to_text_encoder: true
apply_to_geometry_encoder: true
apply_to_detr_encoder: true
apply_to_detr_decoder: true
apply_to_mask_decoder: true
```

---

## 📊 Training Parameter Guide

### Learning Rate Schedule

| Dataset Size | Learning Rate | Warmup Steps | Epochs |
|-------------|---------------|--------------|--------|
| Small (<1K) | 2e-4 | 100 | 5-10 |
| Medium (1K-10K) | 1e-4 | 500 | 10-20 |
| Large (>10K) | 5e-5 | 1000 | 20-50 |

### Batch Size vs GPU Memory

| GPU | VRAM | Minimal Config | Balanced Config | Full Config |
|-----|------|----------------|-----------------|-------------|
| RTX 3060 | 12GB | BS=4 | BS=2 | N/A |
| RTX 3090 | 24GB | BS=8 | BS=4 | BS=2 |
| A100 | 40GB | BS=16 | BS=8 | BS=4 |
| A100 | 80GB | BS=32 | BS=16 | BS=8 |

### LoRA Rank Selection

| Rank | Trainable Params | Training Speed | Performance |
|------|------------------|----------------|-------------|
| 4 | Lowest | Fastest | Good for simple tasks |
| 8 | Low | Fast | Good for most tasks |
| 16 | Medium | Medium | Best for complex tasks |
| 32 | High | Slow | Experimental |

---

## 🎓 Training Best Practices

### 1. Start Small, Scale Up

```bash
# Step 1: Quick validation (minimal config, 1 epoch)
python train_sam3_lora.py \
  --config configs/minimal_lora_config.yaml \
  --num_epochs 1

# Step 2: If working well, scale to balanced
python train_sam3_lora.py \
  --config configs/base_config.yaml \
  --num_epochs 10

# Step 3: If needed, scale to full
python train_sam3_lora.py \
  --config configs/full_lora_config.yaml \
  --num_epochs 20
```

### 2. Monitor Training Progress

Key metrics to watch:
- **Loss**: Should decrease steadily
- **Validation IoU**: Should increase
- **GPU Memory**: Should be <90% utilization
- **Training Speed**: ~0.5-2 iterations/second

```bash
# Monitor GPU during training
watch -n 1 nvidia-smi
```

### 3. Checkpointing Strategy

```yaml
training:
  save_steps: 1000           # Save every 1000 steps
  eval_steps: 500            # Evaluate every 500 steps
  save_total_limit: 3        # Keep only 3 best checkpoints
```

### 4. Data Quality

✅ **Good Practices:**
- High-quality annotations
- Diverse training examples
- Balanced class distribution
- Consistent image quality

❌ **Avoid:**
- Noisy or incorrect labels
- Extreme class imbalance
- Very small training sets (<100 images)
- Inconsistent annotation formats

---

## 🔍 Debugging Common Issues

### Issue 1: Training Loss Not Decreasing

**Symptoms:**
```
Epoch 1: loss=0.543
Epoch 2: loss=0.541
Epoch 3: loss=0.542
...
```

**Solutions:**
```yaml
# 1. Increase learning rate
training:
  learning_rate: 2e-4  # Instead of 1e-4

# 2. Increase LoRA rank
lora:
  rank: 16
  alpha: 32

# 3. Apply LoRA to more components
lora:
  apply_to_vision_encoder: true
  apply_to_detr_encoder: true
  apply_to_detr_decoder: true
```

### Issue 2: Overfitting

**Symptoms:**
```
Training IoU: 0.95
Validation IoU: 0.65
```

**Solutions:**
```yaml
# 1. Add dropout
lora:
  dropout: 0.1

# 2. Increase weight decay
training:
  weight_decay: 0.05

# 3. Use smaller LoRA rank
lora:
  rank: 4
```

### Issue 3: Out of Memory

**Error:** `RuntimeError: CUDA out of memory`

**Solutions:**
```bash
# Option 1: Reduce batch size
python train_sam3_lora.py \
  --config configs/base_config.yaml \
  --batch_size 2

# Option 2: Use gradient accumulation
python train_sam3_lora.py \
  --config configs/base_config.yaml \
  --batch_size 2 \
  --gradient_accumulation_steps 4

# Option 3: Use minimal config
python train_sam3_lora.py \
  --config configs/minimal_lora_config.yaml
```

---

## 📈 Expected Training Timeline

### Minimal Configuration
- **Setup**: 5-10 minutes
- **Data Prep**: 10-30 minutes (depends on dataset size)
- **Training**: 30-60 minutes (1K images, 10 epochs)
- **Total**: ~1-2 hours

### Balanced Configuration
- **Setup**: 5-10 minutes
- **Data Prep**: 10-30 minutes
- **Training**: 2-4 hours (1K images, 10 epochs)
- **Total**: ~3-5 hours

### Full Configuration
- **Setup**: 5-10 minutes
- **Data Prep**: 10-30 minutes
- **Training**: 4-8 hours (1K images, 20 epochs)
- **Total**: ~5-10 hours

---

## 🎯 Use Case Examples

### 1. Retail Product Segmentation

```yaml
# configs/retail_config.yaml
lora:
  rank: 8
  alpha: 16
  apply_to_vision_encoder: true
  apply_to_detr_decoder: true

training:
  train_data_path: "data/retail/train"
  batch_size: 4
  learning_rate: 1e-4
  num_epochs: 15
```

**Text prompts:** "coca cola bottle", "milk carton", "bread package"

### 2. Medical Imaging

```yaml
# configs/medical_config.yaml
lora:
  rank: 16
  alpha: 32
  apply_to_vision_encoder: true
  apply_to_detr_encoder: true
  apply_to_detr_decoder: true

training:
  train_data_path: "data/medical/train"
  batch_size: 2
  learning_rate: 5e-5
  num_epochs: 30
```

**Text prompts:** "tumor", "lesion", "organ"

### 3. Autonomous Driving

```yaml
# configs/driving_config.yaml
lora:
  rank: 12
  alpha: 24
  apply_to_vision_encoder: true
  apply_to_text_encoder: true
  apply_to_detr_decoder: true

training:
  train_data_path: "data/driving/train"
  batch_size: 4
  learning_rate: 1e-4
  num_epochs: 20
```

**Text prompts:** "pedestrian", "vehicle", "traffic sign", "road"

### 4. Agriculture

```yaml
# configs/agriculture_config.yaml
lora:
  rank: 8
  alpha: 16
  apply_to_vision_encoder: true
  apply_to_detr_decoder: true

training:
  train_data_path: "data/crops/train"
  batch_size: 4
  learning_rate: 1e-4
  num_epochs: 15
```

**Text prompts:** "diseased leaf", "ripe fruit", "pest"

---

## 📚 Additional Resources

### Documentation Files
- `README.md` - Complete project documentation
- `TRAINING_GUIDE.md` - This file
- `configs/*.yaml` - Configuration templates

### Code Files
- `train_sam3_lora.py` - Main training script
- `inference.py` - Inference script
- `prepare_data.py` - Data preparation utilities
- `lora_layers.py` - LoRA implementation
- `example_usage.py` - Usage examples

### Scripts
- `quickstart.sh` - Automated setup script

---

## ✅ Pre-Flight Checklist

Before starting training, verify:

- [ ] Python 3.12+ installed
- [ ] CUDA 12.6+ available
- [ ] GPU with 16GB+ VRAM
- [ ] HuggingFace account created
- [ ] HuggingFace CLI login completed
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Dataset prepared and validated
- [ ] Configuration file reviewed
- [ ] Output directory specified
- [ ] Sufficient disk space (>10GB)

---

## 📞 Support

**AI Research Group - KMUTT**

For questions or issues:
1. Check README.md and this guide
2. Review example configurations
3. Run example_usage.py
4. Contact: ai-research@kmutt.ac.th

---

**Last Updated:** December 2025
**Version:** 1.0
**Maintained by:** AI Research Group, KMUTT
