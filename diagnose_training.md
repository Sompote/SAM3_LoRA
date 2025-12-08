# Training Loss Not Reducing - Diagnosis

## Problem: Loss is Highly Fluctuating

From the training output:
```
Step 1: 169
Step 2: 141
Step 3: 169
Step 4: 242 ← INCREASES!
Step 5: 212
Step 6: 182
Step 7: 186
Step 8: 161
Step 9: 189
Step 10: 181
Step 11: 181
Step 12: 172
Step 13: 218 ← INCREASES AGAIN!
Step 14: 186
Step 15: 175
Step 16: 198
```

**This is NOT normal** - loss should decrease more smoothly!

---

## Root Causes

### 1. **CRITICAL: Config Points to Wrong Data Directory**

**File**: `configs/full_lora_config.yaml` (line 35)
```yaml
data_dir: "/workspace/data3"  # ← WRONG! Should be /workspace/data2
```

**Impact**: Unknown - if data3 doesn't exist or has different data, this could cause issues.

---

### 2. **High Gradient Variance from Tiny Batch Size**

**Current setup**:
- `batch_size: 1` (only ONE sample per step)
- `gradient_accumulation_steps: 16`
- **Effective batch size: 16**

**Problem**: With batch_size=1, each gradient update is calculated from **only one sample**, causing:
- **High variance** in gradient estimates
- **Unstable loss** that jumps around
- **Poor convergence** - model can't learn stable patterns

**Why this happens**:
- Sample A might have 3 cracks → loss focuses on detecting 3 objects
- Sample B might have 1 crack → loss focuses on detecting 1 object
- Sample C might have 10 cracks → loss focuses on detecting many objects
- Model gets conflicting signals every step!

---

### 3. **Increased LoRA Rank = More Memory Pressure**

**Config changes**:
- `rank: 64` (was 32) - **DOUBLED**
- `alpha: 128` (was 64)

**Impact**:
- More trainable parameters: **11.8M params** (1.38% of model)
- Higher memory usage: **22.6GB / 32.6GB used** (69% full)
- Training processes keep getting **KILLED by OOM** (exit code 137)

---

### 4. **Learning Rate Warmup Not Working Properly**

**Config**: `warmup_steps: 200`

**Problem**: With gradient accumulation, warmup should be counted in **gradient updates**, not data samples. The model might be getting the full learning rate too early, causing instability.

---

### 5. **No Gradient Clipping Verification**

**Config**: `max_grad_norm: 1.0`

**Question**: Are gradients actually being clipped? Large gradients could cause loss spikes.

---

## Solutions (In Order of Priority)

### IMMEDIATE FIX 1: Correct Data Directory
```yaml
# In configs/full_lora_config.yaml line 35
data_dir: "/workspace/data2"  # ← Fix this!
```

### IMMEDIATE FIX 2: Reduce LoRA Rank to Avoid OOM
```yaml
# In configs/full_lora_config.yaml
lora:
  rank: 32  # Back to original (was 64)
  alpha: 64  # Back to original (was 128)
```

### IMMEDIATE FIX 3: Increase Batch Size to Reduce Variance
```yaml
# In configs/full_lora_config.yaml
training:
  batch_size: 2  # Increase from 1
  gradient_accumulation_steps: 8  # Reduce from 16
  # Effective batch size stays: 2 × 8 = 16
```

**Why this helps**:
- Each gradient step uses **2 samples** instead of 1
- Reduces variance by 50%
- More stable learning
- Same effective batch size (16)

### FIX 4: Add Learning Rate Warmup Properly

The current implementation should handle warmup, but verify it's working by checking if loss stabilizes after ~200 steps.

### FIX 5: Monitor Gradient Norms

Add logging to verify gradients aren't exploding:
```python
# After total_loss.backward()
total_norm = 0
for p in model.parameters():
    if p.grad is not None:
        total_norm += p.grad.data.norm(2).item() ** 2
total_norm = total_norm ** 0.5
print(f"Gradient norm: {total_norm:.4f}")
```

---

## Alternative: Use Light Config

If OOM continues, use the lighter config:
```bash
python3 train_sam3_lora_native.py --config configs/light_lora_config.yaml
```

**Light config**:
- `rank: 16` (less memory)
- `batch_size: 2` (more stable)
- Skips vision encoder LoRA (saves memory)
- Same effective training power

---

## Expected Behavior After Fixes

**First epoch, loss should**:
1. Start high (150-200)
2. Decrease smoothly: 150 → 140 → 130 → 120 → ...
3. Stabilize after warmup (~200 steps)
4. Continue decreasing gradually

**Not**:
- Jump around: 169 → 141 → 169 → 242 → 212
- Increase frequently
- Stay constant for many steps

---

## Summary

| Issue | Impact | Fix |
|-------|--------|-----|
| Wrong data_dir | Unknown/Critical | Change to `/workspace/data2` |
| batch_size=1 | High variance | Increase to 2 |
| rank=64 | OOM kills training | Reduce to 32 |
| gradient_accumulation=16 | Less important | Reduce to 8 |

**Priority**: Fix data_dir and batch_size first!
