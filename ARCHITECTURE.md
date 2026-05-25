# V4 Model Architecture

## Overview

```
Input: [B, 4, 3, 96, 96]  (4 frames, RGB, 96x96)
Output: [B, 4, 2]  (4-step action chunk, 2D pixel coords)
Total Params: 56.6M (34.9M trainable, 21.7M frozen)
```

```
┌─────────────────────────────────────────────────────────────────────┐
│                        V4 Architecture Pipeline                      │
│                                                                      │
│  [B, 4, 3, 96, 96]                                                   │
│       │                                                              │
│       ▼                                                              │
│  ┌─────────────────────────┐                                        │
│  │  ViTBackbone (DINO)     │  21.9M total, 197K trainable           │
│  │  DINO vits8 (frozen)    │  384→512 projection                    │
│  │  144 patch tokens/frame │  Output: [B*4, 144, 512]               │
│  └─────────────────────────┘                                        │
│       │                                                              │
│       ▼                                                              │
│  ┌─────────────────────────┐                                        │
│  │  ToolinitModel          │  25.3M trainable                       │
│  │  3x SerialHybridBlock   │  MHA + 4x Mamba per block              │
│  │  d_model=512            │  Output: [B*4, 144, 512]               │
│  └─────────────────────────┘                                        │
│       │                                                              │
│       ▼  reshape: [B*4, 144, 512]                                    │
│  ┌─────────────────────────┐                                        │
│  │  SpatialSoftmax         │  8.2K trainable                        │
│  │  K=16 keypoints         │  Conv2d 512→16, softmax attention       │
│  │  grid=12x12             │  Output: [B*4, 16, 2], [B*4, 16, 512]  │
│  └─────────────────────────┘                                        │
│       │                                                              │
│       ▼  reshape to [B, 4, 16, 2] + [B, 4, 16, 512]                  │
│  ┌─────────────────────────┐                                        │
│  │  Keypoint Projection    │  1.1M trainable                        │
│  │  feat_compress 512→32   │  Linear(512,32) + Linear(2176,512)     │
│  │  proj 2176→512          │  Output: [B, 512]                      │
│  └─────────────────────────┘                                        │
│       │                                                              │
│       ▼                                                              │
│  ┌─────────────────────────┐                                        │
│  │  DiffusionActionHead    │  8.4M trainable                        │
│  │  ConditionalUnet1d      │  down_dims=(64,128,256)                │
│  │  horizon=16, action=2   │  Output: [B, 4, 2] (first 4 of 16)     │
│  └─────────────────────────┘                                        │
│                                                                      │
│  Final Output: [B, 4, 2]  →  denormalize → pixel coords [12-511]    │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 1. ViTBackbone

**File:** `models/vit_backbone.py`
**Params:** 21,867,392 total / 197,120 trainable / 21,670,272 frozen

### Design

```
Input:  [B, n_obs, 3, 96, 96]  or  [B*n_obs, 3, 96, 96]
Output: [B*n_obs, 144, 512]
```

| Component | Detail |
|-----------|--------|
| Backbone | DINO v1 vits8 (torch.hub: `facebookresearch/dino:main`) |
| Patch size | 8×8 |
| Input resolution | 96×96 → 12×12 grid = 144 patch tokens |
| DINO embed dim | 384 |
| Num heads | 6 |
| CLS token | Discarded (only patch tokens used) |
| Projection | Linear(384 → 512) |
| Freeze | All DINO params frozen, only `proj` trainable |

### Forward Flow

```
x [B, 4, 3, 96, 96]
  → view(B*4, 3, 96, 96)
  → dino.get_intermediate_layers(x, n=1)[0]  → [B*4, 145, 384]
  → remove CLS token ([:, 1:, :])             → [B*4, 144, 384]
  → self.proj()                                → [B*4, 144, 512]
```

### Parameter Breakdown

| Layer | Params | Trainable |
|-------|--------|-----------|
| DINO vits8 backbone | 21,670,272 | 0 |
| Linear(384, 512) | 197,120 | 197,120 |

---

## 2. ToolinitModel (Mamba + Attention Hybrid Core)

**File:** `models/hybrid_core.py`
**Params:** 25,261,056 total / 25,261,056 trainable

### Design

```
Input:  [B*n_obs, 144, 512]
Output: [B*n_obs, 144, 512]
```

### Architecture: 3x SerialHybridBlock

Each block follows a **serial** pattern (not parallel):

```
x → RMSNorm → MultiheadAttention → +x → RMSNorm → Mamba×4 → +x → output
```

| Component | Detail |
|-----------|--------|
| Normalization | RMSNorm (lighter than LayerNorm) |
| Attention | MultiheadAttention, d_model=512, n_heads=8, batch_first |
| Mamba | 4 sequential Mamba blocks per hybrid block |
| Mamba config | d_model=512, d_state=64, d_conv=4, expand=2 |
| Num layers | 3 SerialHybridBlock stacked |

### Forward Flow (per SerialHybridBlock)

```
x [B, 144, 512]
  → norm1(x)
  → spatial_attn(x, x, x) → attn_out
  → x = x + attn_out
  → norm2(x)
  → mamba_out = Mamba1(Mamba2(Mamba3(Mamba4(norm2(x)))))
  → x = x + mamba_out
  → output [B, 144, 512]
```

### Parameter Breakdown (per SerialHybridBlock)

| Layer | Params |
|-------|--------|
| RMSNorm ×2 | 1,024 |
| MultiheadAttention (512, 8 heads) | ~1,049,088 |
| 4× Mamba (d_model=512, d_state=64) | ~8,400,000 |
| **Per block** | ~9,450,000 |
| **3 blocks total** | ~25,261,056 |

---

## 3. SpatialSoftmax

**File:** `models/spatial_softmax.py`
**Params:** 8,208 total / 8,208 trainable

### Design

```
Input:  [B, 144, 512]  (single frame tokens)
Output: keypoints [B, 16, 2], keypoint_features [B, 16, 512]
```

| Component | Detail |
|-----------|--------|
| Conv2d | 1×1 conv: 512 → 16 channels |
| Grid | 12×12, coords normalized to [-1, 1] |
| Num keypoints | 16 |
| Softmax | Over spatial dimension (N=144) |

### Forward Flow

```
x [B, 144, 512]
  → permute(0,2,1).view(B, 512, 12, 12)     → [B, 512, 12, 12]
  → keypoint_conv()                           → [B, 16, 12, 12]
  → flatten(2)                                → [B, 16, 144]
  → softmax(dim=-1)                           → [B, 16, 144]  (attention weights)
  → matmul(attention, coords.T)              → [B, 16, 2]     (keypoints)
  → matmul(attention, x)                     → [B, 16, 512]   (keypoint features)
```

### Parameter Breakdown

| Layer | Params |
|-------|--------|
| Conv2d(512, 16, kernel=1) | 8,192 + 16 = 8,208 |
| Coordinate grid (buffer) | 0 (non-parameter) |

---

## 4. Keypoint Projection

**File:** Inline in `train_v4.py` / `eval_pusht_v4.py`
**Params:** 1,131,040 total / 1,131,040 trainable

### Design

```
Input:  keypoints [B, 4, 16, 2], kp_features [B, 4, 16, 512]
Output: [B, 512]  (global condition for diffusion head)
```

| Component | Detail |
|-----------|--------|
| feat_compress | Linear(512 → 32) |
| Concat | keypoints + compressed features → [B, 4, 16, 34] |
| proj | Linear(2176 → 512) where 2176 = 4 × 16 × 34 |

### Forward Flow

```
keypoints [B, 4, 16, 2]
kp_features [B, 4, 16, 512]
  → feat_compress(kp_features)                → [B, 4, 16, 32]
  → cat(keypoints, compressed, dim=-1)        → [B, 4, 16, 34]
  → flatten(1)                                → [B, 2176]
  → proj()                                    → [B, 512]
```

### Parameter Breakdown

| Layer | Params |
|-------|--------|
| Linear(512, 32) | 16,384 + 32 = 16,416 |
| Linear(2176, 512) | 1,114,112 + 512 = 1,114,624 |

---

## 5. DiffusionActionHead

**File:** `models/diffusion_action_head.py`
**Params:** 8,376,834 total / 8,376,834 trainable

### Design

```
Training:  compute_loss(global_cond [B, 512], action [B, 16, 2]) → MSE loss
Inference: generate(global_cond [B, 512], batch_size=B) → [B, 4, 2]
```

| Component | Detail |
|-----------|--------|
| UNet | ConditionalUnet1d, down_dims=(64, 128, 256) |
| Horizon | 16 action steps |
| Action dim | 2 (x, y pixel coords) |
| Conditioning | FiLM (scale + bias from condition embedding) |
| Time embedding | SinusoidalPosEmb(64) |
| Training scheduler | DDPMScheduler (100 timesteps, squaredcos_cap_v2) |
| Inference scheduler | DDIMScheduler (default 20 steps) |

### ConditionalUnet1d Architecture

```
Input: [B, 16, 2] (noisy action)
Condition: [B, 576] (64 time emb + 512 global_cond)

Encoder:
  Stage 0: ResBlock(2→64) → ResBlock(64→64) → Conv1d(stride=2)  → [B, 8, 64]
  Stage 1: ResBlock(64→128) → ResBlock(128→128) → Conv1d(stride=2) → [B, 4, 128]
  Stage 2: ResBlock(128→256) → ResBlock(256→256)                    → [B, 4, 256]

Bottleneck:
  ResBlock(256→256) → ResBlock(256→256)                              → [B, 4, 256]

Decoder:
  Stage 0: concat skip(256) → ResBlock(512→256) → ResBlock(256→256) → [B, 4, 256]
  Stage 1: ConvTranspose1d(stride=2) → concat skip(128) → ResBlock(384→128) → ResBlock(128→128) → [B, 8, 128]
  Stage 2: ConvTranspose1d(stride=2) → concat skip(64) → ResBlock(192→64) → ResBlock(64→64) → [B, 16, 64]

Output:
  Conv1dBlock(64→64) → Conv1d(64→2) → [B, 16, 2]
```

### Residual Block with FiLM

```
x [B, C_in, T], cond [B, 576]
  → Conv1dBlock(x)                              → [B, C_out, T]
  → cond_mlp(cond) → unsqueeze(-1)              → [B, C_out*2, 1]
  → chunk into scale, bias                      → [B, C_out, 1] each
  → scale * out + bias                          → [B, C_out, T]
  → Conv1dBlock(out)                            → [B, C_out, T]
  → + residual_conv(x)                          → [B, C_out, T]
```

### Training Flow

```
global_cond [B, 512], action [B, 16, 2]
  → eps = randn_like(action)
  → t = random(0, 100)
  → noisy = noise_scheduler.add_noise(action, eps, t)
  → pred = unet(noisy, t, global_cond)
  → loss = MSE(pred, eps)
```

### Inference Flow

```
global_cond [B, 512], batch_size=B
  → sample = randn(B, 16, 2)
  → scheduler.set_timesteps(20)  # DDIM
  → for t in scheduler.timesteps:
      pred = unet(sample, t, global_cond)
      sample = scheduler.step(pred, t, sample).prev_sample
  → return sample[:, :4]  # first 4 actions
```

---

## Parameter Summary

| Module | Total | Trainable | Frozen | % of Total |
|--------|-------|-----------|--------|------------|
| ViTBackbone | 21,867,392 | 197,120 | 21,670,272 | 38.6% |
| ToolinitModel | 25,261,056 | 25,261,056 | 0 | 44.6% |
| SpatialSoftmax | 8,208 | 8,208 | 0 | 0.0% |
| Keypoint Proj | 1,131,040 | 1,131,040 | 0 | 2.0% |
| DiffusionActionHead | 8,376,834 | 8,376,834 | 0 | 14.8% |
| **TOTAL** | **56,644,530** | **34,974,258** | **21,670,272** | **100%** |

## Data Flow Summary

```
[B, 4, 3, 96, 96]
    │
    │  ViTBackbone (DINO vits8, frozen)
    ▼
[B*4, 144, 512]
    │
    │  ToolinitModel (3x MHA + 4x Mamba)
    ▼
[B*4, 144, 512]
    │
    │  reshape → [B, 4, 144, 512] → [B*4, 144, 512]
    │
    │  SpatialSoftmax (K=16, Conv2d 512→16)
    ▼
keypoints: [B*4, 16, 2]    kp_features: [B*4, 16, 512]
    │                              │
    │  reshape → [B, 4, 16, 2]     │  reshape → [B, 4, 16, 512]
    │                              │
    │                              │  feat_compress → [B, 4, 16, 32]
    │                              ▼
    └─────────────────────────  cat → [B, 4, 16, 34]
                                       │
                                       │  flatten → [B, 2176]
                                       │
                                       │  proj → [B, 512]
                                       ▼
                              DiffusionActionHead
                              global_cond [B, 512]
                                       │
                              Training: compute_loss → MSE
                              Inference: generate → [B, 4, 2]
```

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Batch size | 48 |
| Epochs | 35 |
| LR | 1e-4 → 1e-6 (LinearLR warmup + CosineAnnealing) |
| Optimizer | AdamW (weight_decay=0.0) |
| Precision | BF16 autocast + GradScaler |
| Gradient clip | 1.0 |
| Early stopping | Patience=8 |
| Eval warmup | Skip first 10 epochs |
| Inference | DDIM 20 steps |

## Performance

| Metric | Result |
|--------|--------|
| Strict success (≥0.95) | 46.0% (23/50) |
| Effective (≥0.80) | 82.0% (41/50) |
| Mean coverage | 0.864 ± 0.215 |
| Max coverage | 0.991 |
| Training time | ~6-7 hours |
| Dataset | 206 episodes, 25,650 frames |

## keyview
no GPU augmented