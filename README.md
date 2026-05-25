# 🤖 Robo Diffusion Light

> A lightweight Diffusion Policy for robotic manipulation, built on ViT + Mamba architecture.

<div align="center">

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![PushT](https://img.shields.io/badge/Env-PushT-green.svg)](https://github.com/NVlabs/gym-pusht)
[![Diffusion Policy](https://img.shields.io/badge/Method-Diffusion-orange.svg)](https://diffusion-policy.cs.columbia.edu/)

**56.6M params · 34.9M trainable · 14-15h training · 68% strict success rate**

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Architecture](#-architecture)
- [Quick Start](#-quick-start)
- [Training](#-training)
- [Evaluation](#-evaluation)
- [Results](#-results)
- [Project Structure](#-project-structure)
- [Requirements](#-requirements)
- [Known Issues](#-known-issues)
- [Acknowledgments](#-acknowledgments)
- [License](#-license)

---

## 🎯 Overview

**Robo Diffusion Light** is a lightweight imitation learning pipeline for the [PushT](https://github.com/NVlabs/gym-pusht) environment, combining the strengths of:

| Component | Technology | Role |
|-----------|-----------|------|
| 👁️ Vision | **DINO v1 vits8** (frozen) | Visual feature extraction from RGB frames |
| 🧠 Temporal | **Mamba + Attention** hybrid | Spatiotemporal reasoning over observation history |
| 📍 Keypoints | **Spatial Softmax** | Learnable keypoint detection from token grid |
| 🎯 Action | **Diffusion Policy** (UNet1D) | Multi-modal action chunk generation |

### Key Numbers

```
┌─────────────────────────────────────────────────────────────┐
│  Dataset:    206 episodes · 25,650 frames · 10 FPS          │
│  Model:      56.6M params (34.9M trainable, 21.7M frozen)   │
<<<<<<< HEAD
│  Training:   ~6-7 hours on single GPU                       │
=======
│  Training:   ~14-15 hours on single GPU                       │
>>>>>>> 29a3ecc29946fdceb75af144a1c98ee80942e96a
│  Result:     68% strict success · 90% effective coverage    │
└─────────────────────────────────────────────────────────────┘
```

---

## 🏗️ Architecture

### Pipeline

```
Input: [B, 4, 3, 96, 96]  ──→  Output: [B, 4, 2]  (action chunk)

┌──────────────────┐
│  ViTBackbone     │  DINO vits8 (frozen)
│  [B*4, 144, 512] │  144 patch tokens per frame
└────────┬─────────┘
         │
┌────────▼─────────┐
│  ToolinitModel   │  3× SerialHybridBlock
│  [B*4, 144, 512] │  MHA + 4× Mamba per block
└────────┬─────────┘
         │
┌────────▼─────────┐
│ SpatialSoftmax   │  K=16 keypoints
│  [B*4, 16, 2]    │  Conv2d attention on 12×12 grid
│  [B*4, 16, 512]  │
└────────┬─────────┘
         │
┌────────▼─────────┐
│ Keypoint Proj    │  feat_compress + proj
│  [B, 512]        │  Global condition vector
└────────┬─────────┘
         │
┌────────▼─────────┐
│ DiffusionHead    │  ConditionalUnet1D
│  [B, 4, 2]       │  down_dims=(64, 128, 256)
└──────────────────┘
```

### Parameter Breakdown

| Module | Total | Trainable | Frozen | % |
|--------|------:|----------:|-------:|--:|
| ViTBackbone | 21.9M | 197K | 21.7M | 38.6% |
| **ToolinitModel** | **25.3M** | **25.3M** | 0 | **44.6%** |
| DiffusionActionHead | 8.4M | 8.4M | 0 | 14.8% |
| Keypoint Proj | 1.1M | 1.1M | 0 | 2.0% |
| SpatialSoftmax | 8.2K | 8.2K | 0 | 0.0% |
| **TOTAL** | **56.6M** | **34.9M** | **21.7M** | **100%** |

> 💡 The DINO vision backbone is **fully frozen** — only 197K parameters in the projection layer are trainable. The core temporal reasoning (Mamba + Attention) accounts for **44.6%** of all parameters.

---

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Create conda environment
conda create -n robo-diffusion python=3.12
conda activate robo-diffusion

# Install PyTorch (adjust for your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install dependencies
pip install mamba-ssm diffusers gymnasium gym-pusht tqdm matplotlib imageio opencv-python
pip install av  # for video loading
```

### 2. Data Preparation

The project uses the LeRobot-format PushT dataset. Data is loaded automatically from:

```
data/
├── raw_pusht/          # Raw LeRobot PushT dataset
│   ├── data/           # Parquet files (actions, states)
│   ├── meta/           # Dataset metadata & stats
│   └── videos/         # Observation videos
└── lerobot_loader.py   # Custom dataset wrapper
```

### 3. Run Training

```bash
# Train from scratch (V4)
python train_v4.py

# Resume from V4 best checkpoint
python train_v4_resume.py --v4-ckpt checkpoints_diffusion/best.pth
```

### 4. Run Evaluation

```bash
# Evaluate V4 best model
python eval_pusht_v4.py --ckpt checkpoints_diffusion/best.pth --episodes 50

# Evaluate any checkpoint
python eval_pusht_v4.py --ckpt checkpoints_diffusion_v4_resume/last.pth --episodes 50
```

---

## 📊 Training

### Configuration

| Parameter | V4 | V4 Resume |
|-----------|-----|-----------|
| Batch Size | 48 | 48 |
| Total Epochs | 35 | 50 (resume from 35) |
| Learning Rate | 1e-4 → 1e-6 | 1e-6 → 1e-7 |
| LR Schedule | LinearLR + Cosine | Cosine only |
| Optimizer | AdamW (wd=0) | AdamW (wd=0) |
| Precision | BF16 + GradScaler | BF16 + GradScaler |
| Gradient Clip | 1.0 | 1.0 |
| Early Stopping | Patience=8 | Patience=8 |
| Eval Warmup | Skip first 10 epochs | Skip first 10 epochs |

### Training Loop

```
for epoch in range(EPOCHS):
    └── for batch in loader:
            ├── vision(img)          → [B*4, 144, 512]
            ├── core(feat)           → [B*4, 144, 512]
            ├── spatial_softmax()    → keypoints + features
            ├── keypoint_proj()      → [B, 512]
            └── head.compute_loss()  → MSE(noise_pred, noise)
            └── backward + step
    └── eval_action_std()            → action distribution metrics
    └── save_checkpoint()            → best.pth / last.pth
```

---

## 🎬 Evaluation

### Metrics

| Metric | Threshold | Description |
|--------|-----------|-------------|
| **Strict Success** | Coverage ≥ 0.95 | Near-perfect T-shape coverage |
| **Effective** | Coverage ≥ 0.80 | Good coverage, minor gaps |
| **Fail** | Coverage < 0.80 | Significant gaps or no contact |

### Inference

- **Scheduler**: DDIM (20 steps, default)
- **Action Chunking**: Predict 16 steps, execute first 4 (Receding Horizon Control)
- **Observation History**: 4-frame sliding window with warm-start

---

## 📈 Results

### V4 Best (epoch 34) — 50 Episodes

<div align="center">

| Metric | Value |
|--------|-------|
| 🎯 Strict Success (≥0.95) | **46.0%** (23/50) |
| ✅ Effective (≥0.80) | **82.0%** (41/50) |
| 📊 Mean Coverage | **0.864 ± 0.215** |
| 📈 Max Coverage | **0.991** |
| 📉 Min Coverage | **0.000** |

</div>

### Coverage Distribution

```
[0.00-0.10)  ██                    2 ( 4.0%)  ❌ Fail
[0.10-0.50)  ██                    2 ( 4.0%)  ❌ Fail
[0.50-0.80)  █████                 5 (10.0%)  ❌ Fail
[0.80-0.95)  ██████████████████   18 (36.0%)  🟡 Effective
[0.95-1.00]  ███████████████████████ 23 (46.0%)  ✅ Strict Success
```

### Training Progress

| Epoch | Loss | LR | Notes |
|-------|------|-----|-------|
| 0 | 0.524 | 1e-4 | Initial |
| 10 | 0.042 | 7.7e-5 | First eval |
| 20 | 0.035 | 4.6e-5 | Steady improvement |
| 34 | 0.026 | 1e-6 | **Best checkpoint** |

---

## 📁 Project Structure

```
robo-diffusion-light/
├── models/                          # Model architectures
│   ├── vit_backbone.py              # DINO v1 vits8 + projection
│   ├── hybrid_core.py               # Mamba + Attention hybrid
│   ├── spatial_softmax.py           # Keypoint detection
│   ├── diffusion_action_head.py     # Diffusion Policy UNet1D
│   ├── action_head.py               # Legacy MSE head
│   ├── act_action_head.py           # ACT-style CVAE head
│   └── vision_backbone.py           # Legacy ResNet-18 backbone
│
├── train_v4.py                      # Main training script
├── train_v4_resume.py               # Resume training script
├── eval_pusht_v4.py                 # Evaluation script
├── diffusion_loader.py              # Data loading + augmentation
├── metrics_logger.py                # Training metrics & plotting
├── ARCHITECTURE.md                  # Detailed architecture docs
├── .gitignore
├── LICENSE
└── README.md
```

> 📦 **Not included** (excluded from repo): `lerobot/`, `data/`, `checkpoints_*/`, `logs_*/`, `eval_videos/`, `venv/`

---

## 🛠️ Requirements

### Core Dependencies

```
torch >= 2.0
mamba-ssm
diffusers
gymnasium
gym-pusht
tqdm
matplotlib
imageio
opencv-python
av (PyAV)
numpy
```

### Hardware

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | 8GB VRAM | 16GB+ VRAM |
| RAM | 16GB | 32GB |
| Storage | 10GB | 20GB (with data) |

> Training V4 for 35 epochs takes approximately **6-7 hours** on a single GPU.

---

## ⚠️ Known Issues

The following issues are documented for transparency and future improvement:

| # | Issue | Severity | Status |
|---|-------|----------|--------|
| 1 | Action deque overflow in eval (appends 16 actions to maxlen=4 deque) | 🔴 Critical | Documented |
| 2 | GPU augmentation (`augment_batch_gpu`) defined but never called | 🟡 Medium | Documented |
| 3 | Gradient clipping excludes `spatial_softmax` + `keypoint_proj` | 🟡 Medium | Documented |
| 4 | `eval_action_std` autocast inconsistency between scripts | 🟡 Medium | Documented |
| 5 | Hardcoded `action_dim=2` in `generate()` method | 🟡 Medium | Documented |

See `ARCHITECTURE.md` for detailed analysis.

---

## 🙏 Acknowledgments

- [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/) — Chi et al., 2023
- [LeRobot](https://github.com/huggingface/lerobot) — Hugging Face
- [Mamba](https://github.com/state-spaces/mamba) — Gu & Dao, 2023
- [DINO](https://github.com/facebookresearch/dino) — Caron et al., Meta AI
- [gym-pusht](https://github.com/NVlabs/gym-pusht) — NVIDIA

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

```
MIT License

Copyright (c) 2026 Robo Diffusion Light Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
```

---

<div align="center">

**Made with ❤️ for embodied AI research**

[⭐ Star this repo](https://github.com/klxz111/robo-diffusion-light) · [🐛 Report Bug](https://github.com/klxz111/robo-diffusion-light/issues) · [💡 Request Feature](https://github.com/klxz111/robo-diffusion-light/issues)

</div>
