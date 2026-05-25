# Phase 7 模型总结

## 架构

```
ViTBackbone: DINO v1 vits8 (frozen, ~21M)
  fuse_layers=[4,7,11], embed_dim=512
  → patch_tokens [B*2, 144, 512] + cls_token [B*2, 512]

Core: ToolinitModel ×3 SerialHybridBlock (MHA + Mamba×4) — 未动
  RMSNorm → MHA → +x → RMSNorm → Mamba×4 → +x
  Gradient checkpointing enabled
  → [B, 288, 512]

SpatialSoftmax: K=16, grid=12×12
  pre_conv: Conv2d(512,512,3×3,padding=1)
  keypoint_conv: Conv2d(512,16,3×3,padding=1)
  → keypoints [B*2, 16, 2] + features [B*2, 16, 512]

feat_compress: Linear(512, 128) + Dropout(0.15)
cls_proj: Linear(512, 128) + Dropout(0.15)
per_frame: kp[16,2] + feat[16,128] + cls[128] + state[2] = 2210
proj: Linear(4420, 1024) + Dropout(0.15)  ← + state 维度
→ global_cond [B, 1024]

UNet: ConditionalUnet1d, cond_dim=1024, down_dims=(128,256,512), kernel=3
  FiLM conditioning, GroupNorm, Mish activation
  Gradient checkpointing enabled
  DDIM 20 步 → action [B, 8, 2]
```

### 参数量

| 模块 | 参数 | 可训练 | Phase 6 vs 7 |
|:----|:----:|:------:|:------------:|
| DINO v1 vits8 | 21.7M | 0 (frozen) | = |
| vision.proj | 0.6M | 0.6M | = |
| core (SerialHybrid ×3) | 25.3M | 25.3M | = |
| spatial_softmax (Conv2×) | ~1.7M | ~1.7M | = |
| keypoint_proj | **~4.5M** | **~4.5M** | ↑ feat=128, proj=1024 |
| head (ConditionalUNet1d) | **~25M** | **~25M** | ↑ cond=1024 |
| **总计** | **~79M** | **~57M** | |

## 超参

| 参数 | 值 |
|:----|:----|
| BATCH_SIZE | 48 |
| ACCUM_STEPS | 2 (effective BS = 96) |
| EPOCHS | 50 |
| LR | 1e-4, CosineAnnealing → 1e-6, warmup=1 epoch |
| WEIGHT_DECAY | 1e-4 |
| HORIZON | 16 |
| N_OBS_STEPS | 2 |
| N_ACTION_STEPS | 8 |
| GRAD_CLIP | 1.0 |
| 梯度检查点 | ✅ (core + head) |
| 增强 | trans=4, rot=0, noise=0.03, prob=0.5 + **动作裁剪** |
| 数据划分 | train 80% / val 20% |
| Gym Eval | 递增: 10→5, 15→10, ... |
| Dropout | 0.15 (keypoint_proj) |
| **新增** | **observation.state (2D agent_pos) 拼入 per_frame** |

## 参数量轨迹

185M(Phase 4) → 100M → 69M → 51M → 34M → **57M(Phase 7)**

## 历史修复

| # | 问题 | 修复 |
|:-:|:----|:----|
| 1 | 增强平移系数减半 + 动作方向反 | `tx/W→2tx/W`, `+=→-=` |
| 2 | 增强旋转图像/动作方向不一致 | `R(θ)→R(-θ)`, 中心 `48.0→47.5` |
| 3 | Head 过载 | `down_dims=(192,384,768), kernel=3` (130M→45M) |
| 4 | weight_decay=0 | `1e-4` |
| 5 | 无验证集 | `val_split=0.2` |
| 6 | Gym Eval 固定值 | 递增式 |
| 7 | 动作越界 | 增强后 `actions.clamp(ACT_MIN, ACT_MAX)` |
| 8 | 缺 observation.state | 加载 agent_pos 拼入 per_frame |

## 已确认的事实

1. **SR 天花板 ~14%** — 在所有 Phase 中一致，独立于视觉特征和 bottleneck 宽度
2. **0.80-0.95 积压** — 76% episode 卡在这个区间
3. **过拟合已被控制** — 合理增强 + 动作裁剪 + state 直连后预期显著改善

## 未尝试的方向

| 方向 | 预期收益 | 改动量 | 依赖 |
|:----|:--------:|:------:|:----|
| CFG (Classifier-Free Guidance) | **+5~15% SR** ⭐ | ~30 行 | Phase 7 baseline 后 |
| Loss 分步加权（前8步加权） | +2~5% SR | ~3 行 | 独立 |
| 各向异性 loss（X 权重×2） | +1~3% SR | ~3 行 | 独立 |
| 动作加权采样（高 X 过采样） | +3~8% SR | ~20 行 | 独立 |
| DDIM eta > 0（推理随机性） | +1~5% SR | 1 行 | 独立 |

## 关键文件

| 文件 | 说明 |
|:----|:-----|
| `train.py` | 训练入口，EPOCHS=50，含 val_loss + 递增 Gym Eval |
| `test.py` | 50 轮评估 |
| `diffusion_loader.py` | 数据加载 + GPU 增强（已修复 Bug） |
| `models/vit_backbone.py` | ViTBackbone (DINO v1 vits8 frozen) |
| `models/hybrid_core.py` | ToolinitModel ×3 SerialHybridBlock |
| `models/spatial_softmax.py` | SpatialSoftmax K=16, Conv3×3 |
| `models/diffusion_action_head.py` | DiffusionActionHead + ConditionalUNet1d |
