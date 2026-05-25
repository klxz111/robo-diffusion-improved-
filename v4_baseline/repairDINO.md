# CLS Token 修复方案

## 问题

SpatialSoftmax 要求输入 token 数 `N == grid_h × grid_w`（144 = 12×12），
因为内部会将 tokens reshape 回 2D 网格：

```python
x_2d = x.permute(0, 2, 1).view(B, D, self.grid_h, self.grid_w)
```

DINO 输出 145 tokens = 144 patch + 1 CLS。带 CLS 时 N=145，reshape 会 crash。
所以当初的做法是直接 `[:, 1:, :]` 切掉 CLS，但这样丢弃了 DINO 最关键的全局语义特征。

## 方案：拆两路，到 global_cond 汇合

```
ViT → [B, 145, D]
       ├─ patches [B, 144, D] → SpatialSoftmax (原封不动) → kp + feat → feat_compress
       └─ CLS     [B, D]      → cls_proj (新加)
                                                 ↓
                              每帧: [16×(2+128) + 128] = 2208
                              2帧: 4416 → proj(4416, 1024) → global_cond
```

## 实施状态 ✅ 2026-05-21

### 1. `models/vit_backbone.py` — forward 返回 (patch_tokens, cls_token)

```python
layers = [outputs[i] for i in self.fuse_layers]          # 保留 CLS
features = torch.cat(layers, dim=-1)                      # [B, 145, 1152]
out = self.proj(features)                                 # [B, 145, 512]
patch_tokens = out[:, 1:, :]                              # [B, 144, 512]
cls_token = out[:, 0, :]                                  # [B, 512]
return patch_tokens, cls_token
```

### 2. `models/spatial_softmax.py` — 无需改动

### 3. `train.py` / `test.py`

**模型构建：**
```python
feat_compress = nn.Linear(512, 128)
cls_proj = nn.Linear(512, 128)                            # NEW
per_frame_dim = 16 * (2 + 128) + 128                      # = 2208
proj = nn.Linear(N_OBS_STEPS * per_frame_dim, 1024)       # 4416→1024
keypoint_proj = nn.ModuleDict({
    "feat_compress": feat_compress,
    "cls_proj": cls_proj,
    "proj": proj,
})
```

**前向路径（训练循环 / eval_action_std / eval_on_gym / select_action 共 4 处）：**
```python
patch_feat, cls_feat = vision(img)                        # [B*N, 144, 512], [B*N, 512]
feat = patch_feat.view(B, N_OBS_STEPS * 144, -1)          # [B, 288, 512]
out = core(feat)
out = out.view(B * N_OBS_STEPS, 144, out.shape[-1])
keypoints, kp_features = spatial_softmax(out)
keypoints = keypoints.view(B, N_OBS_STEPS, -1, 2)
kp_features = kp_features.view(B, N_OBS_STEPS, -1, -1)

kp_c = feat_compress(kp_features)                         # [B, N, 16, 128]
combined = torch.cat([keypoints, kp_c], dim=-1)            # [B, N, 16, 130]
combined = combined.view(B, N_OBS_STEPS, -1)               # [B, N, 2080]
cls_c = cls_proj(cls_feat).view(B, N_OBS_STEPS, -1)       # [B, N, 128]
full = torch.cat([combined, cls_c], dim=-1)                # [B, N, 2208]
global_cond = proj(full.flatten(1))                        # [B, 4416]→[B, 1024]
```

### 4. proj 输入维

```
旧: N_OBS_STEPS * 16 * (2 + 128)          = 4160
新: N_OBS_STEPS * (16 * (2 + 128) + 128)   = 4416
```

### 5. Checkpoint 不兼容

- ViTBackbone forward 签名改变（1→2 返回值）
- keypoint_proj 新增 `cls_proj` 参数
- `proj` 输入维 4160→4416

**必须从零训练。** checkpoints_phase3 已清空。

## 副作用

- 每帧额外参数量：`cls_proj` = 512×128 = **65K**（可忽略）
- 前向基本无额外开销（CLS 本就在 DINO 输出里）
- SpatialSoftmax 不动，原有一切逻辑不变
- 仅 pipeline 中多一个 `torch.cat` + 一次小 linear

---

## 后续优化方案分析（2026-05-22 追加）

### 核心问题

无论 token 长度（144/288/576）、bottleneck 宽度（32/128）、梯度裁剪策略、数据增强强度如何组合，全部实验的严格成功率卡在 ~15%，大量 episode 积压在 0.80-0.95 coverage。

**根因：优化目标与评估指标脱钩。**

模型优化的是噪声预测的 MSE，而 PushT 的成功取决于能否在长尾区域发力（X≥500 仅 4.1%）。Diffusion Policy 天然倾向"安全均值"输出，不会自发产生激进推动动作。

### 建议优化路线

#### 立即见效（不改训练，只改推理代码）

| # | 方案 | 改动量 | 预期收益 | 说明 |
|:-:|:----|:------:|:--------:|:----|
| 1 | **DDIM eta > 0** | ~1 行 | +1~5% SR | 推理时加少量随机性（eta=0.1~0.3），让动作偶尔突破均值瓶颈。确定性采样（eta=0）永远输出"最安全"的中间动作 |
| 2 | **Action Repetition** | ~5 行 | +1~3% SR | 同一个动作重复执行 2-3 帧，让持续推动积累 coverage |
| 3 | **Checkpoint Averaging** | 独立脚本 | +1~3% SR | 多个 checkpoint 权重平均（model soup），比单点更稳定 |

#### 需要一次完整训练

| # | 方案 | 改动量 | 预期收益 | 说明 |
|:-:|:----|:------:|:--------:|:----|
| 4 | **CFG (Classifier-Free Guidance)** | ~30 行 | **+5~15% SR** ⭐ | 训练时 10% 概率 dropout 条件 cond；推理时双路 forward 放大条件强度。Diffusion Policy 论文标准技巧 |
| 5 | **动作加权采样（长尾过采样）** | ~20 行 | +3~8% SR | 高 X 动作的样本给更高权重，让模型多看"推到边缘"的轨迹。`diffusion_loader.py` 加 `WeightedRandomSampler` |
| 6 | **Loss 分步加权修复** | ~3 行 | +2~5% SR | `diffusion_action_head.py` docstring 声称前 N_ACTION_STEPS 加权但实际未实现。当前所有 16 步 loss 均等，但后 8 步从未执行 |

#### 进阶方向

| # | 方案 | 说明 |
|:-:|:----|:-----|
| 7 | **各向异性 loss（X 方向 2×）** | 3 行代码，针对 PushT 的 X 轴推动特性 |
| 8 | **DINO 部分解冻（最后 2-4 个 block）** | 让视觉 backbone 适应任务，但显存/训练时间增加 |
| 9 | **MinSNR 加权** | 改进 diffusion loss 的信噪比加权策略 |

### `.view()` → `.reshape()` 修复

**问题**：`ViTBackbone.forward` 返回 `patch_tokens = out[:, 1:, :]`（切片视图，non-contiguous），
下游代码对其调 `.view()` 时报错：

> view size is not compatible with input tensor's size and stride

**修复**：所有 `patch_feat.view(...)` 替换为 `patch_feat.reshape(...)`。
`.reshape()` 遇到 non-contiguous tensor 时自动做拷贝，不会报 stride 不兼容。

涉及文件：`train.py`（训练循环 + eval_action_std + eval_on_gym 共 3 处）、`test.py`（select_action 1 处）。
