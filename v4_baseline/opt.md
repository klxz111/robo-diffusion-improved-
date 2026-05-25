# 优化方向记录

下个 Phase 待尝试的优化方向，按预期收益排序。

---

## CFG (Classifier-Free Guidance)

- **预期收益**: **+5~15% SR** ⭐
- **改动量**: ~30 行
- **位置**: `models/diffusion_action_head.py`
- **方案**:
  1. 训练时 `compute_loss` 以 10% 概率将 `global_cond` 置零（dropout condition）
  2. 推理时 `generate` 双路 forward：`cond` 和 `zero_cond`，做 extrapolation
  3. `noise_pred = noise_uncond + scale * (noise_cond - noise_uncond)`
- **独立变量**: ✅ 不耦合其他改动，可随时插拔测试
- **参考**: Diffusion Policy 论文 Section 3.5

---

## Loss 分步加权 + 各向异性加权

- **预期收益**: +3~8% SR
- **改动量**: ~5 行
- **位置**: `models/diffusion_action_head.py:compute_loss`
- **方案**:
  ```python
  step_weights = torch.ones(H, device=eps.device)
  step_weights[:N_ACTION_STEPS] = 2.0           # 前 8 步加倍
  dim_weights = torch.tensor([2.0, 1.0], device=eps.device)  # X 轴 2×
  loss = (pred - eps).pow(2) * step_weights[None, :, None] * dim_weights[None, None, :]
  loss = loss.mean()
  ```
- **独立变量**: ✅

---

## Frame 位置编码

- **预期收益**: +2~5% SR
- **改动量**: ~5 行
- **位置**: `train.py`，core 输入前
- **方案**: 在 `patch_feat.reshape(B, N_OBS_STEPS*144, -1)` 前为每帧 token 加 learnable embedding
  ```python
  self.frame_emb = nn.Embedding(N_OBS_STEPS, 512)
  # 使用: patch_feat = patch_feat + frame_emb(token)  (per frame per token)
  ```
- **不改模块**: ✅ 只在连接处加
- **背景**: 当前两帧 token 拼成 `[B, 288, 512]` 后，MHA 和 Mamba 无法区分帧来源

---

## 动作加权采样

- **预期收益**: +3~8% SR
- **改动量**: ~20 行
- **位置**: `diffusion_loader.py:PreloadedPushTDataset`
- **方案**: 按动作 X 坐标 >450 占比加权采样，让高 X 动作被更多抽到
  ```python
  weights = compute_x_weight(actions)  # X>450 → 高权重
  sampler = WeightedRandomSampler(weights, ...)
  ```
- **独立变量**: ✅

---

## SpatialSoftmax 可学习温度

- **预期收益**: +1~3% SR
- **改动量**: 3 行
- **位置**: `models/spatial_softmax.py`
- **方案**:
  ```python
  self.log_temperature = nn.Parameter(torch.zeros(1))
  attention = F.softmax(attention / self.log_temperature.exp(), dim=-1)
  ```
- **注意**: 需改模块内部，但很小

---

## DDIM eta > 0

- **预期收益**: +1~5% SR
- **改动量**: 1 行
- **位置**: `models/diffusion_action_head.py:268`
- **方案**: `DDIMScheduler(..., eta=0.1)`
- **依赖**: 先训好 baseline，推理时单独调参对比
- **风险**: eta 过大会抖动，建议先试 0.1

---

## Checkpoint Averaging (EMA)

- **预期收益**: +1~3% SR
- **改动量**: 独立脚本
- **方案**: 多个 best.pth checkpoint 的权重做算术平均，用平均权重推理
- **独立变量**: ✅

---

## 过拟合防护（可选项）

- **预期收益**: 间接提升泛化
- **方案**:
  - `keypoint_proj` 各层后加 `nn.Dropout(0.15)`
  - 数据增强概率 `AUG_PROB` 0.8 → 0.95
- **优先级**: 低，Head 缩减 3× + weight_decay 已缓解

---

## Online EMA（训练中滑动平均）

- **预期收益**: +3~8% SR
- **改动量**: ~20 行
- **位置**: `train.py`
- **方案**: 训练过程中同步维护一份滑动平均权重，推理/保存 best 时用 EMA 权重
  ```python
  ema_params = decay * ema_params + (1 - decay) * params  # decay ≈ 0.999
  ```
- **不触及任何模块**: ✅
- **参考**: DDPM (Ho et al. 2020)，Diffusion Policy 官方实现
- **不同于 opt.md 中的 checkpoint averaging**: EMA 是在线更新的，比离线多 checkpoint 平均更稳定

---

## 帧一致增强（Frame-Consistent Augmentation）

- **预期收益**: +1~3% SR
- **改动量**: ~5 行
- **位置**: `diffusion_loader.py:augment_batch_gpu`
- **方案**: 当前每帧独立采样平移/旋转参数，改为 batch 内所有帧共享同一组参数
  ```python
  # 当前: tx = rand(N)  # N = B * n_obs
  # 改为: tx = rand(B).repeat_interleave(n_obs)
  ```
- **背景**: 真实相机抖动是帧间一致的。独立增强让模型学到"帧间差异"这种虚假信号

---

## 时序集成（Temporal Ensemble）

- **预期收益**: +2~5% SR
- **改动量**: ~15 行
- **位置**: `train.py:eval_on_gym` / `test.py`
- **方案**: 相邻两次推理对新旧重叠步的动作做加权平均
  ```python
  # 前一次生成了 [a1,a2,...,a8]，执行了 a1
  # 下一次生成 [b1,b2,...,b8]，将 b1..b7 与剩余旧预测平均
  for i in range(min(len(action_queue), chunk_len)):
      chunk[i] = (chunk[i] + action_queue[i]) * 0.5
  ```
- **参考**: Diffusion Policy 论文 Section 3.3 "Ensemble"
- **针对性**: 直接对抗 0.80-0.95 覆盖率 plateau 的最后抖动问题

---

## 前一步动作条件（Action Conditioning）

- **预期收益**: +3~8% SR
- **改动量**: ~20 行
- **位置**: `train.py` + `models/diffusion_action_head.py`
- **方案**: 将上一步执行的动作（归一化后）拼接到 `global_cond` 末尾
  - cond_dim: 1024 → 1026（参数量可忽略）
  - 训练时需要从数据集对齐前一步动作
- **效果**: 模型获得闭环行为，知道"我刚刚做了什么"
- **风险**: 需注意推理时与训练时的一致性

---

## Position-Aware Keypoint 特征

- **预期收益**: +1~3% SR
- **改动量**: ~3 行
- **位置**: `train.py` forward 路径
- **方案**: 当前 keypoint 坐标和特征是分开压缩再拼接的。改为坐标先拼到特征上再压缩：
  ```
  # 当前: feat_compress(kp_features) → cat with keypoints
  # 改为: cat(keypoints, kp_features) → feat_compress(514→128)
  ```
  `Linear(512, 128)` → `Linear(514, 128)`，参数量 +256
- **效果**: 压缩过程感知每个 keypoint 的空间位置

---

## global_cond LayerNorm 归一化

- **预期收益**: +1~2% SR（训练稳定化）
- **改动量**: 1 行
- **位置**: `train.py:proj` 末尾
- **方案**: `nn.Sequential(Linear(4416, 1024), LayerNorm(1024), Dropout(0.15))`
- **背景**: proj 输出直接进 UNet FiLM 做 scale/bias，分布漂移会干扰 FiLM 稳定性

---

---
## Phase 8 — 容量缩减 + augmentation 调优 (2026-05-23)

### 改动
- **瓶颈**: feat_compress/cls_proj 128 → **64**, cond_dim 1024 → **512**
- **UNet down_dims**: (128,256,512) → **(96,192,384)**
- **Core d_model**: 保留 512 (3层) — 核心推理能力不动
- **Trainable**: 57.7M → **41.8M** (参数/帧: 2,810 → 2,040)
- **Augmentation**: trans=8, rot=5, noise=0.06, prob=0.5 (保持 Phase 7 末)

### 当前结果 (epoch 10/36)
| 指标 | 值 |
|------|----|
| Train Loss | 0.0516 |
| Val Loss | 0.0495 |
| Coverage | 0.267 |
| SR | 0% |
| val>train 次数 | 3 次 (epoch 5,7,9)，每次差<0.003，隔 epoch 自愈 |

### 分析
- 41.8M 处于过拟合敏感区间的下沿 (2,040 参数/帧)，val>train 压力显著缓解
- Coverage 0.267 < Phase 7 同期 0.307，但 val loss 仍在下降

### 实际结果 (epoch 15)
- **Coverage**: 0.267 (ep10) → **0.439** (ep15) ↑，但 SR 仍为 **0%**
- **Val Loss**: 自 epoch 11 的 0.04444 后连续 4 epoch 未更新
- **Val>Train**: epoch 13~15 复发，epoch 15 差距 Δ=0.00814（历史最大）
- **pred_std**: 78.48 (X) / 90.57 (Y)，全面低于 real_std (92.16/102.75) — 模型预测方差不匹配真实分布
- **结论**: 缩减到 41.8M 只是推迟过拟合约 3 epoch，未根本解决

---

## Phase 9 — Core d_model 缩减 512→384 (2026-05-23)

### 改动
- **Core d_model**: 512 → **384**（ViT、SS、keypoint_proj、cond_dim 联动调整）
- **Mamba expand**: 2 → **1**（内部 FFN 砍半）
- **Mamba d_state**: 64 → **32**（SSM 状态压缩）
- **改动量**: ~50 处，均为维度/参数常量替换
- **预估参数量**: 41.8M → ~**26M**

### 文件变更
| 文件 | 改动 |
|------|------|
| `models/hybrid_core.py` | d_model 512→384, d_state 64→32, expand=2→1 |
| `models/vit_backbone.py` | embed_dim 512→384 |
| `models/spatial_softmax.py` | token_dim 512→384 |
| `models/diffusion_action_head.py` | 默认 cond_dim→384, down_dims→(96,192,384) |
| `train.py` | 所有 512→384 维度 + 注释修正 |
| `test.py` | 所有 512→384 维度 + 注释修正 |

### 预期
- 参数/帧: ~2,040 → ~**1,270**
- 3 层 × 4 Mamba 共 12 个，每个从 ~645K → ~326K，净减 ~3.8M

### 实际结果 (epoch 20)
| 指标 | 值 |
|------|-----|
| Coverage@ep10 | 0.411 |
| Coverage@ep15 | 0.696 |
| Coverage@ep20 | 0.700 |
| SR | 0% (0/15 @ep20) |
| Best Val Loss | 0.0412 (ep18) |
| Epoch Time | ~122s (Phase 8 的 3×快) |
| Val>Train | 偶发但自愈，未发散 |
| pred_std | 匹配真实分布 |
| **结论** | ~26M 为容量甜区，Coverage 天花板 ~0.70，首次无过拟合 |

---

## Phase 10 — 冗余剪枝: Mamba 4→2 + 移除 pre_conv (2026-05-23)

### 改动
- **Mamba 每层**: 4个 → **2个**（12→6 Mamba，净减 ~3.3M）
- **SpatialSoftmax**: 删除 `pre_conv`（~1.3M，仅增强层，非核心）
- **预估参数量**: ~26M → ~**21.5M**

### 实际结果
| 指标 | ep10 | ep15 | ep20 | ep25 | ep30 |
|------|------|------|------|------|------|
| Coverage | 0.158 | 0.754 | 0.593 | **0.849** | 0.790 |
| SR | 0% | 10% | 0% | **20%** | 12% |
| Best Val Loss | 0.0471 | 0.0430 | 0.0412 | 0.0385 | **0.0333** |
| Epoch Time | ~98s | ~98s | ~97s | ~96s | ~96s |

### 结论
- 21.5M 为可训练的最小容量。Coverage 在 0.79~0.85 震荡收敛
- SR 天花板约 12~20%，单靠容量缩减和 DDIM 无法突破
- 训练速度快 (~96s/epoch)，60 epoch 计划合适

---

## Phase 11 — CFG + Temporal Ensemble + DDIM eta (2026-05-23)

### 改动

| Trick | 文件 | 说明 |
|-------|------|------|
| **CFG** | `diffusion_action_head.py` | 训练 10% condition dropout；推理双路 forward + guidance_scale |
| **DDIM eta** | `diffusion_action_head.py:generate` | eta=0.1~0.3，增加随机性 |
| **Temporal Ensemble** | `test.py`, `train.py` | 队列过半预生成，新旧动作等权平均 |

### 实际结果

```
CFG guidance_scale=2.0 ep25: Cov 0.408, SR  0%
CFG guidance_scale=3.0 ep25: Cov 0.432, SR 12%
CFG guidance_scale=4.0 ep30: Cov 0.326, SR  0%
```

### 结论

CFG 与 Temporal Ensemble 互相矛盾。CFG 用于放大条件差异，Ensemble 用于拉平动作，两者对抗导致 Coverage 上不去。**已撤销 CFG 和 Ensemble**，保留 DDIM eta=0.3。

---

## Phase 12 — 五项训练侧优化 (2026-05-23)

### 改动

| # | 优化 | 文件 | 行数 | 预期增益 |
|---|------|------|------|---------|
| 1 | **Frame 位置编码** | `train.py`(×4), `test.py` | ~15行 | +2~5% SR |
| 2 | **SS 可学习温度** | `spatial_softmax.py` | 3行 | +1~3% SR |
| 3 | **Loss 分步+各向异性加权** | `diffusion_action_head.py` | 5行 | +3~8% SR |
| 4 | **帧一致增强** | `diffusion_loader.py` | 8行 | +1~3% SR |
| 5 | **Online EMA (decay=0.999)** | `train.py` | ~30行 | +3~8% SR |

### 推理配置
- **DDIM eta=0.3** — 提高随机性，鼓励探索
- **无 CFG** — 已撤回
- **无 Temporal Ensemble** — 已撤回

### 架构
- Phase 10 基础: 21.5M, Mamba×2, expand=1, d_state=32
- 五项优化全部是训练侧，一次性实施、一次性重训

### 实际结果
| 指标 | ep10 | ep15 | ep20 | ep25 | ep30 | ep35 | ep40 | ep45 | ep55 |
|------|------|------|------|------|------|------|------|------|------|
| Coverage | 0.488 | 0.542 | 0.701 | 0.805 | 0.798 | 0.745 | 0.855 | 0.895 | 0.858 |
| SR | 0% | 0% | 6.7% | 20% | 0% | 20% | 11.4% | 20% | **12.0%** |
| Val Loss | 0.118 | 0.097 | 0.081 | 0.081 | 0.069 | 0.068 | 0.068 | 0.060 | 0.057 |
| Train = Val | +0.006 | -0.003 | -0.008 | -0.004 | -0.008 | -0.005 | 0.000 | -0.006 | -0.005 |

### 结论
- 全 Phase 最高 Coverage (0.895)，**50 episode 终局 SR=12%**
- 五项训练优化效果明确，但 21.5M + 25k 帧的容量天花板就是 12% SR
- 数据量和模型容量存在双重瓶颈

---

## Phase 状态汇总

```
Phase 5  ── 已实施: K=16, bottleneck缩减, Dropout, AUG_PROB=0.95, weight_decay
            结果: 69M → 49M，过拟合 epoch 10+

Phase 6  ── 已实施: observation.state (agent_pos) 输入, bottleneck再增 (49→57M)
            结果: 57M 双倍增强 epoch 10 覆盖 0.307, val>train 反复出现

Phase 7  ── 已实施: feat/cls 64, cond 512, down (96,192,384), 41.8M
            结果: 覆盖 0.439 (ep15) 但 SR 0%，val>train 复发

Phase 8  ── 已实施: d_model 512→384, expand 2→1, d_state 64→32, ~26M
            结果: 容量甜区，Coverage 0.70, 无过拟合, 但 SR 0%

Phase 9  ── 已实施: Mamba 4→2, 删除 pre_conv, ~21.5M
            结果: Coverage 0.79~0.85, SR 12~20%, 达容量极限

Phase 10 ─ 已实施: CFG + Temporal Ensemble 实验（已撤回）
            结论: CFG 与 Ensemble 矛盾，退回干净训练 + eta=0.3

Phase 11 ─ 已实施: 位置编码 + SS温度 + Loss加权 + 帧一致增强 + EMA (P12 实际运行)
            结果: Coverage 0.488→0.895, 终局 50ep SR=12%, 全 Phase 最健康

Phase 12 ─ 待实施: 删Dropout + State mean/std + Action delta + Batch 64 + Epoch 200
            状态: 代码已改完, 待训练 (约 7.5h / 200 epoch)

待探索:
  ── ▶ Action Conditioning                                       +3~8% SR
  ── ▶ Checkpoint Averaging / EMA offline                        +1~3% SR

---

## Phase 13 — 对齐官方配置 + Action delta + 数据扩展

### 超参改动

| 项目 | 当前 | 改为 | 原因 |
|------|------|------|------|
| **Dropout** | feat/cls/proj 各 0.15 | **全部删除** | 官方验证 dropout 在 PushT 上掉点 |
| **State norm** | min-max 到 [-1,1] | **mean/std** | 官方使用数据集统计量归一化 |
| **Action** | 绝对坐标 `[12,511]` | **delta** | 相对位移，解耦动作与位置 |
| **Batch size** | 48 | **64** | 官方配置，梯度更稳定 |
| **Epochs** | 60 | **200** | P12 到 ep49 仍在刷新 best val |
| **LR** | 2e-4 | 2e-4 (保留) | P12 曲线健康 |

### 数据可用资源

| 数据集 | 帧数 | 内容 | 状态 |
|-------|------|------|------|
| `raw_pusht/` | 25,650 | 图像+动作 (当前在用) | ✅ 已接入 |
| `raw_pusht_keypoints/` | 25,650 | 同上, 附加 16-dim 关键点标注 | 📝 待接入(co-train) |
| `data-letter/domain0` (T块) | 57,543 | 同 PushT, 零迁移风险 | 📝 待接入(数据扩充) |
| `data-letter/domain7` (T块) | 68,835 | T 块金色+误导 | 📝 评估后接入 |
| `data-letter/domain18` (T块) | 55,480 | T 块深红色 | 📝 评估后接入 |
| `data-letter/domain13/15/20/21/22` | 221,661 | H/V/R/O/B 形状 | 📝 暂不考虑 |

### 预计改动量
- 总计 ~55 行
- 核心模型代码不动，全在数据层和训练配置层
- 一次性改完、一次性重训

### 已实施改动

| # | 改动 | 文件 | 行数 | 状态 |
|---|------|------|------|------|
| 1 | **Batch 48→64** | train.py | 1 | ✅ |
| 2 | **Epoch 60→200** | train.py (余弦 T_max 联动) | 1 | ✅ |
| 3 | **关掉 early stopping** (PATIENCE=200) | train.py | 1 | ✅ |
| 4 | **关掉训练时 GymEval** (INTERVAL=9999) | train.py | 1 | ✅ |
| 5 | **删除 Dropout** (feat_compress/cls_proj/proj) | train.py | 3 | ✅ |
| 6 | **State mean/std 归一化** | diffusion_loader.py + train.py | ~10 | ✅ |
| 7 | **Action delta** (delta = action - current_pos) | diffusion_loader.py, train.py(4处), test.py | ~35 | ✅ |
| 8 | **每 10 epoch 保存 checkpoint** 用于事后批量 eval | train.py | 5 | ✅ |

### 训练策略
- 不早停，跑满 200 epoch
- 每 10 epoch 保存 snapshot (`epoch_010.pth`, `epoch_020.pth`, ...)
- 训练完后用 test.py 批量评估所有 snapshot，挑最佳 SR
- 后续可以用 best.pth 做 RL fine-tune roll out 或 data-letter 混合训练
```
