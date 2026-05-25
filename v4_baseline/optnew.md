# 新阶段优化记录 (Phase 14~17)

## 当前架构配置 (Phase 17)

```
模型总参:   ~26.6M（UNet 21 ResBlocks, +1.5M via depth 2→3）
Batch:      64
Steps:      200,000 (step-based)
LR:         2e-4 (余弦退火到 1e-6, step-based LambdaLR, warmup 2k)
Optimizer:  AdamW, betas=(0.9,0.95), weight_decay=1e-4, clip_grad_norm=1.0
AMP:        BF16, ACCUM_STEPS=4 (effective 256), GradScaler 已移除
Eval:       每 10,000 steps, GymEval (20 次完整评估)
Save:       每 10,000 steps (step_XXXXXX.pth) + best.pth

ToolinitModel: d_model=384, 3 层, Mamba2×2/层, expand=2, d_state=32, headdim=32
UNet:          down_dims=(128,256,512), 21 ResBlocks (depth 2→3), RMSNorm1d + GELU
Backbone:      DINO v1 vits8 (frozen), embed_dim=384
SpatialSoftmax: K=16, token_dim=384, 可学习温度, 无 pre_conv
KeypointProj:  feat_compress(384→64) + cls_proj(384→64) + proj(2244→384)

State:      mean/std 归一化 (预计算)
Action:     delta (action - current_pos), mean/std 归一化
Dropout:    无
EMA:        Online EMA, decay=0.999
Attention:  FlashAttention-2（取代 nn.MultiheadAttention）
Position:   无 Frame 位置编码（已移除）
Aug:        帧一致增强 (trans=8, rot=5, noise=0.06, prob=0.5), 无颜色抖动
Loss:       uniform MSE（已移除 X 轴/前 8 步加权）
Inference:  DPMSolverMultistepScheduler, solver_order=2, use_karras_sigmas=True, 8 步（默认）
```

### 吞吐优化

```
CUDA:       TF32 matmul precision = high, cudnn benchmark = True
Checkpoint: 关闭 (+~30% 吞吐)
GradScaler: 移除（BF16 不需要）
DataLoader: workers=0（纯 RAM，零 IPC 开销）
Compile:    torch.compile(head.unet)
GridSample: 平移+旋转合并为 1 次, 全程 half 精度
Log:        每 50 步记录一次（原每步）
Core:       Mamba-1→Mamba-2（~2-3× scan), FlashAttention-2（~2× attention）
UNet:       GroupNorm→RMSNorm1d（~1.5×), Mish→GELU（~1.3×)
训练吞吐:    ~4.5 it/s (batch=64, accum=4)
```

### 当前数据集

```
raw_pusht/          25,650 帧 (float16 RAM, ~1.3 GB)
domain0/T           57,543 帧 (float16 RAM, ~3.0 GB)
────────────────────────────
合计                83,193 帧 (3.3× pusht)
RAM 占用:           ~4.4 GB / 16 GB
GPU VRAM:           ~3.8 GB / 8 GB (batch=64)
```

### 参数/帧 对比

```
Phase        参数量      帧数      参数/帧
────────────────────────────────────────
P10~12       21.5M     25,650         860   ← 此前甜区
P14          27.0M     25,650       1,052   ← UNet 扩通道
P15          ~12.5M   207,508         ~60   ← 严重欠拟合
P16          ~19.0M    83,193         228   ← 容量升级+仅 domain0
P17 (当前)    ~26.6M   83,193         320   ← ResBlock 加深
```

---

## Phase 14 — UNet 容量扩展 (已撤回)

### 改动
- `down_dims=(96,192,384)` → `(128,256,512)`，UNet 参数量 ~8M→~14M

### 撤回理由
扩通道后 pred_std 卡在 real_std 的 50%，无改善。原因：25k 帧数据下 27M 偏大，模型倾向于保守解。

### 结论
UNet 容量不是 Phase 10~12 的瓶颈。数据才是。

---

## Phase 15 — data-letter 集成 + Step-based 控制台 (已执行)

### 改动

| 模块 | 改动 |
|------|------|
| `diffusion_loader.py` | raw_pusht 预加载 (2.7GB RAM)，data-letter Zarr 按需读取 |
| `diffusion_loader.py` | 统计量只从 raw_pusht 计算（分布一致，避免 Zarr 随机读慢） |
| `train.py` | epoch-based → step-based: STEPS=32k, EVAL_FREQ=4k |
| `train.py` | LR: CosineAnnealingLR → step-based LambdaLR |
| `train.py` | GymEval 重新启用，episode 数递增加载 |
| `train.py` | 删除 unused imports (LinearLR, CosineAnnealingLR, SequentialLR) |
| `test.py` | UNet down_dims 同步回退 |

### 训练控制台

| 参数 | 值 |
|------|-----|
| Total steps | 32,000 |
| Eval freq | 4,000 steps (8 次完整评估) |
| Save freq | 4,000 steps |
| Warmup steps | 500 |
| LR schedule | Cosine, 2e-4→1e-6, step-based LambdaLR |
| Eval ramp | 5→10→15→20→25→30→35→40 episodes |

### 注意事项

- **Zarr v3 兼容**: `zarr.open(path, mode="r")` 而非 `zarr.open(path, "r")`
- **Zarr 按需读取**: `_get_frame(idx)` 根据 idx 判断从 preloaded 还是 Zarr 读取
- **统计量**: state_mean/std 和 delta_mean/std 只从 raw_pusht 计算（~25k 帧足够稳定，分布与 data-letter 一致）

---

## Phase 16 — 模型容量升级 + 吞吐优化 (已执行)

### 背景
P15 的参数/帧仅 ~60，模型严重欠拟合。需要扩容匹配数据量。

### 改动

| 文件 | 改动 | 参数影响 |
|------|------|---------|
| `train.py` | UNet `down_dims=(128,256,512)` | UNet ~14M |
| `test.py` | 同步 `down_dims` | — |
| `models/hybrid_core.py` | Mamba `expand=1→2`（6 层） | +~2M |
| `models/diffusion_action_head.py` | 默认值同步 | — |
| `diffusion_loader.py` | grid_sample 平移+旋转合并为 1 次 | 增广 ~1.6× |
| `diffusion_loader.py` | grid_sample half 精度 | 增广 +~20% |
| `train.py` | TF32 + cuDNN benchmark | 矩阵乘 ~1.3× |
| `train.py` | `train_workers 4→2→0` | 减少 IO/内存 |

### 验证结果 (32k 步训练)

```
Step    Coverage    SR
──────────────────────
 4k     0.099      0%
 8k     0.604      0%
12k     0.574      0%
16k     0.720     15%
20k     0.756      8%
24k     0.724      0%
28k     0.853     14%
32k     0.863     ~%
```

**关键发现**：
1. 模型在 24k→28k 出现二次爆发（val loss -0.022），说明 32k 步仅完成探索期
2. 28k 步后仍在学习，LR 归零后被迫截止，印证训练时间不足
3. 参考官方 25k 数据跑 200k 步、175k 最佳收敛——当前 32k 步远不够
4. 207k 帧数据对 16GB 系统 RAM 压力过大（OOM），回退到 pusht + domain0

---

## Phase 17 — 训练时间扩容 + 去冗余 + 推理升级 (已执行)

### 核心认知转变
此前所有优化围绕**空间维度**（参数/帧比、模型容量、数据量），忽略了**时间维度**（训练步数、LR 退火速度）。训练时间远远不够是根本问题。

### 改动

| 类别 | 改动 | 文件 |
|------|------|------|
| **训练时间** | STEPS 32k → **200k** | `train.py` |
| | WARMUP 500 → 2,000 | `train.py` |
| | EVAL/SAVE_FREQ 4k → 10k | `train.py` |
| **吞吐** | USE_CHECKPOINT False | `train.py` |
| | BATCH_SIZE 64 → **128** | `train.py` |
| | ACCUM_STEPS 2 → 1 | `train.py` |
| | torch.compile(head.unet) | `train.py` |
| **去冗余** | Loss 加权 → uniform MSE | `diffusion_action_head.py` |
| | 删除 frame 位置编码（3 处） | `train.py` + `test.py` |
| **推理** | DDIMScheduler → **DPMSolverMultistepScheduler** | `diffusion_action_head.py` |
| | DDIM 20→10 eval 推理步 | `train.py` |
| | Eval `max_batches` 20→10 | `train.py` |
| **检查点** | 每 eval 保存 `step_XXXXXX.pth` | `train.py` |
| **数据** | 全量 RAM 预加载（pusht float16 + domain0 float16） | `diffusion_loader.py` |
| **架构** | UNet ResBlock 2→3（+7 ResBlocks, +1.5M 参） | `diffusion_action_head.py` |

### VRAM 验证

```
Batch=128, 关 checkpoint, BF16:
  Peak VRAM:  5.3 GB / 8.0 GB
  Headroom:   2.8 GB  ✅ 安全
```

### P17 训练配置

| 参数 | 值 |
|------|-----|
| Total steps | 200,000 |
| Eval freq | 10,000 (20 次) |
| Warmup | 2,000 |
| Batch size | 64 (effective 256, accum=4) |
| LR | 2e-4 → 1e-6 cosine, 200k steps |
| Inference | DPMSolver++, 8 step default, 10 step eval |
| Checkpoints | `best.pth` + `step_000010.pth` ~ `step_200000.pth` |
| Throughput | ~4.5 it/s |

### P17.2 结构性加速 (第二批)

| 类别 | 改动 | 加速 |
|------|------|------|
| **Mamba** | Mamba-1 → **Mamba-2**（ssd 算法） | ~2-3× scan |
| **Mamba 调参** | d_state=32, headdim=32（恢复 SSD 并行度） | ~1.2× |
| **Attention** | nn.MHA → **FlashAttention-2** | ~2× |
| **归一化** | GroupNorm → **RMSNorm1d** | ~1.5× |
| **激活函数** | Mish → **GELU**（Conv1dBlock + cond_mlp） | ~1.3× |
| **优化器** | AdamW betas=(0.9,0.999)→**(0.9,0.95)** | 收敛 +~30% |
| **梯度缩放** | 移除 GradScaler（BF16 无需） | +~5% |
| **增广** | 移除颜色抖动（PushT 灰底无意义） | +~5% |
| **增广精度** | grid_sample 全程 half（无 float() 转换） | +~2% |
| **日志** | log_step 每 50 步（原每步） | +~2% |

---

## 关键教训总结

### 1. 容量与数据的匹配关系

```
参数量       参数/帧     数据量     状态
69M (P5~P6)  2,810      25k      严重过拟合
41.8M (P8)   1,672      25k      过拟合但可用
21.5M (P10~12) 860      25k      甜区, Coverage 0.90
27M (P14)    1,052      25k      保守, pred_std 仅 50%
12.5M (P15)   ~60      207k      欠拟合
19M (P16)     228       83k      平衡, 验证阶段
26.6M (P17)   320       83k      平衡, 训练中
```

### 2. 训练时间 > 模型容量

32k 步仅够模型完成探索期。24k→28k 出现二次爆发、28k→32k 又一次平台→说明模型在 32k 步处**没有学完**。200k 步是修正方向。

### 3. LR 退火是硬约束

余弦退火在 32k 步时将 LR 推到 1e-6，模型被迫在探索中途收敛。200k 步拉长退火曲线，让每个探索-消化周期都停留在更低的 loss 水平。

### 4. CFG 与 Temporal Ensemble

两者互斥。CFG 弱化条件响应，Ensemble 拉平动作，同时用 Coverage 被压在 0.40 以下。

### 5. Action delta

绝对坐标 → delta。解耦动作与位置，但 z-score 空间信噪比低，收敛更慢。

### 6. State mean/std vs min-max

`normalize_action(state)` 用 action 的 [12,511] 归一化 state 无物理意义。改为 `(state - state_mean) / state_std`。

### 7. val>train 的正确判断

不是看"是否出现"而是看"是否恢复":
- ❌ 恶性: val>train 后 val 持续走高不回弹
- ✅ 良性: val>train 后 1~3 epoch 内恢复并创新低

### 8. 结论：非必要不改模型

经过 Phase 14~17 迭代，模型层面已不再修改。后续优化只围绕：
- **训练效率**：吞吐、编译、推理加速
- **训练时间**：给足够长的探索-收敛周期
- 不引入新 trick、不换架构、不加数据

---

## 待定方向

| 方向 | 预期 | 优先级 | 说明 |
|------|------|--------|------|
| Keypoint co-training | +3~8% SR | 🟡 等 P17 结果 | data-letter 有 9 关键点数据 |
| Action Conditioning | +3~8% SR | 🟡 等 P17 结果 | 解决时序保守问题 |
| Mamba2 chunk_size=288 | +~2% it/s | 🟢 低优先级 | 对齐 L=288 减少 chunk 数 |
| 非 T 域 (H/V/R/O/B) 迁移测试 | 泛化性验证 | ⚪ 最后 | T 域验证后评估 |
| RL fine-tune → roll out | 不确定性大 | ⚪ 最后手段 | 仅在 SR 不达标时考虑 |

---

## 技术探索

### 一、Generative Modeling via Drifting

**论文**: Deng et al. 2026, arXiv:2602.04770

#### 核心思想
将扩散/流模型的**推理时迭代**搬到**训练时**。通过漂移场 V(x) = V⁺(x) - V⁻(x) 在训练过程中逐步推动生成分布 q 向数据分布 p 靠近。当 q = p 时 V(x) = 0，达到均衡。

- **V⁺ (吸引)**: 核加权平均拉向数据样本
- **V⁻ (排斥)**: 核加权平均推离生成样本
- **损失**: `L = MSE(f_θ(ε), stopgrad(f_θ(ε) + V(f_θ(ε))))` 即最小化 ‖V‖²
- **推理**: 单步前向 (1-NFE)，无需迭代去噪

#### 关键结果
- ImageNet 256×256: FID 1.54 (latent) / 1.61 (pixel) — 单步 SOTA
- 463M 参数，单次函数评估

#### 衍生工作
| 变体 | 贡献 |
|------|------|
| **Sinkhorn-Drifting** (arXiv:2603.12366) | 证明 Drifting 的 Gibbs 归一化是 Sinkhorn 双端缩放的一侧近似；双端缩放消除 identifiability gap，降低对核温度 τ 的敏感性 |
| **Lookahead Drifting** (arXiv:2605.04060) | 每步顺序计算多个漂移项 V₁, V₂, ..., Vₖ，捕获高阶梯度信息，在 CIFAR10 上有提升 |
| **Unified View** (arXiv:2603.07514) | 证明 Gaussian 核下 mean-shift = 方差缩放的 score mismatch；Lapalce 核可分解为 preconditioned score + 协方差残差 |
| **Drift-AR** (arXiv:2603.28049) | 将自回归预测熵 reinterpret 为反称漂移场的物理方差，实现 1-NFE 视觉解码 |

### 二、Scaling Latent Reasoning via Looped Language Models (Ouro)

**论文**: Zhu et al. 2025, arXiv:2510.25741

#### 核心思想
将**推理能力构建到预训练阶段**，通过**权重共享的循环架构**在**隐空间**中进行迭代计算，而非 CoT 的显式 token 生成。

```
非循环: F(·) = lmhead ∘ T_L ∘ ... ∘ T_1 ∘ emb(·)      # L 个独立层
循环:    F(t)(·) = lmhead ∘ T ∘ T ∘ ... ∘ T (t次) ∘ emb(·)  # 同层块循环 t 次
```

#### 关键设计
| 组件 | 说明 |
|------|------|
| **权重共享循环** | k 层块循环 T 次 = 有效深度 k×T，参数不增 |
| **自适应门控** | 每步输出退出概率 λ_t，推理时用分位数阈值 q 决定何时退出 |
| **熵正则化** | `L = Σ p_ϕℒ^(t) - β·H(p_ϕ)` — 防止门控坍塌到最深步 |
| **两阶段训练** | Stage I: 模型+门控联合训练；Stage II: 冻结 LM 只训门控做深度分配 |
| **KV cache 共享** | 循环间复用 KV cache，推理时循环步仅需增量计算 |

#### 关键结果
- Ouro 1.4B R4 = 匹配 4B Transformer
- Ouro 2.6B R4 = 匹配 8B Transformer (Qwen3-8B)
- **2-3× 参数效率**
- 优势来自**更好的知识操作能力** (fact composition, multi-hop)，而非知识容量 (~2 bits/param 不增)

#### 理论支撑 (ICLR 2025)
"Reasoning with Latent Thoughts: On the Power of Looped Transformers" (Saunshi et al.) 证明循环 Transformer 可模拟 CoT 推理、计算图深度等价于有效推理深度。

### 三、两篇论文的统一视图

| 维度 | Drifting Models | LoopLM (Ouro) |
|------|----------------|---------------|
| **循环主体** | 权重共享的生成器网络 | 权重共享的 Transformer 块 |
| **步进信号** | 漂移场 V(x) 的方向 | 门控退出概率 λ_t |
| **收敛判据** | V(x) = 0 (均衡) | CDF 跨阈值 (退出) |
| **自适应深度** | Lookahead 多阶漂移 | 早期退出 (分位数) |
| **训练 vs 推理** | 训练推分布，推理 1 步 | 训练学退出策略，推理自适应 |

**核心统一抽象**：两者都将推理/生成建模为**隐空间中的迭代精炼过程**，在固定参数预算下实现动态计算深度。

### 四、对当前项目的交叉启示

#### 启示 1：ToolinitModel → LoopedToolinitModel

当前 3 层独立权重 → 改为 1 组权重循环 T 次：

```
当前: 3 × SerialHybridBlock(独立参数) → 固定 3 层
改为: 1 × SerialHybridBlock(共享参数) → 循环 T 步
```

- 参数减少 3×
- 有效深度 T 可调节 (简单场景浅层、复杂场景深层)
- 推理时可选择不同 T 做速度-质量 trade-off

#### 启示 2：自适应深度替代固定 3 层

当前模型对所有输入都用相同深度。但 PushT 场景中：
- 简单场景 (T 块在中间) → 1-2 层精炼足够
- 复杂场景 (T 块被遮挡、靠近边缘) → 需要更深的隐空间精炼

可以结合 LoopLM 的退出门控 + Drifting 的 V(x) 均衡判据实现自适应深度。

#### 启示 3：Drifting 替换 DiffusionActionHead

将动作头从 DDPM noise-prediction 替换为单步漂移场：
- 训练: `L = MSE(x, stopgrad(x + V(x)))`, V = V⁺ - V⁻
- 推理: 单步前向，无需 8~20 步迭代去噪
- 方向：核温度 τ 可学习，支持 Sinkhorn 双端缩放

#### 启示 4：熵正则化抑制预测坍塌

当前模型训练后期 pred_std 偏低 (保守预测)。LoopLM 的 `-β·H(p_ϕ)` 思路可迁移为：
- 对动作分布加熵正则项，鼓励探索不同动作模式
- 对特征分布加熵，防止隐空间坍塌

#### 启示 5：最有趣的融合架构

```
视觉 → DINO (frozen) → LoopedToolinitModel (循环 T 步精炼, 自适应退出)
                                              ↓ 每步独立的特征
                                         选择退出时机
                                              ↓
                                         最终特征 → DriftingActionHead (1-NFE)
```

- 视觉特征通过循环精炼逐步去噪
- 循环深度 T 按场景复杂度自适应
- 动作生成单步完成，无需迭代采样
- 整个架构参数效率 2-3× 于当前

#### 研发优先级

| 方向 | 预期 | 优先级 | 说明 |
|------|------|--------|------|
| **ToolinitModel 循环化** | -3× 参数, 自适应深度 | 🟡 中 | 改动量 ~30 行, 需验证循环收敛性 |
| **ActionHead Drifting 化** | 8~20× 推理加速 | 🟡 中 | 核心 ~80 行, 需重训练对比 baseline |
| **熵正则化** | 缓解 pred_std 偏低 | 🟢 低 | ~5 行, 可独立实验 |
| **Sinkhorn 归一化** | 稳定 Drifting 训练 | ⚪ 待 Drifting 先验证 |
| **自适应退出门控** | 参数效率+场景自适应 | ⚪ 需先完成循环化 |
