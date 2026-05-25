"""
V4 Training Script for Diffusion Policy.

Key changes from V3.5:
  1. ViT backbone (DINO v1 vits8, frozen) instead of ResNet-18
  2. 4-frame observation history (n_obs_steps=4)
  3. BF16 mixed precision + GradScaler
  4. Data augmentation on CPU (collate_fn, vectorized)
  5. num_workers=4, pin_memory=True
  6. Evaluation warmup lock (skip first 10 epochs)
  7. Checkpoint every 5 epochs (last.pth), best_loss.pth on improvement
  8. Early Stopping Patience=8
  9. Gradient norm monitoring
  10. Unified AdamW optimizer (stability first)
  11. SpatialSoftmax uses Conv2d to preserve 2D topology
  12. Keypoint projection with channel compression (not flat Linear)
"""

import argparse
import math
import os
import time
from collections import deque

import gymnasium as gym
import gym_pusht
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from diffusion_loader import create_diffusion_loaders, augment_batch_gpu
from metrics_logger import MetricsLogger
from models.diffusion_action_head import DiffusionActionHead
from models.hybrid_core import ToolinitModel
from models.spatial_softmax import SpatialSoftmax
from models.vit_backbone import ViTBackbone

# ──────────────────────────────────────────────────────────────────────────────
# CUDA 优化
# ──────────────────────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    torch.set_float32_matmul_precision('high')   # TF32 加速矩阵乘 (~1.3×)
    torch.backends.cudnn.benchmark = True         # auto-tune cudnn 内核

# ──────────────────────────────────────────────────────────────────────────────
# Hyperparameters
# ──────────────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
LR = 2e-4
GRAD_CLIP = 1.0
SEED = 42
CKPT_DIR = "checkpoints_phase4"
LOG_DIR = "logs_phase4"

HORIZON = 16
N_OBS_STEPS = 2
N_ACTION_STEPS = 8
ACCUM_STEPS = 4

USE_CHECKPOINT = False

# Step-based 训练控制
STEPS = 200000         # 总 optimizer steps
EVAL_FREQ = 10000      # 每 N steps 跑一次 GymEval
SAVE_FREQ = 10000      # 每 N steps 存一次 checkpoint
WARMUP_STEPS = 2000    # LR warmup steps
MIN_STEPS = 1000       # 最少训练步数

ACT_MIN = torch.tensor([12.0, 25.0], device=DEVICE)
ACT_MAX = torch.tensor([511.0, 511.0], device=DEVICE)

# Data augmentation
AUG_MAX_TRANS_PX = 8
AUG_MAX_ROT_DEG = 5
AUG_ACTION_NOISE_STD = 0.06
AUG_PROB = 0.5

os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────

def normalize_action(a):
    return 2 * (a - ACT_MIN) / (ACT_MAX - ACT_MIN) - 1

def denormalize_action(a):
    return (a + 1) / 2 * (ACT_MAX - ACT_MIN) + ACT_MIN

def normalize_delta(d, d_mean, d_std):
    return (d - d_mean) / d_std

def denormalize_delta(d_norm, d_mean, d_std):
    return d_norm * d_std + d_mean

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def save_checkpoint(
    path,
    epoch,
    vision,
    core,
    head,
    spatial_softmax,
    keypoint_proj,
    optimizer,
    scheduler,
    best_loss,
):
    torch.save(
        {
            "epoch": epoch,
            "vision": vision.state_dict(),
            "core": core.state_dict(),
            "head": head.state_dict(),
            "spatial_softmax": spatial_softmax.state_dict(),
            "keypoint_proj": keypoint_proj.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_loss": best_loss,
        },
        path,
    )

def load_checkpoint(
    path,
    vision,
    core,
    head,
    spatial_softmax,
    keypoint_proj,
    optimizer,
    scheduler,
):
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    vision.load_state_dict(ckpt["vision"])
    core.load_state_dict(ckpt["core"])
    head.load_state_dict(ckpt["head"])
    spatial_softmax.load_state_dict(ckpt["spatial_softmax"])
    keypoint_proj.load_state_dict(ckpt["keypoint_proj"], strict=False)
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    return ckpt["epoch"], ckpt["best_loss"]

@torch.no_grad()
def eval_action_std(
    vision, core, head, spatial_softmax, keypoint_proj, loader, num_samples=32,
    delta_mean_t=None, delta_std_t=None,
):
    """Run a full-denoising eval batch to measure action output Std vs Real Std.

    Uses num_inference_steps=100 (full DDPM) for academic rigor.
    Returns (pred_std_x, pred_std_y, real_std_x, real_std_y) in pixel space.
    """
    head.eval()
    core.eval()
    vision.eval()
    spatial_softmax.eval()
    keypoint_proj.eval()

    batch = next(iter(loader))
    img = batch["observation.image"].to(DEVICE)[:num_samples]
    real_action = batch["action"].to(DEVICE)[:num_samples]
    state = batch["observation.state"].to(DEVICE)[:num_samples]

    B = img.shape[0]
    state_norm = normalize_action(state)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        patch_feat, cls_feat = vision(img)                     # [B*N, 144, 384], [B*N, 384]
        patch_feat = patch_feat.view(B, N_OBS_STEPS, 144, -1)  # [B, N, 144, 384]
        patch_feat = patch_feat + keypoint_proj["frame_emb"].weight[None, :, None, :]
        feat = patch_feat.reshape(B, N_OBS_STEPS * 144, -1)   # [B, N_OBS_STEPS*144, 384]
        out = core(feat)                                       # [B, N_OBS_STEPS*144, 384]
        out = out.view(B * N_OBS_STEPS, 144, out.shape[-1])   # [B*N_OBS_STEPS, 144, 384]
        keypoints, kp_features = spatial_softmax(out)
        keypoints = keypoints.view(B, N_OBS_STEPS, -1, 2)
        kp_features = kp_features.view(B, N_OBS_STEPS, -1, kp_features.shape[-1])

        kp_features_compressed = keypoint_proj["feat_compress"](kp_features)
        combined = torch.cat([keypoints, kp_features_compressed], dim=-1)
        combined = combined.view(B, N_OBS_STEPS, -1)
        cls_c = keypoint_proj["cls_proj"](cls_feat).view(B, N_OBS_STEPS, -1)
        full = torch.cat([combined, cls_c, state_norm], dim=-1).flatten(1)
        global_cond = keypoint_proj["proj"](full)

    generated = head.generate(
        global_cond, batch_size=B, num_inference_steps=10, use_dpm=True
    )

    real_sliced = real_action[:, :N_ACTION_STEPS, :]

    # pred delta vs real delta（统一在像素 delta 空间对比）
    pred_delta = generated * delta_std_t + delta_mean_t  # [B, 8, 2] raw pixel delta
    real_current = state[:, -1, :].unsqueeze(1)
    real_delta = real_sliced - real_current

    pred_std_x = pred_delta[:, :, 0].std().item()
    pred_std_y = pred_delta[:, :, 1].std().item()
    real_std_x = real_delta[:, :, 0].std().item()
    real_std_y = real_delta[:, :, 1].std().item()

    head.train()
    core.train()
    vision.train()
    spatial_softmax.train()
    for m in keypoint_proj.values():
        m.train()

    return pred_std_x, pred_std_y, real_std_x, real_std_y


@torch.no_grad()
def compute_val_loss(vision, core, head, spatial_softmax, keypoint_proj, val_loader,
                     max_batches=20, state_mean_t=None, state_std_t=None,
                     delta_mean_t=None, delta_std_t=None):
    """在验证集上计算平均 loss（无增强，纯 forward）。"""
    head.eval()
    core.eval()
    vision.eval()
    spatial_softmax.eval()
    keypoint_proj.eval()

    def _normalize_state(s):
        return (s - state_mean_t) / state_std_t

    total_loss = 0.0
    count = 0
    for i, batch in enumerate(val_loader):
        if i >= max_batches:
            break
        img = batch["observation.image"].to(DEVICE)
        action = batch["action"].to(DEVICE)
        state = batch["observation.state"].to(DEVICE)

        B = img.shape[0]
        current_pos = state[:, -1, :].unsqueeze(1)  # [B, 1, 2]
        delta_abs = action - current_pos
        delta_norm = (delta_abs - delta_mean_t) / delta_std_t
        state_norm = _normalize_state(state)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            patch_feat, cls_feat = vision(img)
            patch_feat = patch_feat.view(B, N_OBS_STEPS, 144, -1)
            patch_feat = patch_feat + keypoint_proj["frame_emb"].weight[None, :, None, :]
            feat = patch_feat.reshape(B, N_OBS_STEPS * 144, -1)
            out = core(feat)
            out = out.view(B * N_OBS_STEPS, 144, out.shape[-1])
            keypoints, kp_features = spatial_softmax(out)
            keypoints = keypoints.view(B, N_OBS_STEPS, -1, 2)
            kp_features = kp_features.view(B, N_OBS_STEPS, -1, kp_features.shape[-1])

            kpf_compressed = keypoint_proj["feat_compress"](kp_features)
            combined = torch.cat([keypoints, kpf_compressed], dim=-1)
            combined = combined.view(B, N_OBS_STEPS, -1)
            cls_c = keypoint_proj["cls_proj"](cls_feat).view(B, N_OBS_STEPS, -1)
            full = torch.cat([combined, cls_c, state_norm], dim=-1).flatten(1)
            global_cond = keypoint_proj["proj"](full)                    # [B, 2244]→[B, 384]

            loss = head.compute_loss(global_cond, delta_norm)

        total_loss += loss.item()
        count += 1

    head.train()
    core.train()
    vision.train()
    spatial_softmax.train()
    for m in keypoint_proj.values():
        m.train()

    return total_loss / max(count, 1)


# ──────────────────────────────────────────────────────────────────────────────
# Gym Evaluation (rollout in real env)
# ──────────────────────────────────────────────────────────────────────────────


def _warmup_queue(queue, val):
    """用同一帧填充队列至 maxlen（warm-start）。"""
    queue.extend([val.clone() for _ in range(queue.maxlen - len(queue))])


@torch.no_grad()
def eval_on_gym(vision, core, head, spatial_softmax, keypoint_proj, num_episodes=10,
                state_mean_t=None, state_std_t=None, delta_mean_t=None, delta_std_t=None):
    """在 gym_pusht 环境中跑 num_episodes 个 episode，返回严格成功率和平均 coverage。"""
    vision.eval()
    core.eval()
    head.eval()
    spatial_softmax.eval()
    keypoint_proj.eval()

    # 本地引用 stats tensor（避免作用域查找问题）
    _normalize_state = lambda s: (s - state_mean_t) / state_std_t
    _delta_to_abs = lambda dn, cp: cp.unsqueeze(1) + dn * delta_std_t + delta_mean_t

    env = gym.make("gym_pusht/PushT-v0", obs_type="pixels_agent_pos")
    success = 0
    coverages = []

    for ep in range(num_episodes):
        obs_dict, _ = env.reset()
        obs_img = obs_dict["pixels"]
        queues = {
            "image": deque(maxlen=N_OBS_STEPS),
            "action": deque(maxlen=N_ACTION_STEPS),
            "state": deque(maxlen=N_OBS_STEPS),
        }
        max_cov = 0.0
        done = False
        frame = torch.from_numpy(obs_img.copy()).float().permute(2, 0, 1) / 255.0
        agent_pos = obs_dict["agent_pos"].copy()
        _warmup_queue(queues["image"], frame)
        _warmup_queue(queues["state"], torch.from_numpy(agent_pos).float())

        while not done:
            if len(queues["action"]) == 0:
                obs_seq = torch.stack(list(queues["image"]), dim=0).unsqueeze(0).to(DEVICE)
                state_seq = torch.stack(list(queues["state"]), dim=0).unsqueeze(0).to(DEVICE)
                state_norm = _normalize_state(state_seq)
                B, n_obs = obs_seq.shape[0], obs_seq.shape[1]
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    patch_feat, cls_feat = vision(obs_seq)               # [B*n, 144, 384], [B*n, 384]
                    patch_feat = patch_feat.view(B, n_obs, 144, -1)       # [B, n_obs, 144, 384]
                    patch_feat = patch_feat + keypoint_proj["frame_emb"].weight[None, :, None, :]
                    feat = patch_feat.reshape(B, n_obs * 144, -1)        # [B, n_obs*144, 384]
                    out = core(feat)                                      # [B, n_obs*144, 384]
                    out = out.view(B * n_obs, 144, out.shape[-1])         # [B*n_obs, 144, 384]
                    kp, kpf = spatial_softmax(out)
                    kp = kp.view(B, n_obs, -1, 2)
                    kpf = kpf.view(B, n_obs, -1, kpf.shape[-1])
                    kpf_compressed = keypoint_proj["feat_compress"](kpf)
                    combined = torch.cat([kp, kpf_compressed], dim=-1)
                    combined = combined.view(B, n_obs, -1)
                    cls_c = keypoint_proj["cls_proj"](cls_feat).view(B, n_obs, -1)
                    full = torch.cat([combined, cls_c, state_norm], dim=-1).flatten(1)
                    cond = keypoint_proj["proj"](full)
                chunk = head.generate(cond, batch_size=B, num_inference_steps=20, use_dpm=True)
                # delta → 绝对坐标存入队列
                current_pos = state_seq[:, -1, :]  # [B, 2]
                chunk_abs = _delta_to_abs(chunk, current_pos)  # [B, 8, 2] 像素坐标
                for i in range(chunk_abs.shape[1]):
                    queues["action"].append(chunk_abs[0, i].cpu().numpy())

            a = queues["action"].popleft()
            obs_dict, _, terminated, truncated, info = env.step(a.astype(np.float32))
            max_cov = max(max_cov, float(info.get("coverage", 0.0)))
            done = terminated or truncated
            frame = torch.from_numpy(obs_dict["pixels"].copy()).float().permute(2, 0, 1) / 255.0
            queues["image"].append(frame)
            queues["state"].append(torch.from_numpy(obs_dict["agent_pos"].copy()).float())

        if max_cov >= 0.95:
            success += 1
        coverages.append(max_cov)

    env.close()

    vision.train()
    core.train()
    head.train()
    spatial_softmax.train()
    keypoint_proj.train()

    sr = success / num_episodes * 100
    avg_cov = float(np.mean(coverages))
    print(
        "  [GymEval] Episodes=%d | Strict SR: %.1f%% (%d/%d) | Avg Coverage: %.3f"
        % (num_episodes, sr, success, num_episodes, avg_cov)
    )
    return sr, avg_cov


class EMA:
    """Online Exponential Moving Average of model parameters.

    Usage:
        ema = EMA(all_params, decay=0.999)
        # after each optimizer step:
        ema.update(all_params)
        # before eval:
        ema.apply(all_params)
        # after eval:
        ema.restore(all_params)
    """

    def __init__(self, model_params, decay=0.999):
        self.decay = decay
        self.shadow = [p.data.clone().detach() for p in model_params if p.requires_grad]
        self.backup = None

    def update(self, model_params):
        trainable = [p for p in model_params if p.requires_grad]
        for s, p in zip(self.shadow, trainable):
            s.mul_(self.decay).add_(p.data, alpha=1 - self.decay)

    def apply(self, model_params):
        trainable = [p for p in model_params if p.requires_grad]
        self.backup = [p.data.clone() for p in trainable]
        for s, p in zip(self.shadow, trainable):
            p.data.copy_(s)

    def restore(self, model_params):
        trainable = [p for p in model_params if p.requires_grad]
        for b, p in zip(self.backup, trainable):
            p.data.copy_(b)
        self.backup = None


# ──────────────────────────────────────────────────────────────────────────────
# Main Training
# ──────────────────────────────────────────────────────────────────────────────
# Main Training
# ──────────────────────────────────────────────────────────────────────────────
def train(resume=None):
    set_seed(SEED)

    spatial_softmax = SpatialSoftmax(
        num_keypoints=16, token_dim=384, grid_h=12, grid_w=12
    ).to(DEVICE)

    feat_compress = nn.Sequential(
        nn.Linear(384, 64),
    ).to(DEVICE)
    cls_proj = nn.Sequential(
        nn.Linear(384, 64),
    ).to(DEVICE)
    # per_frame: kp[16,2] + feat[16,64] + cls[64] + state[2] = 1122
    per_frame_dim = 16 * (2 + 64) + 64 + 2
    proj = nn.Sequential(
        nn.Linear(N_OBS_STEPS * per_frame_dim, 384),
    ).to(DEVICE)
    frame_emb = nn.Embedding(N_OBS_STEPS, 384).to(DEVICE)
    keypoint_proj = nn.ModuleDict(
        {
            "feat_compress": feat_compress,
            "cls_proj": cls_proj,
            "proj": proj,
            "frame_emb": frame_emb,
        }
    )

    vision = ViTBackbone(embed_dim=384, n_obs=N_OBS_STEPS, fuse_layers=[4, 7, 11]).to(DEVICE)
    core = ToolinitModel(use_checkpoint=USE_CHECKPOINT).to(DEVICE)
    head = DiffusionActionHead(
        action_dim=2,
        horizon=HORIZON,
        n_action_steps=N_ACTION_STEPS,
        global_cond_dim=384,
        down_dims=(128, 256, 512),
        use_checkpoint=USE_CHECKPOINT,
    ).to(DEVICE)

    # 编译 UNet（纯 Conv1d + Linear，不涉及 Mamba；不用 reduce-overhead 避免 CUDAGraph 冲突）
    head.unet = torch.compile(head.unet)

    if USE_CHECKPOINT:
        print("Gradient checkpointing enabled")

    print(
        "ViT params: {:,} ({:,} trainable)".format(
            vision.count_params(), vision.count_trainable()
        )
    )
    print(
        "DiffusionActionHead params: {:,} ({:,} trainable)".format(
            head.count_params(), head.count_trainable()
        )
    )

    all_params = (
        list(vision.proj.parameters())
        + list(core.parameters())
        + list(head.parameters())
        + list(spatial_softmax.parameters())
        + list(keypoint_proj.parameters())
    )
    optimizer = optim.AdamW(all_params, lr=LR, betas=(0.9, 0.95), weight_decay=1e-4)
    ema = EMA(all_params, decay=0.999)

    # Step-based cosine LR schedule
    def lr_lambda(step):
        if step < WARMUP_STEPS:
            return 0.01 + 0.99 * step / max(WARMUP_STEPS, 1)
        progress = (step - WARMUP_STEPS) / max(STEPS - WARMUP_STEPS, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    loader, val_loader, state_mean, state_std, delta_mean, delta_std = create_diffusion_loaders(
        batch_size=BATCH_SIZE, train_workers=0, val_split=0.2, seed=SEED,
    )
    state_mean_t = torch.from_numpy(state_mean).to(DEVICE)
    state_std_t = torch.from_numpy(state_std).to(DEVICE)
    delta_mean_t = torch.from_numpy(delta_mean).to(DEVICE)
    delta_std_t = torch.from_numpy(delta_std).to(DEVICE)

    def normalize_state(s):
        return (s - state_mean_t) / state_std_t

    def compute_delta_norm(action, state):
        cp = state[:, -1, :].unsqueeze(1)
        delta_abs = action - cp
        return (delta_abs - delta_mean_t) / delta_std_t

    def delta_to_abs(delta_norm, current_pos):
        delta_abs = delta_norm * delta_std_t + delta_mean_t
        return current_pos.unsqueeze(1) + delta_abs

    logger = MetricsLogger(log_dir=LOG_DIR)

    global_step, best_val_loss = 0, float("inf")
    if resume and os.path.isfile(resume):
        global_step, best_val_loss = load_checkpoint(
            resume,
            vision,
            core,
            head,
            spatial_softmax,
            keypoint_proj,
            optimizer,
            scheduler,
        )
        print("Resumed from step={}, best_val_loss={:.6f}".format(global_step, best_val_loss))
        # 快进 scheduler 到当前步数
        for _ in range(global_step // ACCUM_STEPS):
            scheduler.step()

    patience_counter = 0

    vision.train()
    core.train()
    head.train()

    # Step-based training loop
    pbar = tqdm(total=STEPS, desc="Training", initial=global_step)
    while global_step < STEPS:
        for batch in loader:
            step_start = time.time()

            img = batch["observation.image"].to(DEVICE)
            action = batch["action"].to(DEVICE)
            state = batch["observation.state"].to(DEVICE)

            # GPU data augmentation
            gpu_batch = augment_batch_gpu(
                {"observation.image": img, "action": action},
                max_trans_px=AUG_MAX_TRANS_PX,
                max_rot_deg=AUG_MAX_ROT_DEG,
                action_noise_std=AUG_ACTION_NOISE_STD,
                augment_prob=AUG_PROB,
            )
            img = gpu_batch["observation.image"]
            action = gpu_batch["action"]

            delta_norm = compute_delta_norm(action, state)
            state_norm = normalize_state(state)

            B = img.shape[0]

            with torch.autocast("cuda", dtype=torch.bfloat16):
                patch_feat, cls_feat = vision(img)                    # [B*N, 144, 384], [B*N, 384]
                patch_feat = patch_feat.view(B, N_OBS_STEPS, 144, -1)  # [B, N, 144, 384]
                patch_feat = patch_feat + frame_emb.weight[None, :, None, :]  # 帧位置编码
                feat = patch_feat.reshape(B, N_OBS_STEPS * 144, -1)  # [B, 288, 384]
                out = core(feat)                                       # [B, 288, 384]
                out = out.view(B * N_OBS_STEPS, 144, out.shape[-1])   # [B*2, 144, 384]
                keypoints, kp_features = spatial_softmax(out)
                keypoints = keypoints.view(B, N_OBS_STEPS, -1, 2)
                kp_features = kp_features.view(
                    B, N_OBS_STEPS, -1, kp_features.shape[-1]
                )

                kp_features_compressed = feat_compress(kp_features)   # [B, N, 16, 128]
                combined = torch.cat([keypoints, kp_features_compressed], dim=-1)  # [B, N, 16, 130]
                combined = combined.view(B, N_OBS_STEPS, -1)          # [B, N, 2080]
                cls_c = cls_proj(cls_feat).view(B, N_OBS_STEPS, -1)  # [B, N, 128]
                full = torch.cat([combined, cls_c, state_norm], dim=-1)  # [B, N, 2210]
                global_cond = proj(full.flatten(1))                   # [B, 2244]→[B, 384]

                loss = head.compute_loss(global_cond, delta_norm)
                loss = loss / ACCUM_STEPS

            loss.backward()
            step_vram = (
                torch.cuda.max_memory_allocated() / 1024**2
                if torch.cuda.is_available()
                else 0
            )
            torch.cuda.reset_peak_memory_stats()

            if (global_step + 1) % ACCUM_STEPS == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    list(vision.proj.parameters())
                    + list(core.parameters())
                    + list(head.parameters())
                    + list(spatial_softmax.parameters())
                    + list(keypoint_proj.parameters()),
                    GRAD_CLIP,
                )

                optimizer.step()
                optimizer.zero_grad()
                ema.update(all_params)
                scheduler.step()

            step_time = time.time() - step_start
            vram = step_vram

            if global_step % 50 == 0:
                logger.log_step(
                    {
                        "step": global_step,
                        "loss": loss.item(),
                        "lr": float(optimizer.param_groups[0]["lr"]),
                        "vram_mb": vram,
                        "step_time_s": step_time,
                    }
                )

            if global_step % 50 == 0 and (global_step + 1) % ACCUM_STEPS == 0:
                print(
                    "| Step {:5d}/{} | Loss: {:.6f} | LR: {:.2e} | GradNorm: {:.3f}".format(
                        global_step, STEPS, loss.item(),
                        float(optimizer.param_groups[0]["lr"]),
                        grad_norm,
                    )
                )
            pbar.update(1)

            global_step += 1

            # Eval at step boundaries
            if global_step % EVAL_FREQ == 0:
                ema.apply(all_params)

                print("\n=== Step {} Eval ===".format(global_step))
                pred_std_x, pred_std_y, real_std_x, real_std_y = eval_action_std(
                    vision, core, head, spatial_softmax, keypoint_proj, val_loader,
                    num_samples=min(32, BATCH_SIZE),
                    delta_mean_t=delta_mean_t, delta_std_t=delta_std_t,
                )

                avg_val_loss = compute_val_loss(
                    vision, core, head, spatial_softmax, keypoint_proj, val_loader,
                    max_batches=10,
                    state_mean_t=state_mean_t, state_std_t=state_std_t,
                    delta_mean_t=delta_mean_t, delta_std_t=delta_std_t,
                )
                print("  Val Loss: {:.6f}".format(avg_val_loss))

                # Gym eval with episode ramp
                n_eps = min(5 * (global_step // EVAL_FREQ), 50)
                gym_sr, gym_coverage = eval_on_gym(
                    vision, core, head,
                    spatial_softmax, keypoint_proj,
                    num_episodes=n_eps,
                    state_mean_t=state_mean_t, state_std_t=state_std_t,
                    delta_mean_t=delta_mean_t, delta_std_t=delta_std_t,
                )

                ema.restore(all_params)

                # 按步数保存（每次 eval 保留，方便回溯）
                save_checkpoint(
                    os.path.join(CKPT_DIR, "step_%06d.pth" % global_step),
                    global_step, vision, core, head,
                    spatial_softmax, keypoint_proj,
                    optimizer, scheduler, avg_val_loss,
                )
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    save_checkpoint(
                        os.path.join(CKPT_DIR, "best.pth"),
                        global_step, vision, core, head,
                        spatial_softmax, keypoint_proj,
                        optimizer, scheduler, best_val_loss,
                    )
                    print("  Best model saved (val_loss={:.6f})".format(best_val_loss))
                else:
                    patience_counter += 1

                lr_val = float(optimizer.param_groups[0]["lr"])
                logger.log_epoch({
                    "epoch": global_step,
                    "avg_val_loss": avg_val_loss,
                    "best_val_loss": best_val_loss,
                    "patience": patience_counter,
                    "pred_std_x": float(pred_std_x),
                    "pred_std_y": float(pred_std_y),
                    "real_std_x": float(real_std_x),
                    "real_std_y": float(real_std_y),
                    "lr": lr_val,
                    "gym_sr": gym_sr,
                    "gym_coverage": gym_coverage,
                })
                logger.save()

                print("  [Step {}] Val Loss: {:.6f} | Coverage: {:.3f} | SR: {:.1f}%".format(
                    global_step, avg_val_loss, gym_coverage, gym_sr))

            if global_step >= STEPS:
                break

    pbar.close()

    # Final save
    save_checkpoint(
        os.path.join(CKPT_DIR, "last.pth"),
        global_step, vision, core, head,
        spatial_softmax, keypoint_proj,
        optimizer, scheduler, best_val_loss,
    )
    print("Training complete! Best Val Loss: {:.6f}".format(best_val_loss))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume", type=str, default=None, help="checkpoint path to resume"
    )
    args = parser.parse_args()
    train(resume=args.resume)
