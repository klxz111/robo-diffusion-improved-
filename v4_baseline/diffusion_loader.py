"""
V4 DataLoader for Diffusion Policy training.

方案 A: 预加载所有帧到 RAM（消除视频解码瓶颈）
方案 B: 增强移到 GPU（消除 CPU-GPU 传输瓶颈）
"""

import os
import random
import sys
import zarr
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

os.environ["HF_HOME"] = os.path.expanduser("~/.cache/huggingface")
os.environ["HF_DATASETS_CACHE"] = os.path.expanduser("~/.cache/huggingface/datasets")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "lerobot/src")))

from lerobot.datasets.lerobot_dataset import LeRobotDataset

SAFE_DATA_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "data/raw_pusht")
)
os.makedirs(SAFE_DATA_ROOT, exist_ok=True)

HORIZON = 16
N_OBS_STEPS = 2

# Action normalization params
ACT_MIN = torch.tensor([12.0, 25.0])
ACT_MAX = torch.tensor([511.0, 511.0])


# ── data-letter T 域 Zarr 路径 ──
T_DOMAINS = [
    ("data-letter/dataset_domain/domain0/domain0.zarr", "domain0"),
]


def _preload_dataset(root):
    """将所有帧（raw_pusht + data-letter domain0）预加载到 RAM（≈ 4.5 GB float16）。"""
    print("  Preloading all data to RAM (float16)...")
    t0 = time.time()
    import gc

    # ── Stage 1: raw_pusht ──
    dataset = LeRobotDataset("lerobot/pusht", root=root, video_backend="pyav")
    n_pusht = len(dataset)

    images = np.empty((n_pusht, 96, 96, 3), dtype=np.float16)
    actions = np.empty((n_pusht, 2), dtype=np.float32)
    states = np.empty((n_pusht, 2), dtype=np.float32)

    for i in range(n_pusht):
        item = dataset[i]
        img = item["observation.image"]
        images[i] = (img.permute(1, 2, 0).numpy() if isinstance(img, torch.Tensor) else np.array(img)).astype(np.float16)
        actions[i] = item["action"].numpy() if isinstance(item["action"], torch.Tensor) else np.array(item["action"])
        states[i] = item["observation.state"].numpy() if isinstance(item["observation.state"], torch.Tensor) else np.array(item["observation.state"])
        if (i + 1) % 5000 == 0:
            print("    PushT %d/%d (%.1fs)" % (i + 1, n_pusht, time.time() - t0))

    fps = dataset.fps
    del dataset
    gc.collect()

    # ── Stage 2: 加载 data-letter Zarr → RAM ──
    offset = n_pusht
    for zarr_path, name in T_DOMAINS:
        if not os.path.exists(zarr_path):
            print("  Skipping %s (not found)" % zarr_path)
            continue
        z = zarr.open(zarr_path, mode="r")
        n = z["data"]["img"].shape[0]
        total = offset + n

        # 扩展到最终大小
        old_img, old_act, old_stt = images, actions, states
        images = np.empty((total, 96, 96, 3), dtype=np.float16)
        actions = np.empty((total, 2), dtype=np.float32)
        states = np.empty((total, 2), dtype=np.float32)
        images[:offset] = old_img
        actions[:offset] = old_act
        states[:offset] = old_stt
        del old_img, old_act, old_stt
        gc.collect()

        # 顺序分块读取
        chunk_size = 10000
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            images[offset + start:offset + end] = z["data"]["img"][start:end].astype(np.float16) / 255.0
            actions[offset + start:offset + end] = z["data"]["action"][start:end].astype(np.float32)
            states[offset + start:offset + end] = z["data"]["state"][start:end, :2].astype(np.float32)

        print("  %s loaded: %d frames → total %d (%.1fs)" % (name, n, total, time.time() - t0))
        offset = total
        del z
        gc.collect()

    total_frames = offset

    # ── 时序样本索引 ──
    valid_indices = list(range(N_OBS_STEPS - 1, total_frames - HORIZON + 1))

    # ── 统计量（只从 raw_pusht 计算）──
    state_mean = states[:n_pusht].mean(axis=0).astype(np.float32)
    state_std = states[:n_pusht].std(axis=0).astype(np.float32) + 1e-8

    pusht_valid = [i for i in range(N_OBS_STEPS - 1, n_pusht - HORIZON + 1)]
    all_deltas = np.concatenate([
        actions[i:i + HORIZON] - states[i][None, :] for i in pusht_valid
    ], axis=0)
    delta_mean = all_deltas.mean(axis=0).astype(np.float32)
    delta_std = all_deltas.std(axis=0).astype(np.float32) + 1e-8
    del all_deltas
    gc.collect()

    elapsed = time.time() - t0
    print("  Preload complete: %d frames, %.1f MB RAM, %.1fs" % (
        total_frames, images.nbytes / 1024 / 1024, elapsed))
    print("  State stats: mean=%s, std=%s" % (state_mean, state_std))
    print("  Delta stats: mean=%s, std=%s" % (delta_mean, delta_std))

    return images, actions, states, valid_indices, fps, state_mean, state_std, delta_mean, delta_std


class PreloadedPushTDataset(Dataset):
    """全部数据预加载到 RAM 的 PushT 数据集（float16 图片 ≈ 4.5 GB）。"""

    def __init__(self, images, actions, states, valid_indices, fps):
        self.images = images          # [N, 96, 96, 3] float16 [0, 1]
        self.actions = actions        # [N, 2] float32
        self.states = states          # [N, 2] float32
        self.valid_indices = valid_indices
        self.n_total = len(images)
        self.fps = fps

    def __len__(self):
        return len(self.valid_indices)

    def _get_frame(self, idx):
        """直接从预加载数组索引（纯 RAM，零 IO）。"""
        return self.images[idx], self.actions[idx], self.states[idx]

    def __getitem__(self, idx):
        center = self.valid_indices[idx]

        # 观测帧
        obs_frames, obs_states = [], []
        for t_offset in range(-(N_OBS_STEPS - 1), 1):
            frame_idx = max(0, center + t_offset)
            img, _, stt = self._get_frame(frame_idx)
            obs_frames.append(img)
            obs_states.append(stt)
        obs_seq = np.stack(obs_frames, axis=0)
        state_seq = np.stack(obs_states, axis=0)

        # 动作
        action_list = []
        for t_offset in range(HORIZON):
            frame_idx = min(center + t_offset, self.n_total - 1)
            _, act, _ = self._get_frame(frame_idx)
            action_list.append(act)
        action_seq = np.stack(action_list, axis=0)

        obs_seq = np.transpose(obs_seq, (0, 3, 1, 2))

        return {
            "observation.image": torch.from_numpy(obs_seq).float(),
            "observation.state": torch.from_numpy(state_seq).float(),
            "action": torch.from_numpy(action_seq).float(),
        }


def augment_batch_gpu(
    batch, max_trans_px=15, max_rot_deg=10, action_noise_std=3.0, augment_prob=0.8
):
    """GPU 端数据增强（合并平移+旋转为单次 grid_sample，吞吐~1.6×）。

    在训练循环中调用，batch 已在 GPU 上。
    """
    if random.random() >= augment_prob:
        return batch

    images = batch["observation.image"]  # [B, n_obs, C, H, W]
    actions = batch["action"]  # [B, horizon, 2]

    B, n_obs, C, H, W = images.shape
    N = B * n_obs
    dev = images.device

    images = images.view(N, C, H, W)

    # 1. 随机平移参数（帧一致）
    tx = (torch.rand(B, device=dev).repeat_interleave(n_obs) * 2 - 1) * max_trans_px
    ty = (torch.rand(B, device=dev).repeat_interleave(n_obs) * 2 - 1) * max_trans_px
    tx_norm = (2 * tx / W).view(N, 1, 1)
    ty_norm = (2 * ty / H).view(N, 1, 1)

    # 2. 随机旋转参数（帧一致）
    theta_deg = (torch.rand(B, device=dev).repeat_interleave(n_obs) * 2 - 1) * max_rot_deg
    theta_rad = torch.deg2rad(theta_deg)
    cos_a, sin_a = torch.cos(theta_rad), torch.sin(theta_rad)
    cos_ = cos_a.view(N, 1, 1)
    sin_ = sin_a.view(N, 1, 1)

    # 3. 单次仿射变换：旋转 + 平移（合并 grid_sample，全程 half 精度）
    #    组合变换: x' = R(θ) * x + t
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, H, device=dev),
        torch.linspace(-1, 1, W, device=dev),
        indexing="ij",
    )
    xx = xx.view(1, H, W).expand(N, -1, -1)
    yy = yy.view(1, H, W).expand(N, -1, -1)
    grid_x = cos_ * xx - sin_ * yy + tx_norm
    grid_y = sin_ * xx + cos_ * yy + ty_norm
    grid = torch.stack([grid_x, grid_y], dim=-1)
    images = F.grid_sample(images.half(), grid.half(), align_corners=False, padding_mode="zeros")

    # 动作补偿：平移 + 旋转
    tx_batch = tx.view(B, n_obs).mean(dim=1)
    ty_batch = ty.view(B, n_obs).mean(dim=1)
    center_x, center_y = 47.5, 47.5
    cos_batch = cos_a.view(B, n_obs).mean(dim=1).unsqueeze(1)
    sin_batch = sin_a.view(B, n_obs).mean(dim=1).unsqueeze(1)
    actions = actions.clone()
    actions[:, :, 0] -= tx_batch.unsqueeze(1)
    actions[:, :, 1] -= ty_batch.unsqueeze(1)
    ax = actions[:, :, 0] - center_x
    ay = actions[:, :, 1] - center_y
    actions[:, :, 0] = cos_batch * ax + sin_batch * ay + center_x
    actions[:, :, 1] = -sin_batch * ax + cos_batch * ay + center_y

    # 4. 动作噪声
    actions += torch.randn_like(actions) * action_noise_std

    images = images.view(B, n_obs, C, H, W)

    # 5. 裁剪动作到合法范围
    act_min = ACT_MIN.to(actions.device)
    act_max = ACT_MAX.to(actions.device)
    actions = actions.clamp(act_min[None, None, :], act_max[None, None, :])

    return {"observation.image": images, "action": actions}


def create_diffusion_loaders(batch_size=48, train_workers=4, eval_workers=0, val_split=0.2, seed=42):
    """创建训练和验证 DataLoader（全部数据在 RAM 中）。

    返回:
        train_loader, val_loader, state_mean, state_std, delta_mean, delta_std
    """
    res = _preload_dataset(SAFE_DATA_ROOT)
    images, actions, states, valid_indices, fps = res[:5]
    state_mean, state_std, delta_mean, delta_std = res[5:9]

    n_all = len(valid_indices)
    n_val = int(n_all * val_split)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n_all)
    train_indices = [valid_indices[i] for i in perm[n_val:]]
    val_indices = [valid_indices[i] for i in perm[:n_val]]

    train_dataset = PreloadedPushTDataset(images, actions, states, train_indices, fps)
    val_dataset = PreloadedPushTDataset(images, actions, states, val_indices, fps)

    print("V4 Diffusion dataset: images=[B, %d, C, H, W], actions=[B, %d, 2]" % (N_OBS_STEPS, HORIZON))
    print("  Split: train=%d, val=%d (all-RAM)" % (
        len(train_indices), len(val_indices)))

    _verify_first_batch(train_dataset, batch_size)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=train_workers, pin_memory=True, drop_last=True,
        persistent_workers=train_workers > 0,
        prefetch_factor=2 if train_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=eval_workers, pin_memory=True, drop_last=False,
    )
    return train_loader, val_loader, state_mean, state_std, delta_mean, delta_std


def _verify_first_batch(dataset, batch_size):
    """验证第一个 batch 的数据完整性。"""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    batch = next(iter(loader))
    img = batch["observation.image"]
    vmin, vmax = img.min().item(), img.max().item()
    vmean, vstd = img.mean().item(), img.std().item()
    has_nan = bool(torch.isnan(img).any().item() or torch.isinf(img).any().item())
    print("  Verify: img shape=%s, range=[%.4f, %.4f], mean=%.4f, std=%.4f%s" % (
          str(list(img.shape)), vmin, vmax, vmean, vstd,
          " ⚠️ NaN!" if has_nan else ""))
    if has_nan or vmax > 1.01 or vmin < -0.01:
        print("  ⚠️  WARNING: image values corrupted!")
    elif vmean < 0.01:
        print("  ⚠️  WARNING: image too dark (mean=%.4f)" % vmean)
    else:
        print("  ✓ Image data OK")
    try:
        import imageio
        sample = (img[0, 0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        imageio.imwrite(os.path.join(os.path.dirname(__file__), "verify_sample.png"), sample)
        print("  ✓ verify_sample.png saved")
    except Exception:
        pass

def get_diffusion_loader(batch_size=32, num_workers=4):
    """Compatibility wrapper — kept for external callers."""
    loader, _, _, _, _, _ = create_diffusion_loaders(
        batch_size=batch_size, train_workers=num_workers, eval_workers=0
    )
    return loader
