"""
V4 Evaluation Script for PushT.

Key changes from V3 eval:
  - ViT backbone (DINO v1 vits8, frozen)
  - 2-frame observation history (n_obs_steps=2)
  - Per-frame Spatial Softmax keypoint extraction
  - Strict (>=0.95) and Effective (>=0.80) success rates
  - Coverage distribution histogram
"""

import os
import sys
from collections import deque

import imageio
import cv2
import gymnasium as gym
import gym_pusht  # noqa: F401
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from models.diffusion_action_head import DiffusionActionHead
from models.hybrid_core import ToolinitModel
from models.spatial_softmax import SpatialSoftmax
from models.vit_backbone import ViTBackbone

# ──────────────────────────────────────────────────────────────────────────────
# 配置
# ──────────────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# V4 架构 best.pth: ViT (DINO v1 vits8, patch=8), 2帧历史, fuse_layers=[4,7,11], DDIM推理
CKPT_PATH = "checkpoints_phase3/best.pth"
VIDEO_DIR = "eval_videos"
NUM_EPISODES = 50
STRICT_COVERAGE_THRESHOLD = 0.95  # 严格成功阈值
EFFECTIVE_COVERAGE_THRESHOLD = 0.80  # 有效覆盖率阈值
HORIZON = 16
N_OBS_STEPS = 2  # 2 帧历史 (-0.1s, 0s)
N_ACTION_STEPS = 8  # Receding Horizon Control
NUM_INFERENCE_STEPS = 20  # DDIM 20 步推理

# Action 归一化参数（来自 lerobot/pusht stats.json）
ACT_MIN = torch.tensor([12.0, 25.0], device=DEVICE)
ACT_MAX = torch.tensor([511.0, 511.0], device=DEVICE)

os.makedirs(VIDEO_DIR, exist_ok=True)

# Delta normalization stats (set by load_model)
DELTA_MEAN_T = None
DELTA_STD_T = None


# ──────────────────────────────────────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────────────────────────────────────


def denormalize_action(a):
    """将 [-1, 1] 归一化动作还原为原始像素坐标。

    参数来源: lerobot/pusht/meta/stats.json
      action.min = [12.0, 25.0]
      action.max = [511.0, 511.0]
    """
    return (a + 1) / 2 * (ACT_MAX - ACT_MIN) + ACT_MIN


def preprocess_frame(obs_image):
    """将环境返回的图像转为 [C, H, W] float tensor。

    对齐 lerobot 的 NormalizerProcessorStep:
      - uint8 [0, 255] → float32 [0, 1]
      - HWC → CHW
    """
    if isinstance(obs_image, np.ndarray):
        frame = torch.from_numpy(obs_image).float() / 255.0
        frame = frame.permute(2, 0, 1)
    elif isinstance(obs_image, torch.Tensor):
        frame = obs_image.float() / 255.0
        if frame.dim() == 3 and frame.shape[0] != 3:
            frame = frame.permute(2, 0, 1)
    else:
        raise TypeError("Unsupported observation type: %s" % type(obs_image))
    return frame


def load_model(ckpt_path):
    """加载 V4 训练好的模型权重。

    架构: ViTBackbone (DINO v1 vits8, frozen)
         → ToolinitModel (SerialHybrid × 3, d_model=384)
         → SpatialSoftmax (K=16, per-frame) + keypoint_proj
         → DiffusionActionHead (UNet1d)
    """
    vision = ViTBackbone(embed_dim=384, n_obs=N_OBS_STEPS, fuse_layers=[4, 7, 11]).to(DEVICE)
    core = ToolinitModel().to(DEVICE)
    spatial_softmax = SpatialSoftmax(
        num_keypoints=16, token_dim=384, grid_h=12, grid_w=12
    ).to(DEVICE)
    feat_compress = nn.Sequential(
        nn.Linear(384, 64),
        nn.Dropout(0.15),
    ).to(DEVICE)
    cls_proj = nn.Sequential(
        nn.Linear(384, 64),
        nn.Dropout(0.15),
    ).to(DEVICE)
    # per_frame: kp[16,2] + feat[16,64] + cls[64] + state[2] = 1122
    per_frame_dim = 16 * (2 + 64) + 64 + 2
    proj = nn.Sequential(
        nn.Linear(N_OBS_STEPS * per_frame_dim, 384),
        nn.Dropout(0.15),
    ).to(DEVICE)
    keypoint_proj = nn.ModuleDict(
        {
            "feat_compress": feat_compress,
            "cls_proj": cls_proj,
            "proj": proj,
        }
    ).to(DEVICE)
    head = DiffusionActionHead(
        action_dim=2,
        horizon=HORIZON,
        n_action_steps=N_ACTION_STEPS,
        global_cond_dim=384,
        down_dims=(128, 256, 512),
    ).to(DEVICE)

    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)

    # 检测是否为完整 checkpoint（V4/V4.2 包含所有模块）
    has_all_modules = all(
        k in ckpt
        for k in ["vision", "core", "head", "spatial_softmax", "keypoint_proj"]
    )

    if has_all_modules:
        # 完整 checkpoint（V4 base, V4.2 等）
        vision.load_state_dict(ckpt["vision"])
        core.load_state_dict(ckpt["core"])
        head.load_state_dict(ckpt["head"])
        spatial_softmax.load_state_dict(ckpt["spatial_softmax"])
        keypoint_proj.load_state_dict(ckpt["keypoint_proj"])
        print(
            "  已加载完整 checkpoint: %s (epoch=%d)"
            % (ckpt_path, ckpt.get("epoch", -1))
        )
    else:
        # 仅 head 的 checkpoint（V4.1 RWR 微调）
        ckpt_v4 = torch.load(
            "checkpoints_diffusion/best.pth", map_location=DEVICE, weights_only=False
        )
        vision.load_state_dict(ckpt_v4["vision"])
        core.load_state_dict(ckpt_v4["core"])
        head.load_state_dict(ckpt_v4["head"])
        spatial_softmax.load_state_dict(ckpt_v4["spatial_softmax"])
        keypoint_proj.load_state_dict(ckpt_v4["keypoint_proj"])
        head.load_state_dict(ckpt["head"])
        print("  V4.1 RWR head 已覆盖")

    # Precompute delta stats from dataset
    from diffusion_loader import _preload_dataset
    _, _, _, _, _, _, _, delta_mean, delta_std = _preload_dataset("data/raw_pusht")  # noqa: E999
    global DELTA_MEAN_T, DELTA_STD_T
    DELTA_MEAN_T = torch.from_numpy(delta_mean).to(DEVICE)
    DELTA_STD_T = torch.from_numpy(delta_std).to(DEVICE)

    vision.eval()
    core.eval()
    head.eval()
    spatial_softmax.eval()
    keypoint_proj.eval()

    return vision, core, head, spatial_softmax, keypoint_proj


def save_video(frames, path, fps=10):
    """保存帧列表为 mp4 视频 (使用 imageio 避免 WSL 下 FFmpeg 问题)"""
    if not frames:
        return
    processed = []
    h, w = frames[0].shape[:2]
    for frame in frames:
        if isinstance(frame, torch.Tensor):
            frame = frame.cpu().numpy()
        if frame.dtype != np.uint8:
            if frame.max() <= 1.0:
                frame = (frame * 255).astype(np.uint8)
            else:
                frame = frame.astype(np.uint8)
        if frame.shape[:2] != (h, w):
            frame = cv2.resize(frame, (w, h))
        processed.append(frame)
    imageio.mimsave(path, processed, fps=fps, codec="libx264", macro_block_size=1)
    print("  视频已保存: %s" % path)


# ──────────────────────────────────────────────────────────────────────────────
# 队列管理（对齐 lerobot 逻辑）
# ──────────────────────────────────────────────────────────────────────────────


def populate_queues(queues, batch):
    """填充观测队列，对齐 lerobot/policies/utils.py:31-48。

    首次调用: 用同一观测重复填充至 maxlen（warm-start）
    后续调用: 追加新观测，最旧的一帧自动弹出（滑动窗口）
    """
    for key, val in batch.items():
        if key not in queues:
            continue
        if len(queues[key]) != queues[key].maxlen:
            while len(queues[key]) != queues[key].maxlen:
                queues[key].append(val.clone())
        else:
            queues[key].append(val)
    return queues


def select_action(queues, vision, core, head, spatial_softmax, keypoint_proj):
    """选择下一个动作，对齐 lerobot 的 select_action 模式。

    V4: 2 帧历史 → ViT (fuse_layers=[4,7,11]) → ToolinitModel → 按帧 Spatial Softmax → UNet
    """
    current_frame = queues["_current_frame"]
    current_state = queues["_current_state"]
    populate_queues(queues, {"image": current_frame, "state": current_state})

    if len(queues["action"]) == 0:
        obs_seq = torch.stack(list(queues["image"]), dim=0).unsqueeze(0).to(DEVICE)
        state_seq = torch.stack(list(queues["state"]), dim=0).unsqueeze(0).to(DEVICE)
        B, n_obs, C, H, W = obs_seq.shape

        # 归一化 state（同 action 范围）
        state_norm = 2 * (state_seq - ACT_MIN[None, None, :]) / (ACT_MAX - ACT_MIN)[None, None, :] - 1

        patch_feat, cls_feat = vision(obs_seq)                    # [B*n, 144, 384], [B*n, 384]
        patch_feat = patch_feat.view(B, n_obs, 144, -1)            # [B, n_obs, 144, 384]
        patch_feat = patch_feat + keypoint_proj["frame_emb"].weight[None, :, None, :]
        feat = patch_feat.reshape(B, n_obs * 144, -1)             # [B, n_obs*144, 384]
        out = core(feat)                                           # [B, n_obs*144, 384]
        out = out.view(B * n_obs, 144, out.shape[-1])              # [B*n_obs, 144, 384]
        keypoints, kp_features = spatial_softmax(out)
        keypoints = keypoints.view(B, n_obs, -1, 2)
        kp_features = kp_features.view(B, n_obs, -1, kp_features.shape[-1])
        kp_features_compressed = keypoint_proj["feat_compress"](kp_features)
        combined = torch.cat([keypoints, kp_features_compressed], dim=-1)
        combined = combined.view(B, n_obs, -1)
        cls_c = keypoint_proj["cls_proj"](cls_feat).view(B, n_obs, -1)
        full = torch.cat([combined, cls_c, state_norm], dim=-1).flatten(1)
        global_cond = keypoint_proj["proj"](full)
        current_pos = state_seq[:, -1, :]  # [B, 2]

        chunk = head.generate(
            global_cond,
            batch_size=B,
            num_inference_steps=NUM_INFERENCE_STEPS,
            use_ddim=True,
        )

        chunk_delta = chunk * DELTA_STD_T + DELTA_MEAN_T  # raw pixel delta
        chunk_abs = current_pos.unsqueeze(1) + chunk_delta  # [B, 8, 2] pixel coords
        for i in range(chunk_abs.shape[1]):
            queues["action"].append(chunk_abs[0, i].cpu().numpy())

    raw_action = queues["action"].popleft()
    return raw_action


# ──────────────────────────────────────────────────────────────────────────────
# 评估主流程
# ──────────────────────────────────────────────────────────────────────────────


def evaluate():
    if not os.path.exists(CKPT_PATH):
        print("模型文件不存在: %s" % CKPT_PATH)
        print("   请先运行 train_v4.py 训练模型")
        sys.exit(1)

    print("=" * 60)
    print("PushT V4 Diffusion Policy 评估")
    print("   设备: %s" % DEVICE)
    print("   模型: %s" % CKPT_PATH)
    print("   Episodes: %d" % NUM_EPISODES)
    print("   n_obs_steps: %d" % N_OBS_STEPS)
    print("   n_action_steps: %d" % N_ACTION_STEPS)
    print("   去噪步数: %d (DDIM)" % NUM_INFERENCE_STEPS)
    print("   严格成功阈值: %.2f" % STRICT_COVERAGE_THRESHOLD)
    print("   有效覆盖率阈值: %.2f" % EFFECTIVE_COVERAGE_THRESHOLD)
    print("=" * 60)

    vision, core, head, spatial_softmax, keypoint_proj = load_model(CKPT_PATH)
    print("模型加载成功")

    env = gym.make("gym_pusht/PushT-v0", obs_type="pixels_agent_pos")
    print("PushT 环境创建成功")

    success_count = 0
    max_coverage_all = []
    all_episode_frames = []

    for ep in tqdm(range(NUM_EPISODES), desc="评估中"):
        obs_dict, info = env.reset()
        obs_img = obs_dict["pixels"]

        # 初始化队列（等价于 lerobot 的 policy.reset()）
        queues = {
            "image": deque(maxlen=N_OBS_STEPS),
            "state": deque(maxlen=N_OBS_STEPS),
            "action": deque(maxlen=N_ACTION_STEPS),
            "_current_frame": preprocess_frame(obs_img),
            "_current_state": torch.from_numpy(obs_dict["agent_pos"].copy()).float(),
        }

        episode_frames = []
        episode_max_coverage = 0.0
        episode_done = False
        obs_image = obs_img

        while not episode_done:
            # 记录当前帧
            if isinstance(obs_image, torch.Tensor):
                vis_frame = obs_image.cpu().numpy()
            else:
                vis_frame = obs_image.copy()
            episode_frames.append(vis_frame)

            # 更新当前帧引用
            queues["_current_frame"] = preprocess_frame(obs_image)

            # 获取动作（对齐 lerobot 的 select_action 模式）
            action = select_action(
                queues, vision, core, head, spatial_softmax, keypoint_proj
            )

            # 反归一化到像素坐标
            action_tensor = torch.from_numpy(action).to(DEVICE)
            action_numpy = action_tensor.cpu().numpy().astype(np.float32)

            # 执行动作
            obs_dict, reward, terminated, truncated, info = env.step(action_numpy)
            episode_done = terminated or truncated

            # 更新 coverage
            coverage = info.get("coverage", 0.0)
            episode_max_coverage = max(episode_max_coverage, float(coverage))

            obs_image = obs_dict["pixels"]
            queues["_current_state"] = torch.from_numpy(obs_dict["agent_pos"].copy()).float()

        # 判断成功
        is_strict_success = episode_max_coverage >= STRICT_COVERAGE_THRESHOLD
        is_effective = episode_max_coverage >= EFFECTIVE_COVERAGE_THRESHOLD
        if is_strict_success:
            success_count += 1

        max_coverage_all.append(episode_max_coverage)
        all_episode_frames.append(
            (episode_frames, is_strict_success, episode_max_coverage)
        )

        if ep % 10 == 0 or ep == NUM_EPISODES - 1:
            print(
                "  Episode %d: coverage=%.3f, strict_success=%s, effective=%s, 累计严格成功率=%d/%d=%.1f%%"
                % (
                    ep,
                    episode_max_coverage,
                    is_strict_success,
                    is_effective,
                    success_count,
                    ep + 1,
                    success_count / (ep + 1) * 100,
                )
            )

    # Save per-episode data to CSV
    eval_csv = os.path.join(VIDEO_DIR, "eval_episodes.csv")
    with open(eval_csv, "w") as f:
        f.write("episode,coverage,strict_success,effective\n")
        for i, cov in enumerate(max_coverage_all):
            is_s = cov >= STRICT_COVERAGE_THRESHOLD
            is_e = cov >= EFFECTIVE_COVERAGE_THRESHOLD
            f.write(f"{i},{cov:.6f},{is_s},{is_e}\n")
    print("  Per-episode data saved: %s" % eval_csv)

    # 统计结果
    strict_success_rate = success_count / NUM_EPISODES * 100
    effective_count = sum(
        1 for c in max_coverage_all if c >= EFFECTIVE_COVERAGE_THRESHOLD
    )
    effective_rate = effective_count / NUM_EPISODES * 100
    mean_coverage = np.mean(max_coverage_all)
    std_coverage = np.std(max_coverage_all)

    # Coverage 分布统计
    bins = [0.0, 0.1, 0.5, 0.8, 0.95, 1.0]
    labels = ["[0.00-0.10)", "[0.10-0.50)", "[0.50-0.80)", "[0.80-0.95)", "[0.95-1.00]"]
    dist = np.histogram(max_coverage_all, bins=bins)[0]

    print("\n" + "=" * 60)
    print("评估结果")
    print("=" * 60)
    print(
        "   严格成功率 (≥%.2f):  %.1f%% (%d/%d)"
        % (STRICT_COVERAGE_THRESHOLD, strict_success_rate, success_count, NUM_EPISODES)
    )
    print(
        "   有效覆盖率 (≥%.2f):  %.1f%% (%d/%d)"
        % (EFFECTIVE_COVERAGE_THRESHOLD, effective_rate, effective_count, NUM_EPISODES)
    )
    print("   Mean Coverage:     %.3f +/- %.3f" % (mean_coverage, std_coverage))
    print("   Max Coverage:      %.3f" % np.max(max_coverage_all))
    print("   Min Coverage:      %.3f" % np.min(max_coverage_all))
    print("")
    print("   Coverage 分布:")
    for label, count in zip(labels, dist):
        print(
            "     %s:  %2d episodes (%.1f%%)"
            % (label, count, count / NUM_EPISODES * 100)
        )
    print("=" * 60)

    # 保存视频
    print("\n保存视频...")

    all_frames_concat = []
    for frames, is_strict, cov in all_episode_frames:
        frame_h, frame_w = frames[0].shape[:2]
        info_frame = np.zeros((40, frame_w, 3), dtype=np.uint8)
        status = (
            "SUCCESS"
            if is_strict
            else ("EFFECTIVE" if cov >= EFFECTIVE_COVERAGE_THRESHOLD else "FAIL")
        )
        if is_strict:
            color = (0, 255, 0)
        elif cov >= EFFECTIVE_COVERAGE_THRESHOLD:
            color = (0, 255, 255)
        else:
            color = (0, 0, 255)
        text = "Ep | cov=%.3f | %s" % (cov, status)
        cv2.putText(info_frame, text, (5, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)
        all_frames_concat.append(info_frame)
        all_frames_concat.extend(frames)

    concat_path = os.path.join(VIDEO_DIR, "eval_all_episodes.mp4")
    save_video(all_frames_concat, concat_path, fps=10)

    strict_eps = [(i, f, c) for i, (f, s, c) in enumerate(all_episode_frames) if s]
    effective_eps = [
        (i, f, c)
        for i, (f, s, c) in enumerate(all_episode_frames)
        if c >= EFFECTIVE_COVERAGE_THRESHOLD and not s
    ]
    fail_eps = [
        (i, f, c)
        for i, (f, s, c) in enumerate(all_episode_frames)
        if c < EFFECTIVE_COVERAGE_THRESHOLD
    ]

    if strict_eps:
        mid_idx = len(strict_eps) // 2
        ep_idx, frames, cov = strict_eps[mid_idx]
        path = os.path.join(
            VIDEO_DIR, "typical_strict_success_ep%d_cov%.3f.mp4" % (ep_idx, cov)
        )
        save_video(frames, path, fps=10)

    if effective_eps:
        mid_idx = len(effective_eps) // 2
        ep_idx, frames, cov = effective_eps[mid_idx]
        path = os.path.join(
            VIDEO_DIR, "typical_effective_ep%d_cov%.3f.mp4" % (ep_idx, cov)
        )
        save_video(frames, path, fps=10)

    if fail_eps:
        best_fail_idx = max(range(len(fail_eps)), key=lambda i: fail_eps[i][2])
        ep_idx, frames, cov = fail_eps[best_fail_idx]
        path = os.path.join(VIDEO_DIR, "typical_fail_ep%d_cov%.3f.mp4" % (ep_idx, cov))
        save_video(frames, path, fps=10)

    print("视频保存完成")
    env.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt", type=str, default=None, help="Checkpoint path to evaluate"
    )
    parser.add_argument("--episodes", type=int, default=None, help="Number of episodes")
    args = parser.parse_args()

    if args.ckpt:
        CKPT_PATH = args.ckpt
    if args.episodes:
        NUM_EPISODES = args.episodes

    evaluate()