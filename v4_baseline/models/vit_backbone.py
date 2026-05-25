"""
ViT Backbone for V4 Training Script.

Uses DINO v1 vits8 (frozen) as the vision backbone.
Outputs token sequence compatible with SpatialSoftmax.
"""

import torch
import torch.nn as nn


class ViTBackbone(nn.Module):
    """ViT backbone using DINO v1 vits8 (frozen) with multi-layer feature fusion.

    Args:
        embed_dim: Output embedding dimension (default: 384)
        n_obs: Number of observation frames (default: 4)
        fuse_layers: DINO block indices to fuse (default: [11] for single last layer)

    Input:
        x: [B, n_obs, C, H, W] or [B * n_obs, C, H, W]

    Output:
        patch_tokens: [B * n_obs, 144, embed_dim]  (patch tokens for SpatialSoftmax)
        cls_token:    [B * n_obs, embed_dim]        (global CLS token)
    """

    def __init__(self, embed_dim=384, n_obs=4, fuse_layers=None):
        super().__init__()
        self.n_obs = n_obs
        if fuse_layers is None:
            fuse_layers = [11]  # default: single last layer (backwards compatible)
        self.fuse_layers = fuse_layers

        # Load DINO v1 vits8 from torch.hub
        self.dino = torch.hub.load("facebookresearch/dino:main", "dino_vits8")

        # Freeze DINO backbone
        for param in self.dino.parameters():
            param.requires_grad = False

        # DINO vits8: patch_size=8, embed_dim=384, num_heads=6
        # Input 96x96 -> 12x12 patches = 144 tokens + 1 cls token = 145 tokens
        dino_dim = 384

        # Projection: [B, num_tokens, dino_dim * n_layers] -> [B, num_tokens, embed_dim]
        self.proj = nn.Linear(dino_dim * len(fuse_layers), embed_dim)

    def forward(self, x):
        # Handle both [B, n_obs, C, H, W] and [B * n_obs, C, H, W]
        if x.ndim == 5:
            B, n_obs, C, H, W = x.shape
            x = x.view(B * n_obs, C, H, W)
        else:
            B = x.shape[0]

        # DINO forward pass (no grad since frozen)
        with torch.no_grad():
            max_layer = max(self.fuse_layers)
            outputs = {}
            h = self.dino.prepare_tokens(x)  # [B, 145, 384]
            for i, blk in enumerate(self.dino.blocks):
                h = blk(h)
                if i in self.fuse_layers:
                    outputs[i] = self.dino.norm(h)
                if i >= max_layer:
                    break

            # Concatenate selected layers along feature dim (保留 CLS)
            # Each: [B, 145, 384]
            layers = [outputs[i] for i in self.fuse_layers]
            features = torch.cat(layers, dim=-1)  # [B, 145, n_layers * 384]

        # Project to target dimension
        out = self.proj(features)  # [B, 145, 384]
        patch_tokens = out[:, 1:, :]  # [B, 144, 384]  — SpatialSoftmax 通路
        cls_token = out[:, 0, :]      # [B, 384]       — 全局语义旁路
        return patch_tokens, cls_token

    def count_params(self):
        return sum(p.numel() for p in self.parameters())

    def count_trainable(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
