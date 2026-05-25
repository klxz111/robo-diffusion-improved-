import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from mamba_ssm import Mamba2
from flash_attn import flash_attn_func


class RMSNorm(nn.Module):
    """Pre-RMSNorm: 比 LayerNorm 更轻量，深层网络更稳定。"""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        norm = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * norm * self.weight


class SerialHybridBlock(nn.Module):
    """串行混合块: MHA (全局空间注意力) → Mamba×2 (时序推演)。

    结构:
      x → RMSNorm → MHA → +x → RMSNorm → Mamba×2 → +x
    """

    def __init__(self, d_model=384, n_heads=8, d_state=32, headdim=32):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.spatial_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

        self.norm2 = RMSNorm(d_model)
        self.temporal_mamba = nn.Sequential(
            Mamba2(d_model=d_model, d_state=d_state, d_conv=4, expand=2, headdim=headdim),
            Mamba2(d_model=d_model, d_state=d_state, d_conv=4, expand=2, headdim=headdim),
        )

    def forward(self, x):
        # FlashAttention-2 (全局空间注意力, 需要 bf16/fp16 输入)
        x_norm = self.norm1(x)
        B, L, D = x_norm.shape
        nh = self.spatial_attn.num_heads
        orig_dtype = x_norm.dtype
        if orig_dtype not in (torch.float16, torch.bfloat16):
            x_norm = x_norm.to(torch.bfloat16)
        q = x_norm.view(B, L, nh, D // nh)
        attn_out = flash_attn_func(q, q, q, causal=False)
        attn_out = attn_out.reshape(B, L, D).to(orig_dtype)
        x = x + attn_out

        # Mamba×2 (时序推演)
        x_norm = self.norm2(x)
        mamba_out = self.temporal_mamba(x_norm)
        x = x + mamba_out

        return x


class ToolinitModel(nn.Module):
    def __init__(self, d_model=384, num_layers=3, use_checkpoint=False):
        super().__init__()
        self.layers = nn.ModuleList(
            [SerialHybridBlock(d_model) for _ in range(num_layers)]
        )
        self.use_checkpoint = use_checkpoint

    def forward(self, x):
        for layer in self.layers:
            if self.use_checkpoint and self.training:
                x = checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)
        return x
