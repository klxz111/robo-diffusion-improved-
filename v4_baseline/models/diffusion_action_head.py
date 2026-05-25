import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

try:
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
    from diffusers.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler
except ImportError:
    raise ImportError(
        "diffusers is required for DiffusionActionHead. "
        "Install with: pip install diffusers"
    )


class SinusoidalPosEmb(nn.Module):
    """1D sinusoidal positional embedding for diffusion timesteps."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=x.device, dtype=torch.float32) * -emb)
        emb = x.unsqueeze(-1) * emb.unsqueeze(0)
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class RMSNorm1d(nn.Module):
    """1D RMS Norm: x → x / RMS(x) * weight。比 GroupNorm 轻量，适合短序列。"""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        # x: [B, C, L] — 沿通道维归一化
        norm = x.pow(2).mean(1, keepdim=True).add(self.eps).rsqrt()
        return x * norm * self.weight.view(1, -1, 1)


class Conv1dBlock(nn.Module):
    """Conv1d -> RMSNorm -> GELU"""

    def __init__(self, inp, out, kernel, n_groups=8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(inp, out, kernel, padding=kernel // 2),
            RMSNorm1d(out),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class ConditionalResidualBlock1d(nn.Module):
    """1D residual block with FiLM conditioning."""

    def __init__(self, in_ch, out_ch, cond_dim, kernel=3, n_groups=8):
        super().__init__()
        self.conv1 = Conv1dBlock(in_ch, out_ch, kernel, n_groups)
        self.conv2 = Conv1dBlock(out_ch, out_ch, kernel, n_groups)
        self.cond_mlp = nn.Sequential(nn.GELU(), nn.Linear(cond_dim, out_ch * 2))
        self.residual_conv = (
            nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x, cond):
        out = self.conv1(x)
        cond_embed = self.cond_mlp(cond).unsqueeze(-1)
        scale, bias = cond_embed.chunk(2, dim=1)
        out = scale * out + bias
        out = self.conv2(out)
        out = out + self.residual_conv(x)
        return out


class ConditionalUnet1d(nn.Module):
    """1D temporal U-Net with FiLM conditioning.

    Architecture:
        Encoder: 3 stages of (ResBlock x2 + StridedConv1d)
        Bottleneck: ResBlock x2
        Decoder: 3 stages of (Concat skip + ResBlock x2 + ConvTranspose1d)
        Final: Conv1d to action dimension
    """

    def __init__(
        self,
        action_dim=2,
        horizon=16,
        down_dims=(128, 256, 512),
        cond_dim=64,
        kernel_size=3,
        n_groups=8,
        use_checkpoint=False,
    ):
        super().__init__()
        self.horizon = horizon
        self.use_checkpoint = use_checkpoint
        total_cond = 64 + cond_dim

        # Encoder channel plan
        # Stage 0: action_dim -> down_dims[0]
        # Stage 1: down_dims[0] -> down_dims[1]
        # Stage 2: down_dims[1] -> down_dims[2]
        enc_in = [action_dim] + list(down_dims[:-1])  # [2, 96, 192]
        enc_out = list(down_dims)  # [96, 192, 384]

        self.down_modules = nn.ModuleList()
        for i in range(len(down_dims)):
            is_last = i == len(down_dims) - 1
            self.down_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1d(
                            enc_in[i], enc_out[i], total_cond, kernel_size, n_groups
                        ),
                        ConditionalResidualBlock1d(
                            enc_out[i], enc_out[i], total_cond, kernel_size, n_groups
                        ),
                        ConditionalResidualBlock1d(
                            enc_out[i], enc_out[i], total_cond, kernel_size, n_groups
                        ),
                        nn.Conv1d(enc_out[i], enc_out[i], 3, stride=2, padding=1)
                        if not is_last
                        else nn.Identity(),
                    ]
                )
            )

        bottleneck_ch = down_dims[-1]
        self.mid_modules = nn.ModuleList(
            [
                ConditionalResidualBlock1d(
                    bottleneck_ch, bottleneck_ch, total_cond, kernel_size, n_groups
                ),
                ConditionalResidualBlock1d(
                    bottleneck_ch, bottleneck_ch, total_cond, kernel_size, n_groups
                ),
                ConditionalResidualBlock1d(
                    bottleneck_ch, bottleneck_ch, total_cond, kernel_size, n_groups
                ),
            ]
        )

        # Decoder channel plan (reversed)
        # Stage 0: bottleneck(384) + skip[2](384) -> 384 -> upsample -> 384
        # Stage 1: prev_up(384) + skip[1](192) -> 192 -> upsample -> 192
        # Stage 2: prev_up(192) + skip[0](96) -> 96
        dec_out = list(reversed(down_dims))  # [384, 192, 96]
        skips = list(
            reversed(down_dims)
        )  # skip channels from encoder: [384, 192, 96]

        self.up_modules = nn.ModuleList()
        for i in range(len(down_dims)):
            is_last = i == len(down_dims) - 1
            if i == 0:
                dec_in_ch = bottleneck_ch + skips[i]  # 384 + 384 = 768
            else:
                dec_in_ch = dec_out[i - 1] + skips[i]  # 384+192=576, 192+96=288

            self.up_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1d(
                            dec_in_ch, dec_out[i], total_cond, kernel_size, n_groups
                        ),
                        ConditionalResidualBlock1d(
                            dec_out[i], dec_out[i], total_cond, kernel_size, n_groups
                        ),
                        ConditionalResidualBlock1d(
                            dec_out[i], dec_out[i], total_cond, kernel_size, n_groups
                        ),
                        nn.ConvTranspose1d(
                            dec_out[i], dec_out[i], 4, stride=2, padding=1
                        )
                        if not is_last
                        else nn.Identity(),
                    ]
                )
            )

        self.final_conv = nn.Sequential(
            Conv1dBlock(down_dims[0], down_dims[0], kernel_size, n_groups),
            nn.Conv1d(down_dims[0], action_dim, 1),
        )

    def _forward_block(self, block, x, cond):
        """Encoder block helper for checkpoint compatibility."""
        res1, res2, res3, downsample = block
        x = res1(x, cond)
        x = res2(x, cond)
        x = res3(x, cond)
        skip = x
        x = downsample(x)
        return x, skip

    def _forward_mid(self, block, x, cond):
        """Mid block helper for checkpoint compatibility."""
        return block(x, cond)

    def _forward_up_block(self, block, x, skip, cond):
        """Decoder block helper for checkpoint compatibility."""
        res1, res2, res3, upsample = block
        x = torch.cat([x, skip], dim=1)
        x = res1(x, cond)
        x = res2(x, cond)
        x = res3(x, cond)
        x = upsample(x)
        return x

    def forward(self, x, timestep, global_cond=None):
        x = x.permute(0, 2, 1)
        t_emb = SinusoidalPosEmb(64)(timestep.to(x.device).float())
        cond = t_emb
        if global_cond is not None:
            cond = torch.cat([cond, global_cond], dim=-1)

        skips = []
        for block in self.down_modules:
            if self.use_checkpoint and self.training:
                x, skip = checkpoint(self._forward_block, block, x, cond, use_reentrant=False)
            else:
                x, skip = self._forward_block(block, x, cond)
            skips.append(skip)

        for mid in self.mid_modules:
            if self.use_checkpoint and self.training:
                x = checkpoint(self._forward_mid, mid, x, cond, use_reentrant=False)
            else:
                x = self._forward_mid(mid, x, cond)

        for block, skip in zip(self.up_modules, reversed(skips)):
            if self.use_checkpoint and self.training:
                x = checkpoint(self._forward_up_block, block, x, skip, cond, use_reentrant=False)
            else:
                x = self._forward_up_block(block, x, skip, cond)

        return self.final_conv(x).permute(0, 2, 1)


class DiffusionActionHead(nn.Module):
    """Diffusion Policy action head.

    Wraps a ConditionalUnet1d + DDPMScheduler for training and inference.

    Training:
        loss = head.compute_loss(global_cond, action)

    Inference:
        action_chunk = head.generate(global_cond, batch_size)
    """

    def __init__(
        self,
        action_dim=2,
        horizon=16,
        n_action_steps=4,
        global_cond_dim=384,
        down_dims=(96, 192, 384),
        num_train_timesteps=100,
        clip_sample_range=1.0,
        use_checkpoint=False,
    ):
        super().__init__()
        self.horizon = horizon
        self.n_action_steps = n_action_steps

        self.unet = ConditionalUnet1d(
            action_dim=action_dim,
            horizon=horizon,
            down_dims=down_dims,
            use_checkpoint=use_checkpoint,
            cond_dim=global_cond_dim,
        )

        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            clip_sample_range=clip_sample_range,
            prediction_type="epsilon",
        )

        self.inference_scheduler = DPMSolverMultistepScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_schedule="squaredcos_cap_v2",
            prediction_type="epsilon",
            solver_order=2,
            use_karras_sigmas=True,
        )

    def compute_loss(self, global_cond, action):
        """Compute diffusion loss.

        Args:
            global_cond: [B, global_cond_dim]  (Mamba output)
            action:      [B, horizon, action_dim]  (normalized to [-1, 1])

        Returns:
            Weighted MSE loss between predicted noise and actual noise
        """
        B, H, D = action.shape
        eps = torch.randn_like(action)
        t = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (B,),
            device=action.device,
        )
        noisy = self.noise_scheduler.add_noise(action, eps, t)
        pred = self.unet(noisy, t, global_cond)

        loss = (pred - eps).pow(2).mean()
        return loss

    @torch.no_grad()
    def generate(
        self, global_cond, batch_size=1, num_inference_steps=None, use_dpm=True,
    ):
        """Generate action chunk via iterative denoising (DPM-Solver++).

        Args:
            global_cond: [B, global_cond_dim]
            batch_size: number of samples to generate
            num_inference_steps: defaults to 8 for DPM, 100 for DDPM
            use_dpm: if True, use DPMSolverMultistepScheduler (faster); else DDPMScheduler

        Returns:
            [B, n_action_steps, action_dim]  (still in [-1, 1] space)
        """
        device = global_cond.device
        sample = torch.randn(batch_size, self.horizon, 2, device=device)

        scheduler = self.inference_scheduler if use_dpm else self.noise_scheduler
        if num_inference_steps is None:
            num_inference_steps = (
                8 if use_dpm else scheduler.config.num_train_timesteps
            )
        scheduler.set_timesteps(num_inference_steps)

        for t in scheduler.timesteps:
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            pred = self.unet(sample, t_batch, global_cond)
            sample = scheduler.step(pred, t, sample).prev_sample

        return sample[:, : self.n_action_steps]

    def count_params(self):
        return sum(p.numel() for p in self.parameters())

    def count_trainable(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
