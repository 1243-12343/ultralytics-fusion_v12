import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class FFM(nn.Module):
    """
    Fusion Focus Module baseline for RGB-IR feature fusion.

    Inputs:
        x: [x_vis, x_ir], each shaped [B, C, H, W]
    Output:
        fused feature map shaped [B, C_out, H, W]
    """

    def __init__(
        self,
        c1,
        c2=None,
        num_heads=4,
        ffn_ratio=2.0,
        dropout=0.0,
        cutoff_ratio=0.25,
        max_transformer_tokens=256,
        fft_downsample=2,
    ):
        super().__init__()
        c2 = c1 if c2 is None else c2
        self.c1 = c1
        self.c2 = c2
        self.cutoff_ratio = float(cutoff_ratio)
        self.max_transformer_tokens = int(max_transformer_tokens)
        self.fft_downsample = max(1, int(fft_downsample))
        self._hp_mask = None

        # Frequency-aware detail enhancement (Hadamard gate).
        self.freq_gate = nn.Sequential(
            nn.Conv2d(c1, c1, 1, bias=False),
            nn.BatchNorm2d(c1),
            nn.Sigmoid(),
        )

        # Multi-pattern branches.
        self.raw_proj = nn.Sequential(
            nn.Conv2d(2 * c1, c2, 1, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True),
        )
        self.pool_proj = nn.Sequential(
            nn.Conv2d(2 * c1, c2, 1, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True),
        )
        self.freq_proj = nn.Sequential(
            nn.Conv2d(2 * c1, c2, 1, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True),
        )
        self.mix = nn.Sequential(
            nn.Conv2d(3 * c2, c2, 1, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True),
        )

        # Lightweight transformer fusion in token space.
        nhead = max(1, min(int(num_heads), c2))
        while c2 % nhead != 0 and nhead > 1:
            nhead -= 1
        ff_dim = max(c2, int(c2 * float(ffn_ratio)))
        self.transformer = nn.TransformerEncoderLayer(
            d_model=c2,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.out = nn.Sequential(
            nn.Conv2d(c2, c2, 3, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True),
        )

    def __getstate__(self):
        """Drop cached FFT masks during deepcopy."""
        state = self.__dict__.copy()
        state["_hp_mask"] = None
        return state

    def _get_highpass_mask(self, h, w, device, dtype):
        if (
            self._hp_mask is None
            or self._hp_mask.shape[-2:] != (h, w)
            or self._hp_mask.device != device
            or self._hp_mask.dtype != dtype
        ):
            yy, xx = torch.meshgrid(
                torch.arange(h, device=device),
                torch.arange(w, device=device),
                indexing="ij",
            )
            cy = (h - 1) / 2.0
            cx = (w - 1) / 2.0
            radius = torch.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
            cutoff = max(1.0, self.cutoff_ratio * min(h, w) / 2.0)
            self._hp_mask = (radius >= cutoff).to(dtype)[None, None, :, :]
        return self._hp_mask

    def _highpass_spatial(self, x):
        """Extract high-frequency components with FFT high-pass mask."""
        b, c, h, w = x.shape
        x_dtype = x.dtype
        x_small = x
        if self.fft_downsample > 1 and min(h, w) >= self.fft_downsample * 2:
            x_small = F.avg_pool2d(x, kernel_size=self.fft_downsample, stride=self.fft_downsample)
        hs, ws = x_small.shape[-2:]

        xf = x_small.float()
        spec = torch.fft.fft2(xf, norm="ortho")
        spec = torch.fft.fftshift(spec, dim=(-2, -1))
        highpass = self._get_highpass_mask(hs, ws, spec.device, spec.dtype)
        spec_high = spec * highpass
        x_high = torch.fft.ifft2(torch.fft.ifftshift(spec_high, dim=(-2, -1)), norm="ortho").real
        if (hs, ws) != (h, w):
            x_high = F.interpolate(x_high, size=(h, w), mode="bilinear", align_corners=False)
        return x_high.to(x_dtype)

    def _detail_enhance(self, x):
        x_high = self._highpass_spatial(x)
        gate = self.freq_gate(x_high)
        # Hadamard-style modulation: enhance detail-sensitive regions.
        x_enh = x * (1.0 + gate)
        return x_enh, x_high

    def forward(self, x):
        if not isinstance(x, (list, tuple)) or len(x) != 2:
            raise ValueError("FFM expects a list/tuple of two feature maps: [vis, ir].")

        x_vis, x_ir = x
        if x_vis.shape[-2:] != x_ir.shape[-2:]:
            x_ir = F.interpolate(x_ir, size=x_vis.shape[-2:], mode="bilinear", align_corners=False)
        if x_vis.shape[1] != x_ir.shape[1]:
            raise ValueError(f"FFM expects matched channels, got {x_vis.shape[1]} and {x_ir.shape[1]}.")

        vis_enh, vis_hf = self._detail_enhance(x_vis)
        ir_enh, ir_hf = self._detail_enhance(x_ir)

        raw = self.raw_proj(torch.cat([vis_enh, ir_enh], dim=1))
        pool = self.pool_proj(torch.cat([F.avg_pool2d(vis_enh, 2, 2), F.avg_pool2d(ir_enh, 2, 2)], dim=1))
        pool = F.interpolate(pool, size=raw.shape[-2:], mode="bilinear", align_corners=False)
        freq = self.freq_proj(torch.cat([vis_hf, ir_hf], dim=1))

        fused = self.mix(torch.cat([raw, pool, freq], dim=1))
        b, c, h, w = fused.shape
        t_in = fused
        if h * w > self.max_transformer_tokens:
            side = max(4, int(math.sqrt(self.max_transformer_tokens)))
            t_in = F.adaptive_avg_pool2d(fused, (side, side))

        bt, ct, ht, wt = t_in.shape
        tokens = t_in.flatten(2).transpose(1, 2)  # [B, HW, C]
        tokens = self.transformer(tokens)
        t_out = tokens.transpose(1, 2).reshape(bt, ct, ht, wt)
        if (ht, wt) != (h, w):
            t_out = F.interpolate(t_out, size=(h, w), mode="bilinear", align_corners=False)

        return self.out(t_out + raw)


class CLPF(nn.Module):
    """
    Cross-modal Latent Predictive Fusion.

    It replaces FFM's transformer fusion with RGB<->IR latent prediction. The prediction error is used as a
    modality reliability score, so unreliable modality features receive lower fusion weights.
    """

    def __init__(
        self,
        c1,
        c2=None,
        latent_ratio=0.25,
        cutoff_ratio=0.25,
        max_latent_hw=16,
        fft_downsample=2,
        temperature=1.0,
    ):
        super().__init__()
        c2 = c1 if c2 is None else c2
        latent_c = max(16, int(c2 * float(latent_ratio)))
        self.c1 = c1
        self.c2 = c2
        self.cutoff_ratio = float(cutoff_ratio)
        self.max_latent_hw = max(4, int(max_latent_hw))
        self.fft_downsample = max(1, int(fft_downsample))
        self.temperature = max(float(temperature), 1e-3)
        self._hp_mask = None
        self.cross_modal_loss = None

        self.freq_gate = nn.Sequential(
            nn.Conv2d(c1, c1, 1, bias=False),
            nn.BatchNorm2d(c1),
            nn.Sigmoid(),
        )
        self.vis_proj = nn.Sequential(
            nn.Conv2d(c1, c2, 1, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True),
        )
        self.ir_proj = nn.Sequential(
            nn.Conv2d(c1, c2, 1, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True),
        )
        self.freq_proj = nn.Sequential(
            nn.Conv2d(2 * c1, c2, 1, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True),
        )

        self.rgb_encoder = nn.Sequential(
            nn.Conv2d(c2, latent_c, 1, bias=False),
            nn.BatchNorm2d(latent_c),
            nn.SiLU(inplace=True),
        )
        self.ir_encoder = nn.Sequential(
            nn.Conv2d(c2, latent_c, 1, bias=False),
            nn.BatchNorm2d(latent_c),
            nn.SiLU(inplace=True),
        )
        self.rgb_to_ir = nn.Sequential(
            nn.Conv2d(latent_c, latent_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(latent_c),
            nn.SiLU(inplace=True),
            nn.Conv2d(latent_c, latent_c, 1),
        )
        self.ir_to_rgb = nn.Sequential(
            nn.Conv2d(latent_c, latent_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(latent_c),
            nn.SiLU(inplace=True),
            nn.Conv2d(latent_c, latent_c, 1),
        )
        self.reliability_refine = nn.Sequential(
            nn.Conv2d(2 * c2 + c2, c2, 1, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True),
            nn.Conv2d(c2, c2, 3, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True),
        )

    def __getstate__(self):
        """Drop temporary tensors so YOLO EMA can deepcopy the model after AMP checks."""
        state = self.__dict__.copy()
        state["cross_modal_loss"] = None
        state["_hp_mask"] = None
        return state

    def _get_highpass_mask(self, h, w, device, dtype):
        if (
            self._hp_mask is None
            or self._hp_mask.shape[-2:] != (h, w)
            or self._hp_mask.device != device
            or self._hp_mask.dtype != dtype
        ):
            yy, xx = torch.meshgrid(
                torch.arange(h, device=device),
                torch.arange(w, device=device),
                indexing="ij",
            )
            cy = (h - 1) / 2.0
            cx = (w - 1) / 2.0
            radius = torch.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
            cutoff = max(1.0, self.cutoff_ratio * min(h, w) / 2.0)
            self._hp_mask = (radius >= cutoff).to(dtype)[None, None, :, :]
        return self._hp_mask

    def _highpass_spatial(self, x):
        b, c, h, w = x.shape
        x_dtype = x.dtype
        x_small = x
        if self.fft_downsample > 1 and min(h, w) >= self.fft_downsample * 2:
            x_small = F.avg_pool2d(x, kernel_size=self.fft_downsample, stride=self.fft_downsample)
        hs, ws = x_small.shape[-2:]
        spec = torch.fft.fftshift(torch.fft.fft2(x_small.float(), norm="ortho"), dim=(-2, -1))
        spec_high = spec * self._get_highpass_mask(hs, ws, spec.device, spec.dtype)
        x_high = torch.fft.ifft2(torch.fft.ifftshift(spec_high, dim=(-2, -1)), norm="ortho").real
        if (hs, ws) != (h, w):
            x_high = F.interpolate(x_high, size=(h, w), mode="bilinear", align_corners=False)
        return x_high.to(x_dtype)

    def _detail_enhance(self, x):
        x_high = self._highpass_spatial(x)
        gate = self.freq_gate(x_high)
        return x * (1.0 + gate), x_high

    def _latent(self, feat, encoder):
        if max(feat.shape[-2:]) > self.max_latent_hw:
            feat = F.adaptive_avg_pool2d(feat, (self.max_latent_hw, self.max_latent_hw))
        return encoder(feat)

    def forward(self, x):
        if not isinstance(x, (list, tuple)) or len(x) != 2:
            raise ValueError("CLPF expects a list/tuple of two feature maps: [vis, ir].")
        x_vis, x_ir = x
        if x_vis.shape[-2:] != x_ir.shape[-2:]:
            x_ir = F.interpolate(x_ir, size=x_vis.shape[-2:], mode="bilinear", align_corners=False)
        if x_vis.shape[1] != x_ir.shape[1]:
            raise ValueError(f"CLPF expects matched channels, got {x_vis.shape[1]} and {x_ir.shape[1]}.")

        vis_enh, vis_hf = self._detail_enhance(x_vis)
        ir_enh, ir_hf = self._detail_enhance(x_ir)
        f_vis = self.vis_proj(vis_enh)
        f_ir = self.ir_proj(ir_enh)
        f_freq = self.freq_proj(torch.cat([vis_hf, ir_hf], dim=1))

        z_rgb = self._latent(f_vis, self.rgb_encoder)
        z_ir = self._latent(f_ir, self.ir_encoder)
        z_ir_pred = self.rgb_to_ir(z_rgb)
        z_rgb_pred = self.ir_to_rgb(z_ir)

        err_rgb = (z_rgb_pred - z_rgb).pow(2).mean(dim=(1, 2, 3), keepdim=True)
        err_ir = (z_ir_pred - z_ir).pow(2).mean(dim=(1, 2, 3), keepdim=True)
        errors = torch.cat([err_rgb, err_ir], dim=1).flatten(1)
        weights = torch.softmax(-errors / self.temperature, dim=1).view(-1, 2, 1, 1)
        w_rgb, w_ir = weights[:, 0:1], weights[:, 1:2]

        # 记录权重和误差用于可视化分析
        self.recorded_w_rgb = w_rgb.detach()
        self.recorded_w_ir = w_ir.detach()
        self.recorded_err_rgb = err_rgb.detach()
        self.recorded_err_ir = err_ir.detach()
        self.recorded_z_rgb = z_rgb.detach()
        self.recorded_z_ir = z_ir.detach()
        self.recorded_z_rgb_pred = z_rgb_pred.detach()
        self.recorded_z_ir_pred = z_ir_pred.detach()

        if self.training:
            self.cross_modal_loss = F.mse_loss(z_ir_pred, z_ir.detach()) + F.mse_loss(z_rgb_pred, z_rgb.detach())
        else:
            self.cross_modal_loss = None

        fused = w_rgb * f_vis + w_ir * f_ir
        fused = self.reliability_refine(torch.cat([fused, f_vis, f_ir + f_freq], dim=1))
        return fused


class CLPFRes(CLPF):
    """
    CLPF with explicit modality-preserving residual path.

    Besides CLPF's predictive fusion, it adds a direct skip from original RGB/IR features so fine-grained
    modality details are less likely to be over-smoothed by cross-modal prediction.
    """

    def __init__(self, c1, c2=None, latent_ratio=0.25, cutoff_ratio=0.25, max_latent_hw=16, fft_downsample=2, temperature=1.0):
        super().__init__(
            c1=c1,
            c2=c2,
            latent_ratio=latent_ratio,
            cutoff_ratio=cutoff_ratio,
            max_latent_hw=max_latent_hw,
            fft_downsample=fft_downsample,
            temperature=temperature,
        )
        self.rgb_skip = (
            nn.Identity()
            if self.c1 == self.c2
            else nn.Sequential(
                nn.Conv2d(self.c1, self.c2, 1, bias=False),
                nn.BatchNorm2d(self.c2),
            )
        )
        self.ir_skip = (
            nn.Identity()
            if self.c1 == self.c2
            else nn.Sequential(
                nn.Conv2d(self.c1, self.c2, 1, bias=False),
                nn.BatchNorm2d(self.c2),
            )
        )
        self.skip_gain = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        fused = super().forward(x)
        x_vis, x_ir = x
        if x_vis.shape[-2:] != x_ir.shape[-2:]:
            x_ir = F.interpolate(x_ir, size=x_vis.shape[-2:], mode="bilinear", align_corners=False)
        skip = 0.5 * (self.rgb_skip(x_vis) + self.ir_skip(x_ir))
        return fused + self.skip_gain * skip


class MFSIGReg(nn.Module):
    """
    Modality-Fusion Sketched Isotropic Gaussian Regularization.

    Train-time regularizer for fused P2 features. The forward output is unchanged, while a SIGReg-style loss is stored
    in `reg_loss` and later collected by the detection loss.
    """

    def __init__(self, c1, proj_dim=64, loss_weight=0.005, eps=1e-4):
        super().__init__()
        self.c1 = c1
        self.loss_weight = float(loss_weight)
        self.eps = float(eps)
        self.reg_loss = None
        self.projector = nn.Sequential(
            nn.Conv2d(c1, int(proj_dim), 1, bias=False),
            nn.BatchNorm2d(int(proj_dim)),
            nn.SiLU(inplace=True),
        )

    def __getstate__(self):
        state = self.__dict__.copy()
        state["reg_loss"] = None
        return state

    def forward(self, x):
        if not self.training or self.loss_weight <= 0:
            self.reg_loss = None
            return x

        z = self.projector(x)
        z = z.flatten(2).transpose(1, 2).reshape(-1, z.shape[1])
        if z.shape[0] < 2:
            self.reg_loss = z.sum() * 0.0
            return x

        z = (z - z.mean(dim=0, keepdim=True)) / (z.std(dim=0, keepdim=True) + self.eps)
        cov = (z.T @ z) / (z.shape[0] - 1)
        eye = torch.eye(cov.shape[0], device=cov.device, dtype=cov.dtype)
        self.reg_loss = (cov - eye).pow(2).mean() * self.loss_weight
        return x


class SAAF(nn.Module):
    """
    Small-object Adaptive Attention Focus module for P2 features.

    It enhances regions with strong local contrast and high-frequency responses while preserving the original feature
    through residual-style modulation.
    """

    def __init__(self, c1, hidden_ratio=0.25):
        super().__init__()
        hidden = max(16, int(c1 * float(hidden_ratio)))
        self.edge = nn.Sequential(
            nn.Conv2d(c1, c1, 3, padding=1, groups=c1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
        )
        self.mask = nn.Sequential(
            nn.Conv2d(c1 * 2, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, c1, 1),
            nn.Sigmoid(),
        )
        self.refine = nn.Sequential(
            nn.Conv2d(c1, c1, 3, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        local_mean = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        local_contrast = (x - local_mean).abs()
        edge = self.edge(x)
        focus = self.mask(torch.cat([local_contrast, edge], dim=1))
        return self.refine(x * (1.0 + focus))


class FCM(nn.Module):
    """
    Feature Coupling Module for lightweight P3 small-object enhancement.

    The module splits channels into a fine-detail branch and a semantic branch, then uses spatial and channel gates
    to exchange information between them.
    """

    def __init__(self, c1, split_ratio=0.25):
        super().__init__()
        self.c1 = c1
        self.one = max(1, int(c1 * float(split_ratio)))
        self.two = c1 - self.one
        hidden = max(16, c1 // 4)

        self.detail = nn.Sequential(
            nn.Conv2d(self.one, self.one, 3, padding=1, bias=False),
            nn.BatchNorm2d(self.one),
            nn.SiLU(inplace=True),
            nn.Conv2d(self.one, self.one, 3, padding=1, bias=False),
            nn.BatchNorm2d(self.one),
            nn.SiLU(inplace=True),
            nn.Conv2d(self.one, c1, 1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
        )
        self.semantic = nn.Sequential(
            nn.Conv2d(self.two, c1, 1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(c1, 1, 7, padding=3, bias=False),
            nn.Sigmoid(),
        )
        self.channel = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c1, hidden, 1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, c1, 1, bias=False),
            nn.Sigmoid(),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(c1, c1, 1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        x1, x2 = torch.split(x, [self.one, self.two], dim=1)
        detail = self.detail(x1)
        semantic = self.semantic(x2)
        coupled = self.spatial(semantic) * detail + self.channel(detail) * semantic
        return self.fuse(coupled) + x


class HCFM(nn.Module):
    """
    High-frequency Contextual Focus Module.

    This improves FCM with local-contrast cues, multi-dilation depthwise context, and residual gated refinement for P3
    small-object detection.
    """

    def __init__(self, c1, split_ratio=0.25, hidden_ratio=0.25):
        super().__init__()
        self.fcm = FCM(c1, split_ratio)
        hidden = max(16, int(c1 * float(hidden_ratio)))

        self.edge = nn.Sequential(
            nn.Conv2d(c1, c1, 3, padding=1, groups=c1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
        )
        self.context1 = nn.Conv2d(c1, c1, 3, padding=1, groups=c1, bias=False)
        self.context2 = nn.Conv2d(c1, c1, 3, padding=2, dilation=2, groups=c1, bias=False)
        self.context3 = nn.Conv2d(c1, c1, 3, padding=3, dilation=3, groups=c1, bias=False)
        self.context_mix = nn.Sequential(
            nn.Conv2d(c1 * 3, c1, 1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
        )
        self.focus_gate = nn.Sequential(
            nn.Conv2d(c1 * 3, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, c1, 1),
            nn.Sigmoid(),
        )
        self.out = nn.Sequential(
            nn.Conv2d(c1 * 2, c1, 1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
            nn.Conv2d(c1, c1, 3, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
        )
        self.gamma = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        fcm = self.fcm(x)
        local_contrast = (x - F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)).abs()
        edge = self.edge(x)
        context = self.context_mix(torch.cat([self.context1(x), self.context2(x), self.context3(x)], dim=1))
        gate = self.focus_gate(torch.cat([local_contrast, edge, context], dim=1))
        enhanced = self.out(torch.cat([fcm, context * (1.0 + gate)], dim=1))
        return x + self.gamma * enhanced
