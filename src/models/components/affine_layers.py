"""Affine coupling layers used by the decoupled Glow architecture."""

from __future__ import annotations

import torch
import torch.nn as nn

from .glow_layers import GatedConvNet


def create_checkerboard_mask(height: int, width: int, invert: bool = False) -> torch.Tensor:
    """Generate a checkerboard mask for spatial coupling."""
    x = torch.arange(height, dtype=torch.int32)
    y = torch.arange(width, dtype=torch.int32)
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    mask = torch.fmod(xx + yy, 2).to(torch.float32).view(1, 1, height, width)
    if invert:
        mask = 1 - mask
    return mask


def create_channel_mask(num_channels: int, invert: bool = False) -> torch.Tensor:
    """Create a channel-wise mask for splitting operations."""
    mask = torch.cat(
        [
            torch.ones(num_channels // 2, dtype=torch.float32),
            torch.zeros(num_channels - num_channels // 2, dtype=torch.float32),
        ]
    ).view(1, num_channels, 1, 1)
    if invert:
        mask = 1 - mask
    return mask


class AffineCoupling(nn.Module):
    """Stabilised affine coupling layer for the density model."""

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        mask: torch.Tensor | None = None,
        num_layers: int = 3,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.use_mask = mask is not None

        if self.use_mask:
            self.register_buffer("mask", mask)
            net_in_channels = in_channels
            net_out_channels = in_channels * 2
            scale_channels = in_channels
        else:
            net_in_channels = in_channels // 2
            net_out_channels = (in_channels // 2) * 2
            scale_channels = in_channels // 2

        self.net = GatedConvNet(
            c_in=net_in_channels,
            c_hidden=hidden_dim,
            c_out=net_out_channels,
            num_layers=num_layers,
        )
        self.log_scaling_factor = nn.Parameter(torch.zeros(scale_channels, 1, 1))

    def forward(self, x: torch.Tensor, log_det_jac: torch.Tensor, reverse: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        if x.shape[0] == 0:
            return x, log_det_jac

        if self.use_mask:
            x_masked = x * self.mask
            s_m = self.net(x_masked)
            log_s_raw, m = s_m.chunk(2, dim=1)
            s_fac = torch.exp(torch.clamp(self.log_scaling_factor, -3.0, 3.0))
            s = torch.tanh(log_s_raw / (s_fac + 1e-6)) * s_fac
            s = s * (1 - self.mask)
            m = m * (1 - self.mask)
            if not reverse:
                y = (x + m) * torch.exp(s)
                log_det_jac = log_det_jac + s.sum(dim=[1, 2, 3])
            else:
                y = (x * torch.exp(-s)) - m
                log_det_jac = log_det_jac - s.sum(dim=[1, 2, 3])
            return y, log_det_jac

        x_a, x_b = x.chunk(2, dim=1)
        s_m = self.net(x_b)
        log_s_raw, m = s_m.chunk(2, dim=1)
        s_fac = torch.exp(torch.clamp(self.log_scaling_factor, -3.0, 3.0))
        s = torch.tanh(log_s_raw / (s_fac + 1e-6)) * s_fac
        if not reverse:
            y_a = torch.exp(s) * x_a + m
            log_det_jac = log_det_jac + s.sum(dim=[1, 2, 3])
        else:
            y_a = (x_a - m) * torch.exp(-s)
            log_det_jac = log_det_jac - s.sum(dim=[1, 2, 3])
        y = torch.cat([y_a, x_b], dim=1)
        return y, log_det_jac


class NeuralFlowCoupling(nn.Module):
    """Time-conditioned affine coupling used by the dynamics model."""

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        mask: torch.Tensor | None = None,
        num_layers: int = 3,
        reverse_split: bool = False,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.use_mask = mask is not None
        self.reverse_split = reverse_split

        if self.use_mask:
            self.register_buffer("mask", mask)
            net_in_channels = in_channels + 1
            net_out_channels = in_channels * 2
            scale_channels = in_channels
        else:
            net_in_channels = in_channels // 2 + 1
            net_out_channels = (in_channels // 2) * 2
            scale_channels = in_channels // 2

        self.net = GatedConvNet(
            c_in=net_in_channels,
            c_hidden=hidden_dim,
            c_out=net_out_channels,
            num_layers=num_layers,
        )
        self.log_scaling_factor = nn.Parameter(torch.zeros(scale_channels, 1, 1))

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        log_det_jac: torch.Tensor,
        reverse: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if x.shape[0] == 0:
            return x, log_det_jac

        if t.shape[0] != x.shape[0]:
            if t.shape[0] == 1:
                t = t.expand(x.shape[0], *t.shape[1:])
            else:
                raise ValueError("Time tensor batch size mismatch")
        time_scale = t.view(-1, 1, 1, 1)

        if self.use_mask:
            x_masked = x * self.mask
            t_channel = time_scale.expand(-1, 1, x.shape[2], x.shape[3])
            net_in = torch.cat([x_masked, t_channel], dim=1)
            s_m = self.net(net_in)
            log_s_raw, m = s_m.chunk(2, dim=1)
            s_fac = torch.exp(torch.clamp(self.log_scaling_factor, -3.0, 3.0))
            s = torch.tanh(log_s_raw / (s_fac + 1e-6)) * s_fac
            s = s * time_scale
            m = m * time_scale
            s = s * (1 - self.mask)
            m = m * (1 - self.mask)
            if not reverse:
                y = (x + m) * torch.exp(s)
                log_det_jac = log_det_jac + s.sum(dim=[1, 2, 3])
            else:
                y = (x * torch.exp(-s)) - m
                log_det_jac = log_det_jac - s.sum(dim=[1, 2, 3])
            return y, log_det_jac

        x_a, x_b = x.chunk(2, dim=1)
        x_trans, x_cond = (x_a, x_b) if not self.reverse_split else (x_b, x_a)
        t_channel = time_scale.expand(-1, 1, x_cond.shape[2], x_cond.shape[3])
        net_in = torch.cat([x_cond, t_channel], dim=1)
        s_m = self.net(net_in)
        log_s_raw, m = s_m.chunk(2, dim=1)
        s_fac = torch.exp(torch.clamp(self.log_scaling_factor, -3.0, 3.0))
        s = torch.tanh(log_s_raw / (s_fac + 1e-6)) * s_fac
        s = s * time_scale
        m = m * time_scale
        if not reverse:
            y_trans = torch.exp(s) * x_trans + m
            log_det_jac = log_det_jac + s.sum(dim=[1, 2, 3])
        else:
            y_trans = (x_trans - m) * torch.exp(-s)
            log_det_jac = log_det_jac - s.sum(dim=[1, 2, 3])
        if not self.reverse_split:
            y_a, y_b = y_trans, x_cond
        else:
            y_a, y_b = x_cond, y_trans
        y = torch.cat([y_a, y_b], dim=1)
        return y, log_det_jac
