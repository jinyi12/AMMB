"""Core building blocks for Glow-based normalizing flows."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConcatELU(nn.Module):
    """Applies ELU to both x and -x, then concatenates along channel dim."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([F.elu(x), F.elu(-x)], dim=1)


class LayerNormChannels(nn.Module):
    """Layer norm that operates channel-wise for image tensors."""

    def __init__(self, c_in: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1, c_in, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, c_in, 1, 1))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, unbiased=False, keepdim=True)
        y = (x - mean) / torch.sqrt(var + self.eps)
        return y * self.gamma + self.beta


class GatedConv(nn.Module):
    """Residual gated convolutional block."""

    def __init__(self, c_in: int, c_hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            ConcatELU(),
            nn.Conv2d(c_in * 2, c_hidden, kernel_size=5, padding=2),
            ConcatELU(),
            nn.Conv2d(c_hidden * 2, c_in * 2, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        val, gate = out.chunk(2, dim=1)
        return x + val * torch.sigmoid(gate)


class GatedConvNet(nn.Module):
    """Stack of gated convolutional blocks used inside coupling layers."""

    def __init__(
        self,
        c_in: int,
        c_hidden: int = 32,
        c_out: int = -1,
        num_layers: int = 3,
    ) -> None:
        super().__init__()
        c_out = c_out if c_out > 0 else 2 * c_in
        layers = [nn.Conv2d(c_in, c_hidden, kernel_size=5, padding=2)]
        for _ in range(num_layers):
            layers.extend([GatedConv(c_hidden, c_hidden), LayerNormChannels(c_hidden)])
        layers.extend([ConcatELU(), nn.Conv2d(c_hidden * 2, c_out, kernel_size=3, padding=1)])
        self.nn = nn.Sequential(*layers)
        self.nn[-1].weight.data.zero_()
        self.nn[-1].bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.nn(x)


class ActNorm(nn.Module):
    """Activation Normalization layer with clamped log-scale."""

    def __init__(self, num_features: int, eps: float = 1e-6, log_scale_clamp: float = 5.0) -> None:
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.log_scale_clamp = log_scale_clamp
        self.register_buffer("initialized", torch.tensor(0, dtype=torch.uint8))
        self.log_scale = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_features, 1, 1))

    def forward(self, x: torch.Tensor, log_det_jac: torch.Tensor, reverse: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.initialized:
            with torch.no_grad():
                mean = torch.mean(x, dim=[0, 2, 3], keepdim=True)
                std = torch.std(x, dim=[0, 2, 3], keepdim=True)
                std_clamped = torch.clamp(std, min=1e-3)
                log_scale_init = torch.log(1 / (std_clamped + self.eps))
                log_scale_init = torch.clamp(log_scale_init, -self.log_scale_clamp, self.log_scale_clamp)
                self.bias.data.copy_(-mean)
                self.log_scale.data.copy_(log_scale_init)
                self.initialized.fill_(1)

        log_scale = torch.clamp(self.log_scale, -self.log_scale_clamp, self.log_scale_clamp)
        bias = self.bias
        H, W = x.shape[2], x.shape[3]

        if not reverse:
            x = torch.exp(log_scale) * (x + bias)
            d_log_det = torch.sum(log_scale) * H * W
            log_det_jac = log_det_jac + d_log_det
        else:
            x = x * torch.exp(-log_scale) - bias
            d_log_det = torch.sum(log_scale) * H * W
            log_det_jac = log_det_jac - d_log_det
        return x, log_det_jac


class Invertible1x1Conv(nn.Module):
    """PLU-parameterised invertible 1x1 convolution."""

    def __init__(self, num_channels: int, log_s_clamp: float = 5.0) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.log_s_clamp = log_s_clamp

        W_init = torch.randn(num_channels, num_channels)
        Q, _ = torch.linalg.qr(W_init)
        P, L, U = torch.linalg.lu(Q)

        self.register_buffer("P", P)
        self.register_buffer("sign_s", torch.sign(torch.diag(U)))

        self.L = nn.Parameter(L)
        self.log_s = nn.Parameter(torch.log(torch.abs(torch.diag(U)) + 1e-6))
        self.U = nn.Parameter(torch.triu(U, diagonal=1))

        self.register_buffer("L_mask", torch.tril(torch.ones(num_channels, num_channels), -1))
        self.register_buffer("I", torch.eye(num_channels))

    def _get_clamped_log_s(self) -> torch.Tensor:
        return torch.clamp(self.log_s, -self.log_s_clamp, self.log_s_clamp)

    def _get_weight(self, reverse: bool) -> torch.Tensor:
        L = self.L * self.L_mask + self.I
        log_s_clamped = self._get_clamped_log_s()
        U = self.U * self.L_mask.T + torch.diag(self.sign_s * torch.exp(log_s_clamped))

        if not reverse:
            W = self.P @ L @ U
        else:
            try:
                W = torch.linalg.inv(U) @ torch.linalg.inv(L) @ self.P.T
            except torch.linalg.LinAlgError:
                W_fwd = self.P @ L @ U
                W = torch.linalg.inv(W_fwd)
        return W.view(self.num_channels, self.num_channels, 1, 1)

    def forward(self, x: torch.Tensor, log_det_jac: torch.Tensor, reverse: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        W = self._get_weight(reverse)
        x = F.conv2d(x, W)
        log_s_clamped = self._get_clamped_log_s()
        d_log_det = log_s_clamped.sum() * x.shape[2] * x.shape[3]
        if reverse:
            d_log_det *= -1
        log_det_jac = log_det_jac + d_log_det
        return x, log_det_jac


class TimeDependentActNorm(nn.Module):
    """ActNorm variant that is identity when t = 0."""

    def __init__(self, num_features: int, log_scale_clamp: float = 5.0) -> None:
        super().__init__()
        self.log_scale_clamp = log_scale_clamp
        self.log_scale_params = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.bias_params = nn.Parameter(torch.zeros(1, num_features, 1, 1))

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        log_det_jac: torch.Tensor,
        reverse: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        time_scale = t.view(-1, 1, 1, 1)
        log_scale = torch.clamp(time_scale * self.log_scale_params, -self.log_scale_clamp, self.log_scale_clamp)
        bias = time_scale * self.bias_params
        H, W = x.shape[2], x.shape[3]

        if not reverse:
            x = torch.exp(log_scale) * (x + bias)
            d_log_det = torch.sum(log_scale, dim=[1, 2, 3]) * H * W
            log_det_jac = log_det_jac + d_log_det
        else:
            x = x * torch.exp(-log_scale) - bias
            d_log_det = torch.sum(log_scale, dim=[1, 2, 3]) * H * W
            log_det_jac = log_det_jac - d_log_det
        return x, log_det_jac


class TimeDependentInvertible1x1Conv(Invertible1x1Conv):
    """Invertible 1x1 conv that smoothly deviates from identity as t increases."""

    def __init__(self, num_channels: int, log_s_clamp: float = 5.0) -> None:
        super().__init__(num_channels, log_s_clamp)
        self.P.data.copy_(torch.eye(num_channels))
        self.sign_s.data.fill_(1.0)
        self.L.data.zero_()
        self.log_s.data.zero_()
        self.U.data.zero_()

    def _get_weight(self, t: torch.Tensor, reverse: bool) -> torch.Tensor:
        time_scale = t.view(-1, 1, 1)
        L_scaled = (self.L * time_scale) * self.L_mask + self.I
        log_s_scaled = self.log_s * time_scale.squeeze(-1)
        U_scaled = (self.U * time_scale) * self.L_mask.T + torch.diag_embed(self.sign_s * torch.exp(log_s_scaled))
        if not reverse:
            W = self.P @ L_scaled @ U_scaled
        else:
            W = torch.linalg.inv(U_scaled) @ torch.linalg.inv(L_scaled) @ self.P.T
        return W.view(-1, self.num_channels, self.num_channels, 1, 1)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        log_det_jac: torch.Tensor,
        reverse: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, channels, height, width = x.shape
        if t.shape[0] != batch:
            t = t.expand(batch, -1)
        weights = self._get_weight(t, reverse)
        x_reshaped = x.view(1, batch * channels, height, width)
        weights_reshaped = weights.view(batch * channels, channels, 1, 1)
        out = F.conv2d(x_reshaped, weights_reshaped, groups=batch).view(batch, channels, height, width)
        time_scale = t.view(-1, 1)
        log_s_scaled = self.log_s.unsqueeze(0) * time_scale
        d_log_det = log_s_scaled.sum(dim=1) * height * width
        if reverse:
            log_det_jac = log_det_jac - d_log_det
        else:
            log_det_jac = log_det_jac + d_log_det
        return out, log_det_jac


class TimeCondAffineCoupling(nn.Module):
    """Time-conditioned affine coupling layer."""

    def __init__(self, in_channels: int, hidden_dim: int, scale_clamp: float = 2.0) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.scale_clamp = scale_clamp
        input_channels = in_channels // 2 + 1
        output_channels = (in_channels // 2) * 2
        self.net = nn.Sequential(
            nn.utils.weight_norm(nn.Conv2d(input_channels, hidden_dim, kernel_size=3, padding=1)),
            nn.ReLU(inplace=True),
            nn.utils.weight_norm(nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)),
            nn.ReLU(inplace=True),
            nn.utils.weight_norm(nn.Conv2d(hidden_dim, output_channels, kernel_size=3, padding=1)),
        )
        self.net[-1].weight.data.zero_()
        self.net[-1].bias.data.zero_()

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        log_det_jac: torch.Tensor,
        reverse: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if x.shape[0] == 0:
            return x, log_det_jac
        x_a, x_b = x.chunk(2, dim=1)
        if t.shape[0] != x.shape[0]:
            if t.shape[0] == 1:
                t = t.expand(x.shape[0], -1)
            else:
                raise ValueError("Time tensor batch size mismatch")
        t_channel = t.view(-1, 1, 1, 1).expand(-1, 1, x_b.shape[2], x_b.shape[3])
        net_in = torch.cat([x_b, t_channel], dim=1)
        s_m = self.net(net_in)
        log_s_raw, m = s_m.chunk(2, dim=1)
        s = torch.tanh(log_s_raw) * self.scale_clamp
        if not reverse:
            y_a = torch.exp(s) * x_a + m
            log_det_jac = log_det_jac + s.sum(dim=[1, 2, 3])
        else:
            y_a = (x_a - m) * torch.exp(-s)
            log_det_jac = log_det_jac - s.sum(dim=[1, 2, 3])
        y = torch.cat([y_a, x_b], dim=1)
        return y, log_det_jac


def squeeze(x: torch.Tensor) -> torch.Tensor:
    batch, channels, height, width = x.shape
    x = x.view(batch, channels, height // 2, 2, width // 2, 2)
    x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
    return x.view(batch, channels * 4, height // 2, width // 2)


def unsqueeze(x: torch.Tensor) -> torch.Tensor:
    batch, channels, height, width = x.shape
    x = x.view(batch, channels // 4, 2, 2, height, width)
    x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
    return x.view(batch, channels // 4, height * 2, width * 2)
