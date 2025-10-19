"""Decoupled Glow architecture reused in the Lightning refactor."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Normal

from .affine_layers import AffineCoupling, NeuralFlowCoupling
from .glow_layers import ActNorm, Invertible1x1Conv, squeeze, unsqueeze


def spatial_data_conversion(data: Tensor, resolution: int, to_spatial: bool = True) -> Tensor:
    """Convert flattened tensors to spatial layout and back."""
    if to_spatial:
        batch_size = data.shape[0]
        data_dim = data.shape[1]
        spatial_dim = resolution * resolution
        num_channels = data_dim // spatial_dim
        if data_dim != num_channels * spatial_dim:
            raise ValueError(f"Data dimension {data_dim} incompatible with resolution {resolution}")
        return data.view(batch_size, num_channels, resolution, resolution)
    batch_size = data.shape[0]
    return data.view(batch_size, -1)


class FlowStepStatic(nn.Module):
    """Single Glow step for the time-independent density model."""

    def __init__(self, num_channels: int, hidden_dim: int) -> None:
        super().__init__()
        self.act_norm = ActNorm(num_channels)
        self.inv_conv = Invertible1x1Conv(num_channels)
        self.affine_coupling = AffineCoupling(num_channels, hidden_dim)

    def forward(self, x: Tensor, log_det_jac: Tensor, reverse: bool = False) -> tuple[Tensor, Tensor]:
        if not reverse:
            x, log_det_jac = self.act_norm(x, log_det_jac, reverse=False)
            x, log_det_jac = self.inv_conv(x, log_det_jac, reverse=False)
            x, log_det_jac = self.affine_coupling(x, log_det_jac, reverse=False)
        else:
            x, log_det_jac = self.affine_coupling(x, log_det_jac, reverse=True)
            x, log_det_jac = self.inv_conv(x, log_det_jac, reverse=True)
            x, log_det_jac = self.act_norm(x, log_det_jac, reverse=True)
        return x, log_det_jac


class FlowScaleStatic(nn.Module):
    """Sequence of static Glow steps at a single scale."""

    def __init__(self, num_channels: int, hidden_dim: int, num_steps: int) -> None:
        super().__init__()
        self.steps = nn.ModuleList([FlowStepStatic(num_channels, hidden_dim) for _ in range(num_steps)])

    def forward(self, x: Tensor, log_det_jac: Tensor, reverse: bool = False) -> tuple[Tensor, Tensor]:
        flow_steps: Iterable[nn.Module]
        flow_steps = self.steps if not reverse else reversed(self.steps)
        for step in flow_steps:
            x, log_det_jac = step(x, log_det_jac, reverse=reverse)
            x = torch.clamp(x, -1e5, 1e5)
            if torch.isnan(x).any() or torch.isinf(x).any():
                x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        return x, log_det_jac


class StaticGlow(nn.Module):
    """Static Glow model G_phi used for the initial density."""

    def __init__(self, input_shape: Tuple[int, int, int], hidden_dim: int, n_blocks_flow: int, num_scales: int = 2) -> None:
        super().__init__()
        self.num_scales = num_scales
        self.scales = nn.ModuleList()
        channels, height, width = input_shape
        self.latent_shapes: list[Tuple[int, int, int]] = []

        current_c, current_h, current_w = channels, height, width
        for i in range(num_scales):
            if current_h < 2 or current_w < 2:
                self.num_scales = i
                break
            current_c *= 4
            current_h //= 2
            current_w //= 2
            self.scales.append(FlowScaleStatic(current_c, hidden_dim, n_blocks_flow))
            if i < num_scales - 1:
                self.latent_shapes.append((current_c // 2, current_h, current_w))
                current_c = current_c // 2
            else:
                self.latent_shapes.append((current_c, current_h, current_w))

        self.register_buffer("base_mean", torch.zeros(1))
        self.register_buffer("base_std", torch.ones(1))
        self.total_latent_dim = sum(int(np.prod(shape)) for shape in self.latent_shapes)

    @property
    def base_dist(self) -> Normal:
        return Normal(self.base_mean, self.base_std)

    def forward(self, data: Tensor) -> tuple[Tensor, Tensor]:
        z = data
        log_det = torch.zeros(z.shape[0], device=z.device)
        latents: list[Tensor] = []
        for i in range(self.num_scales):
            z = squeeze(z)
            z, log_det = self.scales[i](z, log_det, reverse=False)
            if i < self.num_scales - 1:
                z_this_scale, z = z.chunk(2, dim=1)
                latents.append(z_this_scale)
        latents.append(z)
        flat_latents = [torch.flatten(latent, start_dim=1) for latent in latents]
        epsilon = torch.cat(flat_latents, dim=1)
        return epsilon, log_det

    def inverse(self, epsilon: Tensor) -> tuple[Tensor, Tensor]:
        batch = epsilon.shape[0]
        if epsilon.dim() == 4:
            epsilon = epsilon.view(batch, -1)
        latents: list[Tensor] = []
        cursor = 0
        for shape in self.latent_shapes:
            num_elements = int(np.prod(shape))
            if cursor + num_elements > epsilon.shape[1]:
                padding = cursor + num_elements - epsilon.shape[1]
                epsilon = torch.cat([epsilon, torch.zeros(batch, padding, device=epsilon.device)], dim=1)
            flat_latent = epsilon[:, cursor : cursor + num_elements]
            latents.append(flat_latent.view(batch, *shape))
            cursor += num_elements

        z = latents.pop()
        log_det = torch.zeros(z.shape[0], device=z.device)
        for i in reversed(range(self.num_scales)):
            if i < self.num_scales - 1:
                z = torch.cat([latents.pop(), z], dim=1)
            z, log_det = self.scales[i](z, log_det, reverse=True)
            z = unsqueeze(z)
        return z, log_det

    def log_prob(self, data: Tensor) -> Tensor:
        epsilon, log_det_inv = self.forward(data)
        if epsilon.dim() != 2:
            epsilon = torch.flatten(epsilon, start_dim=1)
        epsilon = torch.nan_to_num(epsilon, nan=0.0, posinf=0.0, neginf=0.0)
        log_det_inv = torch.nan_to_num(log_det_inv, nan=0.0)
        base_log_prob = Normal(0, 1).log_prob(epsilon).sum(dim=1)
        return base_log_prob + log_det_inv


class EfficientFlowStep(nn.Module):
    """Single flow step for the dynamics model."""

    def __init__(self, num_channels: int, hidden_dim: int, reverse_split: bool = False) -> None:
        super().__init__()
        if num_channels % 2 != 0:
            raise ValueError("Number of channels must be even for splitting")
        self.affine_coupling = NeuralFlowCoupling(num_channels, hidden_dim, reverse_split=reverse_split)

    def forward(self, x: Tensor, t: Tensor, log_det_jac: Tensor, reverse: bool = False) -> tuple[Tensor, Tensor]:
        return self.affine_coupling(x, t, log_det_jac, reverse=reverse)


class InvertibleNeuralFlow(nn.Module):
    """Transport map T_theta implemented as an invertible neural flow."""

    def __init__(self, input_shape: Tuple[int, int, int], hidden_dim: int, n_blocks_flow: int, num_scales: int = 2) -> None:
        super().__init__()
        channels, height, width = input_shape
        channels_squeezed = channels * 4
        self.flow_steps = nn.ModuleList(
            [EfficientFlowStep(channels_squeezed, hidden_dim, reverse_split=i % 2 == 1) for i in range(n_blocks_flow)]
        )

    def _ensure_time_batch(self, t: Tensor | float, batch_size: int, device: torch.device) -> Tensor:
        if not torch.is_tensor(t):
            t = torch.tensor(float(t), device=device)
        if t.dim() == 0:
            t = t.view(1, 1).expand(batch_size, 1)
        elif t.shape[0] == 1:
            t = t.expand(batch_size, *t.shape[1:])
        if t.dim() == 1:
            t = t.unsqueeze(1)
        return t

    def forward(self, x_0: Tensor, t: Tensor | float) -> tuple[Tensor, Tensor]:
        t_batch = self._ensure_time_batch(t, x_0.shape[0], x_0.device)
        log_det = torch.zeros(x_0.shape[0], device=x_0.device)
        z = squeeze(x_0)
        for step in self.flow_steps:
            z, log_det = step(z, t_batch, log_det, reverse=False)
        x_t = unsqueeze(z)
        return x_t, log_det

    def inverse(self, x_t: Tensor, t: Tensor | float) -> tuple[Tensor, Tensor]:
        t_batch = self._ensure_time_batch(t, x_t.shape[0], x_t.device)
        log_det = torch.zeros(x_t.shape[0], device=x_t.device)
        z = squeeze(x_t)
        for step in reversed(self.flow_steps):
            z, log_det = step(z, t_batch, log_det, reverse=True)
        x_0 = unsqueeze(z)
        return x_0, log_det


def _t_dir(func, t: Tensor, create_graph: Optional[bool] = None):
    if create_graph is None:
        create_graph = torch.is_grad_enabled()
    return torch.autograd.functional.jvp(func, t, torch.ones_like(t), create_graph=create_graph)


@dataclass
class DecoupledBridgeConfig:
    data_dim: int
    hidden_size: int
    resolution: int
    n_blocks_flow: int = 2
    num_scales: int = 2
    T: float = 1.0
    sigma_reverse: float = 0.5
    training_noise_std: float = 1e-4
    score_clamp_norm: float = 100.0


class DecoupledBridge(nn.Module):
    """Container wrapping the density (G_phi) and dynamics (T_theta) models."""

    def __init__(
        self,
        data_dim: int,
        hidden_size: int,
        resolution: int,
        n_blocks_flow: int = 2,
        num_scales: int = 2,
        T: float = 1.0,
        sigma_reverse: float = 0.5,
        data_mean: Optional[Tensor] = None,
        data_std: Optional[Tensor] = None,
        training_noise_std: float = 1e-4,
        score_clamp_norm: float = 100.0,
    ) -> None:
        super().__init__()
        self.data_dim = data_dim
        self.resolution = resolution
        self.T = T
        self.sigma_reverse = sigma_reverse
        self.training_noise_std = training_noise_std
        self.score_clamp_norm = score_clamp_norm

        spatial_dim = resolution * resolution
        self.num_channels = data_dim // spatial_dim
        if data_dim != self.num_channels * spatial_dim:
            raise ValueError(f"Data dimension {data_dim} incompatible with resolution {resolution}")
        self.input_shape = (self.num_channels, resolution, resolution)

        self.density_model = StaticGlow(
            input_shape=self.input_shape,
            hidden_dim=hidden_size,
            n_blocks_flow=n_blocks_flow,
            num_scales=num_scales,
        )
        self.dynamics_model = InvertibleNeuralFlow(
            input_shape=self.input_shape,
            hidden_dim=hidden_size,
            n_blocks_flow=n_blocks_flow,
            num_scales=num_scales,
        )

        if data_mean is None:
            data_mean = torch.zeros(data_dim)
        if data_std is None:
            data_std = torch.ones(data_dim)
        self.register_buffer("data_mean", data_mean)
        self.register_buffer("data_std", data_std)

    def update_normalization(self, mean: Tensor, std: Tensor) -> None:
        with torch.no_grad():
            self.data_mean.copy_(mean)
            self.data_std.copy_(std)

    def normalize(self, x: Tensor) -> Tensor:
        return (x - self.data_mean) / self.data_std

    def denormalize(self, z: Tensor) -> Tensor:
        return z * self.data_std + self.data_mean

    def _format_time(self, t: Tensor | float, batch_size: int) -> Tensor:
        if not torch.is_tensor(t):
            t = torch.tensor(float(t), device=self.data_mean.device)
        if t.dim() == 0:
            t = t.view(1, 1).expand(batch_size, 1)
        elif t.shape[0] == 1:
            t = t.expand(batch_size, *t.shape[1:])
        if t.dim() == 1:
            t = t.unsqueeze(1)
        if t.dim() == 2 and t.shape[0] == batch_size and t.shape[1] == 1:
            return t
        raise ValueError(f"Time tensor shape {t.shape} incompatible with batch size {batch_size}")

    def _apply_noise_if_training(self, x: Tensor) -> Tensor:
        if self.training and self.training_noise_std > 0:
            noise = torch.randn_like(x) * self.training_noise_std
            return x + noise
        return x

    def sample_initial(self, n_samples: int, device: torch.device | str = "cpu") -> Tensor:
        epsilon = torch.randn(n_samples, self.data_dim, device=device)
        epsilon_spatial = spatial_data_conversion(epsilon, self.resolution, to_spatial=True)
        x0_spatial, _ = self.density_model.inverse(epsilon_spatial)
        return spatial_data_conversion(x0_spatial, self.resolution, to_spatial=False)

    def transport(self, x0: Tensor, t: Tensor | float) -> Tensor:
        t_batch = self._format_time(t, x0.shape[0])
        x0_spatial = spatial_data_conversion(x0, self.resolution, to_spatial=True)
        xt_spatial, _ = self.dynamics_model.forward(x0_spatial, t_batch)
        return spatial_data_conversion(xt_spatial, self.resolution, to_spatial=False)

    def inverse_transport(self, xt: Tensor, t: Tensor | float) -> Tensor:
        t_batch = self._format_time(t, xt.shape[0])
        xt_spatial = spatial_data_conversion(xt, self.resolution, to_spatial=True)
        x0_spatial, _ = self.dynamics_model.inverse(xt_spatial, t_batch)
        return spatial_data_conversion(x0_spatial, self.resolution, to_spatial=False)

    def log_prob_initial(self, x0: Tensor) -> Tensor:
        x0_spatial = spatial_data_conversion(x0, self.resolution, to_spatial=True)
        return self.density_model.log_prob(x0_spatial)

    def log_prob_pushforward(self, xt: Tensor, t: Tensor | float) -> Tensor:
        t_batch = self._format_time(t, xt.shape[0])
        xt_spatial = spatial_data_conversion(xt, self.resolution, to_spatial=True)
        x0_spatial, log_det_inv = self.dynamics_model.inverse(xt_spatial, t_batch)
        log_p0 = self.density_model.log_prob(x0_spatial)
        return log_p0 + log_det_inv

    def forward_velocity(self, xt: Tensor, t: Tensor | float) -> Tensor:
        t_batch = self._format_time(t, xt.shape[0])
        x0 = self.inverse_transport(xt, t_batch)
        x0_detached = x0.detach()

        def transport_fixed(time_tensor: Tensor) -> Tensor:
            return self.transport(x0_detached, time_tensor)

        grad_mode = torch.is_grad_enabled()
        torch.set_grad_enabled(True)
        try:
            velocity_result = _t_dir(transport_fixed, t_batch, create_graph=grad_mode and self.training)
            velocity = velocity_result[1] if isinstance(velocity_result, tuple) else velocity_result
        finally:
            torch.set_grad_enabled(grad_mode)
        return velocity

    def score_function(self, xt: Tensor, t: Tensor | float) -> Tensor:
        t_batch = self._format_time(t, xt.shape[0])
        xt_grad = xt.detach().clone().requires_grad_(True)
        grad_mode = torch.is_grad_enabled()
        torch.set_grad_enabled(True)
        try:
            log_pt = self.log_prob_pushforward(xt_grad, t_batch)
            score = torch.autograd.grad(log_pt.sum(), xt_grad, retain_graph=False, create_graph=False)[0]
        finally:
            torch.set_grad_enabled(grad_mode)
        score = score.detach()
        if torch.isnan(score).any() or torch.isinf(score).any():
            score = torch.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
        if self.score_clamp_norm is not None and self.score_clamp_norm > 0:
            max_norm = self.score_clamp_norm
            score_norm = torch.norm(score.view(score.shape[0], -1), p=2, dim=1)
            mask = score_norm > max_norm
            if mask.any():
                scale = max_norm / (score_norm[mask] + 1e-6)
                score[mask] = score[mask] * scale.view(-1, 1)
        return score

    def reverse_drift(self, xt: Tensor, t: Tensor | float) -> Tensor:
        velocity = self.forward_velocity(xt, t)
        score = self.score_function(xt, t)
        return velocity - (self.sigma_reverse**2 / 2) * score

    @property
    def base_dist(self):
        return self.density_model.base_dist

    @property
    def flow(self):
        return self.dynamics_model

    def get_params(self, t: Tensor) -> tuple[Tensor, Tensor]:
        batch = t.shape[0] if t.dim() > 0 else 1
        device = t.device if torch.is_tensor(t) else self.data_mean.device
        mu = self.data_mean.unsqueeze(0).expand(batch, -1).to(device)
        gamma = torch.ones_like(mu)
        return mu, gamma
