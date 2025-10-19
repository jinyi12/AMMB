"""Glow network wrapper exposing the sampling utilities required by Lightning."""

from __future__ import annotations

from typing import Sequence, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from .decoupled_glow import StaticGlow


class GlowNetwork(nn.Module):
    """Thin wrapper around :class:`StaticGlow` adding helper APIs for sampling."""

    def __init__(
        self,
        input_dims: Sequence[int],
        hidden_size: int,
        n_blocks_flow: int,
        num_scales: int = 2,
    ) -> None:
        super().__init__()
        self.input_dims: Tuple[int, int, int] = tuple(int(dim) for dim in input_dims)  # type: ignore[assignment]
        self.model = StaticGlow(
            input_shape=self.input_dims,
            hidden_dim=hidden_size,
            n_blocks_flow=n_blocks_flow,
            num_scales=num_scales,
        )
        self.initialized: bool = False

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Run the forward Glow pass returning latent variables and log determinant."""
        return self.model(x)

    def reverse(self, latent: Tensor) -> Tensor:
        """Invert the flow to generate samples from a latent tensor."""
        samples, _ = self.model.inverse(latent)
        return samples

    def log_prob(self, x: Tensor) -> Tensor:
        """Compute the log likelihood for the given batch."""
        return self.model.log_prob(x)

    def get_latent_shape(self, input_dims: Sequence[int] | None = None) -> Tuple[int]:
        """Return the flattened latent dimensionality required for sampling."""
        if input_dims is not None:
            expected = tuple(int(dim) for dim in input_dims)
            if expected != self.input_dims:
                raise ValueError(f"Mismatched input dimensions: expected {self.input_dims}, got {expected}")
        return (self.model.total_latent_dim,)

    def sample(self, num_samples: int, temperature: float = 1.0, device: torch.device | None = None) -> Tensor:
        """Draw ``num_samples`` samples by reversing the flow at the requested temperature."""
        device = device if device is not None else next(self.parameters()).device
        latent_dim, = self.get_latent_shape(self.input_dims)
        noise = torch.randn((num_samples, latent_dim), device=device) * float(temperature)
        with torch.no_grad():
            generated = self.reverse(noise)
        return generated
