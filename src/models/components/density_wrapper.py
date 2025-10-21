"""Lightweight wrapper for density models with normalization and noise utilities.

This module encapsulates the density model and associated utilities (normalization,
training-time noise, log probability calculation), implementing the Interface 
Segregation Principle by providing only density-specific functionality without
transport or dynamics-related methods.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


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


class DensityWrapper(nn.Module):
    """Encapsulates a density model with normalization utilities and noise handling.
    
    This wrapper provides a focused interface for density operations:
    - Training-time noise application (dequantization, regularization)
    - Spatial data conversion (flattened ↔ spatial)
    - Log probability calculation in normalized space
    
    IMPORTANT: This wrapper assumes input data is ALREADY NORMALIZED by the datamodule.
    The typical data flow is:
    1. Datamodule loads raw data x and normalizes: z = (x - mean) / std
    2. DensityWrapper.log_prob(z) receives normalized z
    3. DensityWrapper adds training noise and converts to spatial
    4. Model computes log probability in z-space
    5. Result returned (no Jacobian correction: input already transformed)
    
    The data_mean and data_std are stored ONLY for visualization/interpretation
    (denormalization of samples), not for additional normalization.
    
    By separating density concerns from transport logic, this wrapper adheres
    to the Interface Segregation Principle and eliminates unnecessary coupling
    between density and dynamics training phases.
    """

    def __init__(
        self,
        density_model: nn.Module,
        resolution: int,
        data_mean: Optional[Tensor] = None,
        data_std: Optional[Tensor] = None,
        training_noise_std: float = 1e-4,
    ) -> None:
        """Initialize DensityWrapper.
        
        Args:
            density_model: The underlying density model (e.g., StaticGlow)
            resolution: Spatial resolution of data (assumes square: resolution x resolution)
            data_mean: Optional normalization mean (can be set later via update_normalization)
            data_std: Optional normalization std (can be set later via update_normalization)
            training_noise_std: Standard deviation of regularization noise during training
        """
        super().__init__()
        self.model = density_model
        self.resolution = resolution
        self.training_noise_std = training_noise_std

        # Initialize normalization buffers
        if data_mean is None:
            data_mean = torch.zeros(1)
        if data_std is None:
            data_std = torch.ones(1)

        self.register_buffer("data_mean", data_mean.clone() if isinstance(data_mean, Tensor) else torch.tensor(data_mean))
        self.register_buffer("data_std", data_std.clone() if isinstance(data_std, Tensor) else torch.tensor(data_std))

    def update_normalization(self, mean: Tensor, std: Tensor) -> None:
        """Update normalization statistics.
        
        Args:
            mean: New normalization mean
            std: New normalization standard deviation
        """
        with torch.no_grad():
            self.data_mean.copy_(mean.to(self.data_mean.device, self.data_mean.dtype))
            self.data_std.copy_(std.to(self.data_std.device, self.data_std.dtype))

    def _normalize(self, x: Tensor) -> Tensor:
        """Normalize data using stored statistics.
        
        Args:
            x: Input tensor
            
        Returns:
            Normalized tensor: z = (x - mean) / std
        """
        return (x - self.data_mean) / self.data_std

    def _denormalize(self, z: Tensor) -> Tensor:
        """Denormalize data using stored statistics.
        
        Args:
            z: Normalized tensor
            
        Returns:
            Original-scale tensor: x = z * std + mean
        """
        return z * self.data_std + self.data_mean

    def _apply_noise_if_training(self, x: Tensor) -> Tensor:
        """Apply regularization noise during training (e.g., dequantization).
        
        Args:
            x: Input tensor
            
        Returns:
            Tensor with noise applied if training, otherwise unchanged
        """
        if self.training and self.training_noise_std > 0:
            noise = torch.randn_like(x) * self.training_noise_std
            return x + noise
        return x

    def log_prob(self, z0: Tensor) -> Tensor:
        """Compute log probability.
        
        IMPORTANT ASSUMPTION: Input z0 is already normalized by the datamodule!
        
        The datamodule performs normalization once during setup():
            z = (x - data_mean) / data_std
        
        This method receives z0 in normalized space and computes log p(z).
        NO Jacobian correction is applied here because:
        1. No transformation occurs in this method (input already transformed)
        2. The Jacobian of the datamodule transformation was already accounted for
        3. We compute log probability in z-space directly
        
        The computation follows:
        1. Apply training-time noise (if in training mode, for regularization)
        2. Convert to spatial layout for model (e.g., flatten → [batch, channels, H, W])
        3. Compute model log probability: log p(z)
        4. Return result (no Jacobian correction)
        
        Args:
            z0: Input data tensor in NORMALIZED space (batch_size, data_dim) - flattened
            
        Returns:
            Log probability with shape (batch_size,)
        """
        # Apply noise if training (for regularization, not denormalization)
        z0_noisy = self._apply_noise_if_training(z0)
        
        # Convert to spatial layout for model (model expects 4D tensors like [B, C, H, W])
        z0_spatial = spatial_data_conversion(z0_noisy, self.resolution, to_spatial=True)
        
        # Compute model log probability in normalized space
        log_prob_z = self.model.log_prob(z0_spatial)
        
        # No Jacobian correction needed: input already in normalized space
        return log_prob_z

    def forward(self, x0: Tensor) -> Tensor:
        """Forward pass computes log probability.
        
        Args:
            x0: Input tensor
            
        Returns:
            Log probability
        """
        return self.log_prob(x0)
