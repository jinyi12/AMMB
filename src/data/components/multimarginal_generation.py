"""Utility functions for generating multi-marginal datasets.

This module migrates the bespoke data generation utilities from the legacy
3MASB codebase into the Lightning + Hydra project structure. The functions
here expose the same APIs that the training scripts historically relied upon
so that higher-level components can remain focused on orchestration logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
from scipy.spatial.distance import pdist, squareform
import scipy.stats as stats
from torch import Tensor
from tqdm import trange


# ---------------------------------------------------------------------------
# Gaussian smoothing utilities
# ---------------------------------------------------------------------------


def gaussian_blur_periodic(input_tensor: Tensor, kernel_size: int, sigma: float) -> Tensor:
    """Apply Gaussian blur with circular padding to emulate periodic boundaries."""
    if sigma <= 1e-9 or kernel_size <= 1:
        return input_tensor

    if kernel_size % 2 == 0:
        kernel_size += 1

    k = torch.arange(kernel_size, dtype=torch.float32, device=input_tensor.device)
    center = (kernel_size - 1) / 2
    gauss_1d = torch.exp(-0.5 * ((k - center) / sigma) ** 2)
    gauss_1d = gauss_1d / gauss_1d.sum()
    gauss_2d = torch.outer(gauss_1d, gauss_1d)

    if input_tensor.dim() != 4:
        raise ValueError("Input tensor must be [B, C, H, W]")

    channels = input_tensor.shape[1]
    kernel = gauss_2d.expand(channels, 1, kernel_size, kernel_size)
    padding = (kernel_size - 1) // 2

    padded = torch.nn.functional.pad(
        input_tensor,
        (padding, padding, padding, padding),
        mode="circular",
    )
    output = torch.nn.functional.conv2d(padded, kernel, padding=0, groups=channels)
    return output


# ---------------------------------------------------------------------------
# Non-Gaussian transforms
# ---------------------------------------------------------------------------


def transform_to_non_gaussian(
    gaussian_field: np.ndarray,
    mu_target: float,
    sigma_target: float,
    distribution: str = "gamma",
) -> np.ndarray:
    """Map a Gaussian random field to a target distribution via PIT."""
    g_mean = np.mean(gaussian_field)
    g_std = np.std(gaussian_field)

    if g_std < 1e-9:
        standard_gaussian = np.zeros_like(gaussian_field)
    else:
        standard_gaussian = (gaussian_field - g_mean) / g_std

    z_normcdf = stats.norm.cdf(standard_gaussian, 0, 1)
    z_normcdf = np.clip(z_normcdf, 1e-9, 1 - 1e-9)

    if distribution == "gamma":
        shape = (mu_target / sigma_target) ** 2
        scale = (sigma_target**2) / mu_target
        return stats.gamma.ppf(z_normcdf, shape, scale=scale)

    if distribution == "lognormal":
        sigma_ln = np.sqrt(np.log(1 + (sigma_target / mu_target) ** 2))
        mu_ln = np.log(mu_target) - 0.5 * sigma_ln**2
        return stats.lognorm.ppf(z_normcdf, s=sigma_ln, scale=np.exp(mu_ln))

    raise ValueError(f"Unsupported distribution type: {distribution}")


# ---------------------------------------------------------------------------
# Random field generation
# ---------------------------------------------------------------------------


@dataclass
class RandomFieldGenerator2D:
    """Generate 2D Gaussian random fields with either FFT or KL methods."""

    nx: int = 100
    ny: int = 100
    lx: float = 1.0
    ly: float = 1.0
    device: str = "cpu"
    generation_method: str = "kl"
    kl_error_threshold: float = 1e-3

    def __post_init__(self) -> None:
        method = self.generation_method.lower()
        if method not in {"fft", "kl"}:
            raise ValueError("generation_method must be 'fft' or 'kl'")
        self.generation_method = method

        self.kl_cache: Dict[Tuple[float, str, float], np.ndarray] = {}
        if self.generation_method == "kl":
            self._initialize_kl_mesh()

    def _initialize_kl_mesh(self) -> None:
        x = np.linspace(0, self.lx, self.nx)
        y = np.linspace(0, self.ly, self.ny)
        X, Y = np.meshgrid(x, y, indexing="ij")
        self.xy_coords = np.column_stack((X.flatten(), Y.flatten()))

    def _get_kl_transform_matrix(self, correlation_length: float, covariance_type: str) -> np.ndarray:
        cache_key = (correlation_length, covariance_type, self.kl_error_threshold)
        if cache_key in self.kl_cache:
            return self.kl_cache[cache_key]

        distances = squareform(pdist(self.xy_coords, "euclidean"))
        l_val = correlation_length

        if covariance_type == "exponential":
            cov_matrix = np.exp(-distances / l_val)
        elif covariance_type == "gaussian":
            cov_matrix = np.exp(-((distances / l_val) ** 2) / 2.0)
        else:
            raise ValueError(f"Invalid covariance_type for KL method: {covariance_type}")

        eig_vals, eig_vecs = np.linalg.eigh(cov_matrix)
        idx = np.argsort(eig_vals)[::-1]
        eig_vals = np.maximum(0, eig_vals[idx])
        eig_vecs = eig_vecs[:, idx]

        total_variance = np.sum(eig_vals)
        if total_variance > 1e-9:
            error_func = 1 - (np.cumsum(eig_vals) / total_variance)
            truncation_idx = np.where(error_func <= self.kl_error_threshold)[0]
            n_truncate = truncation_idx[0] + 1 if truncation_idx.size > 0 else len(eig_vals)
        else:
            n_truncate = 0

        if n_truncate == 0:
            kl_transform = np.empty((len(self.xy_coords), 0))
        else:
            sqrt_eigs = np.sqrt(eig_vals[:n_truncate])
            kl_transform = eig_vecs[:, :n_truncate] * sqrt_eigs[np.newaxis, :]

        self.kl_cache[cache_key] = kl_transform
        return kl_transform

    def generate_random_field(
        self,
        mean: float = 10.0,
        std: float = 2.0,
        correlation_length: float = 0.2,
        covariance_type: str = "exponential",
    ) -> np.ndarray:
        if self.generation_method == "fft":
            return self._generate_fft(mean, std, correlation_length, covariance_type)
        return self._generate_kl(mean, std, correlation_length, covariance_type)

    def _generate_fft(
        self,
        mean: float,
        std: float,
        correlation_length: float,
        covariance_type: str,
    ) -> np.ndarray:
        dx = self.lx / self.nx
        dy = self.ly / self.ny
        white_noise = np.random.normal(0, 1, (self.nx, self.ny))
        fourier_coeff = np.fft.fft2(white_noise)

        kx = 2 * np.pi * np.fft.fftfreq(self.nx, d=dx)
        ky = 2 * np.pi * np.fft.fftfreq(self.ny, d=dy)
        Kx, Ky = np.meshgrid(kx, ky, indexing="ij")
        K = np.sqrt(Kx**2 + Ky**2)

        l_val = correlation_length
        if covariance_type == "exponential":
            denom = 1 + (l_val * K) ** 2
            power = (2 * np.pi * l_val**2) / np.maximum(1e-9, denom ** 1.5)
        elif covariance_type == "gaussian":
            power = np.pi * l_val**2 * np.exp(-((l_val * K) ** 2) / 4)
        else:
            raise ValueError("Invalid covariance_type")

        power = np.nan_to_num(power)
        fourier_coeff *= np.sqrt(power)
        field = np.fft.ifft2(fourier_coeff).real

        field_std = np.std(field)
        if field_std > 1e-9:
            field = (field - np.mean(field)) / field_std * std + mean
        else:
            field = np.full_like(field, mean)
        return field

    def _generate_kl(
        self,
        mean: float,
        std: float,
        correlation_length: float,
        covariance_type: str,
    ) -> np.ndarray:
        kl_transform = self._get_kl_transform_matrix(correlation_length, covariance_type)
        n_kl = kl_transform.shape[1]
        if n_kl == 0:
            return np.full((self.nx, self.ny), mean)

        xi = np.random.normal(0, 1, n_kl)
        field_flat = kl_transform @ xi
        field_flat = field_flat * std + mean
        return field_flat.reshape((self.nx, self.ny))

    def coarsen_field(self, field: Tensor | np.ndarray, H: float) -> Tensor:
        if isinstance(field, np.ndarray):
            field = torch.from_numpy(field).to(self.device)

        original_dim = field.dim()
        if original_dim == 2:
            field = field.unsqueeze(0).unsqueeze(0)
        elif original_dim == 3:
            field = field.unsqueeze(1)
        elif original_dim != 4:
            raise ValueError("Unsupported field dimensions")

        pixel_size = self.lx / self.nx
        filter_sigma_phys = H / 6.0
        filter_sigma_pix = filter_sigma_phys / pixel_size

        if filter_sigma_pix < 1e-6:
            smooth = field
        else:
            kernel_size = int(6 * filter_sigma_pix)
            if kernel_size % 2 == 0:
                kernel_size += 1
            kernel_size = max(3, kernel_size)
            smooth = gaussian_blur_periodic(field, kernel_size=kernel_size, sigma=filter_sigma_pix)

        if original_dim == 2:
            return smooth.squeeze(0).squeeze(0)
        if original_dim == 3:
            return smooth.squeeze(1)
        return smooth


# ---------------------------------------------------------------------------
# Scheduling utilities
# ---------------------------------------------------------------------------


def calculate_filter_schedule(
    num_constraints: int,
    H_max: float,
    L_domain: float,
    resolution: int,
    micro_corr_length: float,
    schedule_type: str,
) -> np.ndarray:
    if num_constraints <= 1:
        return np.array([H_max] if num_constraints == 1 else [])

    if schedule_type == "linear" or num_constraints == 2:
        return np.linspace(0, H_max, num_constraints)

    if schedule_type == "geometric":
        delta_x = L_domain / resolution
        H_min_numerical = 12 * delta_x
        H_min = max(H_min_numerical, micro_corr_length, 1e-6)

        if H_max <= H_min * 1.01:
            return np.linspace(0, H_max, num_constraints)

        H_non_zero = np.geomspace(H_min, H_max, num_constraints - 1)
        return np.concatenate(([0.0], H_non_zero))

    raise ValueError(f"Unknown schedule type: {schedule_type}")


# ---------------------------------------------------------------------------
# Data generation entry points
# ---------------------------------------------------------------------------


def generate_multiscale_grf_data(
    N_samples: int,
    T: float = 1.0,
    N_constraints: int = 5,
    resolution: int = 32,
    L_domain: float = 1.0,
    micro_corr_length: float = 0.1,
    H_max_factor: float = 0.5,
    mean_val: float = 10.0,
    std_val: float = 2.0,
    covariance_type: str = "exponential",
    device: str = "cpu",
    generation_method: str = "fft",
    kl_error_threshold: float = 1e-3,
    schedule_type: str = "geometric",
) -> Tuple[Dict[float, Tensor], int]:
    time_steps = torch.linspace(0, T, N_constraints)
    H_max = L_domain * H_max_factor
    H_schedule = calculate_filter_schedule(
        N_constraints,
        H_max,
        L_domain,
        resolution,
        micro_corr_length,
        schedule_type,
    )

    marginal_data: Dict[float, Tensor] = {}
    data_dim = resolution * resolution

    generator = RandomFieldGenerator2D(
        nx=resolution,
        ny=resolution,
        lx=L_domain,
        ly=L_domain,
        device=device,
        generation_method=generation_method,
        kl_error_threshold=kl_error_threshold,
    )

    if generation_method == "kl":
        generator._get_kl_transform_matrix(micro_corr_length, covariance_type)

    micro_fields = []
    for _ in trange(N_samples, leave=False):
        field = generator.generate_random_field(
            mean=mean_val,
            std=std_val,
            correlation_length=micro_corr_length,
            covariance_type=covariance_type,
        )
        micro_fields.append(field)

    micro_fields_tensor = torch.tensor(np.array(micro_fields), dtype=torch.float32).to(device)

    for i, t in enumerate(time_steps):
        t_val = float(t.item())
        H_t = H_schedule[i]
        coarsened = generator.coarsen_field(micro_fields_tensor, H=H_t)
        flattened = coarsened.reshape(N_samples, data_dim)
        marginal_data[t_val] = flattened

    return marginal_data, data_dim


def generate_spiral_distributional_data(
    N_constraints: int = 5,
    T: float = 1.0,
    data_dim: int = 3,
    N_samples_per_marginal: int = 128,
    noise_std: float = 0.1,
    device: str = "cpu",
) -> Tuple[Dict[float, Tensor], Tensor]:
    time_steps = torch.linspace(0, T, N_constraints)
    marginal_data: Dict[float, Tensor] = {}

    for t in time_steps:
        t_val = float(t.item())
        if data_dim != 3:
            raise NotImplementedError("Only data_dim=3 supported for spiral data.")

        mu = torch.tensor(
            [
                torch.sin(t * 2 * torch.pi / T),
                torch.cos(t * 2 * torch.pi / T),
                t / T,
            ]
        )
        samples = mu + noise_std * torch.randn(N_samples_per_marginal, data_dim)
        marginal_data[t_val] = samples.to(device)

    return marginal_data, time_steps


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------


def normalize_multimarginal_data(
    marginal_data: Dict[float, Tensor]
) -> Tuple[Dict[float, Tensor], Tensor, Tensor]:
    concatenated = torch.cat(list(marginal_data.values()), dim=0)
    mean = torch.mean(concatenated, dim=0)
    std = torch.std(concatenated, dim=0)
    std[std < 1e-6] = 1.0

    normalized: Dict[float, Tensor] = {}
    for t, samples in marginal_data.items():
        normalized[t] = (samples - mean) / std

    return normalized, mean, std
