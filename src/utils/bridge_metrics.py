"""Utility functions for validating and visualising Decoupled Glow bridges."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F

TensorDict = Dict[float, torch.Tensor]


def _ensure_same_sample_count(target: torch.Tensor, generated: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Trim both tensors to the same number of samples to allow paired metrics."""
    n = min(target.shape[0], generated.shape[0])
    if n == 0:
        raise ValueError("Expected at least one sample per marginal for validation metrics")
    return target[:n], generated[:n]


def sliced_wasserstein_distance(
    target: torch.Tensor,
    generated: torch.Tensor,
    num_projections: int = 128,
) -> torch.Tensor:
    """Compute the sliced Wasserstein distance between two batches of samples.

    This implementation follows the standard random-projection approach and works in any
    dimensionality without requiring external optimal transport dependencies.
    """
    target, generated = _ensure_same_sample_count(target, generated)
    if target.shape != generated.shape:
        raise ValueError("Sliced Wasserstein requires matching tensor shapes after trimming")

    device = target.device
    dim = target.shape[1]
    projections = torch.randn(dim, num_projections, device=device)
    projections = F.normalize(projections, dim=0)

    projected_target = target @ projections
    projected_generated = generated @ projections

    projected_target = torch.sort(projected_target, dim=0).values
    projected_generated = torch.sort(projected_generated, dim=0).values

    distance = torch.mean((projected_target - projected_generated) ** 2)
    return torch.sqrt(torch.clamp(distance, min=0.0))


def compute_spatial_acf_2d(samples: torch.Tensor) -> torch.Tensor:
    """Estimate the mean spatial autocorrelation function for 2D fields."""
    if samples.dim() != 3:
        raise ValueError("Expected samples with shape [N, H, W] for ACF computation")

    centered = samples - samples.mean(dim=(1, 2), keepdim=True)
    freq_domain = torch.fft.fft2(centered, norm="ortho")
    power_spectrum = torch.abs(freq_domain) ** 2
    autocovariance = torch.fft.ifft2(power_spectrum, norm="ortho").real
    acf = autocovariance.mean(dim=0)

    variance = acf[0, 0]
    if variance.abs() < 1e-9:
        return torch.zeros_like(acf)

    acf = acf / variance.clamp(min=1e-9)
    return torch.fft.fftshift(acf)


def compute_sample_covariance_matrix(samples: torch.Tensor) -> torch.Tensor:
    """Compute the sample covariance matrix for a batch of flattened samples."""
    if samples.dim() != 2:
        raise ValueError("Covariance matrix expects input with shape [N, D]")

    samples = samples - samples.mean(dim=0, keepdim=True)
    denom = max(samples.shape[0] - 1, 1)
    covariance = samples.t().matmul(samples) / denom
    covariance = torch.nan_to_num(covariance, nan=0.0, posinf=0.0, neginf=0.0)
    return covariance


def relative_covariance_frobenius_distance(target_cov: torch.Tensor, generated_cov: torch.Tensor) -> float:
    """Compute the relative Frobenius norm between two covariance matrices."""
    if target_cov.shape != generated_cov.shape:
        raise ValueError("Covariance matrices must share the same shape")

    target_norm = torch.norm(target_cov, p="fro")
    diff_norm = torch.norm(target_cov - generated_cov, p="fro")
    if target_norm < 1e-9:
        return 0.0 if diff_norm < 1e-9 else float("inf")
    return (diff_norm / target_norm).item()


def compute_sample_correlation_matrix(samples: torch.Tensor) -> torch.Tensor:
    """Compute the sample correlation matrix (normalized covariance) for a batch of samples.
    
    Args:
        samples: Tensor of shape [N, D] containing N samples of dimension D
        
    Returns:
        Correlation matrix of shape [D, D] with values in [-1, 1]
    """
    if samples.dim() != 2:
        raise ValueError("Correlation matrix expects input with shape [N, D]")
    
    # Compute covariance first
    cov = compute_sample_covariance_matrix(samples)
    
    # Extract standard deviations from diagonal
    std = torch.sqrt(torch.diagonal(cov).clamp(min=1e-8))
    
    # Normalize by outer product of std
    # corr[i,j] = cov[i,j] / (std[i] * std[j])
    correlation = cov / (std.unsqueeze(1) * std.unsqueeze(0))
    
    # Clamp to [-1, 1] to handle numerical errors
    correlation = torch.clamp(correlation, min=-1.0, max=1.0)
    
    return correlation


def compute_sample_correlation_matrix_with_eigen(
    samples: torch.Tensor,
    truncate: bool = False,
    variance_threshold: float = 0.999,
) -> Tuple[torch.Tensor, dict]:
    """Compute correlation matrix and its eigendecomposition.
    
    Args:
        samples: Tensor of shape [N, D] containing N samples
        truncate: If True, truncate eigenvalues to variance threshold
        variance_threshold: Variance threshold for truncation (default: 99.9%)
        
    Returns:
        Tuple of (correlation_matrix, eigen_info_dict)
        
    eigen_info_dict contains:
        - eigenvalues: Sorted eigenvalues in descending order
        - eigenvectors: Corresponding eigenvectors
        - variance_ratio: Cumulative variance explained ratio
        - n_components: Number of components needed for threshold
    """
    if samples.dim() != 2:
        raise ValueError("Expected samples with shape [N, D]")
    
    # Compute correlation matrix
    correlation = compute_sample_correlation_matrix(samples)
    
    # Eigendecomposition
    eigenvalues, eigenvectors = torch.linalg.eigh(correlation)
    
    # Sort in descending order
    sorted_indices = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]
    
    # Clamp negative eigenvalues to zero (due to numerical errors in eigendecomposition)
    eigenvalues = torch.clamp(eigenvalues, min=0.0)
    
    # Compute variance (explained variance ratio)
    total_variance = eigenvalues.sum().clamp(min=1e-8)
    variance_ratio = eigenvalues / total_variance
    cumsum_variance = torch.cumsum(variance_ratio, dim=0)
    
    # Find number of components for threshold
    n_components = (cumsum_variance < variance_threshold).sum().item() + 1
    n_components = min(n_components, len(eigenvalues))
    
    eigen_info = {
        "eigenvalues": eigenvalues,
        "eigenvectors": eigenvectors,
        "variance_ratio": cumsum_variance,
        "n_components": n_components,
    }
    
    if truncate:
        # Return only truncated eigenvalues
        eigenvalues_trunc = eigenvalues[:n_components]
        return correlation, {
            "eigenvalues": eigenvalues_trunc,
            "eigenvectors": eigenvectors[:, :n_components],
            "variance_ratio": cumsum_variance[:n_components],
            "n_components": n_components,
        }
    
    return correlation, eigen_info


def calculate_validation_metrics(
    target_marginals: TensorDict,
    generated_marginals: TensorDict,
    resolution: int,
    num_projections: int = 128,
) -> Dict[str, Dict[float, float]]:
    """Compute quantitative validation metrics for each marginal time.
    
    IMPORTANT: Metrics are computed in the data space of the input tensors.
    If inputs are normalized (z-space from datamodule), metrics reflect performance
    in normalized space. For interpretation in original data space (x), multiply
    distances by the scale of normalization (data_std).
    
    Args:
        target_marginals: Dictionary mapping times to target sample tensors (normalized)
        generated_marginals: Dictionary mapping times to generated sample tensors (normalized)
        resolution: Spatial resolution for ACF computation (assumes square: resolution x resolution)
        num_projections: Number of random projections for sliced Wasserstein
        
    Returns:
        Dictionary with metrics per time and summary statistics
    """
    shared_times = sorted(set(target_marginals.keys()) & set(generated_marginals.keys()))
    if not shared_times:
        raise ValueError("No overlapping time points between target and generated marginals")

    per_time: Dict[float, Dict[str, float]] = {}
    w2_values = []
    mse_acf_values = []
    rel_cov_values = []

    for time_point in shared_times:
        target = target_marginals[time_point]
        generated = generated_marginals[time_point]

        target, generated = _ensure_same_sample_count(target, generated)
        # Metrics operate on CPU to avoid unnecessary GPU synchronisation overhead
        target_cpu = target.detach().cpu()
        generated_cpu = generated.detach().cpu()

        sw_distance = sliced_wasserstein_distance(target_cpu, generated_cpu, num_projections=num_projections)
        w2_value = float(sw_distance.item())

        flat_target = target_cpu.view(target_cpu.shape[0], -1)
        flat_generated = generated_cpu.view(generated_cpu.shape[0], -1)
        target_cov = compute_sample_covariance_matrix(flat_target)
        generated_cov = compute_sample_covariance_matrix(flat_generated)
        rel_cov = relative_covariance_frobenius_distance(target_cov, generated_cov)

        channels = flat_target.shape[1] // (resolution * resolution)
        if channels <= 0:
            raise ValueError("Invalid channel count derived from data dimension and resolution")
        target_fields = target_cpu.view(target_cpu.shape[0], channels, resolution, resolution)
        generated_fields = generated_cpu.view(generated_cpu.shape[0], channels, resolution, resolution)
        # Average across channels to obtain a single representative field for ACF
        target_field_mean = target_fields.mean(dim=1)
        generated_field_mean = generated_fields.mean(dim=1)
        target_acf = compute_spatial_acf_2d(target_field_mean)
        generated_acf = compute_spatial_acf_2d(generated_field_mean)
        mse_acf = torch.mean((target_acf - generated_acf) ** 2).item()

        per_time[time_point] = {
            "w2": w2_value,
            "mse_acf": float(mse_acf),
            "rel_fro_cov": float(rel_cov),
        }
        w2_values.append(w2_value)
        mse_acf_values.append(mse_acf)
        rel_cov_values.append(rel_cov)

    summary = {
        "w2_mean": float(torch.tensor(w2_values).mean().item()) if w2_values else float("nan"),
        "w2_std": float(torch.tensor(w2_values).std(unbiased=False).item()) if len(w2_values) > 1 else 0.0,
        "mse_acf_mean": float(torch.tensor(mse_acf_values).mean().item()) if mse_acf_values else float("nan"),
        "rel_fro_cov_mean": float(torch.tensor(rel_cov_values).mean().item()) if rel_cov_values else float("nan"),
    }

    return {"per_time": per_time, "summary": summary}
