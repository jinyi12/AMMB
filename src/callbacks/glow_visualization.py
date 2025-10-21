"""Callbacks specific to Glow-based models."""

from __future__ import annotations

import math
import traceback
from typing import Dict, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from lightning import Callback, LightningModule, Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities import rank_zero_only
from torchvision.utils import make_grid

from src.utils.bridge_metrics import (
    calculate_validation_metrics,
    compute_sample_correlation_matrix,
    compute_sample_correlation_matrix_with_eigen,
)


def format_for_paper():
    """Standard formatting for publication-ready figures."""
    plt.rcParams.update({'image.cmap': 'viridis'})
    plt.rcParams.update({'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif',
                                        'Bitstream Vera Serif', 'Computer Modern Roman', 'New Century Schoolbook',
                                        'Century Schoolbook L',  'Utopia', 'ITC Bookman', 'Bookman',
                                        'Nimbus Roman No9 L', 'Palatino', 'Charter', 'serif']})
    plt.rcParams.update({'font.family': 'serif'})
    plt.rcParams.update({'font.size': 10})
    plt.rcParams.update({'mathtext.fontset': 'custom'})
    plt.rcParams.update({'mathtext.rm': 'serif'})
    plt.rcParams.update({'mathtext.it': 'serif:italic'})
    plt.rcParams.update({'mathtext.bf': 'serif:bold'})
    plt.close('all')


def _visualize_eigenvalue_spectra_comparison_multitimes(
    target_samples: Dict[float, torch.Tensor],
    generated_samples: Dict[float, torch.Tensor],
    time_points: list,
    variance_threshold: float = 0.999,
) -> tuple[plt.Figure, plt.Figure]:
    """Create figures comparing eigenvalue spectra across multiple time points.
    
    Creates two separate figures:
    1. Eigenvalue spectra (one subplot per time point)
    2. Cumulative variance explained (one subplot per time point)
    
    Shows only the k retained eigenvalues based on variance_threshold (99.9%).
    
    Args:
        target_samples: Dictionary mapping times to target tensors [N, D] in DATA SPACE
        generated_samples: Dictionary mapping times to generated tensors [N, D] in DATA SPACE
        time_points: List of time points to visualize
        variance_threshold: Variance threshold for truncation (default: 99.9%)
        
    Returns:
        Tuple of (eigenvalue_figure, cumvar_figure)
    """
    format_for_paper()
    
    n_times = len(time_points)
    # Calculate grid layout: prefer wider layouts
    ncols = min(3, n_times)
    nrows = (n_times + ncols - 1) // ncols
    
    # Figure 1: Eigenvalue spectra
    fig_eigen, axes_eigen = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
    if n_times == 1:
        axes_eigen = np.array([axes_eigen])
    axes_eigen = axes_eigen.flatten()
    
    # Figure 2: Cumulative variance
    fig_cumvar, axes_cumvar = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
    if n_times == 1:
        axes_cumvar = np.array([axes_cumvar])
    axes_cumvar = axes_cumvar.flatten()
    
    for idx, time_point in enumerate(time_points):
        target_data = target_samples[time_point]
        generated_data = generated_samples[time_point]
        
        # Compute correlation matrices with eigendecomposition
        _, target_eigen_info = compute_sample_correlation_matrix_with_eigen(
            target_data, truncate=False, variance_threshold=variance_threshold
        )
        _, generated_eigen_info = compute_sample_correlation_matrix_with_eigen(
            generated_data, truncate=False, variance_threshold=variance_threshold
        )
        
        target_eigenvalues = target_eigen_info["eigenvalues"].cpu().numpy()
        target_cumvar = target_eigen_info["variance_ratio"].cpu().numpy()
        target_n_comp = target_eigen_info["n_components"]
        
        generated_eigenvalues = generated_eigen_info["eigenvalues"].cpu().numpy()
        generated_cumvar = generated_eigen_info["variance_ratio"].cpu().numpy()
        generated_n_comp = generated_eigen_info["n_components"]
        
        n_show = max(target_n_comp, generated_n_comp)
        x_axis = np.arange(1, n_show + 1)
        
        # Plot eigenvalue spectrum
        ax_eigen = axes_eigen[idx]
        ax_eigen.loglog(x_axis, target_eigenvalues[:n_show], 'b-', label='Target', linewidth=2, alpha=0.7)
        ax_eigen.loglog(x_axis, generated_eigenvalues[:n_show], 'r--', label='Generated', linewidth=2, alpha=0.7)
        ax_eigen.axvline(target_n_comp, color='blue', linestyle=':', linewidth=1.5, alpha=0.5)
        ax_eigen.axvline(generated_n_comp, color='red', linestyle=':', linewidth=1.5, alpha=0.5)
        ax_eigen.set_xlabel('Mode Index')
        ax_eigen.set_ylabel('Eigenvalue')
        ax_eigen.set_title(f't = {time_point:.2f}\nTarget k={target_n_comp}, Gen k={generated_n_comp}')
        ax_eigen.grid(True, which='both', alpha=0.3)
        if idx == 0:
            ax_eigen.legend(fontsize=8, loc='best')
        
        # Plot cumulative variance
        ax_cumvar = axes_cumvar[idx]
        ax_cumvar.semilogx(x_axis, target_cumvar[:n_show] * 100, 'b-', label='Target', linewidth=2, alpha=0.7)
        ax_cumvar.semilogx(x_axis, generated_cumvar[:n_show] * 100, 'r--', label='Generated', linewidth=2, alpha=0.7)
        ax_cumvar.axhline(99.9, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax_cumvar.axvline(target_n_comp, color='blue', linestyle=':', linewidth=1.5, alpha=0.5)
        ax_cumvar.axvline(generated_n_comp, color='red', linestyle=':', linewidth=1.5, alpha=0.5)
        ax_cumvar.set_xlabel('Number of Components')
        ax_cumvar.set_ylabel('Cumulative Variance (%)')
        ax_cumvar.set_ylim([90, 100.5])
        ax_cumvar.set_title(f't = {time_point:.2f}')
        ax_cumvar.grid(True, which='both', alpha=0.3)
        if idx == 0:
            ax_cumvar.legend(fontsize=8, loc='lower right')
    
    # Hide unused subplots
    for idx in range(n_times, len(axes_eigen)):
        axes_eigen[idx].set_visible(False)
        axes_cumvar[idx].set_visible(False)
    
    fig_eigen.suptitle('Eigenvalue Spectra Comparison', fontsize=14, y=0.995)
    fig_cumvar.suptitle('Cumulative Variance Explained', fontsize=14, y=0.995)
    
    fig_eigen.tight_layout()
    fig_cumvar.tight_layout()
    
    return fig_eigen, fig_cumvar


def _visualize_covariance_heatmaps_comparison_multitimes(
    target_samples: Dict[float, torch.Tensor],
    generated_samples: Dict[float, torch.Tensor],
    time_points: list,
    max_dim_for_heatmap: int = 32,
) -> plt.Figure:
    """Create a figure comparing correlation matrices across multiple time points.
    
    Shows for each time point:
    - Target correlation matrix
    - Generated correlation matrix
    - Absolute difference between them
    
    Args:
        target_samples: Dictionary mapping times to target tensors [N, D] in DATA SPACE
        generated_samples: Dictionary mapping times to generated tensors [N, D] in DATA SPACE
        time_points: List of time points to visualize
        max_dim_for_heatmap: Maximum dimensionality to show (for readability)
        
    Returns:
        matplotlib Figure object with grid layout (3 rows x n_times columns)
    """
    format_for_paper()
    
    n_times = len(time_points)
    cmap_corr = 'coolwarm'
    cmap_diff = 'RdBu_r'
    
    # Create figure: 3 rows (target, generated, diff) x n_times columns
    fig, axes = plt.subplots(3, n_times, figsize=(5*n_times, 12))
    if n_times == 1:
        axes = axes.reshape(3, 1)
    
    for col_idx, time_point in enumerate(time_points):
        target_data = target_samples[time_point]
        generated_data = generated_samples[time_point]
        
        # Compute correlation matrices
        target_corr = compute_sample_correlation_matrix(target_data).cpu().numpy()
        generated_corr = compute_sample_correlation_matrix(generated_data).cpu().numpy()
        
        # Limit to max_dim_for_heatmap for readability
        max_dim = min(max_dim_for_heatmap, target_corr.shape[0], generated_corr.shape[0])
        target_corr = target_corr[:max_dim, :max_dim]
        generated_corr = generated_corr[:max_dim, :max_dim]
        
        diff_corr = target_corr - generated_corr
        
        # Target correlation heatmap (row 0)
        im_target = axes[0, col_idx].imshow(target_corr, cmap=cmap_corr, vmin=-1, vmax=1)
        axes[0, col_idx].set_title(f't = {time_point:.2f}')
        if col_idx == 0:
            axes[0, col_idx].set_ylabel('Target', fontsize=10)
        plt.colorbar(im_target, ax=axes[0, col_idx], fraction=0.046, pad=0.04)
        
        # Generated correlation heatmap (row 1)
        im_gen = axes[1, col_idx].imshow(generated_corr, cmap=cmap_corr, vmin=-1, vmax=1)
        if col_idx == 0:
            axes[1, col_idx].set_ylabel('Generated', fontsize=10)
        plt.colorbar(im_gen, ax=axes[1, col_idx], fraction=0.046, pad=0.04)
        
        # Difference heatmap (row 2)
        vmax_diff = np.abs(diff_corr).max()
        im_diff = axes[2, col_idx].imshow(diff_corr, cmap=cmap_diff, vmin=-vmax_diff, vmax=vmax_diff)
        if col_idx == 0:
            axes[2, col_idx].set_ylabel('Difference', fontsize=10)
        plt.colorbar(im_diff, ax=axes[2, col_idx], fraction=0.046, pad=0.04)
    
    fig.suptitle('Correlation Matrix Comparison', fontsize=14, y=0.995)
    plt.tight_layout()
    return fig



class GlowVisualizationCallback(Callback):
    """Periodically sample from the Glow model and log images to W&B."""

    def __init__(self, num_samples: int = 16, temperature: float = 0.7, log_every_n_epochs: int = 5) -> None:
        super().__init__()
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        self.num_samples = int(num_samples)
        self.temperature = float(temperature)
        self.log_every_n_epochs = max(1, int(log_every_n_epochs))

    @rank_zero_only
    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if trainer.sanity_checking:
            return
        if (trainer.current_epoch + 1) % self.log_every_n_epochs != 0:
            return

        original_mode = pl_module.training
        pl_module.eval()
        device = pl_module.device

        try:
            input_dims: Sequence[int] | None = getattr(pl_module.hparams, "input_dims", None)
            if input_dims is None:
                raise AttributeError("GlowVisualizationCallback expects 'input_dims' in module hyperparameters")

            latent_shape = pl_module.net.get_latent_shape(input_dims)
            if not latent_shape:
                raise RuntimeError("Received empty latent shape from Glow network")
            latent_shape_tuple = tuple(int(dim) for dim in latent_shape)
            z_shape = (self.num_samples, *latent_shape_tuple)

            noise = torch.randn(z_shape, device=device) * self.temperature
            with torch.no_grad():
                samples = pl_module.net.reverse(noise)

            samples_cpu = samples.detach().cpu()
            nrow = max(1, int(math.sqrt(self.num_samples)))
            grid = make_grid(samples_cpu, nrow=nrow, normalize=True)

            for logger in trainer.loggers or []:
                if isinstance(logger, WandbLogger):
                    caption = f"Epoch {trainer.current_epoch} | Temperature {self.temperature:.2f}"
                    logger.log_image(
                        key="generated_samples",
                        images=[grid],
                        caption=[caption],
                    )
                    break
        except Exception as exc:  # pragma: no cover - safety net for training runs
            print(f"Error during visualization sampling: {exc}")
        finally:
            if original_mode:
                pl_module.train()


class DecoupledBridgeVisualizationCallback(Callback):
    """Periodically sample from the DecoupledBridge model and log images to W&B."""

    def __init__(
        self,
        num_samples: int = 16,
        log_every_n_epochs: int = 5,
        max_times_to_log: int = 3,
        num_projections: int = 128,
    ) -> None:
        super().__init__()
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        self.num_samples = int(num_samples)
        self.log_every_n_epochs = max(1, int(log_every_n_epochs))
        self.max_times_to_log = max(1, int(max_times_to_log))
        self.num_projections = max(1, int(num_projections))

    @rank_zero_only
    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if trainer.sanity_checking:
            return
        if (trainer.current_epoch + 1) % self.log_every_n_epochs != 0:
            return

        # Guard: Only run visualization for dynamics phase (which has a bridge)
        # Density phase uses DecoupledDensityLitModule which doesn't have bridge
        if not hasattr(pl_module, "bridge"):
            return

        original_mode = pl_module.training
        pl_module.eval()
        device = pl_module.device

        try:
            # Get parameters from the bridge
            bridge = getattr(pl_module, "bridge", None)
            datamodule = getattr(trainer, "datamodule", None)
            if bridge is None:
                raise AttributeError("DecoupledBridgeVisualizationCallback requires module.bridge attribute")
            if datamodule is None or not hasattr(datamodule, "normalized_marginals"):
                raise AttributeError("Visualization callback requires datamodule with normalized marginals")
            if not hasattr(datamodule, "get_batch_by_indices"):
                raise AttributeError("Datamodule must expose get_batch_by_indices for visualization")

            times_tensor = datamodule.constraint_times
            times = [float(t) for t in times_tensor.tolist()]
            if not times:
                raise RuntimeError("No constraint times available for visualization")

            resolution = bridge.resolution
            num_channels = bridge.num_channels

            # Get normalization statistics for denormalization
            data_mean = getattr(datamodule, "data_mean", None)
            data_std = getattr(datamodule, "data_std", None)
            denormalize_for_viz = data_mean is not None and data_std is not None

            max_plots = min(self.max_times_to_log, len(times))
            if max_plots == len(times):
                selected_times = list(times)
            elif max_plots == 1:
                selected_times = [times[-1]]
            else:
                indices = torch.linspace(0, len(times) - 1, steps=max_plots)
                selected_indices = {times[int(round(idx.item()))] for idx in indices}
                selected_times = list(selected_indices)
            selected_times = sorted(set(selected_times), reverse=True)

            initial_time = times[0]
            terminal_time = times[-1]
            total_samples = datamodule.normalized_marginals[terminal_time].shape[0]
            sample_count = min(self.num_samples, total_samples)
            if sample_count == 0:
                raise RuntimeError("No samples available for visualization")
            indices = torch.randperm(total_samples)[:sample_count]
            subset = datamodule.get_batch_by_indices(indices)
            dtype = getattr(pl_module, "dtype", torch.float32)
            evaluation_times = sorted(set(selected_times) | {initial_time, terminal_time})
            target_samples: Dict[float, torch.Tensor] = {}
            for time_point in evaluation_times:
                data_tensor = subset[time_point]
                target_samples[time_point] = data_tensor.to(device=device, dtype=dtype)

            generated_samples: Dict[float, torch.Tensor] = {}
            with torch.no_grad():
                t_terminal = torch.full((sample_count, 1), terminal_time, device=device, dtype=dtype)
                x0_estimate = bridge.inverse_transport(target_samples[terminal_time], t_terminal)
                for time_point in evaluation_times:
                    t_tensor = torch.full((sample_count, 1), time_point, device=device, dtype=dtype)
                    generated = bridge.transport(x0_estimate, t_tensor)
                    generated_samples[time_point] = generated.detach()

            # Denormalize samples BEFORE computing metrics (compute in data space)
            if denormalize_for_viz:
                target_samples_data_space = {}
                generated_samples_data_space = {}
                for t in evaluation_times:
                    # Denormalize in flat space
                    target_flat = target_samples[t].detach().cpu()
                    generated_flat = generated_samples[t].detach().cpu()
                    target_samples_data_space[t] = target_flat * data_std + data_mean
                    generated_samples_data_space[t] = generated_flat * data_std + data_mean
            else:
                target_samples_data_space = {t: target_samples[t].detach().cpu() for t in evaluation_times}
                generated_samples_data_space = {t: generated_samples[t].detach().cpu() for t in evaluation_times}

            metrics = calculate_validation_metrics(
                target_samples_data_space,
                generated_samples_data_space,
                resolution=resolution,
                num_projections=self.num_projections,
            )

            nrow = max(1, int(math.sqrt(sample_count)))
            for time_point in selected_times:
                # Get samples in flattened form (as stored)
                target_flat = target_samples[time_point].detach().cpu()  # Shape: (16, 256)
                generated_flat = generated_samples[time_point].detach().cpu()  # Shape: (16, 256)

                # Denormalize in flat space BEFORE reshaping
                # (denormalization stats are defined in flat space)
                if denormalize_for_viz:
                    target_flat = target_flat * data_std + data_mean
                    generated_flat = generated_flat * data_std + data_mean

                # NOW reshape to spatial for visualization
                target = target_flat.view(sample_count, num_channels, resolution, resolution)
                generated = generated_flat.view(sample_count, num_channels, resolution, resolution)

                target_grid = make_grid(target, nrow=nrow, normalize=True)
                generated_grid = make_grid(generated, nrow=nrow, normalize=True)
                comparison_grid = torch.cat([target_grid, generated_grid], dim=1)

                metrics_for_time = metrics["per_time"].get(time_point)
                if metrics_for_time is None:
                    closest_time = min(metrics["per_time"].keys(), key=lambda val: abs(val - time_point))
                    metrics_for_time = metrics["per_time"][closest_time]
                caption = (
                    f"t={time_point:.3f} | W2={metrics_for_time['w2']:.4f} | "
                    f"MSE_ACF={metrics_for_time['mse_acf']:.3e} | "
                    f"RelCov={metrics_for_time['rel_fro_cov']:.4f} | Top=target, Bottom=generated"
                )

                for logger in trainer.loggers or []:
                    if isinstance(logger, WandbLogger):
                        logger.log_image(
                            key=f"bridge_samples/t_{time_point:.3f}",
                            images=[comparison_grid],
                            caption=[caption],
                        )
                        break

            # Log multi-time eigenvalue and covariance visualizations (outside the loop)
            for logger in trainer.loggers or []:
                if isinstance(logger, WandbLogger):
                    # Log eigenvalue spectra comparison across all selected times
                    try:
                        fig_eigenspec, fig_cumvar = _visualize_eigenvalue_spectra_comparison_multitimes(
                            target_samples_data_space,
                            generated_samples_data_space,
                            selected_times,
                            variance_threshold=0.999,
                        )
                        logger.experiment.log({
                            "eigenvalue_spectra/all_times": wandb.Image(fig_eigenspec),
                            "cumulative_variance/all_times": wandb.Image(fig_cumvar),
                        })
                        plt.close(fig_eigenspec)
                        plt.close(fig_cumvar)
                    except Exception as e:
                        print(f"Error logging multi-time eigenvalue spectra: {e}")
                    
                    # Log covariance heatmap comparison across all selected times
                    try:
                        fig_heatmap = _visualize_covariance_heatmaps_comparison_multitimes(
                            target_samples_data_space,
                            generated_samples_data_space,
                            selected_times,
                            max_dim_for_heatmap=32,
                        )
                        logger.experiment.log({
                            "covariance_heatmaps/all_times": wandb.Image(fig_heatmap),
                        })
                        plt.close(fig_heatmap)
                    except Exception as e:
                        print(f"Error logging multi-time covariance heatmaps: {e}")
                    
                    break

            # Log summary metrics only (avoid per-time step conflicts with wandb step ordering)
            log_payload = {}
            for metric_name, metric_value in metrics["summary"].items():
                if math.isfinite(metric_value):
                    log_payload[f"viz/{metric_name}"] = metric_value
            if log_payload and getattr(trainer, "logger", None) is not None:
                trainer.logger.log_metrics(log_payload, step=trainer.global_step)
        except Exception as exc:  # pragma: no cover - safety net for training runs
            error_msg = (
                f"[DecoupledBridgeVisualizationCallback] Error at epoch {trainer.current_epoch}: {exc}\n"
                f"Traceback:\n{traceback.format_exc()}"
            )
            print(error_msg)
            # Optionally log to wandb if available
            if getattr(trainer, "logger", None) is not None:
                try:
                    trainer.logger.log_metrics({"viz/error_count": 1}, step=trainer.global_step)
                except Exception:
                    pass  # If logging error also fails, don't crash
        finally:
            if original_mode:
                pl_module.train()
