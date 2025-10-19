"""Callbacks specific to Glow-based models."""

from __future__ import annotations

import math
from typing import Dict, Sequence

import torch
from lightning import Callback, LightningModule, Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities import rank_zero_only
from torchvision.utils import make_grid

from src.utils.bridge_metrics import calculate_validation_metrics


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

            metrics = calculate_validation_metrics(
                {t: target_samples[t].detach() for t in evaluation_times},
                {t: generated_samples[t].detach() for t in evaluation_times},
                resolution=resolution,
                num_projections=self.num_projections,
            )

            nrow = max(1, int(math.sqrt(sample_count)))
            for time_point in selected_times:
                target = target_samples[time_point].detach().cpu().view(sample_count, num_channels, resolution, resolution)
                generated = generated_samples[time_point].detach().cpu().view(sample_count, num_channels, resolution, resolution)

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

            log_payload = {}
            for time_point, values in metrics["per_time"].items():
                for metric_name, metric_value in values.items():
                    if math.isfinite(metric_value):
                        log_payload[f"viz/{metric_name}_t{time_point:.3f}"] = metric_value
            for metric_name, metric_value in metrics["summary"].items():
                if math.isfinite(metric_value):
                    log_payload[f"viz/{metric_name}"] = metric_value
            if log_payload and getattr(trainer, "logger", None) is not None:
                trainer.logger.log_metrics(log_payload, step=trainer.global_step)
        except Exception as exc:  # pragma: no cover - safety net for training runs
            print(f"Error during visualization sampling: {exc}")
        finally:
            if original_mode:
                pl_module.train()
