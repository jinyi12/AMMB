"""Lightning module for training Glow-based density models."""

from __future__ import annotations

import math
import os
from typing import Any, Tuple

import torch
from lightning import LightningModule
from lightning.pytorch.loggers import WandbLogger
from torch import Tensor
from torchmetrics import MeanMetric
from torchvision.utils import save_image

try:  # pragma: no cover - optional dependency at runtime
    import wandb
except ImportError:  # pragma: no cover - handles environments without WandB
    wandb = None  # type: ignore[assignment]


class GlowLitModule(LightningModule):
    """Minimal Lightning wrapper around a Glow network."""

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: Any,
        scheduler: Any | None = None,
        input_dims: Tuple[int, int, int] = (3, 64, 64),
        eval_batch_size: int = 64,
        eval_temperature: float = 1.0,
        log_train_metrics: bool = True,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["net"])
        self.net = net
        self.optimizer_partial = optimizer
        self.scheduler_partial = scheduler

        self.input_dims = tuple(int(dim) for dim in input_dims)
        self.n_pixels = math.prod(self.input_dims)
        self.eval_batch_size = int(eval_batch_size)
        self.eval_temperature = float(eval_temperature)
        self.log_train_metrics = log_train_metrics

        self.train_nll = MeanMetric()
        self.train_bpd = MeanMetric()
        self.val_nll = MeanMetric()
        self.val_bpd = MeanMetric()
        self.test_nll = MeanMetric()
        self.test_bpd = MeanMetric()

        self.test_output_dir: str | None = None
        self.generated_count: int = 0

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    def calculate_bpd(self, nll: Tensor) -> Tensor:
        """Convert negative log-likelihood to bits-per-dimension."""
        return nll / (math.log(2.0) * float(self.n_pixels))

    def _unpack_batch(self, batch: Any) -> Tensor:
        if isinstance(batch, dict):
            if "x" in batch:
                return batch["x"]
            if "image" in batch:
                return batch["image"]
            if "x0" in batch:
                return batch["x0"]
        if isinstance(batch, (tuple, list)):
            return batch[0]
        if torch.is_tensor(batch):
            return batch
        raise TypeError(f"Unsupported batch structure: {type(batch)}")

    def model_step(self, batch: Any) -> Tuple[Tensor, Tensor]:
        inputs = self._unpack_batch(batch)
        inputs = inputs.to(self.device).float()
        if inputs.dim() == 2 and inputs.shape[1] == self.n_pixels:
            inputs = inputs.view(-1, *self.input_dims)

        log_prob = self.net.log_prob(inputs)
        if log_prob.dim() == 0:
            log_prob = log_prob.unsqueeze(0)

        nll = -log_prob
        mean_nll = nll.mean()
        mean_bpd = self.calculate_bpd(nll).mean()
        return mean_nll, mean_bpd

    # ------------------------------------------------------------------
    # Lightning steps
    # ------------------------------------------------------------------

    def training_step(self, batch: Any, batch_idx: int) -> Tensor:  # noqa: D401
        nll, bpd = self.model_step(batch)

        if self.log_train_metrics:
            self.train_nll(nll.detach())
            self.train_bpd(bpd.detach())
            self.log("train/nll", self.train_nll, on_step=True, on_epoch=True, prog_bar=True)
            self.log("train/bpd", self.train_bpd, on_step=False, on_epoch=True, prog_bar=True)
            self.log("train/bpd_step", bpd.detach(), on_step=True, on_epoch=False)

        return nll

    def validation_step(self, batch: Any, batch_idx: int) -> None:  # noqa: D401
        nll, bpd = self.model_step(batch)
        self.val_nll(nll.detach())
        self.val_bpd(bpd.detach())
        self.log("val/nll", self.val_nll, on_step=False, on_epoch=True)
        self.log("val/bpd", self.val_bpd, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch: Any, batch_idx: int) -> None:  # noqa: D401
        nll, bpd = self.model_step(batch)
        self.test_nll(nll.detach())
        self.test_bpd(bpd.detach())
        self.log("test/nll", self.test_nll, on_step=False, on_epoch=True)
        self.log("test/bpd", self.test_bpd, on_step=False, on_epoch=True)

        if self.trainer.is_global_zero and self.eval_batch_size > 0:
            self._generate_and_save_samples()

    # ------------------------------------------------------------------
    # ActNorm initialisation
    # ------------------------------------------------------------------

    def on_fit_start(self) -> None:  # noqa: D401
        if not self.trainer or not self.trainer.is_global_zero:
            return
        if getattr(self.net, "initialized", False):
            return

        datamodule = getattr(self.trainer, "datamodule", None)
        if datamodule is None:
            self.print("Warning: Unable to perform ActNorm initialization (missing datamodule).")
            return

        train_loader = datamodule.train_dataloader()
        if train_loader is None:
            self.print("Warning: train_dataloader unavailable for ActNorm initialization.")
            return

        batch = next(iter(train_loader))
        inputs = self._unpack_batch(batch).to(self.device).float()
        if inputs.dim() == 2 and inputs.shape[1] == self.n_pixels:
            inputs = inputs.view(-1, *self.input_dims)

        was_training = self.net.training
        self.net.eval()
        with torch.no_grad():
            _ = self.net(inputs)
        if was_training:
            self.net.train()

        setattr(self.net, "initialized", True)
        self.print("Performed data-dependent ActNorm initialization.")

    # ------------------------------------------------------------------
    # Evaluation data handling
    # ------------------------------------------------------------------

    def on_test_start(self) -> None:  # noqa: D401
        if not self.trainer:
            return
        log_dir = getattr(self.trainer, "log_dir", None) or self.trainer.default_root_dir
        self.test_output_dir = os.path.join(log_dir, "generated_evaluation_samples")
        if self.trainer.is_global_zero and self.test_output_dir is not None:
            os.makedirs(self.test_output_dir, exist_ok=True)
        self.generated_count = 0

    def _generate_and_save_samples(self) -> None:
        assert self.test_output_dir is not None, "test_output_dir should be initialised in on_test_start"
        latent_shape = self.net.get_latent_shape(self.input_dims)
        latent_shape_tuple = tuple(int(dim) for dim in latent_shape)
        noise = torch.randn((self.eval_batch_size, *latent_shape_tuple), device=self.device) * self.eval_temperature
        with torch.no_grad():
            samples = self.net.reverse(noise)

        samples = samples.detach().cpu()
        for sample in samples:
            filename = f"sample_{self.trainer.global_rank}_{self.generated_count:05d}.png"
            save_path = os.path.join(self.test_output_dir, filename)
            save_image(sample, save_path, normalize=True)
            self.generated_count += 1

    def on_test_end(self) -> None:  # noqa: D401
        if not self.trainer or not self.trainer.is_global_zero or self.test_output_dir is None:
            return
        if wandb is None:
            return

        wandb_logger = None
        for logger in self.trainer.loggers or []:
            if isinstance(logger, WandbLogger):
                wandb_logger = logger
                break
        if wandb_logger is None:
            return

        run = getattr(wandb_logger, "experiment", None)
        if run is None or not getattr(run, "id", None):
            return

        artifact = wandb.Artifact(
            name=f"evaluation-samples-{run.id}",
            type="dataset",
            description=f"Generated evaluation samples at temperature {self.eval_temperature:.2f}.",
        )
        artifact.add_dir(self.test_output_dir)
        run.log_artifact(artifact)
        self.print(f"Uploaded evaluation samples from {self.test_output_dir} to WandB Artifacts.")

    # ------------------------------------------------------------------
    # Optimiser configuration
    # ------------------------------------------------------------------

    def configure_optimizers(self) -> Any:  # noqa: D401
        optimizer = self.optimizer_partial(params=self.parameters())
        if self.scheduler_partial is not None:
            scheduler = self.scheduler_partial(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                },
            }
        return optimizer
