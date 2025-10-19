"""LightningModule wrapping the decoupled Glow bridge architecture."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from lightning import LightningModule
from torch import Tensor
from torchmetrics import MeanMetric

from .components.decoupled_glow import DecoupledBridge
from src.utils import RankedLogger


log = RankedLogger(__name__, rank_zero_only=True)


class DecoupledGlowLitModule(LightningModule):
    """Lightning integration for the decoupled Glow bridge."""

    def __init__(
        self,
        bridge: DecoupledBridge,
        optimizer: Any,
        scheduler: Optional[Any] = None,
        lambda_dynamics: float = 1.0,
        lambda_path: float = 0.0,
        log_train_metrics: bool = True,
        density_phase_epochs: int = 0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["bridge"])
        self.bridge = bridge
        self.optimizer_partial = optimizer
        self.scheduler_partial = scheduler
        self.lambda_dynamics = lambda_dynamics
        self.lambda_path = lambda_path
        self.log_train_metrics = log_train_metrics
        self.density_phase_epochs = max(0, int(density_phase_epochs))

        self._sequential_training = self.density_phase_epochs > 0
        if self._sequential_training:
            self._optimize_density = True
            self._optimize_dynamics = False
        else:
            self._optimize_density = True
            self._optimize_dynamics = True
        self._density_frozen = False
        self._current_phase = "dynamics"

        self.train_total = MeanMetric()
        self.train_density = MeanMetric()
        self.train_dynamics = MeanMetric()
        self.train_path = MeanMetric()

        self.val_total = MeanMetric()
        self.val_density = MeanMetric()
        self.val_dynamics = MeanMetric()
        self.val_path = MeanMetric()

        self.test_total = MeanMetric()
        self.test_density = MeanMetric()
        self.test_dynamics = MeanMetric()
        self.test_path = MeanMetric()

        self._constraint_times: Optional[Tensor] = None

    @property
    def active_times(self) -> Tensor:
        if self._constraint_times is None:
            raise RuntimeError("Constraint times are not set; call setup() first.")
        return self._constraint_times[1:]

    def _ensure_constraint_times(self) -> None:
        if self._constraint_times is not None:
            return
        datamodule = getattr(self.trainer, "datamodule", None)
        if datamodule is None or not hasattr(datamodule, "constraint_times"):
            raise RuntimeError("Datamodule must expose constraint_times after setup().")
        times = datamodule.constraint_times.to(self.device)
        self._constraint_times = times
        if getattr(datamodule, "data_mean", None) is not None and getattr(datamodule, "data_std", None) is not None:
            mean = datamodule.data_mean.to(self.device)
            std = datamodule.data_std.to(self.device)
            self.bridge.update_normalization(mean, std)

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------

    def forward(self, x0: Tensor, t: Tensor | float) -> Tensor:
        return self.bridge.transport(x0, t)

    def model_step(self, batch: Dict[str, Tensor], stage: str) -> Dict[str, Tensor]:
        self._ensure_constraint_times()
        x0 = batch["x0"].view(batch["x0"].shape[0], -1)
        xt = batch.get("xt")
        if xt is not None:
            xt = xt.view(x0.shape[0], -1, x0.shape[1])
        else:
            xt = torch.zeros(x0.shape[0], 0, x0.shape[1], device=x0.device, dtype=x0.dtype)

        # Guard against dataloaders that do not provide intermediate constraints during dynamics phase.
        available_targets = xt.shape[1]
        expected_targets = self.active_times.numel()
        if available_targets < expected_targets:
            if available_targets > 0:
                log.warning(
                    "Received %s transport targets but expected %s constraint times; skipping dynamics loss for this batch.",
                    available_targets,
                    expected_targets,
                )
            expected_targets = available_targets

        is_training_step = stage == "train"
        optimize_density = is_training_step and self._optimize_density
        compute_dynamics = self.active_times.numel() > 0 and (
            self._optimize_dynamics or not is_training_step
        )
        if expected_targets == 0:
            compute_dynamics = False
        optimize_dynamics = is_training_step and self._optimize_dynamics and compute_dynamics
        compute_path = self.lambda_path > 0 and compute_dynamics
        optimize_path = (
            is_training_step and self._optimize_dynamics and self.lambda_path > 0 and compute_dynamics
        )

        if optimize_density:
            x0_density = self.bridge._apply_noise_if_training(x0)
        else:
            x0_density = x0
        density_loss = -self.bridge.log_prob_initial(x0_density).mean()

        dynamics_loss = torch.tensor(0.0, device=self.device)
        if compute_dynamics:
            losses = []
            for idx, t_val in enumerate(self.active_times[:expected_targets]):
                target = xt[:, idx, :]
                t_tensor = torch.full((x0.shape[0], 1), float(t_val.item()), device=self.device, dtype=x0.dtype)
                prediction = self.bridge.transport(x0, t_tensor)
                losses.append(F.mse_loss(prediction, target))
            if losses:
                dynamics_loss = torch.stack(losses).mean()

        path_loss = torch.tensor(0.0, device=self.device)
        if compute_path:
            t_choice = self.active_times[torch.randint(0, self.active_times.numel(), (1,), device=self.device)]
            t_tensor = torch.full((x0.shape[0], 1), float(t_choice.item()), device=self.device, dtype=x0.dtype)
            x_t = self.bridge.transport(x0, t_tensor).requires_grad_(True)
            velocity = self.bridge.forward_velocity(x_t, t_tensor)
            path_loss = 0.5 * (velocity**2).mean()

        train_loss = torch.tensor(0.0, device=self.device)
        if optimize_density:
            train_loss = train_loss + density_loss
        if optimize_dynamics:
            train_loss = train_loss + self.lambda_dynamics * dynamics_loss
        if optimize_path:
            train_loss = train_loss + self.lambda_path * path_loss

        eval_loss = density_loss + self.lambda_dynamics * dynamics_loss + self.lambda_path * path_loss
        total_loss = train_loss if is_training_step else eval_loss

        return {
            "loss": total_loss,
            "density": density_loss.detach(),
            "dynamics": dynamics_loss.detach(),
            "path": path_loss.detach(),
        }

    def training_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Tensor:
        outputs = self.model_step(batch, stage="train")
        self.train_total(outputs["loss"])
        self.train_density(outputs["density"])
        self.train_dynamics(outputs["dynamics"])
        self.train_path(outputs["path"])
        if self.log_train_metrics:
            self.log("train/loss", self.train_total, on_step=True, on_epoch=True, prog_bar=True)
            self.log("train/density", self.train_density, on_step=False, on_epoch=True)
            self.log("train/dynamics", self.train_dynamics, on_step=False, on_epoch=True)
            if self.lambda_path > 0:
                self.log("train/path", self.train_path, on_step=False, on_epoch=True)
        return outputs["loss"]

    def validation_step(self, batch: Dict[str, Tensor], batch_idx: int) -> None:
        outputs = self.model_step(batch, stage="val")
        self.val_total(outputs["loss"])
        self.val_density(outputs["density"])
        self.val_dynamics(outputs["dynamics"])
        self.val_path(outputs["path"])
        self.log("val/loss", self.val_total, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/density", self.val_density, on_step=False, on_epoch=True)
        self.log("val/dynamics", self.val_dynamics, on_step=False, on_epoch=True)
        if self.lambda_path > 0:
            self.log("val/path", self.val_path, on_step=False, on_epoch=True)

    def test_step(self, batch: Dict[str, Tensor], batch_idx: int) -> None:
        outputs = self.model_step(batch, stage="test")
        self.test_total(outputs["loss"])
        self.test_density(outputs["density"])
        self.test_dynamics(outputs["dynamics"])
        self.test_path(outputs["path"])
        self.log("test/loss", self.test_total, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/density", self.test_density, on_step=False, on_epoch=True)
        self.log("test/dynamics", self.test_dynamics, on_step=False, on_epoch=True)
        if self.lambda_path > 0:
            self.log("test/path", self.test_path, on_step=False, on_epoch=True)

    def configure_optimizers(self):
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

    def _set_training_phase(self, phase: str) -> None:
        if self._current_phase == phase:
            return
        datamodule = getattr(self.trainer, "datamodule", None)
        if datamodule is not None and hasattr(datamodule, "set_training_phase"):
            changed = datamodule.set_training_phase(phase)
            if changed and self.trainer is not None and hasattr(self.trainer, "reset_train_dataloader"):
                self.trainer.reset_train_dataloader()
        self._current_phase = phase

    def on_fit_start(self) -> None:  # type: ignore[override]
        if not self._sequential_training:
            return
        self._set_training_phase("density")

    def on_train_start(self) -> None:  # type: ignore[override]
        if not self._sequential_training:
            return
        self._set_training_phase("density")

    def on_train_epoch_start(self) -> None:
        if not self._sequential_training:
            return
        if self.current_epoch >= self.density_phase_epochs:
            if self._optimize_density:
                self._optimize_density = False
                log.info("Completed density phase; freezing density model and switching to dynamics training.")
                if not self._density_frozen:
                    for param in self.bridge.density_model.parameters():
                        param.requires_grad = False
                    self._density_frozen = True
                self._set_training_phase("dynamics")
            self._optimize_dynamics = True
        else:
            self._set_training_phase("density")
