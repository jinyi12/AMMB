from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from lightning import LightningModule
from torch import Tensor
from torchmetrics import MeanMetric

from src.models.components.decoupled_glow import DecoupledBridge


class DecoupledDynamicsLitModule(LightningModule):
    def __init__(
        self,
        bridge: DecoupledBridge,
        optimizer: Any,
        scheduler: Optional[Any] = None,
        lambda_dynamics: float = 1.0,
        lambda_path: float = 0.0,
        log_train_metrics: bool = True,
        freeze_density: bool = True,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["bridge"])
        self.bridge = bridge
        self.optimizer_partial = optimizer
        self.scheduler_partial = scheduler
        self.lambda_dynamics = lambda_dynamics
        self.lambda_path = lambda_path
        self.log_train_metrics = log_train_metrics
        self.freeze_density = freeze_density

        self.train_total = MeanMetric()
        self.train_dynamics = MeanMetric()
        self.train_path = MeanMetric()
        self.train_density = MeanMetric()

        self.val_total = MeanMetric()
        self.val_dynamics = MeanMetric()
        self.val_path = MeanMetric()
        self.val_density = MeanMetric()

        self.test_total = MeanMetric()
        self.test_dynamics = MeanMetric()
        self.test_path = MeanMetric()
        self.test_density = MeanMetric()

        self._constraint_times: Optional[Tensor] = None
        self._normalization_synced = False
        self._density_frozen = False

    @property
    def active_times(self) -> Tensor:
        if self._constraint_times is None:
            raise RuntimeError("Constraint times are not set; call setup() first.")
        return self._constraint_times[1:]

    def _ensure_datamodule_state(self) -> None:
        datamodule = getattr(self.trainer, "datamodule", None)
        if datamodule is None:
            raise RuntimeError("Datamodule is not attached; call Trainer.fit first.")

        if not self._normalization_synced:
            if getattr(datamodule, "data_mean", None) is None or getattr(datamodule, "data_std", None) is None:
                raise RuntimeError("Datamodule must expose data_mean and data_std after setup().")
            mean = datamodule.data_mean.to(self.device)
            std = datamodule.data_std.to(self.device)
            self.bridge.update_normalization(mean, std)
            self._normalization_synced = True

        if self._constraint_times is None:
            if not hasattr(datamodule, "constraint_times"):
                raise RuntimeError("Datamodule must expose constraint_times after setup().")
            self._constraint_times = datamodule.constraint_times.to(self.device)

    def forward(self, x0: Tensor, t: Tensor | float) -> Tensor:
        return self.bridge.transport(x0, t)

    def model_step(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        self._ensure_datamodule_state()
        x0 = batch["x0"].view(batch["x0"].shape[0], -1)
        xt = batch.get("xt")
        if xt is not None:
            xt = xt.view(x0.shape[0], -1, x0.shape[1])
        else:
            xt = torch.zeros(x0.shape[0], 0, x0.shape[1], device=x0.device, dtype=x0.dtype)

        has_constraints = self.active_times.numel() > 0

        dynamics_loss = torch.tensor(0.0, device=self.device)
        if has_constraints:
            losses = []
            for idx, t_val in enumerate(self.active_times):
                target = xt[:, idx, :]
                t_tensor = torch.full((x0.shape[0], 1), float(t_val.item()), device=self.device, dtype=x0.dtype)
                prediction = self.bridge.transport(x0, t_tensor)
                losses.append(F.mse_loss(prediction, target))
            if losses:
                dynamics_loss = torch.stack(losses).mean()

        path_loss = torch.tensor(0.0, device=self.device)
        if self.lambda_path > 0 and has_constraints:
            t_choice = self.active_times[torch.randint(0, self.active_times.numel(), (1,), device=self.device)]
            t_tensor = torch.full((x0.shape[0], 1), float(t_choice.item()), device=self.device, dtype=x0.dtype)
            x_t = self.bridge.transport(x0, t_tensor).requires_grad_(True)
            velocity = self.bridge.forward_velocity(x_t, t_tensor)
            path_loss = 0.5 * (velocity**2).mean()

        density_loss = -self.bridge.log_prob_initial(x0).mean()
        total_loss = self.lambda_dynamics * dynamics_loss + self.lambda_path * path_loss

        return {
            "loss": total_loss,
            "dynamics": dynamics_loss.detach(),
            "path": path_loss.detach(),
            "density": density_loss.detach(),
        }

    def training_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Tensor:
        outputs = self.model_step(batch)
        self.train_total(outputs["loss"].detach())
        self.train_dynamics(outputs["dynamics"])
        self.train_path(outputs["path"])
        self.train_density(outputs["density"])
        if self.log_train_metrics:
            self.log("train/loss", self.train_total, on_step=True, on_epoch=True, prog_bar=True)
            self.log("train/dynamics", self.train_dynamics, on_step=False, on_epoch=True)
            if self.lambda_path > 0:
                self.log("train/path", self.train_path, on_step=False, on_epoch=True)
            self.log("train/density", self.train_density, on_step=False, on_epoch=True)
        return outputs["loss"]

    def validation_step(self, batch: Dict[str, Tensor], batch_idx: int) -> None:
        outputs = self.model_step(batch)
        self.val_total(outputs["loss"].detach())
        self.val_dynamics(outputs["dynamics"])
        self.val_path(outputs["path"])
        self.val_density(outputs["density"])
        self.log("val/loss", self.val_total, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/dynamics", self.val_dynamics, on_step=False, on_epoch=True)
        if self.lambda_path > 0:
            self.log("val/path", self.val_path, on_step=False, on_epoch=True)
        self.log("val/density", self.val_density, on_step=False, on_epoch=True)

    def test_step(self, batch: Dict[str, Tensor], batch_idx: int) -> None:
        outputs = self.model_step(batch)
        self.test_total(outputs["loss"].detach())
        self.test_dynamics(outputs["dynamics"])
        self.test_path(outputs["path"])
        self.test_density(outputs["density"])
        self.log("test/loss", self.test_total, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/dynamics", self.test_dynamics, on_step=False, on_epoch=True)
        if self.lambda_path > 0:
            self.log("test/path", self.test_path, on_step=False, on_epoch=True)
        self.log("test/density", self.test_density, on_step=False, on_epoch=True)

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

    def on_fit_start(self) -> None:  # type: ignore[override]
        self._ensure_datamodule_state()
        if self.freeze_density and not self._density_frozen:
            for param in self.bridge.density_model.parameters():
                param.requires_grad = False
            self.bridge.density_model.eval()
            self._density_frozen = True