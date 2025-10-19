from __future__ import annotations

from typing import Any, Dict, Optional
from lightning import LightningModule
from torch import Tensor
from torchmetrics import MeanMetric

from src.models.components.decoupled_glow import DecoupledBridge


class DecoupledDensityLitModule(LightningModule):
    def __init__(
        self,
        bridge: DecoupledBridge,
        optimizer: Any,
        scheduler: Optional[Any] = None,
        log_train_metrics: bool = True,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["bridge"])
        self.bridge = bridge
        self.optimizer_partial = optimizer
        self.scheduler_partial = scheduler
        self.log_train_metrics = log_train_metrics

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        self._normalization_synced = False

    def forward(self, x0: Tensor) -> Tensor:
        x0 = x0.view(x0.shape[0], -1)
        return self.bridge.log_prob_initial(x0)

    def _ensure_normalization(self) -> None:
        if self._normalization_synced:
            return
        datamodule = getattr(self.trainer, "datamodule", None)
        if datamodule is None:
            raise RuntimeError("Datamodule is not attached; call Trainer.fit first.")
        if getattr(datamodule, "data_mean", None) is None or getattr(datamodule, "data_std", None) is None:
            raise RuntimeError("Datamodule must expose data_mean and data_std after setup().")
        mean = datamodule.data_mean.to(self.device)
        std = datamodule.data_std.to(self.device)
        self.bridge.update_normalization(mean, std)
        self._normalization_synced = True

    def _step(self, batch: Dict[str, Tensor]) -> Tensor:
        self._ensure_normalization()
        x0 = batch["x0"].view(batch["x0"].shape[0], -1)
        x0_input = self.bridge._apply_noise_if_training(x0)
        return -self.bridge.log_prob_initial(x0_input).mean()

    def training_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Tensor:
        loss = self._step(batch)
        self.train_loss(loss.detach())
        if self.log_train_metrics:
            self.log("train/loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: Dict[str, Tensor], batch_idx: int) -> None:
        loss = self._step(batch)
        self.val_loss(loss.detach())
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch: Dict[str, Tensor], batch_idx: int) -> None:
        loss = self._step(batch)
        self.test_loss(loss.detach())
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)

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
        self._ensure_normalization()