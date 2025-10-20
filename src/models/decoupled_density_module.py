from __future__ import annotations

from typing import Any, Dict, Optional

from torch import Tensor
from torchmetrics import MeanMetric

from src.models.components.decoupled_glow import DecoupledBridge
from src.models.base_decoupled_module import BaseDecoupledLitModule


class DecoupledDensityLitModule(BaseDecoupledLitModule):
    def __init__(
        self,
        bridge: DecoupledBridge,
        optimizer: Any,
        scheduler: Optional[Any] = None,
        log_train_metrics: bool = True,
        data_mean: Optional[Tensor] = None,
        data_std: Optional[Tensor] = None,
    ) -> None:
        super().__init__(
            bridge=bridge,
            optimizer=optimizer,
            scheduler=scheduler,
            log_train_metrics=log_train_metrics,
        )

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        self._normalization_synced = False
        if (data_mean is None) != (data_std is None):
            raise ValueError("Both data_mean and data_std must be provided together.")
        if data_mean is not None and data_std is not None:
            self.set_normalization(data_mean, data_std)

    def forward(self, x0: Tensor) -> Tensor:
        self._ensure_normalization()
        x0 = x0.view(x0.shape[0], -1)
        return self.bridge.log_prob_initial(x0)

    def _ensure_normalization(self) -> None:
        if not self._normalization_synced:
            raise RuntimeError(
                "Normalization statistics not provided; call set_normalization() before using the module."
            )

    def set_normalization(self, mean: Tensor, std: Tensor) -> None:
        mean = mean.to(device=self.bridge.data_mean.device, dtype=self.bridge.data_mean.dtype)
        std = std.to(device=self.bridge.data_std.device, dtype=self.bridge.data_std.dtype)
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

    def get_optimized_parameters(self):
        return list(self.bridge.density_model.parameters())
