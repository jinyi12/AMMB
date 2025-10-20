from __future__ import annotations

from typing import Any, Iterable, Optional

from lightning import LightningModule

from src.models.components.decoupled_glow import DecoupledBridge


class BaseDecoupledLitModule(LightningModule):
    """Shared LightningModule utilities for decoupled density and dynamics phases."""

    def __init__(
        self,
        *,
        bridge: DecoupledBridge,
        optimizer: Any,
        scheduler: Optional[Any] = None,
        log_train_metrics: bool = True,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(
            logger=False,
            ignore=["bridge", "optimizer", "scheduler"],
        )
        self.bridge = bridge
        self.optimizer_partial = optimizer
        self.scheduler_partial = scheduler
        self.log_train_metrics = log_train_metrics

    # ------------------------------------------------------------------
    # Optimization utilities
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        params = list(self.get_optimized_parameters())
        if not params:
            raise ValueError("No parameters returned by get_optimized_parameters().")

        optimizer = self.optimizer_partial(params=params)
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

    def get_optimized_parameters(self) -> Iterable:
        """Return an iterable of parameters that should be optimized."""
        raise NotImplementedError
