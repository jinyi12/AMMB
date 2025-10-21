from __future__ import annotations

from typing import Any, Dict, Optional

from torch import Tensor
from torchmetrics import MeanMetric

from src.models.components.density_wrapper import DensityWrapper
from src.models.base_decoupled_module import BaseDecoupledLitModule


class DecoupledDensityLitModule(BaseDecoupledLitModule):
    def __init__(
        self,
        density_wrapper: DensityWrapper,
        optimizer: Any,
        scheduler: Optional[Any] = None,
        log_train_metrics: bool = True,
    ) -> None:
        """Initialize density training module with lightweight DensityWrapper.
        
        This module is fully decoupled from dynamics components, adhering to
        the Interface Segregation Principle. It depends only on:
        - DensityWrapper: density model + normalization + noise utilities
        - Optimizer/scheduler configurations
        
        This eliminates unnecessary coupling and mutable state issues.
        
        Args:
            density_wrapper: DensityWrapper with pre-instantiated density model
            optimizer: Optimizer factory (partial)
            scheduler: Optional scheduler factory
            log_train_metrics: Whether to log training metrics
        """
        super().__init__(
            optimizer=optimizer,
            scheduler=scheduler,
            log_train_metrics=log_train_metrics,
        )
        
        # Explicit dependency: density wrapper is stored directly (no wrapper)
        self.density_wrapper = density_wrapper

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # Ensure density model is trainable
        self.density_wrapper.model.train()
        for param in self.density_wrapper.model.parameters():
            param.requires_grad = True

    def forward(self, x0: Tensor) -> Tensor:
        """Forward pass computes log probability."""
        x0_flat = x0.view(x0.shape[0], -1) if x0.dim() > 2 else x0
        return self.density_wrapper.log_prob(x0_flat)

    def _step(self, batch: Dict[str, Tensor]) -> Tensor:
        """Compute loss for a batch."""
        x0 = batch["x0"].view(batch["x0"].shape[0], -1)
        # Negative log-likelihood: loss = -log p(x)
        return -self.density_wrapper.log_prob(x0).mean()

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
        """Return only density model parameters for optimization."""
        return list(self.density_wrapper.model.parameters())
