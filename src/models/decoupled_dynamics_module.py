from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torchmetrics import MeanMetric

from src.models.components.decoupled_glow import TransportBridge
from src.models.base_decoupled_module import BaseDecoupledLitModule


class DecoupledDynamicsLitModule(BaseDecoupledLitModule):
    def __init__(
        self,
        bridge: TransportBridge,
        constraint_times: Tensor,
        optimizer: Any,
        scheduler: Optional[Any] = None,
        lambda_dynamics: float = 1.0,
        lambda_path: float = 0.0,
        log_train_metrics: bool = True,
        freeze_density: bool = True,
    ) -> None:
        """Initialize dynamics training module with TransportBridge.
        
        This module receives a pre-trained TransportBridge with:
        - DensityWrapper: pre-trained density model + normalization
        - InvertibleNeuralFlow: dynamics model to be trained
        
        Freezing happens in __init__ because sequential orchestration ensures
        density training is complete before this module is created.
        
        Args:
            bridge: TransportBridge with pre-trained density and fresh dynamics
            constraint_times: 1D tensor of constraint time points
            optimizer: Optimizer factory (partial)
            scheduler: Optional scheduler factory
            lambda_dynamics: Weight for dynamics loss
            lambda_path: Weight for path regularization
            log_train_metrics: Whether to log training metrics
            freeze_density: Whether to freeze density model (should be True)
        """
        super().__init__(
            optimizer=optimizer,
            scheduler=scheduler,
            log_train_metrics=log_train_metrics,
        )
        
        # Explicit dependency: transport bridge with both models
        self.bridge = bridge
        self.lambda_dynamics = lambda_dynamics
        self.lambda_path = lambda_path
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

        # Store constraint times immediately
        if constraint_times.ndim != 1:
            raise ValueError("constraint_times tensor must be 1D with monotonic entries.")
        if constraint_times.numel() < 2:
            raise ValueError("constraint_times must contain at least start and end times.")
        self.register_buffer("_constraint_times", constraint_times.detach().clone(), persistent=False)
        
        # Freeze density model if requested (safe now that initialization is sequential)
        if self.freeze_density:
            self._freeze_density()

    @property
    def active_times(self) -> Tensor:
        """Return constraint times excluding t=0."""
        return self._constraint_times[1:]

    def _freeze_density(self) -> None:
        """Freeze density model parameters and set to eval mode."""
        for param in self.bridge.density.model.parameters():
            param.requires_grad = False
        self.bridge.density.model.eval()

    def forward(self, x0: Tensor, t: Tensor | float) -> Tensor:
        return self.bridge.transport(x0, t)

    def model_step(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
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
            t_choice = self.active_times[torch.randint(0, self.active_times.numel(), (1,), device=self.active_times.device)]
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

    def get_optimized_parameters(self) -> Iterable:
        """Return only dynamics model parameters for optimization."""
        return list(self.bridge.dynamics.parameters())