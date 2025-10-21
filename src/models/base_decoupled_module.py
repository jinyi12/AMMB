from __future__ import annotations

from typing import Any, Iterable, Optional

from lightning import LightningModule


class BaseDecoupledLitModule(LightningModule):
    """Lightweight base class providing only optimizer configuration utilities.
    
    This base class provides shared optimizer/scheduler configuration logic
    without imposing specific component requirements (bridge, wrapper, etc).
    
    Subclasses should:
    1. Call super().__init__() 
    2. Set self.optimizer_partial and self.scheduler_partial
    3. Implement get_optimized_parameters()
    4. Manage their own dependencies (DensityWrapper, TransportBridge, etc)
    
    This design adheres to the Interface Segregation Principle by providing
    only the minimal shared utilities without forcing unnecessary coupling.
    """

    def __init__(
        self,
        *,
        optimizer: Any,
        scheduler: Optional[Any] = None,
        log_train_metrics: bool = True,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(
            logger=False,
            ignore=["optimizer", "scheduler"],
        )
        self.optimizer_partial = optimizer
        self.scheduler_partial = scheduler
        self.log_train_metrics = log_train_metrics

    # ------------------------------------------------------------------
    # Optimization utilities
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        """Configure optimizer and optional scheduler.
        
        This implementation calls get_optimized_parameters() to retrieve
        parameters, then instantiates the optimizer and scheduler.
        
        Subclasses should not override this unless they have special needs.
        Instead, override get_optimized_parameters() to specify which params
        should be optimized.
        """
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
        """Return an iterable of parameters that should be optimized.
        
        Subclasses MUST implement this to specify which model parameters
        should receive gradient updates.
        
        Examples:
            - Density module: return density_wrapper.model.parameters()
            - Dynamics module: return bridge.dynamics.parameters()
        
        Returns:
            Iterable of torch.nn.Parameter objects
            
        Raises:
            NotImplementedError: If not overridden by subclass
        """
        raise NotImplementedError
