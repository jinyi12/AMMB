"""LightningDataModule for the decoupled Glow bridge experiments.

This module centralises data generation, normalisation, and batching logic so
that the LightningModule can focus solely on optimisation concerns. It wraps
legacy utilities (ported into ``src.data.components``) and exposes batched
trajectories that stay aligned across all constraint times.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split

from .components.multimarginal_generation import (
    generate_multiscale_grf_data,
    generate_spiral_distributional_data,
    normalize_multimarginal_data,
)


class InitialMarginalDataset(Dataset):
    """Dataset containing only the initial marginal x_0."""

    def __init__(self, data: torch.Tensor) -> None:
        self.data = data

    def __len__(self) -> int:  # noqa: D401
        return self.data.shape[0]

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:  # noqa: D401
        return {"x0": self.data[index]}


class TrajectoryDataset(Dataset):
    """Dataset returning aligned trajectories across constraint times."""

    def __init__(self, normalized_data: Dict[float, torch.Tensor], times: Iterable[float]) -> None:
        sorted_times = sorted(times)
        if not sorted_times:
            raise ValueError("TrajectoryDataset requires at least one time point")

        self.initial_time = sorted_times[0]
        self.other_times: List[float] = [t for t in sorted_times if abs(t - self.initial_time) > 1e-9]

        self.x0 = normalized_data[self.initial_time]
        self.marginals: Dict[float, torch.Tensor] = {
            t: normalized_data[t] for t in self.other_times
        }
        self.length = self.x0.shape[0]

    def __len__(self) -> int:  # noqa: D401
        return self.length

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:  # noqa: D401
        x0 = self.x0[index]
        if self.other_times:
            xt = torch.stack([self.marginals[t][index] for t in self.other_times], dim=0)
        else:
            xt = x0.new_zeros((0, x0.shape[0]))
        return {"x0": x0, "xt": xt}


class MultiMarginalDataModule(LightningDataModule):
    """LightningDataModule encapsulating the synthetic multi-marginal datasets."""

    def __init__(
        self,
        data_dir: str = "data/",
        data_type: str = "grf",
        batch_size: int = 64,
        dynamics_batch_size: Optional[int] = None,
        val_split: float = 0.1,
        num_workers: int = 0,
        pin_memory: bool = False,
        # Shared configuration parameters
        T: float = 1.0,
        n_constraints: int = 5,
        # GRF-specific configuration
        n_samples: int = 1024,
        resolution: int = 32,
        L_domain: float = 1.0,
        micro_corr_length: float = 0.1,
        H_max_factor: float = 0.5,
        mean_val: float = 10.0,
        std_val: float = 2.0,
        covariance_type: str = "gaussian",
        generation_method: str = "fft",
        kl_error_threshold: float = 1e-3,
        schedule_type: str = "geometric",
        # Spiral-specific configuration
        spiral_data_dim: int = 3,
        spiral_samples_per_marginal: int = 512,
        spiral_noise_std: float = 0.1,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        self._raw_marginals: Optional[Dict[float, torch.Tensor]] = None
        self._normalized_marginals: Optional[Dict[float, torch.Tensor]] = None
        self.data_mean: Optional[torch.Tensor] = None
        self.data_std: Optional[torch.Tensor] = None
        self.times: Optional[List[float]] = None
        self.times_tensor: Optional[torch.Tensor] = None

        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None
        self.density_dataset: Optional[Dataset] = None
        self._training_phase: str = "dynamics"

    # ------------------------------------------------------------------
    # Lightning API
    # ------------------------------------------------------------------

    def prepare_data(self) -> None:  # noqa: D401
        # Data is generated on-the-fly during setup; nothing to download.
        return

    def setup(self, stage: Optional[str] = None) -> None:  # noqa: D401
        if self.train_dataset is not None and stage in {"fit", None}:
            return

        device = "cpu"  # Data generation happens on CPU to avoid unnecessary transfers

        if self.hparams.data_type == "grf":
            raw_marginals, _ = generate_multiscale_grf_data(
                N_samples=self.hparams.n_samples,
                T=self.hparams.T,
                N_constraints=self.hparams.n_constraints,
                resolution=self.hparams.resolution,
                L_domain=self.hparams.L_domain,
                micro_corr_length=self.hparams.micro_corr_length,
                H_max_factor=self.hparams.H_max_factor,
                mean_val=self.hparams.mean_val,
                std_val=self.hparams.std_val,
                covariance_type=self.hparams.covariance_type,
                device=device,
                generation_method=self.hparams.generation_method,
                kl_error_threshold=self.hparams.kl_error_threshold,
                schedule_type=self.hparams.schedule_type,
            )
        elif self.hparams.data_type == "spiral":
            raw_marginals, _ = generate_spiral_distributional_data(
                N_constraints=self.hparams.n_constraints,
                T=self.hparams.T,
                data_dim=self.hparams.spiral_data_dim,
                N_samples_per_marginal=self.hparams.spiral_samples_per_marginal,
                noise_std=self.hparams.spiral_noise_std,
                device=device,
            )
        else:
            raise ValueError(f"Unsupported data_type: {self.hparams.data_type}")

        normalized_marginals, mean, std = normalize_multimarginal_data(raw_marginals)

        self._raw_marginals = raw_marginals
        self._normalized_marginals = normalized_marginals
        self.data_mean = mean
        self.data_std = std
        self.times = sorted(normalized_marginals.keys())
        self.times_tensor = torch.tensor(self.times, dtype=torch.float32)

        trajectory_dataset = TrajectoryDataset(normalized_marginals, self.times)
        density_dataset = InitialMarginalDataset(normalized_marginals[self.times[0]])

        if self.hparams.val_split > 0:
            val_length = max(1, int(len(trajectory_dataset) * self.hparams.val_split))
            train_length = len(trajectory_dataset) - val_length
            if train_length <= 0:
                raise ValueError("Validation split too large; no samples left for training")
            self.train_dataset, self.val_dataset = random_split(
                trajectory_dataset,
                [train_length, val_length],
                generator=torch.Generator().manual_seed(42),
            )
            # Reuse the validation subset for testing until a dedicated split is required.
            self.test_dataset = self.val_dataset
        else:
            self.train_dataset = trajectory_dataset
            self.val_dataset = None
            self.test_dataset = None
        self.density_dataset = density_dataset

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    @property
    def normalized_marginals(self) -> Dict[float, torch.Tensor]:
        if self._normalized_marginals is None:
            raise RuntimeError("Data has not been prepared yet. Call setup().")
        return self._normalized_marginals

    @property
    def raw_marginals(self) -> Dict[float, torch.Tensor]:
        if self._raw_marginals is None:
            raise RuntimeError("Data has not been prepared yet. Call setup().")
        return self._raw_marginals

    @property
    def constraint_times(self) -> torch.Tensor:
        if self.times_tensor is None:
            raise RuntimeError("Data has not been prepared yet. Call setup().")
        return self.times_tensor

    def get_batch_by_indices(self, indices: torch.Tensor) -> Dict[float, torch.Tensor]:
        """Return a dictionary of marginals filtered by sample indices."""
        data = {
            t: marginal[indices] for t, marginal in self.normalized_marginals.items()
        }
        return data

    # ------------------------------------------------------------------
    # DataLoaders
    # ------------------------------------------------------------------

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise RuntimeError("train_dataloader called before setup")

        if self._training_phase == "density":
            if self.density_dataset is None:
                raise RuntimeError("Density dataset unavailable; ensure setup() has run.")
            return DataLoader(
                self.density_dataset,
                batch_size=self.hparams.batch_size,
                shuffle=True,
                num_workers=self.hparams.num_workers,
                pin_memory=self.hparams.pin_memory,
            )

        batch_size = self.hparams.dynamics_batch_size or self.hparams.batch_size
        return DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )

    def val_dataloader(self) -> Optional[DataLoader]:
        if self.val_dataset is None:
            return None
        batch_size = self.hparams.dynamics_batch_size or self.hparams.batch_size
        return DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )

    def test_dataloader(self) -> Optional[DataLoader]:
        dataset = self.test_dataset or self.val_dataset
        if dataset is None:
            return None
        batch_size = self.hparams.dynamics_batch_size or self.hparams.batch_size
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )

    def density_dataloader(self) -> DataLoader:
        if self.density_dataset is None:
            raise RuntimeError("Density dataset unavailable; ensure setup() has run.")
        return DataLoader(
            self.density_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )

    # ------------------------------------------------------------------
    # Checkpoint support
    # ------------------------------------------------------------------

    def state_dict(self) -> Dict[str, torch.Tensor]:
        if self.data_mean is None or self.data_std is None or self.times_tensor is None:
            return {}
        return {
            "data_mean": self.data_mean,
            "data_std": self.data_std,
            "times_tensor": self.times_tensor,
        }

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:  # noqa: D401
        if not state_dict:
            return
        if "data_mean" in state_dict:
            self.data_mean = state_dict["data_mean"]
        if "data_std" in state_dict:
            self.data_std = state_dict["data_std"]
        if "times_tensor" in state_dict:
            self.times_tensor = state_dict["times_tensor"]
            self.times = state_dict["times_tensor"].tolist()

    def teardown(self, stage: Optional[str] = None) -> None:  # noqa: D401
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.density_dataset = None

    def set_training_phase(self, phase: str) -> bool:
        if phase not in {"density", "dynamics"}:
            raise ValueError("phase must be either 'density' or 'dynamics'")
        if self._training_phase == phase:
            return False
        self._training_phase = phase
        return True
