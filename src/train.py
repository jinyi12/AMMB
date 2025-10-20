from typing import Any, Dict, List, Optional, Tuple

import hydra
import lightning as L
import rootutils
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, OmegaConf

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #

from src.models.decoupled_density_module import DecoupledDensityLitModule  # noqa: E402
from src.models.decoupled_dynamics_module import DecoupledDynamicsLitModule  # noqa: E402
from src.utils import (
    RankedLogger,
    extras,
    get_metric_value,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)  # noqa: E402

log = RankedLogger(__name__, rank_zero_only=True)


@task_wrapper
def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A tuple with metrics and dict with all instantiated objects.
    """
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    sequential_cfg = cfg.get("sequential")
    sequential_enabled = bool(sequential_cfg.get("enabled")) if sequential_cfg is not None else False
    if sequential_enabled:
        log.info("Running sequential decoupled training pipeline.")
        return _train_decoupled_sequential(cfg, datamodule, sequential_cfg)

    model_cfg = cfg.get("model")
    if model_cfg is None:
        raise ValueError("Model configuration is required when sequential training is disabled.")
    model_target = getattr(model_cfg, "_target_", "<unknown>")
    log.info(f"Instantiating model <{model_target}>")
    model: LightningModule = hydra.utils.instantiate(model_cfg)

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    if cfg.get("train"):
        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

    train_metrics = trainer.callback_metrics

    if cfg.get("test"):
        log.info("Starting testing!")
        checkpoint_callback = getattr(trainer, "checkpoint_callback", None)
        ckpt_path = None
        if checkpoint_callback is not None and getattr(checkpoint_callback, "best_model_path", ""):
            ckpt_path = checkpoint_callback.best_model_path
            log.info(f"Using best ckpt for testing: {ckpt_path}")
        elif getattr(trainer, "fast_dev_run", False):
            log.info("Fast dev run detected; skipping checkpoint-based testing.")
        else:
            log.warning("Best ckpt not found! Using current weights for testing...")
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        log.info(f"Best ckpt path: {ckpt_path}")

    test_metrics = trainer.callback_metrics

    # merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict, object_dict


def _config_to_kwargs(cfg_section: Optional[DictConfig]) -> Dict[str, Any]:
    if not cfg_section:
        return {}
    container = OmegaConf.to_container(cfg_section, resolve=True)
    if not isinstance(container, dict):
        raise TypeError("Expected configuration section to resolve to a mapping.")
    return container


def _maybe_instantiate(config: Optional[DictConfig]) -> Optional[Any]:
    if not config:
        return None
    return hydra.utils.instantiate(config)


def _train_decoupled_sequential(
    cfg: DictConfig,
    datamodule: LightningDataModule,
    sequential_cfg: DictConfig,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if cfg.get("ckpt_path"):
        raise NotImplementedError("Sequential training does not yet support resuming from checkpoints.")

    bridge_cfg = sequential_cfg.get("bridge")
    if bridge_cfg is None:
        raise ValueError("Sequential training requires a bridge configuration.")
    density_cfg = sequential_cfg.get("density")
    dynamics_cfg = sequential_cfg.get("dynamics")
    if density_cfg is None or density_cfg.get("optimizer") is None:
        raise ValueError("Sequential density configuration must define an optimizer.")
    if dynamics_cfg is None or dynamics_cfg.get("optimizer") is None:
        raise ValueError("Sequential dynamics configuration must define an optimizer.")

    bridge = hydra.utils.instantiate(bridge_cfg)

    if hasattr(datamodule, "setup"):
        datamodule.setup(stage="fit")

    data_mean = getattr(datamodule, "data_mean", None)
    data_std = getattr(datamodule, "data_std", None)
    if data_mean is None or data_std is None:
        raise RuntimeError(
            "Sequential training requires the datamodule to expose data_mean and data_std after setup()."
        )

    constraint_times = getattr(datamodule, "constraint_times", None)
    if constraint_times is None:
        raise RuntimeError(
            "Sequential training requires the datamodule to expose constraint_times after setup()."
        )

    density_optimizer_partial = hydra.utils.instantiate(density_cfg.optimizer)
    density_scheduler_partial = _maybe_instantiate(density_cfg.get("scheduler"))
    density_kwargs = _config_to_kwargs(density_cfg.get("module"))
    density_module = DecoupledDensityLitModule(
        bridge=bridge,
        optimizer=density_optimizer_partial,
        scheduler=density_scheduler_partial,
        data_mean=data_mean,
        data_std=data_std,
        **density_kwargs,
    )

    dynamics_optimizer_partial = hydra.utils.instantiate(dynamics_cfg.optimizer)
    dynamics_scheduler_partial = _maybe_instantiate(dynamics_cfg.get("scheduler"))
    dynamics_kwargs = _config_to_kwargs(dynamics_cfg.get("module"))
    dynamics_module = DecoupledDynamicsLitModule(
        bridge=bridge,
        optimizer=dynamics_optimizer_partial,
        scheduler=dynamics_scheduler_partial,
        data_mean=data_mean,
        data_std=data_std,
        constraint_times=constraint_times,
        **dynamics_kwargs,
    )

    density_epochs = int(sequential_cfg.get("density_epochs", 0))
    dynamics_epochs = sequential_cfg.get("dynamics_epochs")
    if dynamics_epochs is None:
        dynamics_epochs = cfg.trainer.get("max_epochs", 0)

    density_metrics: Dict[str, Any] = {}
    density_trainer: Optional[Trainer] = None

    if cfg.get("train") and density_epochs > 0:
        log.info("Starting density phase training.")
        if hasattr(datamodule, "set_training_phase"):
            datamodule.set_training_phase("density")
        density_callbacks = instantiate_callbacks(cfg.get("callbacks"))
        density_trainer = hydra.utils.instantiate(
            cfg.trainer,
            callbacks=density_callbacks,
            logger=None,
            max_epochs=density_epochs,
        )
        density_trainer.fit(density_module, datamodule=datamodule)
        density_metrics = {f"density/{k}": v for k, v in dict(density_trainer.callback_metrics).items()}

    if hasattr(datamodule, "set_training_phase"):
        datamodule.set_training_phase("dynamics")

    logger_instances: List[Logger] = instantiate_loggers(cfg.get("logger"))
    dynamics_callbacks = instantiate_callbacks(cfg.get("callbacks"))
    dynamics_trainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=dynamics_callbacks,
        logger=logger_instances if logger_instances else None,
        max_epochs=dynamics_epochs,
    )

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": dynamics_module,
        "density_module": density_module,
        "dynamics_module": dynamics_module,
        "callbacks": dynamics_callbacks,
        "logger": logger_instances,
        "trainer": dynamics_trainer,
        "density_trainer": density_trainer,
    }

    if logger_instances:
        log.info("Logging hyperparameters for dynamics phase.")
        log_hyperparameters(object_dict)

    train_metrics: Dict[str, Any] = density_metrics.copy()
    if cfg.get("train"):
        log.info("Starting dynamics phase training.")
        dynamics_trainer.fit(dynamics_module, datamodule=datamodule)
        train_metrics.update(dict(dynamics_trainer.callback_metrics))

    test_metrics: Dict[str, Any] = {}
    if cfg.get("test"):
        log.info("Starting sequential testing.")
        checkpoint_callback = getattr(dynamics_trainer, "checkpoint_callback", None)
        ckpt_path = None
        if checkpoint_callback is not None and getattr(checkpoint_callback, "best_model_path", ""):
            ckpt_path = checkpoint_callback.best_model_path
            log.info(f"Using best ckpt for testing: {ckpt_path}")
        elif getattr(dynamics_trainer, "fast_dev_run", False):
            log.info("Fast dev run detected; skipping checkpoint-based testing.")
        else:
            log.warning("Best ckpt not found! Using current weights for testing...")
        dynamics_trainer.test(model=dynamics_module, datamodule=datamodule, ckpt_path=ckpt_path)
        test_metrics = dict(dynamics_trainer.callback_metrics)

    metric_dict = {**train_metrics, **test_metrics}
    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    # Register eval resolver for arithmetic expressions in configs
    OmegaConf.register_new_resolver("eval", eval)

    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    # train the model
    metric_dict, _ = train(cfg)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )

    # return optimized metric
    return metric_value


if __name__ == "__main__":
    main()
