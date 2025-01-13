"""Perform hyperparameter tuning using the Optuna framework.

This script is configured using the Hydra framework, with configuration
details specified in the "src/conf/" directory.  The configuration file
associated with this script is named "train_classifier_optuna.yaml".

Typical usage example:

  >>> python train_classifier_optuna.py experiment=lenet_fashionmnist/hparam_tuning/optuna/step_lr_random
"""


import logging

import hydra
import optuna
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from optuna.trial import Trial
from torchmetrics import MetricCollection

from src.base_classes.base_loader import BaseLoader
from src.config import TrainClassifierConf
from src.training.classification_trainer import ClassificationTrainer
from src.utils.training import (
    get_collate_fn,
    get_lr_scheduler,
    get_train_transform,
    get_val_transform,
    log_study_results,
    log_trial_parameters,
    set_device,
    set_seeds,
    suggest_trial_parameters
)

logger = logging.getLogger(__name__)

cs = ConfigStore.instance()
cs.store(name="train_classifier_conf", node=TrainClassifierConf)


@hydra.main(version_base=None, config_path="conf", config_name="train_classifier_optuna")
def main(cfg: TrainClassifierConf) -> None:
    """Perform hyperparameter tuning using the Optuna framework."""

    def objective(trial: Trial) -> float:
        """Objective function to be optimized by Optuna."""

        # Prepare Optuna trial
        logger.info(
            "OPTUNA TRIAL [%0*d/%d]",
            len(str(cfg.optuna.n_trials)), trial.number + 1, cfg.optuna.n_trials
        )
        suggest_trial_parameters(cfg, trial)
        log_trial_parameters(logger, trial)

        # Set target device
        device = set_device()
        logger.info("Target device is set to: %s.", device.type.upper())

        # Set seeds for reproducibility
        set_seeds(cfg.reproducibility)

        # Prepare transforms and dataset
        logger.info("Preparing transforms and dataset ...")
        train_transform = get_train_transform(cfg.transform.train)
        val_transform = get_val_transform(cfg.transform.val)
        dataset = instantiate(cfg.dataset.train_set)

        # Set up dataloaders
        logger.info("Preparing dataloaders ...")
        collate_fn = get_collate_fn(
            transform_train_params=cfg.transform.train,
            num_classes=cfg.dataset.num_classes
        )
        multiprocessing_context = None
        if device.type == "mps" and cfg.dataloader.num_workers > 0:
            multiprocessing_context = "fork"
        base_loader = BaseLoader(
            dataset=dataset,
            main_transform=train_transform,
            val_transform=val_transform,
            val_split=cfg.dataloader.val_split,
            batch_size=cfg.dataloader.batch_size,
            shuffle=True,
            num_workers=cfg.dataloader.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            multiprocessing_context=multiprocessing_context,
            seeds=cfg.reproducibility
        )
        train_loader = base_loader.get_dataloader(mode="main")
        val_loader = base_loader.get_dataloader(mode="val")

        # Set up criterion
        logger.info("Setting up criterion ...")
        criterion = instantiate(cfg.criterion.kwargs)

        # Instantiate metrics to track during training
        logger.info("Instantiating metrics ...")
        prediction_metrics = MetricCollection({
            name: instantiate(metric) for name, metric in cfg.performance.metrics.items()
        })

        # Instantiate model and optimizer
        logger.info("Instantiating model and optimizer ...")
        model = instantiate(cfg.model.kwargs).to(device)

        optimizer = instantiate(
            cfg.optimizer.kwargs,
            params=model.parameters()
        )

        # Set up learning rate scheduler
        logger.info("Setting up learning rate scheduler (if specified) ...")
        lr_scheduler = get_lr_scheduler(
            cfg=cfg,
            optimizer=optimizer
        )

        # Instantiate trainer and start training
        # NOTE: Training is automatically resumed if a checkpoint is provided.  It's the user's
        #       responsibility to ensure that the checkpoint is compatible with the current model,
        #       optimizer, and scheduler.
        logger.info("Setting up trainer ...")
        trainer = ClassificationTrainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            train_loader=train_loader,
            val_loader=val_loader,
            prediction_metrics=prediction_metrics,
            device=device,
            cfg=cfg,
            lr_scheduler=lr_scheduler,
            run_id=trial.number + 1,
            trial=trial
        )
        best_score, _ = trainer.train()

        return best_score

    # Suppress Optuna logging
    optuna.logging.set_verbosity(optuna.logging.CRITICAL)

    # Run Optuna study
    logger.info("Creating new Optuna study with name %s ...", cfg.optuna.study_name)
    pruner = instantiate(cfg.optuna.pruner.kwargs) if "pruner" in cfg.optuna \
        else optuna.pruners.NopPruner()
    study = optuna.create_study(
        storage="sqlite:///../optuna_studies.sqlite3",
        sampler=instantiate(cfg.optuna.sampler.kwargs),
        pruner=pruner,
        study_name=cfg.optuna.study_name,
        direction=("minimize" if cfg.optuna.minimize else "maximize")
    )
    study.optimize(
        func=objective,
        n_trials=cfg.optuna.n_trials
    )

    # Log results
    log_study_results(logger, study)


if __name__ == "__main__":
    main()
