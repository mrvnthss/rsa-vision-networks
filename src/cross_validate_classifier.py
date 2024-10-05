"""Train a classification model in PyTorch w/ cross-validation.

This script is configured using the Hydra framework, with configuration
details specified in the "src/conf/" directory.  The configuration file
associated with this script is named "cross_validate_classifier.yaml".

Typical usage example:

  >>> python cross_validate_classifier.py experiment=lenet_fashionmnist/grid_search/overfitting
"""


import logging
from pathlib import Path

import hydra
import torch
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torchmetrics import MetricCollection

from src.config import TrainClassifierConf
from src.dataloaders.stratified_k_fold_loader import StratifiedKFoldLoader
from src.training.classification_trainer import ClassificationTrainer
from src.utils.training import get_collate_fn, get_lr_scheduler, get_train_transform, \
    get_val_transform, set_device, set_seeds

logger = logging.getLogger(__name__)

cs = ConfigStore.instance()
cs.store(name="train_classifier_conf", node=TrainClassifierConf)


@hydra.main(version_base=None, config_path="conf", config_name="cross_validate_classifier")
def main(cfg: TrainClassifierConf) -> None:
    """Train a classification model in PyTorch w/ cross-validation."""

    if "num_folds" not in cfg.training or cfg.training.num_folds is None:
        raise ValueError(
            "The number 'cfg.training.num_folds' of folds for cross-validation must be specified, "
            "to run the 'cross_validate_classifier.py' script. To train a model without "
            "cross-validation, use the 'train_classifier.py' script."
        )

    if cfg.training.num_folds < 2:
        raise ValueError(
            "The number of folds for cross-validation should be an integer greater than 1, "
            f"but got {cfg.training.num_folds}."
        )

    # Do not resume training for cross-validation runs
    if "resume_from" in cfg.training:
        cfg.training.resume_from = None
    else:
        OmegaConf.set_struct(cfg, False)
        cfg.training.resume_from = None
        OmegaConf.set_struct(cfg, True)

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

    # Set up folds for stratified k-fold cross-validation
    logger.info(
        "Preparing folds for stratified %s-fold cross-validation ...",
        cfg.training.num_folds
    )
    collate_fn = get_collate_fn(
        transform_train_params=cfg.transform.train,
        num_classes=cfg.dataset.num_classes
    )
    multiprocessing_context = None
    if device.type == "mps" and cfg.dataloader.num_workers > 0:
        multiprocessing_context = "fork"
    stratified_k_fold_loader = StratifiedKFoldLoader(
        dataset=dataset,
        train_transform=train_transform,
        val_transform=val_transform,
        num_folds=cfg.training.num_folds,
        batch_size=cfg.dataloader.batch_size,
        shuffle=True,
        num_workers=cfg.dataloader.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        multiprocessing_context=multiprocessing_context,
        seeds=cfg.reproducibility
    )

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

    # Save model, optimizer, and learning rate scheduler states to reset them for each fold
    init_model_state_dict_path = Path(cfg.experiment.dir) / "init_model_state_dict.pt"
    torch.save(model.state_dict(), init_model_state_dict_path)

    init_optimizer_state_dict_path = Path(cfg.experiment.dir) / "init_optimizer_state_dict.pt"
    torch.save(optimizer.state_dict(), init_optimizer_state_dict_path)

    init_scheduler_state_dict_path = None
    if lr_scheduler is not None:
        init_scheduler_state_dict_path = Path(
            cfg.experiment.dir) / "init_scheduler_state_dict.pt"
        torch.save(lr_scheduler.state_dict(), init_scheduler_state_dict_path)

    # Iterate over individual folds
    for fold_idx in range(cfg.training.num_folds):
        logger.info(
            "CROSS-VALIDATION RUN %0*d/%d",
            len(str(cfg.training.num_folds)), fold_idx + 1, cfg.training.num_folds
        )

        # Set up dataloaders
        logger.info("Preparing dataloaders ...")
        train_loader = stratified_k_fold_loader.get_dataloader(
            fold_idx=fold_idx,
            mode="train"
        )
        val_loader = stratified_k_fold_loader.get_dataloader(
            fold_idx=fold_idx,
            mode="val"
        )

        if fold_idx > 0:
            # Reset seeds
            set_seeds(cfg.reproducibility)

            # Reset model, optimizer, and learning rate scheduler states
            logger.info("Resetting model and optimizer states ...")
            model.load_state_dict(
                torch.load(init_model_state_dict_path, weights_only=True)
            )

            optimizer.load_state_dict(
                torch.load(init_optimizer_state_dict_path, weights_only=True)
            )

            if init_scheduler_state_dict_path is not None:
                logger.info("Resetting scheduler state ...")
                lr_scheduler.load_state_dict(
                    torch.load(init_scheduler_state_dict_path, weights_only=True)
                )

        # Instantiate trainer and start training
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
            run_id=fold_idx + 1
        )
        trainer.train()

    # Remove initial model and optimizer state dicts after last fold
    init_model_state_dict_path.unlink(missing_ok=True)
    init_optimizer_state_dict_path.unlink(missing_ok=True)
    if init_scheduler_state_dict_path is not None:
        init_scheduler_state_dict_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
