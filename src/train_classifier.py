"""Train a model for image classification in PyTorch.

This script is configured using the Hydra framework, with configuration
details specified in the "src/conf/" directory.  The configuration file
associated with this script is named "train_classifier.yaml".

Typical usage example:

  >>> python train_classifier.py experiment=lenet_fashionmnist/grid_search/batch_size_lr
"""


import logging
from pathlib import Path

import hydra
import torch
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from torchmetrics import MetricCollection

from src.base_classes.base_loader import BaseLoader
from src.config import TrainClassifierConf
from src.dataloaders.stratified_k_fold_loader import StratifiedKFoldLoader
from src.training.classification_trainer import ClassificationTrainer
from src.utils.classification_presets import ClassificationPresets
from src.utils.utils import set_seeds

logger = logging.getLogger(__name__)

cs = ConfigStore.instance()
cs.store(name="train_classifier_conf", node=TrainClassifierConf)


@hydra.main(version_base=None, config_path="conf", config_name="train_classifier")
def main(cfg: TrainClassifierConf) -> None:
    """Train a model for image classification in PyTorch."""

    # Reproducibility
    set_seeds(
        seed=cfg.reproducibility.torch_seed,
        cudnn_deterministic=cfg.reproducibility.cudnn_deterministic,
        cudnn_benchmark=cfg.reproducibility.cudnn_benchmark
    )

    # Set target device
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    logger.info("Target device is set to: %s.", device.type.upper())

    # Instantiate model, optimizer, and criterion
    logger.info("Instantiating model, setting up optimizer and criterion ...")
    model = instantiate(cfg.model.architecture).to(device)
    optimizer = instantiate(
        {k: cfg.optimizer[k] for k in cfg.optimizer if k not in ["name", "params"]},
        params=model.parameters()
    )
    cfg.optimizer.params = optimizer.state_dict()["param_groups"]
    criterion = instantiate(cfg.criterion)

    # Prepare transforms and dataset
    logger.info("Preparing transforms and dataset ...")
    train_transform = ClassificationPresets(
        mean=cfg.dataset.transform_params.mean,
        std=cfg.dataset.transform_params.std,
        crop_size=cfg.dataset.transform_params.crop_size,
        crop_scale=(
            cfg.dataset.transform_params.crop_scale["lower"],
            cfg.dataset.transform_params.crop_scale["upper"]
        ),
        crop_ratio=(
            cfg.dataset.transform_params.crop_ratio["lower"],
            cfg.dataset.transform_params.crop_ratio["upper"]
        ),
        flip_prob=cfg.dataset.transform_params.flip_prob,
        is_training=True
    )
    val_transform = ClassificationPresets(
        mean=cfg.dataset.transform_params.mean,
        std=cfg.dataset.transform_params.std,
        crop_size=cfg.dataset.transform_params.crop_size,
        resize_size=cfg.dataset.transform_params.resize_size,
        is_training=False
    )
    dataset = instantiate(cfg.dataset.train_set)

    # Instantiate metrics to track during training
    logger.info("Instantiating metrics ...")
    prediction_metrics = MetricCollection({
        name: instantiate(metric) for name, metric in cfg.metrics.items()
    })

    # SINGLE TRAINING RUN
    if cfg.training.num_folds is None:
        # Set up dataloaders
        logger.info("Preparing dataloaders ...")
        base_loader = BaseLoader(
            dataset=dataset,
            main_transform=train_transform,
            val_transform=val_transform,
            val_split=cfg.dataloader.val_split,
            batch_size=cfg.dataloader.batch_size,
            shuffle=True,
            num_workers=cfg.dataloader.num_workers,
            pin_memory=True,
            split_seed=cfg.reproducibility.split_seed,
            shuffle_seed=cfg.reproducibility.shuffle_seed
        )
        train_loader = base_loader.get_dataloader(mode="Main")
        val_loader = base_loader.get_dataloader(mode="Val")

        # Instantiate trainer and start training
        # NOTE: Training is automatically resumed if a checkpoint is provided.
        logger.info("Setting up trainer ...")
        trainer = ClassificationTrainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            train_loader=train_loader,
            val_loader=val_loader,
            prediction_metrics=prediction_metrics,
            device=device,
            cfg=cfg
        )
        trainer.train()
    # STRATIFIED K-FOLD CROSS-VALIDATION
    elif cfg.training.num_folds > 1:
        # Set up dataloader for stratified k-fold cross-validation
        logger.info(
            "Preparing folds for stratified %s-fold cross-validation ...",
            cfg.training.num_folds
        )
        stratified_k_fold_loader = StratifiedKFoldLoader(
            dataset=dataset,
            train_transform=train_transform,
            val_transform=val_transform,
            num_folds=cfg.training.num_folds,
            batch_size=cfg.dataloader.batch_size,
            shuffle=True,
            num_workers=cfg.dataloader.num_workers,
            pin_memory=True,
            fold_seed=cfg.reproducibility.split_seed,
            shuffle_seed=cfg.reproducibility.shuffle_seed
        )

        # Store model and optimizer states to reset them for each fold
        init_model_state_dict_path = Path(cfg.experiment.dir) / "init_model_state_dict.pt"
        init_optimizer_state_dict_path = Path(cfg.experiment.dir) / "init_optimizer_state_dict.pt"
        torch.save(model.state_dict(), init_model_state_dict_path)
        torch.save(optimizer.state_dict(), init_optimizer_state_dict_path)

        # Iterate over individual folds
        for fold_idx in range(cfg.training.num_folds):
            logger.info(
                "CROSS-VALIDATION RUN %0*d/%d",
                len(str(cfg.training.num_folds)), fold_idx + 1, cfg.training.num_folds
            )

            # Reset random seeds and model and optimizer states
            if fold_idx > 0:
                # Reset random seeds
                set_seeds(
                    seed=cfg.reproducibility.torch_seed,
                    cudnn_deterministic=cfg.reproducibility.cudnn_deterministic,
                    cudnn_benchmark=cfg.reproducibility.cudnn_benchmark
                )

                # Reset model and optimizer states
                logger.info("Resetting model and optimizer states ...")
                model.load_state_dict(
                    torch.load(init_model_state_dict_path, weights_only=True)
                )
                optimizer.load_state_dict(
                    torch.load(init_optimizer_state_dict_path, weights_only=True)
                )

            # Set up dataloaders
            logger.info("Preparing dataloaders ...")
            train_loader = stratified_k_fold_loader.get_dataloader(
                fold_idx=fold_idx,
                mode="Train"
            )
            val_loader = stratified_k_fold_loader.get_dataloader(
                fold_idx=fold_idx,
                mode="Val"
            )

            # Instantiate trainer and start training
            # NOTE: Training is automatically resumed if a checkpoint is provided.
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
                run_id=fold_idx + 1
            )
            trainer.train()

        # Remove initial model and optimizer state dicts after last fold
        init_model_state_dict_path.unlink(missing_ok=True)
        init_optimizer_state_dict_path.unlink(missing_ok=True)
    else:
        raise ValueError(
            f"'cfg.training.num_folds' should be either None or an integer greater than 1, "
            f"but got {cfg.training.num_folds}."
        )


if __name__ == "__main__":
    main()
