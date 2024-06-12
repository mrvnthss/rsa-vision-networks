"""Train a model for image classification in PyTorch.

This script is configured using the Hydra framework, with configuration
details specified in the "src/conf/" directory.  The configuration file
associated with this script is named "train_classifier.yaml".

Typical usage example:

  >>> python train_classifier.py model=lenet dataset=fashionmnist
  ...                            training.num_epochs=10
"""


import logging

import hydra
import torch
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate

from src.config import ClassifierConf
from src.training import ClassificationTrainer
from src.utils import BalancedSampler


logger = logging.getLogger(__name__)

cs = ConfigStore.instance()
cs.store(name="classifier_conf", node=ClassifierConf)


@hydra.main(version_base=None, config_path="conf", config_name="train_classifier")
def main(cfg: ClassifierConf) -> None:
    """Train a model for image classification in PyTorch."""

    # Set random seeds for reproducibility
    torch.manual_seed(cfg.training.seed)
    torch.cuda.manual_seed_all(cfg.training.seed)
    logger.info("Random seed is set to: %d", cfg.training.seed)

    # Set target device
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    logger.info("Target device is set to: %s", device)

    # Prepare datasets
    logger.info("Preparing datasets")
    train_set = instantiate(cfg.dataset.train_set)
    val_set = instantiate(cfg.dataset.val_set)

    # Instantiate dataset samplers
    logger.info("Instantiating dataset samplers")
    train_sampler = BalancedSampler(
        dataset=train_set,
        shuffle=True,
        seed=cfg.training.seed
    )
    val_sampler = BalancedSampler(
        dataset=val_set,
        shuffle=False,
        seed=cfg.training.seed
    )

    # Prepare dataloaders
    logger.info("Setting up dataloaders")
    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=cfg.dataloader.batch_size,
        sampler=train_sampler,
        num_workers=cfg.dataloader.num_workers,
        pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=val_set,
        batch_size=cfg.dataloader.batch_size,
        sampler=val_sampler,
        num_workers=cfg.dataloader.num_workers,
        pin_memory=True
    )

    # Instantiate model, loss function, and optimizer
    logger.info("Instantiating model, setting up loss function and optimizer")
    model = instantiate(cfg.model.architecture).to(device)
    loss_fn = instantiate(cfg.loss)
    optimizer = instantiate(
        cfg.optimizer,
        params=model.parameters()
    )

    # Instantiate trainer and start training
    # NOTE: Training is automatically resumed if a checkpoint is provided
    logger.info("Setting up trainer")
    trainer = ClassificationTrainer(
        model,
        train_loader,
        val_loader,
        loss_fn,
        optimizer,
        device,
        cfg
    )
    trainer.train()


if __name__ == "__main__":
    main()
