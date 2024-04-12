import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import torch

from training import BalancedSampler, Trainer


@hydra.main(version_base=None, config_path="conf", config_name="train_classifier")
def main(cfg: DictConfig) -> None:
    # Set random seed for reproducibility
    torch.manual_seed(cfg.training.seed)

    # Prepare datasets
    train_set = instantiate(cfg.dataset.train_set)
    val_set = instantiate(cfg.dataset.val_set)

    # Prepare samplers
    train_sampler = BalancedSampler(train_set, shuffle=True, seed=cfg.training.seed)
    val_sampler = BalancedSampler(val_set, shuffle=False, seed=cfg.training.seed)
    # NOTE: Setting the epoch index is not necessary for the val_sampler since
    #       it does not (deterministically) shuffle the indices
    train_sampler.set_epoch(cfg.logging.epoch_index)

    # Prepare dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=cfg.dataloader.batch_size,
        sampler=train_sampler,
        num_workers=cfg.dataloader.num_workers,
        pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=cfg.dataloader.batch_size,
        sampler=val_sampler,
        num_workers=cfg.dataloader.num_workers,
        pin_memory=True
    )

    # Set target device
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    # Instantiate model and move to target device
    model = instantiate(cfg.model.architecture).to(device)

    # Instantiate loss function
    loss_fn = instantiate(cfg.loss)

    # Instantiate optimizer
    optimizer = instantiate(
        cfg.optimizer,
        model.parameters()
    )

    # Train model
    trainer = Trainer(
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
