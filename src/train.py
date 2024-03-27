import hydra
from hydra.utils import instantiate
import torch

from trainer import Trainer


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg) -> None:
    # Set random seed for reproducibility
    torch.manual_seed(cfg.training.seed)

    # Prepare datasets
    train_set = instantiate(cfg.dataset.train_set)
    val_set = instantiate(cfg.dataset.val_set)

    # Prepare dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=cfg.dataset.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
        pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=cfg.dataset.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        pin_memory=True
    )

    # Set target device
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    # Instantiate model and move to target device
    model = instantiate(cfg.model).to(device)

    # Set up loss function and optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
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
    _ = trainer.train()


if __name__ == "__main__":
    main()
