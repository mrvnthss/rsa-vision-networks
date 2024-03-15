import hydra
from hydra.utils import instantiate
import torch

from utils.training import train_model


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg) -> None:
    # Set random seed for reproducibility
    torch.manual_seed(cfg.training.seed)

    # Prepare datasets
    train_set = instantiate(cfg.dataset.train)
    val_set = instantiate(cfg.dataset.val)

    # Prepare dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=cfg.params.batch_size,
        shuffle=True,
        num_workers=cfg.params.num_workers
    )
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=cfg.params.batch_size,
        shuffle=False,
        num_workers=cfg.params.num_workers
    )

    # Set target device
    cfg.training.device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    # Instantiate model and move to target device
    model = instantiate(cfg.model).to(cfg.training.device)

    # Set up loss function and optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=cfg.params.lr,
        momentum=cfg.params.momentum
    )

    # Train model
    _ = train_model(
        model,
        train_loader,
        val_loader,
        loss_fn,
        optimizer,
        cfg
    )


if __name__ == "__main__":
    main()
