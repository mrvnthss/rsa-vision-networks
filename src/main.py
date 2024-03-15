import hydra
import torch
import torchvision
from torchvision.transforms import v2 as transforms

from models.lenet import LeNet
from utils.training import train_model


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg) -> None:
    # Set random seed for reproducibility
    torch.manual_seed(cfg.training.seed)

    # Prepare transform to use for datasets
    transform = transforms.Compose([
        transforms.ToImage(),  # convert to Image
        transforms.ToDtype(torch.float32, scale=True),  # scale data to have values in [0, 1]
        transforms.Normalize((0.5,), (0.5,))  # normalize
    ])

    # Prepare datasets
    train_set = torchvision.datasets.FashionMNIST(
        cfg.paths.data,
        train=True,
        transform=transform,
        download=True
    )
    val_set = torchvision.datasets.FashionMNIST(
        cfg.paths.data,
        train=False,
        transform=transform,
        download=True
    )

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
    model = LeNet().to(cfg.training.device)

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
