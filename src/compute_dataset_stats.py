"""This script computes the mean and standard deviation of a dataset.

This script is configured using the Hydra framework, with configuration
details specified in the 'src/conf/' directory.  The configuration file
associated with this script is named 'compute_dataset_stats.yaml'.

Typical usage example:

  python compute_dataset_stats.py model=vgg dataset=cifar10
"""


import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import torch

from utils.data_utils import compute_dataset_stats


@hydra.main(version_base=None, config_path="conf", config_name="compute_dataset_stats")
def main(cfg: DictConfig) -> None:
    # Prepare dataloader
    dataset = instantiate(cfg.dataset.train_set)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.dataloader.batch_size,
        shuffle=False,
        num_workers=cfg.dataloader.num_workers,
        pin_memory=True
    )

    # Compute dataset statistics
    mean, std = compute_dataset_stats(dataloader)

    # Print dataset statistics to console
    output = (
        "norm_constants:\n"
        f"  mean: {[round(ch_mean, 6) for ch_mean in mean]}\n"
        f"  std: {[round(ch_std, 6) for ch_std in std]}"
    )
    print(output)


if __name__ == "__main__":
    main()
