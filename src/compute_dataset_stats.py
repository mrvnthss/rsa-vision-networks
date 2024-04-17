"""This script computes the mean and standard deviation of a dataset.

This script is configured using the Hydra framework, with configuration
details specified in the 'src/conf/' directory.  The configuration file
associated with this script is named 'compute_dataset_stats.yaml'.

Typical usage example:

  python compute_dataset_stats.py model=vgg dataset=cifar10
"""


from typing import Tuple

import hydra
from hydra.utils import instantiate
import numpy as np
from omegaconf import DictConfig
import torch
from tqdm import tqdm


def compute_dataset_stats(
        dataloader: torch.utils.data.DataLoader
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the mean and standard deviation of a dataset.

    Args:
        dataloader: The dataloader providing the dataset.

    Returns:
        A tuple containing the mean and standard deviation of the
          dataset as NumPy arrays.
    """
    with torch.no_grad():
        # Determine number of channels
        num_channels = next(iter(dataloader))[0].size(1)

        # Initialize running totals
        running_sum = torch.zeros(num_channels)
        running_squared_sum = torch.zeros(num_channels)

        # Prepare progress bar
        total_samples = len(dataloader.dataset)
        desc = "Computing dataset statistics"
        pbar = tqdm(total=total_samples, desc=desc, unit="image")

        for inputs, _ in dataloader:
            # Update running totals
            running_sum += torch.sum(inputs, dim=(0, 2, 3))
            running_squared_sum += torch.sum(inputs ** 2, dim=(0, 2, 3))

            # Update progress bar
            pbar.update(inputs.size(0))

        # Close progress bar
        pbar.close()

        # Calculate total number of pixels
        _, _, height, width = inputs.size()
        total_pixels = total_samples * height * width

        # Calculate sample mean and corrected sample standard deviation per channel
        # https://de.wikipedia.org/wiki/Empirische_Varianz#Darstellung_mittels_Verschiebungssatz
        mean = running_sum / total_pixels
        std = torch.sqrt(
            (running_squared_sum / (total_pixels - 1))
            - (total_pixels / (total_pixels - 1)) * mean ** 2
            )

        return mean.numpy(), std.numpy()


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
