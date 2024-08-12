"""Compute the average mean and standard deviation of a dataset.

This script is configured using the Hydra framework, with configuration
details specified in the "src/conf/" directory.  The configuration file
associated with this script is named "compute_dataset_stats.yaml".

Typical usage example:

  >>> python compute_dataset_stats.py dataset=cifar10
  norm_constants:
    mean: [0.4914, 0.4822, 0.4465]
    std: [0.2023, 0.1994, 0.201]
"""


from typing import Tuple

import hydra
import numpy as np
import torch
import torchvision.transforms.v2 as T
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from tqdm import tqdm

from src.config import ComputeStatsConf


def compute_dataset_stats(
        dataset: torch.utils.data.Dataset
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the average mean and standard deviation of a dataset.

    Args:
        dataset: The dataset for which to compute statistics.

    Returns:
        A tuple containing the average mean and standard deviation of
          the ``dataset`` as NumPy arrays.
    """

    # Prepare dataloader
    # NOTE: We use a batch size of 1 to allow for varying image sizes within the dataset
    dataloader = torch.utils.data.DataLoader(dataset)

    with torch.no_grad():
        # Determine number of channels
        num_channels = next(iter(dataloader))[0].size(1)

        # Initialize running totals
        running_mean = torch.zeros(num_channels)
        running_std = torch.zeros(num_channels)

        # Prepare progress bar
        total_samples = len(dataset)
        desc = "Computing dataset statistics"
        pbar = tqdm(desc=desc, total=total_samples, unit="image")

        for image, _ in dataloader:
            # Compute mean and standard deviation (across pixels) for each channel
            image.squeeze_(0)  # --> (C, H, W)
            running_mean += torch.mean(image, dim=(1, 2))
            running_std += torch.std(image, dim=(1, 2))

            # Update progress bar, i.e., increment by 1
            pbar.update()

        # Close progress bar
        pbar.close()

        # Calculate average mean and standard deviation (i.e., average across images) per channel
        avg_mean = running_mean / total_samples
        avg_std = running_std / total_samples

        return avg_mean.numpy(), avg_std.numpy()


cs = ConfigStore.instance()
cs.store(name="compute_stats_conf", node=ComputeStatsConf)


@hydra.main(version_base=None, config_path="conf", config_name="compute_dataset_stats")
def main(cfg: ComputeStatsConf) -> None:
    """Compute the average mean and standard deviation of a dataset."""

    transforms = []
    if cfg.dataset.is_grayscale:
        transforms.append(T.Grayscale())
    transforms += [
        T.PILToTensor(),
        T.ToDtype(torch.float, scale=True),
        T.ToPureTensor()
    ]

    # Load dataset
    dataset = instantiate(
        cfg.dataset.train_set,
        transform=T.Compose(transforms)
    )

    # Compute dataset statistics
    mean, std = compute_dataset_stats(dataset)

    # Print dataset statistics to console
    output = (
        "norm_constants:\n"
        f"  mean: {[round(ch_mean, 4) for ch_mean in mean]}\n"
        f"  std: {[round(ch_std, 4) for ch_std in std]}"
    )
    print(output)


if __name__ == "__main__":
    main()
