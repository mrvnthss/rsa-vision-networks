"""This module provides utility functions for handling/processing data.

Functions:
    compute_dataset_stats: Compute the mean and standard deviation of a
      dataset.
"""


from typing import Tuple

import numpy as np
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
