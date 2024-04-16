"""This module provides utility functions for training in PyTorch.

Functions:
    compute_log_indices: Compute indices at which logs should be
      recorded.
"""


from typing import List

import torch


def compute_log_indices(
        dataloader: torch.utils.data.DataLoader,
        num_updates: int
) -> List[int]:
    """Compute indices at which logs should be recorded.

    This function computes indices at which logs should be recorded
    during training or validation.  The indices are computed based on
    the total number of samples in the dataset and the desired number of
    updates.

    Args:
        dataloader: The dataloader providing the dataset.
        num_updates: The desired number of updates.

    Returns:
        A list containing the indices at which logs should be recorded.
    """
    total_samples = len(dataloader.dataset)
    sample_intervals = torch.linspace(
        0, total_samples, num_updates + 1
    )
    log_indices = (
        torch.ceil(sample_intervals / dataloader.batch_size) - 1
    ).int().tolist()[1:]

    return log_indices
