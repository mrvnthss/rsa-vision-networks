from typing import List

import torch


def compute_log_indices(
        dataloader: torch.utils.data.DataLoader,
        num_updates: int
) -> List[int]:
    total_samples = len(dataloader.dataset)
    sample_intervals = torch.linspace(
        0, total_samples, num_updates + 1
    )
    log_indices = (
        torch.ceil(sample_intervals / dataloader.batch_size) - 1
    ).int().tolist()[1:]

    return log_indices
