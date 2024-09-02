"""A class for sampling PyTorch datasets."""


from typing import Dict, Iterator

import numpy as np
import torch
from torch.utils.data.sampler import Sampler


class BaseSampler(Sampler):
    """A base sampler to use with PyTorch dataloaders.

    The BaseSampler class is used to sample elements from a dataset
    while preserving dataset characteristics in the sense that the
    distribution of classes in a (set of) mini-batch(es) resembles the
    overall distribution of classes in the dataset.

    Attributes:
        class_frequencies: A list of class frequencies, where each
          frequency is repeated for the number of samples in that class,
          used for sampling.
        epoch_idx: The current epoch index, starting from 1.  A value of
          -1 indicates that the epoch index has not been set yet.  Used
          to implement deterministic shuffling if ``shuffle`` is True.
        sample_indices_by_class: A dictionary mapping class indices to
          lists of indices pointing to samples of that class.
        sample_indices_unpacked: All sample indices of the
          ``sample_indices_by_class`` dictionary flattened into a NumPy
          array.
        seed: The random seed used for deterministic shuffling.  Has no
          effect if ``shuffle`` is False.
        shuffle: Whether to shuffle the data before sampling.
        total_samples: The total number of samples in the dataset.

    Methods:
        set_epoch_idx(epoch_idx): Set the current epoch index used for
          deterministic shuffling.
    """

    def __init__(
            self,
            sample_indices_by_class: Dict[int, list[int]],
            shuffle: bool = False,
            seed: int = 0
    ) -> None:
        """Initialize the BaseSampler instance.

        Args:
            sample_indices_by_class: A dictionary mapping class indices
              to lists of indices pointing to samples of that class.
            shuffle: Whether to shuffle the data before sampling.
            seed: The random seed used for deterministic shuffling.  Has
              no effect if ``shuffle`` is False.
        """

        super().__init__()

        self.sample_indices_by_class = sample_indices_by_class
        self.shuffle = shuffle
        self.seed = seed

        self.class_frequencies = []
        self.sample_indices_unpacked = []
        for sample_indices in self.sample_indices_by_class.values():
            num_samples = len(sample_indices)
            self.class_frequencies.extend([num_samples] * num_samples)
            self.sample_indices_unpacked.extend(sample_indices)
        self.sample_indices_unpacked = np.array(self.sample_indices_unpacked)
        self.total_samples = len(self.sample_indices_unpacked)

        self.epoch_idx = -1

    def __iter__(self) -> Iterator[int]:
        """Sample elements while preserving dataset characteristics.

        Sample indices are drawn randomly under the condition that the
        distribution of classes in a (set of) mini-batch(es) resembles
        the overall distribution of classes in the dataset.

        Yields:
            An index pointing to a sample in the dataset.

        Raises:
            ValueError: If ``shuffle`` is True and ``epoch_idx`` has not
              been set yet.
        """

        # Make sure ``epoch_idx`` has been set if shuffling is enabled
        if self.shuffle and self.epoch_idx == -1:
            raise ValueError(
                "'shuffle' is set to True, but 'epoch_idx' has not been set yet. "
                "Please set 'epoch_idx' using the 'set_epoch_idx' method."
            )

        # NOTE: We use an epoch-dependent seed if shuffling is enabled and a fixed seed otherwise
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch_idx if self.shuffle else 0)

        raw_indices = torch.multinomial(
            torch.tensor(self.class_frequencies, dtype=torch.float),
            self.total_samples,
            replacement=False,
            generator=g
        )
        return iter(self.sample_indices_unpacked[raw_indices])

    def __len__(self) -> int:
        """The number of samples in the dataset."""

        return self.total_samples

    def set_epoch_idx(
            self,
            epoch_idx: int
    ) -> None:
        """Set the current epoch index used for deterministic shuffling.

        Args:
            epoch_idx: The current epoch index, starting from 1.
        """

        self.epoch_idx = epoch_idx
