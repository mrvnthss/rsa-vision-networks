"""A base dataloader class for PyTorch datasets."""


from typing import Any, Callable, List, Optional, Tuple, TypeVar, Union

import numpy as np
import torch
from torch.utils.data import Sampler

from src.base_classes.base_sampler import BaseSampler

T = TypeVar('T')
_collate_fn_t = Callable[[List[T]], Any]


class BaseLoader(torch.utils.data.DataLoader):
    """A base dataloader for PyTorch datasets.

    This class provides a base dataloader for PyTorch datasets that
    supports splitting the dataset into training and validation sets.
    This is done by constructing two dataloaders on top of the same
    dataset, but equipped with two samplers that sample disjoint
    subsets of the dataset.  The main dataloader (for training or
    testing) corresponds to the BaseLoader instance itself, while the
    validation dataloader (providing validation samples during the
    training process) can be obtained by calling the ``get_val_loader``
    method of that instance.

    Attributes:
        main_sampler: The main sampler used to sample data from the
          dataset.
        shared_kwargs: Keyword arguments shared between the main
          dataloader and the validation dataloader.
        val_sampler: The sampler used to sample validation data from
          the dataset.

    Methods:
        get_val_loader: Construct a dataloader for validation purposes.
    """

    def __init__(
            self,
            dataset: torch.utils.data.Dataset,
            val_split: Optional[float] = None,
            batch_size: int = 1,
            shuffle: bool = False,
            num_workers: int = 0,
            collate_fn: Optional[_collate_fn_t] = None,
            pin_memory: bool = False,
            drop_last: bool = False,
            split_seed: int = 0,
            shuffle_seed: int = 0
    ) -> None:
        """Initialize the BaseLoader instance.

        Args:
            dataset: The dataset to load samples from.
            val_split: The proportion of the dataset to use for
              validation.
            batch_size: The number of samples to load per batch.
            shuffle: Whether to enable shuffling for the main sampler.
              See also the ``shuffle`` argument of the BaseSampler
              class.
            num_workers: The number of subprocesses to use for data
              loading.
            collate_fn: The function used to merge a list of samples
              into a mini-batch.
            pin_memory: Whether to use pinned memory for faster GPU
              transfers.
            drop_last: Whether to drop the last incomplete batch in case
              the dataset size is not divisible by the batch size.
            split_seed: The random seed that controls the random split
              of the dataset into main samples and samples used
              for validation.  Has no effect if ``val_split`` is None.
            shuffle_seed: The random seed that controls deterministic
              shuffling of the main sampler.  See also the ``seed``
              argument of the BaseSampler class.

        Raises:
            ValueError: If the dataset does not have a ``targets``
              attribute or if ``val_split`` is not None but outside the
              range (0, 1).
        """

        if not hasattr(dataset, "targets"):
            raise ValueError(
                "The dataset must have a 'targets' attribute to be used with the BaseLoader class."
            )

        if val_split is not None:
            if not 0 < val_split < 1:
                raise ValueError(
                    "The validation split should be either None or a float in the range (0, 1), "
                    f"but got {val_split}."
                )

        self.main_sampler, self.val_sampler = self._get_samplers(
            dataset.targets,
            val_split,
            split_seed,
            shuffle,
            shuffle_seed
        )

        self.shared_kwargs = {
            "dataset": dataset,
            "batch_size": batch_size,
            "num_workers": num_workers,
            "collate_fn": collate_fn,
            "pin_memory": pin_memory,
            "drop_last": drop_last
        }

        super().__init__(sampler=self.main_sampler, **self.shared_kwargs)

    def get_val_loader(self) -> Optional[torch.utils.data.DataLoader]:
        """Construct a dataloader providing validation samples."""

        if self.val_sampler is None:
            return None

        return torch.utils.data.DataLoader(
            sampler=self.val_sampler,
            **self.shared_kwargs
        )

    @staticmethod
    def _get_samplers(
            targets: Union[torch.Tensor, List[int]],
            val_split: Optional[float] = None,
            split_seed: int = 0,
            shuffle: bool = True,
            shuffle_seed: int = 0
    ) -> Tuple[Sampler, Optional[Sampler]]:
        """Build samplers for the main and validation dataloaders.

        Args:
            targets: A list or tensor containing the class index for
              each image in the dataset.
            val_split: The proportion of the dataset to use for
              validation.
            split_seed: The random seed that controls the random split
              of the dataset indices into main indices and indices used
              for validation.  Has no effect if ``val_split`` is None.
            shuffle: Whether to enable shuffling for the main sampler.
              See also the ``shuffle`` argument of the BaseSampler
              class.
            shuffle_seed: The random seed that controls deterministic
              shuffling of the main sampler.  See also the ``seed``
              argument of the BaseSampler class.

        Returns:
            A tuple containing the samplers for the main and validation
            dataloaders.
        """

        # Arrange sample indices by class labels
        if isinstance(targets, torch.Tensor):
            targets = targets.tolist()
        unique_class_indices = sorted(list(set(targets)))
        sample_indices_by_class = {class_idx: [] for class_idx in unique_class_indices}
        for sample_idx, class_idx in enumerate(targets):
            sample_indices_by_class[class_idx].append(sample_idx)

        if val_split is None:
            main_sampler = BaseSampler(
                sample_indices_by_class=sample_indices_by_class,
                shuffle=shuffle,
                seed=shuffle_seed
            )
            return main_sampler, None

        # Shuffle sample indices within classes to perform a (reproducible) random split
        for idx, class_idx in enumerate(sample_indices_by_class):
            rng = np.random.default_rng(split_seed + idx)
            rng.shuffle(sample_indices_by_class[class_idx])

        # Perform split by class
        main_sample_indices = {}
        val_sample_indices = {}
        for class_idx, indices in sample_indices_by_class.items():
            split_idx = int(np.floor(val_split * len(indices)))
            # NOTE: By sorting the indices after the split, we ensure that the ``split_seed`` only
            #       affects the split into main and validation indices itself, but not the order of
            #       the samples within each set.  The latter is controlled by the ``shuffle_seed``
            #       argument of the BaseSampler class.
            main_sample_indices.update({class_idx: sorted(indices[split_idx:])})
            val_sample_indices.update({class_idx: sorted(indices[:split_idx])})

        main_sampler = BaseSampler(
            sample_indices_by_class=main_sample_indices,
            shuffle=shuffle,
            seed=shuffle_seed
        )
        val_sampler = BaseSampler(
            sample_indices_by_class=val_sample_indices,
            shuffle=False
        )
        return main_sampler, val_sampler
