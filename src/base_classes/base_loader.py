"""A base dataloader class for PyTorch datasets."""


import copy
import random
from typing import Any, Callable, List, Literal, Optional, Tuple, TypeVar, Union

import numpy as np
import torch
from torch.utils.data import Sampler

from src.base_classes.base_sampler import BaseSampler
from src.config import ReproducibilityConf

T = TypeVar('T')
_collate_fn_t = Callable[[List[T]], Any]


class BaseLoader:
    """A base dataloader for PyTorch datasets.

    This class provides a base dataloader for PyTorch datasets that
    supports splitting the dataset into training and validation sets.
    This is done by constructing two dataloaders on top of the same
    dataset, but equipped with two samplers that sample disjoint
    subsets of the dataset.  These two dataloaders can be accessed
    via the ``get_dataloader`` method.

    Attributes:
        collate_fn: The function used by the main loader to merge a
          list of samples into a mini-batch.
        dataset: The dataset to load samples from.
        main_sampler: The main sampler used to sample data from the
          dataset.
        main_transform: The transformation to apply to the samples
          provided by the main loader.
        shared_kwargs: Keyword arguments shared between the main
          dataloader and the validation dataloader.
        val_sampler: The sampler used to sample validation data from
          the dataset.
        val_transform: The transformation to apply to the samples
          provided by the validation loader.

    Methods:
        get_dataloader(mode): Construct the dataloader for a particular
          mode.
    """

    def __init__(
            self,
            dataset: torch.utils.data.Dataset,
            main_transform: Optional[Callable] = None,
            val_transform: Optional[Callable] = None,
            val_split: Optional[float] = None,
            batch_size: int = 1,
            shuffle: bool = False,
            num_workers: int = 0,
            collate_fn: Optional[_collate_fn_t] = None,
            pin_memory: bool = False,
            drop_last: bool = False,
            multiprocessing_context: Optional[str] = None,
            seeds: Optional[ReproducibilityConf] = None
    ) -> None:
        """Initialize the BaseLoader instance.

        Args:
            dataset: The dataset to load samples from.
            main_transform: The transformation to apply to the samples
              provided by the main loader.
            val_transform: The transformation to apply to the samples
              provided by the validation loader.  Has no effect if
              ``val_split`` is None.
            val_split: The proportion of the dataset to use as
              validation samples.
            batch_size: The number of samples to load per batch.
            shuffle: Whether the main sampler ought to shuffle the data
              differently in every epoch.  See also the ``shuffle``
              argument of the BaseSampler class.
            num_workers: The number of subprocesses to use for data
              loading.
            collate_fn: The function used by the main loader to merge a
              list of samples into a mini-batch.
            pin_memory: Whether to use pinned memory for faster GPU
              transfers.
            drop_last: Whether to drop the last incomplete batch in case
              the dataset size is not divisible by the batch size.
            multiprocessing_context: The multiprocessing context to use
              (e.g., "fork" or "spawn").  If None, the operating
              system's default context is used.
            seeds: The configuration that contains the random seeds for
              (a) the split of the dataset into main and validation
              samples and (b) the shuffling behavior of the main sampler
              across epochs.

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
                    "'val_split' should be either None or a float in the range (0, 1), "
                    f"but got {val_split}."
                )

        if seeds is not None:
            split_seed = seeds.split_seed
            shuffle_seed = seeds.shuffle_seed if shuffle else None
        else:
            split_seed = 0
            shuffle_seed = 0 if shuffle else None

        self.dataset = dataset
        self.main_transform = main_transform
        self.val_transform = val_transform
        self.collate_fn = collate_fn

        self.main_sampler, self.val_sampler = self._get_samplers(
            dataset.targets,
            val_split,
            split_seed,
            shuffle,
            shuffle_seed
        )

        # https://pytorch.org/docs/stable/notes/randomness.html#dataloader
        def seed_worker(worker_id: int) -> None:
            worker_seed = torch.initial_seed() % 2 ** 32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        g = torch.Generator()
        g.manual_seed(0)

        self.shared_kwargs = {
            "batch_size": batch_size,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
            "drop_last": drop_last,
            "worker_init_fn": seed_worker,
            "multiprocessing_context": multiprocessing_context,
            "generator": g
        }

    def get_dataloader(
            self,
            mode: Literal["main", "val"]
    ) -> Optional[torch.utils.data.DataLoader]:
        """Construct the dataloader for a particular mode.

        Args:
            mode: Whether to construct the main (training/testing) or
              validation dataloader.

        Returns:
            The dataloader for the specified mode.

        Raises:
            ValueError: If ``mode`` is neither "main" nor "val".
        """

        if mode not in ["main", "val"]:
            raise ValueError(f"'mode' should be either 'main' or 'val', but got {mode}.")

        if mode == "val" and self.val_sampler is None:
            return None

        dataset = copy.deepcopy(self.dataset)
        transform = self.main_transform if mode == "main" else self.val_transform
        dataset.transform = transform
        sampler = self.main_sampler if mode == "main" else self.val_sampler
        collate_fn = self.collate_fn if mode == "main" else None
        return torch.utils.data.DataLoader(
            dataset=dataset,
            sampler=sampler,
            collate_fn=collate_fn,
            **self.shared_kwargs
        )

    @staticmethod
    def _get_samplers(
            targets: Union[List[int], torch.Tensor],
            val_split: Optional[float] = None,
            split_seed: int = 0,
            shuffle: bool = True,
            shuffle_seed: int = 0
    ) -> Tuple[Sampler, Optional[Sampler]]:
        """Build samplers for the main and validation dataloaders.

        Args:
            targets: A list or tensor containing the class index for
              each image in the dataset.
            val_split: The proportion of the dataset to use as
              validation samples.
            split_seed: The random seed that controls the random split
              of the dataset into main samples and validation samples.
              Has no effect if ``val_split`` is None.
            shuffle: Whether the main sampler ought to shuffle the data
              differently in every epoch.  See also the ``shuffle``
              argument of the BaseSampler class.
            shuffle_seed: The random seed that controls the shuffling
              behavior of the main sampler across epochs.  See also the
              ``seed`` argument of the BaseSampler class.

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
