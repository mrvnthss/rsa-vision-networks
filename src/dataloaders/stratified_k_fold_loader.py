"""A dataloader implementing stratified k-fold cross-validation."""


import copy
import random
from typing import Any, Callable, List, Optional, TypeVar, Literal

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold

from src.base_classes.base_sampler import BaseSampler
from src.config import ReproducibilityConf

T = TypeVar('T')
_collate_fn_t = Callable[[List[T]], Any]


class StratifiedKFoldLoader:
    """A dataloader implementing stratified k-fold cross-validation.

    This class provides a dataloader for PyTorch datasets that generates
    stratified k-fold splits of the dataset.  It provides dataloaders
    for each fold and mode (training or validation) that can be
    accessed via the ``get_dataloader`` method.

    Attributes:
        all_folds: A list of tuples, where each tuple contains the
          indices of the training and validation samples for a single
          fold.
        collate_fn: The function used by the training loader to merge a
          list of samples into a mini-batch.
        dataset: The dataset to load samples from.
        shared_kwargs: Keyword arguments shared between the training
          dataloader and the validation dataloader.
        shuffle: Whether the training sampler ought to shuffle the data
          differently in every epoch.  See also the ``shuffle`` argument
          of the BaseSampler class.
        shuffle_seed: The random seed that controls the shuffling
          behavior of the training sampler across epochs.  See also the
          ``seed`` argument of the BaseSampler class.
        targets_np: A NumPy array containing the class index for each
          image in the dataset.
        train_transform: The transformation to apply to the samples
          provided by the training loader.
        val_transform: The transformation to apply to the samples
          provided by the validation loader.

    Methods:
        get_dataloader(fold_idx, mode): Construct the dataloader for a
          particular fold and mode.
    """

    def __init__(
            self,
            dataset: torch.utils.data.Dataset,
            train_transform: Optional[Callable] = None,
            val_transform: Optional[Callable] = None,
            num_folds: int = 5,
            batch_size: int = 1,
            shuffle: bool = False,
            num_workers: int = 0,
            collate_fn: Optional[_collate_fn_t] = None,
            pin_memory: bool = False,
            drop_last: bool = False,
            seeds: Optional[ReproducibilityConf] = None
    ) -> None:
        """Initialize the StratifiedKFoldLoader instance.

        Args:
            dataset: The dataset to load samples from.
            train_transform: The transformation to apply to the samples
              provided by the training loader.
            val_transform: The transformation to apply to the samples
              provided by the validation loader.
            num_folds: The number of folds for cross-validation.  Must
              be an integer greater than 1.
            batch_size: The number of samples to load per batch.
            shuffle: Whether the training sampler ought to shuffle the
              data differently in every epoch.  See also the ``shuffle``
              argument of the BaseSampler class.
            num_workers: The number of subprocesses to use for data
              loading.
            collate_fn: The function used by the training loader to
              merge a list of samples into a mini-batch.
            pin_memory: Whether to use pinned memory for faster GPU
              transfers.
            drop_last: Whether to drop the last incomplete batch in case
              the dataset size is not divisible by the batch size.
            seeds: The configuration that contains the random seeds for
              (a) constructing the folds and (b) the shuffling behavior
              of the training sampler across epochs.

        Raises:
            ValueError: If the dataset does not have a ``targets``
              attribute of type np.ndarray, torch.Tensor, or list, or if
              ``num_folds`` is not an integer greater than 1.
        """

        if not hasattr(dataset, "targets"):
            raise ValueError(
                "The dataset must have a 'targets' attribute to be used with the "
                "StratifiedKFoldLoader class."
            )

        # Convert ``targets`` attribute to NumPy array
        if isinstance(dataset.targets, np.ndarray):
            self.targets_np = dataset.targets
        elif isinstance(dataset.targets, torch.Tensor):
            self.targets_np = dataset.targets.detach().cpu().numpy()
        elif isinstance(dataset.targets, list):
            self.targets_np = np.array(dataset.targets)
        else:
            raise ValueError(
                "The 'targets' attribute of the dataset should be of type np.ndarray, "
                f"torch.Tensor, or list, but has type {type(dataset.targets)}."
            )

        if num_folds <= 1:
            raise ValueError(
                f"'num_folds' should be an integer greater than 1, but got {num_folds}."
            )

        self.shuffle = shuffle

        if seeds is not None:
            split_seed = seeds.split_seed
            self.shuffle_seed = seeds.shuffle_seed if shuffle else None
        else:
            split_seed = 0
            self.shuffle_seed = 0 if shuffle else None

        self.dataset = dataset
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.collate_fn = collate_fn

        # Set up folds
        skf = StratifiedKFold(
            n_splits=num_folds,
            shuffle=True,
            random_state=split_seed
        )
        self.all_folds = list(skf.split(
            np.zeros(len(dataset)),
            dataset.targets
        ))

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
            "generator": g
        }

    def get_dataloader(
            self,
            fold_idx: int,
            mode: Literal["train", "val"]
    ) -> torch.utils.data.DataLoader:
        """Construct the dataloader for a particular fold and mode.

        Args:
            fold_idx: The index of the fold to construct the dataloader
              for.
            mode: Whether to construct the training or validation
              dataloader.

        Returns:
            The dataloader for the specified fold and mode.

        Raises:
            ValueError: If ``mode`` is neither "train" nor "val".
        """

        if mode not in ["train", "val"]:
            raise ValueError(f"'mode' should be either 'train' or 'val', but got {mode}.")

        dataset = copy.deepcopy(self.dataset)
        transform = self.train_transform if mode == "train" else self.val_transform
        dataset.transform = transform
        sampler = self._get_sampler(fold_idx, mode)
        collate_fn = self.collate_fn if mode == "train" else None
        return torch.utils.data.DataLoader(
            dataset=dataset,
            sampler=sampler,
            collate_fn=collate_fn,
            **self.shared_kwargs
        )

    def _get_sampler(
            self,
            fold_idx: int,
            mode: Literal["train", "val"]
    ) -> BaseSampler:
        """Construct the sampler for a particular fold and mode.

        Args:
            fold_idx: The index of the fold to construct the sampler
              for.
            mode: Whether to construct the training or validation
              sampler.

        Returns:
            The sampler for the specified fold and mode.

        Raises:
            ValueError: If ``mode`` is neither "train" nor "val".
        """

        if mode not in ["train", "val"]:
            raise ValueError(f"'mode' should be either 'train' or 'val', but got {mode}.")

        train_indices, val_indices = self.all_folds[fold_idx]
        indices = train_indices if mode == "train" else val_indices

        # Arrange sample indices by class
        unique_class_indices = sorted(list(set(self.targets_np)))
        indices_by_class = {class_idx: [] for class_idx in unique_class_indices}
        for sample_idx in indices:
            indices_by_class[self.targets_np[sample_idx]].append(sample_idx)

        # Create sampler
        shuffle = self.shuffle if mode == "train" else False
        seed = self.shuffle_seed if mode == "train" else None
        sampler = BaseSampler(
            sample_indices_by_class=indices_by_class,
            shuffle=shuffle,
            seed=seed
        )
        return sampler
