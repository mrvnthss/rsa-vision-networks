"""A class for balanced sampling of ImageFolder datasets."""


import torch


class BalancedSampler(torch.utils.data.sampler.Sampler):
    """A sampler providing balanced batches for ImageFolder datasets.

    The BalancedSampler class is most useful in conjunction with
    dataloaders for datasets inheriting from PyTorch's ImageFolder
    class.  These dataloaders usually sample images from one class
    at a time (if shuffling is disabled), which leads to imbalanced
    validation batches.  The BalancedSampler class addresses this issue
    by sampling indices based on weights that are inversely proportional
    to the class frequencies in the dataset.

    Attributes:
        dataset: The dataset to sample from.
        epoch_idx: The epoch index used for deterministic shuffling.
        seed: The random seed used for deterministic shuffling.
        shuffle: Whether to shuffle the data before sampling.

    Methods:
        set_epoch_idx(epoch_idx): Set the epoch index used for
          deterministic shuffling.
    """

    def __init__(
            self,
            dataset: torch.utils.data.Dataset,
            shuffle: bool = True,
            seed: int = 0
    ) -> None:
        """Initialize the BalancedSampler instance.

        Args:
            dataset: The dataset to sample from.
            shuffle: Whether to shuffle the data before sampling.
            seed: The random seed used for deterministic shuffling.
        """

        super().__init__()
        self.dataset = dataset
        self.shuffle = shuffle
        self.seed = seed
        self.epoch_idx = -1

    def __iter__(self):
        """Generate indices for balanced sampling of the dataset.

        This method checks if an epoch index has been set when shuffling
        is enabled, initializes a random number generator with a seed
        that is either epoch-dependent (for deterministic shuffling) or
        fixed (for sampling without shuffling), and samples indices
        without replacement based on class weights.

        Yields:
            An index pointing to a sample in the dataset.
        """

        # Check if ``epoch_idx`` has been set if shuffling is enabled
        if self.shuffle and self.epoch_idx == -1:
            raise ValueError(
                "'shuffle' is set to True, but the 'epoch_idx' has not been set. "
                "Please set 'epoch_idx' using the 'set_epoch_idx()' method."
            )

        # Initialize random number generator
        g = torch.Generator()
        if self.shuffle:
            # Epoch-dependent seed to shuffle deterministically
            g.manual_seed(self.seed + self.epoch_idx)
        else:
            # Fixed seed for sampling without shuffling
            g.manual_seed(self.seed)

        # Sample indices without replacement based on class weights
        targets = torch.IntTensor(self.dataset.targets)
        class_weights = torch.bincount(targets) / len(targets)
        indices = torch.multinomial(
            class_weights[targets],
            len(targets),
            replacement=False,
            generator=g
        )
        return iter(indices)

    def __len__(self):
        """Return the number of samples in the dataset."""

        return len(self.dataset)

    def set_epoch_idx(
            self,
            epoch_idx: int
    ) -> None:
        """Set the epoch index used for deterministic shuffling.

        Args:
            epoch_idx: The epoch index to set.
        """

        self.epoch_idx = epoch_idx
