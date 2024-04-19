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

    Params:
        dataset: The dataset to sample from.
        shuffle: Whether to shuffle the data before sampling.
        seed: The random seed used for deterministic shuffling.

    (Additional) Attributes:
        epoch: The epoch used for deterministic shuffling.
    """

    def __init__(
            self,
            dataset: torch.utils.data.Dataset,
            shuffle: bool = True,
            seed: int = 0
    ) -> None:
        self.dataset = dataset
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = -1

    def __iter__(self):
        """Generate indices for balanced sampling of the dataset.

        This method checks if the epoch has been set when shuffling is
        enabled, initializes a random number generator with a seed that
        is either epoch-dependent (for deterministic shuffling) or fixed
        (for sampling without shuffling), and samples indices without
        replacement based on class weights.

        Yields:
            An index pointing to a sample in the dataset.
        """
        # Check if ``self.epoch`` has been set if shuffling is enabled
        if self.shuffle and self.epoch == -1:
            raise ValueError(
                "self.shuffle is set to True, but self.epoch has not been set. "
                "Please set self.epoch using the set_epoch method."
            )

        # Initialize random number generator
        g = torch.Generator()
        if self.shuffle:
            # Epoch-dependent seed to shuffle deterministically
            g.manual_seed(self.seed + self.epoch)
        else:
            # Fixed seed for sampling without shuffling
            g.manual_seed(self.seed)

        # Sample indices without replacement based on class weights
        targets = torch.IntTensor(self.dataset.targets)
        class_weights = torch.bincount(targets) / len(targets)
        indices = torch.multinomial(
            class_weights[targets], len(targets), replacement=False, generator=g
        )
        return iter(indices)

    def __len__(self):
        return len(self.dataset)

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch used for deterministic shuffling."""
        self.epoch = epoch
