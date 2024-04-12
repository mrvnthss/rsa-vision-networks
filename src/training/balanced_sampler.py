import torch


class BalancedSampler(torch.utils.data.sampler.Sampler):
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
        # Check if self.epoch has been set if shuffling is enabled
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
            # Fixed seed for deterministic sampling without shuffling
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
        self.epoch = epoch
