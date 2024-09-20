"""A custom implementation of PyTorch's SequentialLR class.

The ``step`` method of the ``SequentialLR`` class implemented in PyTorch
2.4 raises a UserWarning each time the ``step`` method is called for the
first time after switching to the next scheduler.  Check the following
GitHub issue for more details:
    https://github.com/pytorch/pytorch/issues/116776
"""


from bisect import bisect_right
from typing import List

from torch.optim.lr_scheduler import LRScheduler
from torch.optim.lr_scheduler import SequentialLR as _SequentialLR
from torch.optim.optimizer import Optimizer
from typing_extensions import override


class SequentialLR(_SequentialLR):
    def __init__(
            self,
            optimizer: Optimizer,
            schedulers: List[LRScheduler],
            milestones: List[int],
            last_epoch: int = -1,
            verbose: str = "deprecated",
    ):
        super().__init__(
            optimizer,
            schedulers,
            milestones,
            last_epoch,
            verbose
        )

    @override
    def step(self) -> None:
        self.last_epoch += 1
        idx = bisect_right(self._milestones, self.last_epoch)
        scheduler = self._schedulers[idx]
        scheduler.step()
