"""A modification of PyTorch's CosineAnnealingWarmRestarts class."""


import math
from typing import List, Optional, Union

from torch.optim.lr_scheduler import (
    LRScheduler,
    _enable_get_lr_call,
    _warn_get_lr_called_within_step
)
from torch.optim.optimizer import Optimizer


class CosineAnnealingWarmRestarts(LRScheduler):
    """A modification of PyTorch's CosineAnnealingWarmRestarts class.

    Note:
        This class differs from the one in PyTorch by incorporating an
        additional ``gamma`` parameter that is used to decay the initial
        learning rate at every restart.  The default value of 1.0 yields
        the same behavior as PyTorch's implementation.
    """

    def __init__(
            self,
            optimizer: Optimizer,
            T_0: int,
            T_mult: int = 1,
            eta_min: float = 0.0,
            gamma: float = 1.0,
            last_epoch: int = -1,
            verbose: Union[bool, str] = "deprecated"
    ) -> None:
        """Initialize the LR scheduler.

        Args:
            optimizer: Wrapped optimizer.
            T_0: Number of iterations until the first restart.
            T_mult: Factor by which to increase the number of iterations
              between consecutive restarts.
            eta_min: Minimum learning rate.
            gamma: Multiplicative factor of learning rate decay applied
              at every restart.
            last_epoch: The index of the last epoch.
            verbose: If True, prints a message to stdout for each
              update.
        """

        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError(f"Expected positive integer T_0, but got {T_0}.")
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError(f"Expected integer T_mult >= 1, but got {T_mult}.")
        if not isinstance(eta_min, (float, int)):
            raise ValueError(
                f"Expected float or int eta_min, but got {eta_min} of type {type(eta_min)}."
            )
        self.T_0 = T_0
        self.T_i = T_0
        self.T_mult = T_mult
        self.T_cur = last_epoch

        self.eta_min = eta_min
        self.gamma = gamma
        self.lr_mult = 1.0

        super().__init__(
            optimizer=optimizer,
            last_epoch=last_epoch,
            verbose=verbose
        )

    def get_lr(self) -> List[float]:
        _warn_get_lr_called_within_step(self)

        return [
            self.eta_min
            + (base_lr - self.eta_min)
            * (1 + math.cos(math.pi * self.T_cur / self.T_i))
            / 2
            * self.lr_mult
            for base_lr in self.base_lrs
        ]

    def step(
            self,
            epoch: Optional[int] = None
    ) -> None:
        if epoch is None and self.last_epoch < 0:
            epoch = 0

        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.T_cur = self.T_cur - self.T_i
                self.T_i = self.T_i * self.T_mult
                self.lr_mult = self.lr_mult * self.gamma
        else:
            if epoch < 0:
                raise ValueError(f"Expected non-negative epoch, but got {epoch}.")
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                    self.lr_mult = self.gamma ** (epoch // self.T_0)
                else:
                    n = int(
                        math.log(
                            (epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult
                        )
                    )
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** n
                    self.lr_mult = self.gamma ** n
            else:
                self.T_i = self.T_0
                self.T_cur = epoch
                self.lr_mult = 1.0

        self.last_epoch = math.floor(epoch)

        with _enable_get_lr_call(self):
            for i, data in enumerate(zip(self.optimizer.param_groups, self.get_lr())):
                param_group, lr = data
                param_group["lr"] = lr

        self._last_lr = [group["lr"] for group in self.optimizer.param_groups]
