"""Modified LeNet-5 architecture by LeCun et al. (1998)."""


import torch
from einops.layers.torch import Rearrange
from torch import nn


class LeNetTwoFeatures(nn.Module):
    """Modified LeNet-5 architecture.

    A modified implementation of the LeNet-5 architecture that can be
    used to classify 32x32 grayscale images.  The network outputs logits
    for each class.  The number of classes to predict can be specified.

    Note:
        This implementation deviates from the one in src/models/lenet.py
        in that the number of units in the penultimate layer has been
        reduced to 2 (from 84).

    Attributes:
        net: The network architecture.

    Methods:
        forward(x): Perform forward pass through the network.
    """

    def __init__(
            self,
            num_classes: int = 10
    ) -> None:
        """Initialize the LeNet-5 network.

        Args:
            num_classes: The number of classes to predict.
        """

        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(1, 6, 5),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(6, 16, 5),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(16, 120, 5),
            nn.ReLU(),
            Rearrange("b c h w -> b (c h w)", c=120, h=1, w=1),
            nn.Linear(120, 2),
            nn.ReLU(),
            nn.Linear(2, num_classes)
        )

        # NOTE: The implementation of ``self.net`` above is equivalent to the following, as long as
        #       the input shape is 32x32:
        # self.net = nn.Sequential(
        #     nn.Conv2d(1, 6, 5),
        #     nn.MaxPool2d(2),
        #     nn.ReLU(),
        #     nn.Conv2d(6, 16, 5),
        #     nn.MaxPool2d(2),
        #     nn.ReLU(),
        #     Rearrange("b c h w -> b (c h w)", c=16, h=5, w=5),
        #     nn.Linear(16 * 5 * 5, 120),
        #     nn.ReLU(),
        #     nn.Linear(120, 2),
        #     nn.ReLU(),
        #     nn.Linear(2, num_classes)
        # )

    def forward(
            self,
            x: torch.Tensor
    ) -> torch.Tensor:
        """Perform forward pass through the network."""

        return self.net(x)
