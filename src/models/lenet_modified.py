"""Modified LeNet-5 architecture."""


from typing import Tuple

import torch
from einops.layers.torch import Rearrange
from torch import nn


class LeNetModified(nn.Module):
    """Modified LeNet-5 architecture.

    A modified implementation of the LeNet-5 architecture that can be
    used to classify 32x32 grayscale images.  The network outputs logits
    for each class.  The number of classes to predict can be specified,
    as can the width of the network's layers.

    Note:
        This implementation deviates from the one in src/models/lenet.py
        in that one fully-connected layer has been removed entirely.
        Also, the widths of all remaining layers can be adjusted.

    Attributes:
        net: The network architecture.

    Methods:
        forward(x): Perform forward pass through the network.
    """

    def __init__(
            self,
            layer_widths: Tuple[int] = (6, 16, 120),
            num_classes: int = 10
    ) -> None:
        """Initialize the LeNet-5 network.

        Args:
            layer_widths: The number of filters in each of the two
              convolutional layers and the number of units in the
              fully-connected layer, in that order.
            num_classes: The number of classes to predict.
        """

        super().__init__()

        conv_1, conv_2, fc = layer_widths

        self.net = nn.Sequential(
            nn.Conv2d(1, conv_1, 5),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(conv_1, conv_2, 5),
            nn.MaxPool2d(2),
            nn.ReLU(),
            Rearrange("b c h w -> b (c h w)", c=conv_2, h=5, w=5),
            nn.Linear(conv_2 * 5 * 5, fc),
            nn.ReLU(),
            nn.Linear(fc, num_classes)
        )

    def forward(
            self,
            x: torch.Tensor
    ) -> torch.Tensor:
        """Perform forward pass through the network."""

        return self.net(x)
