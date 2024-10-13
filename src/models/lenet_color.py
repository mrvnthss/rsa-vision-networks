"""LeNet-5 architecture modified to classify RGB images."""


from typing import Tuple

import torch
from einops.layers.torch import Rearrange
from torch import nn


class LeNetColor(nn.Module):
    """LeNet-5 architecture modified to classify RGB images.

    An implementation of the LeNet-5 architecture that can be used to
    classify 32x32 RGB images.  The network outputs logits for each
    class.  The number of classes to predict can be specified, as can
    the width of the network's layers.

    Note:
        This implementation deviates from the one in src/models/lenet.py
        in that it expects RGB images instead of grayscale images.

    Attributes:
        net: The network architecture.

    Methods:
        forward(x): Perform forward pass through the network.
    """

    def __init__(
            self,
            num_classes: int = 10,
            layer_widths: Tuple[int] = (6, 16, 120, 84),
    ) -> None:
        """Initialize the LeNet-5 network.

        Args:
            layer_widths: The number of filters in each of the two
              convolutional layers and the number of units in each of
              the two fully-connected layers, in that order.
            num_classes: The number of classes to predict.
        """

        super().__init__()

        conv_1, conv_2, fc_1, fc_2 = layer_widths

        self.net = nn.Sequential(
            nn.Conv2d(3, conv_1, 5),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(conv_1, conv_2, 5),
            nn.MaxPool2d(2),
            nn.ReLU(),
            Rearrange("b c h w -> b (c h w)", c=conv_2, h=5, w=5),
            nn.Linear(conv_2 * 5 * 5, fc_1),
            nn.ReLU(),
            nn.Linear(fc_1, fc_2),
            nn.ReLU(),
            nn.Linear(fc_2, num_classes)
        )

    def forward(
            self,
            x: torch.Tensor
    ) -> torch.Tensor:
        """Perform forward pass through the network."""

        return self.net(x)
