"""Modified LeNet-5 architecture by LeCun et al. (1998)."""


import torch
from einops.layers.torch import Rearrange
from torch import nn


class LeNet(nn.Module):
    """Modified LeNet-5 architecture (LeCun et al., 1998).

    An implementation of the LeNet-5 architecture that can be used to
    classify 32x32 grayscale images.  The network outputs logits for
    each class.  The number of classes to predict can be specified.

    Attributes:
        features: The feature extractor of LeNet-5.
        classifier: The classifier of LeNet-5.

    Methods:
        forward(x): Perform forward pass through the network.

    Note:
        This implementation deviates from the one suggested by LeCun et
        al. (1998) in the following ways:
          1) In the original implementation, outputs of convolutional
             layers are sub-sampled using average pooling, and are then
             passed through a hyperbolic tangent activation function.
             In this implementation, outputs are first passed through a
             ReLU activation function, and are then sub-sampled using
             max pooling.
          2) In the original implementation, not every feature map of
             S2 is connected to every feature map of C3.  See Table 1
             on page 8 of LeCun et al. (1998) for details.  In contrast,
             this implementation does connect every feature map of S2 to
             every feature map of C3.
          3) The output layer in the original implementation uses
             Euclidean Radial Basis Functions (RBF), whereas this
             implementation employs a fully connected layer.
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

        self.features = nn.Sequential(
            # NOTE: Layer names (C1, S2, ...) are taken from the paper, "B" = batch size
            nn.Conv2d(1, 6, 5),                                # C1: Bx6x28x28
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                                # S2: Bx6x14x14
            nn.Conv2d(6, 16, 5),                               # C3: Bx16x10x10
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                                # S4: Bx16x5x5
            Rearrange("b c h w -> b (c h w)", c=16, h=5, w=5)  # OUTPUT: Bx400
        )

        self.classifier = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),  # C5: Bx120
            nn.ReLU(inplace=True),
            nn.Linear(120, 84),          # F6: Bx84
            nn.ReLU(inplace=True),
            nn.Linear(84, num_classes),  # OUTPUT: Bx10
        )

    def forward(
            self,
            x: torch.Tensor
    ) -> torch.Tensor:
        """Perform forward pass through the network."""

        return self.classifier(self.features(x))
