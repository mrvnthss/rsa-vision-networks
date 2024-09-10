"""Modified LeNet-5 architecture by LeCun et al. (1998)."""


import torch
from einops.layers.torch import Rearrange
from torch import nn


class LeNet(nn.Module):
    """Modified LeNet-5 architecture (LeCun et al., 1998).

    An implementation of the LeNet-5 architecture that can be used to
    classify 32x32 grayscale images.  The network outputs logits for
    each class.  The number of classes to predict can be specified.

    Note:
        This implementation deviates from the one suggested by LeCun et
        al. (1998) in the following ways:
          1) In the original implementation, outputs of convolutional
             layers are sub-sampled by 2x2 receptive fields.  The four
             inputs to each receptive field are summed, multiplied by a
             trainable parameter, and are added to a trainable bias.
             In this implementation, sub-sampling is performed using
             max pooling.
          2) The activations of layers up to and including F6 are passed
             through a scaled hyperbolic tangent function in the
             original implementation.  Here, we replace the scaled
             hyperbolic tangent by the ReLU activation function.
          3) In the original implementation, not every feature map of
             S2 is connected to every feature map of C3 (cf. Table 1
             on page 8 of LeCun et al. (1998) for details).  In
             contrast, this implementation does connect every feature
             map of S2 to every feature map of C3.
          4) The output layer in the original implementation uses
             Euclidean Radial Basis Functions (RBF), which here has been
             replaced with a fully connected layer.

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
            nn.Conv2d(1, 6, 5),                                  # C1
            nn.MaxPool2d(2),                                     # S2
            nn.ReLU(),
            nn.Conv2d(6, 16, 5),                                 # C3
            nn.MaxPool2d(2),                                     # S4
            nn.ReLU(),
            nn.Conv2d(16, 120, 5),                               # C5 (equiv. to full connection)
            nn.ReLU(inplace=True),
            Rearrange("b c h w -> b (c h w)", c=120, h=1, w=1),
            nn.Linear(120, 84),                                  # F6
            nn.ReLU(),
            nn.Linear(84, num_classes)                           # Output
        )

    def forward(
            self,
            x: torch.Tensor
    ) -> torch.Tensor:
        """Perform forward pass through the network."""

        return self.net(x)
