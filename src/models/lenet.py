"""Modified LeNet-5 architecture by LeCun et al. (1998)."""


import torch
from einops import rearrange
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

    Attributes:
        conv1: The first convolutional layer of LeNet-5.
        conv2: The second convolutional layer of LeNet-5.
        fc1: The first fully connected layer of LeNet-5.
        fc2: The second fully connected layer of LeNet-5.
        fc3: The output layer of LeNet-5.

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

        self.conv1 = self._make_conv_layer(1, 6)
        self.conv2 = self._make_conv_layer(6, 16)

        self.fc1 = self._make_fc_layer(400, 120)
        self.fc2 = self._make_fc_layer(120, 84)
        self.fc3 = self._make_fc_layer(84, num_classes, add_relu=False)

    @staticmethod
    def _make_conv_layer(
            in_channels: int,
            out_channels: int,
            conv_kernel_size: int = 5,
            pool_kernel_size: int = 2,
    ) -> nn.Module:
        """Create a conv. layer w/ ReLU activation and max pooling.

        Args:
            in_channels: The number of input channels.
            out_channels: The number of output channels.
            conv_kernel_size: The size of the convolutional kernel.
            pool_kernel_size: The size of the max pooling kernel.

        Returns:
            A convolutional layer, followed by a ReLU activation and a
            max pooling layer.
        """

        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, conv_kernel_size),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(pool_kernel_size)
        )

    @staticmethod
    def _make_fc_layer(
            in_features: int,
            out_features: int,
            add_relu: bool = True
    ) -> nn.Module:
        """Create a fully connected layer w/ ReLU activation.

        Args:
            in_features: The number of input features.
            out_features: The number of output features.
            add_relu: Whether to add a ReLU activation after the layer.

        Returns:
            A fully connected layer, followed by a ReLU activation if
            specified.
        """

        fc_layer = [nn.Linear(in_features, out_features)]
        if add_relu:
            fc_layer.append(nn.ReLU(inplace=True))
        return nn.Sequential(*fc_layer)

    def forward(
            self,
            x: torch.Tensor
    ) -> torch.Tensor:
        """Perform forward pass through the network."""

        x = self.conv1(x)
        x = self.conv2(x)
        x = rearrange(x, "b c h w -> b (c h w)", c=16, h=5, w=5)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
