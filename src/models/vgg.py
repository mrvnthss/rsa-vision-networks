"""VGG architecture by Simonyan and Zisserman (2015)."""


import torch
from torch import nn
from torchvision import models


# VGG configurations as described in Simonyan and Zisserman (2015)
#   https://doi.org/10.48550/arXiv.1409.1556
# NOTE: The configurations below correspond to "ConvNet Configuration[s]" A, B, D, and E in Table 1
#       of the paper.  These architectures are commonly referred to as VGG11, VGG13, VGG16, and
#       VGG19, respectively, indicating the number of layers with trainable parameters.
configurations = {
    11: [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    13: [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    16: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    19: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M',
         512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    """VGG architecture (Simonyan and Zisserman, 2015).

    An implementation of the VGG architecture that can be used to
    classify 224x224 color images.  The model outputs logits for each
    class.  The number of layers (11, 13, 16, or 19) and the number of
    classes to predict can be specified.

    Attributes:
        features: The feature extractor of VGG.
        classifier: The classifier of VGG.

    Methods:
        forward(x): Perform forward pass through the network.
    """

    def __init__(
            self,
            num_layers: int,
            num_classes: int = 1000,
            pretrained: bool = False
    ) -> None:
        """Initialize the VGG network.

        Args:
            num_layers: The number of layers with trainable parameters.
              Must be one of 11, 13, 16, or 19.
            num_classes: The number of classes to predict.
            pretrained: Whether to initialize the weights of the network
              with pretrained weights (trained on ImageNet).
        """

        super().__init__()

        # Create feature extractor
        self._make_layers(num_layers)

        # Create classifier
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes),
        )

        # Initialize weights
        self._initialize_weights(
            num_layers,
            num_classes,
            pretrained
        )

    def forward(
            self,
            x: torch.Tensor
    ) -> torch.Tensor:
        """Perform forward pass through the network."""

        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _make_layers(
            self,
            num_layers: int
    ) -> None:
        """Create the feature extractor of VGG."""

        layers = []
        in_channels = 3
        for v in configurations[num_layers]:
            if v == 'M':
                layers += [nn.MaxPool2d(2, 2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        self.features = nn.Sequential(*layers)

    def _initialize_weights(
            self,
            num_layers: int,
            num_classes: int,
            pretrained: bool
    ) -> None:
        """Initialize the weights of VGG."""

        if pretrained:
            # Load weights from TorchVision
            weights = models.get_weight(f"VGG{num_layers}_Weights.IMAGENET1K_V1")
            state_dict = weights.get_state_dict()

            # Delete weights of the last FC layer from `state_dict` if num_classes != 1000
            if num_classes != 1000:
                state_dict.pop("classifier.6.weight")
                state_dict.pop("classifier.6.bias")

            # Load weights
            missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)

            # Make sure that loading was successful
            assert len(unexpected_keys) == 0
            if num_classes != 1000:
                assert missing_keys == ["classifier.6.weight", "classifier.6.bias"]
            else:
                assert len(missing_keys) == 0

            # Initialize the weights of the last FC layer
            if num_classes != 1000:
                nn.init.normal_(self.classifier[6].weight, 0, 0.01)
                nn.init.zeros_(self.classifier[6].bias)
        else:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.zeros_(m.bias)
