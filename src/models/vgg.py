"""The VGG architecture by Simonyan and Zisserman (2015)."""


import torch
import torch.nn as nn

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
    """VGG architecture introduced by Simonyan and Zisserman (2015).

    An implementation of the VGG architecture that can be used to
    classify 224x224 color images.  The model outputs logits for each
    class.  The number of layers (11, 13, 16, or 19) and the number of
    classes to predict can be specified.

    Params:
        num_layers: The number of layers with trainable parameters.
        num_classes: The number of classes to predict.
    """

    def __init__(
            self,
            num_layers: int,
            num_classes: int = 1000
    ) -> None:
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

        # Initialize weights based on remarks in Section 3.1 of Simonyan and Zisserman (2015)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)  # Glorot & Bengio (2010)
                nn.init.zeros_(m.bias)  # "The biases were initialized with zero."

    def _make_layers(self, num_layers: int) -> None:
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
