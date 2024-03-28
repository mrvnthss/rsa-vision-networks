import torch.nn as nn

# VGG configurations as described in Simonyan and Zisserman (2015)
#   https://doi.org/10.48550/arXiv.1409.1556
# NOTE: The configurations below correspond to "ConvNet Configuration" A, B, D, and E in Table 1
#       of the paper. These architectures are commonly referred to as VGG11, VGG13, VGG16, and
#       VGG19, respectively, indicating the number of layers with learnable parameters.
configurations = {
    11: [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    13: [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    16: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    19: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M',
         512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    """VGG architecture (Simonyan and Zisserman, 2015) adapted to
    accommodate inputs of varying spatial dimensions.
    """
    def __init__(self, num_layers, img_size=224, num_classes=1000):
        super(VGG, self).__init__()
        # Compute spatial dimension of images after being passed through the feature extractor
        # NOTE: img_size must be divisible by 2^5=32 since we have 5 max-pool layers with stride=2
        if img_size % (2 ** 5) != 0:
            raise ValueError(f"img_size {img_size} is not divisible by 32!")
        out_size = img_size // (2 ** 5)

        # Create feature extractor
        self.features = make_layers(num_layers)

        # Create classifier
        self.classifier = nn.Sequential(
            nn.Linear(512 * out_size ** 2, 4096),
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

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def make_layers(num_layers):
    cfg = configurations[num_layers]
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(2, 2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)
