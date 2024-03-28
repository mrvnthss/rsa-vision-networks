import torch.nn as nn


class LeNet(nn.Module):
    """Modified LeNet-5 architecture (LeCun et al., 1998)."""
    def __init__(self, img_size=32, num_classes=10):
        super(LeNet, self).__init__()
        # Compute spatial dimension of images after being passed through the feature extractor
        if img_size % 4 != 0:
            raise ValueError(f"img_size {img_size} is not divisible by 4!")
        out_size = img_size // 4 - 3

        # Create feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, 5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        # Create classifier
        self.classifier = nn.Sequential(
            nn.Linear(16 * out_size ** 2, 120),
            nn.ReLU(inplace=True),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Linear(84, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
