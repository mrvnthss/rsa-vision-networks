import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):
    """LeNet-5 architecture proposed by LeCun et al. (1998)."""
    def __init__(self, num_classes=10):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        # Expected input size: 1x28x28
        x = F.relu(self.conv1(x))   # 1x28x28 -> 6x24x24
        x = self.pool(x)            # 6x24x24 -> 6x12x12
        x = F.relu(self.conv2(x))   # 6x12x12 -> 16x8x8
        x = self.pool(x)            # 16x8x8 -> 16x4x4
        x = x.view(-1, 16 * 4 * 4)  # 16x4x4 -> 256
        x = F.relu(self.fc1(x))     # 256 -> 120
        x = F.relu(self.fc2(x))     # 120 -> 84
        x = self.fc3(x)             # 84 -> num_classes
        return x
