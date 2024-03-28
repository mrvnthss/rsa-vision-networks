import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):
    """Modified LeNet-5 architecture (LeCun et al., 1998)."""
    def __init__(self, img_size=32, num_classes=10):
        super(LeNet, self).__init__()
        if img_size % 4 != 0:
            raise ValueError(f"img_size {img_size} is not divisible by 4!")
        out_size = img_size // 4 - 3
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * out_size ** 2, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
