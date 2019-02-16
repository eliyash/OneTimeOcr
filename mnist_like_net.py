import torch.nn as nn
import torch.nn.functional as f


class Net(nn.Module):
    def __init__(self, num_classes):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(9680, 1024)
        self.fc2 = nn.Linear(1024, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = f.relu(f.max_pool2d(self.conv1(x), 2))
        x = f.relu(f.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 9680)
        x = f.relu(self.fc1(x))
        x = f.dropout(x, training=self.training)
        x = f.relu(self.fc2(x))
        x = f.dropout(x, training=self.training)
        x = self.fc3(x)
        return f.log_softmax(x, dim=1)