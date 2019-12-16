import torch.nn as nn
import torch.nn.functional as F
from functions import babysit

size_in = 28 * 28
size_out = 10


class Linear(nn.Module):
    dataset = 'MNIST'

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(size_in, size_out),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = x.view(-1, size_in)
        x = self.layers(x)
        return x


class WLP(nn.Module):
    dataset = 'MNIST'

    def __init__(self):
        super().__init__()
        self.size_hidden = 100
        self.fc1 = nn.Linear(size_in, self.size_hidden)
        self.fc2 = nn.Linear(self.size_hidden, size_out)

    def forward(self, x):
        x = x.view(-1, size_in)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class CNN(nn.Module):
    dataset = 'MNIST'

    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class CNNDeep(nn.Module):
    def __init__(self):
        super(CNNDeep, self).__init__()
        ch = 64
        hidden = 128
        self.features = nn.Sequential(
            nn.Conv2d(1, ch, 3),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
        )
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(6*6*ch, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 10),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.layers(x)
        return x


if __name__ == '__main__':
    babysit(CNN)
