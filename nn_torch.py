import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from functions import babysit

dataset = 'MNIST'
#dataset = 'CIFAR10'

cwh_in = np.array({
    'MNIST': [1, 28, 28],
    'CIFAR10': [3, 32, 32]
}[dataset])
size_in = np.prod(cwh_in)
size_out = 10

class Linear(nn.Module):
    dataset = dataset
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(size_in, size_out),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.layers(x)
        return x


class WLP(nn.Module):
    dataset = dataset
    def __init__(self):
        super().__init__()
        size_hidden = 100
        self.fc1 = nn.Linear(size_in, size_hidden)
        self.fc2 = nn.Linear(size_hidden, size_out)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class CNN(nn.Module):
    dataset = dataset
    def __init__(self):
        super(CNN, self).__init__()
        ch1 = 20
        ch2 = 50
        hidden = 500

        self.conv1 = nn.Conv2d(cwh_in[0], ch1, 5, 1)
        self.conv2 = nn.Conv2d(ch1, ch2, 5, 1)

        wh_middle = cwh_in[1:]//4 - 3

        self.fc1 = nn.Linear(wh_middle[0]*wh_middle[1]*ch2, hidden)
        self.fc2 = nn.Linear(hidden, size_out)

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
    dataset = dataset
    def __init__(self):
        super(CNNDeep, self).__init__()
        ch = 64
        hidden = 128
        self.features = nn.Sequential(
            nn.Conv2d(cwh_in[0], ch, 3),
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
        wh_middle = cwh_in[1:] - 22
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(wh_middle[0]*wh_middle[1]*ch, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, size_out),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.layers(x)
        return x


if __name__ == '__main__':
    babysit(CNN)
