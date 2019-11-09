import torch.nn as nn
import torch.nn.functional as F

size_in = 28 * 28
size_out = 10


class KumaFC(nn.Module):
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

class KumaFCSeq(nn.Module):
    def __init__(self):
        super().__init__()
        self.size_hidden = 100
        self.layers = nn.Sequential(
            nn.Linear(size_in, self.size_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(self.size_hidden, size_out),
            nn.LogSoftmax()
        )

    def forward(self, x):
        x = x.view(-1, size_in)
        x = self.layers(x)
        return x


class KumaCNN(nn.Module):
    def __init__(self):
        super(KumaCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class KumaCNNSeq(nn.Module):
    def __init__(self):
        super(KumaCNNSeq, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 20, 5, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(20, 50, 5, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.layers = nn.Sequential(
            nn.Linear(4*4*50, 500),
            nn.ReLU(inplace=True),
            nn.Linear(500, 10),
            nn.LogSoftmax()
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 4*4*50)
        x = self.layers(x)
        return x
