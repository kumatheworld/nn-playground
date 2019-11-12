import numpy as np
import math
from scipy.special import logsumexp, softmax
from torchvision import datasets, transforms
import torch

class HandmadeFC():
    def __init__(self):
        self.size_in = 28 * 28
        self.size_hidden = 100
        self.size_out = 10

        bd = 1 / math.sqrt(self.size_in)
        self.w1 = np.random.uniform(-bd, bd, [self.size_in, self.size_hidden])
        self.b1 = np.random.uniform(-bd, bd, self.size_hidden)
        bd = 1 / math.sqrt(self.size_hidden)
        self.w2 = np.random.uniform(-bd, bd, [self.size_hidden, self.size_out])
        self.b2 = np.random.uniform(-bd, bd, self.size_out)

    def forward(self, x):
        self.x = x.reshape(-1, self.size_in)
        self.z1 = self.x.dot(self.w1) + self.b1
        self.a1 = np.maximum(self.z1, 0)
        self.z2 = self.a1.dot(self.w2) + self.b2
        self.a2 = self.z2 - logsumexp(self.z2, axis=1).reshape(-1, 1)

    def loss(self, a2, y):
        n = y.shape[0]
        l = 0
        for i in range(n):
            l -= a2[i][y[i]]
        l = l / n
        return l

    def backward(self, y):
        n = y.shape[0]

        self.grad_a2 = np.zeros(self.a2.shape)
        for i in range(n):
            self.grad_a2[i][y[i]] -= 1 / n
        self.grad_z2 = softmax(self.z2, axis=1) / n + self.grad_a2

        self.grad_b2 = self.grad_z2.sum(axis=0)
        self.grad_w2 = np.transpose(self.a1).dot(self.grad_z2)

        self.grad_a1 = self.grad_z2.dot(np.transpose(self.w2))
        self.grad_z1 = (self.a1 > 0) * self.grad_a1

        self.grad_b1 = self.grad_z1.sum(axis=0)
        self.grad_w1 = np.transpose(self.x).dot(self.grad_z1)

        self.grad_x = self.grad_z1.dot(np.transpose(self.w1))

    def update(self, lr):
        self.b2 -= lr * self.grad_b2
        self.w2 -= lr * self.grad_w2
        self.b1 -= lr * self.grad_b1
        self.w1 -= lr * self.grad_w1

    def train(self, x, y, lr):
        self.forward(x)
        self.loss_train = self.loss(self.a2, y)
        self.backward(y)
        self.update(lr)

    def test(self, x, y):
        self.forward(x)
        self.loss_test = self.loss(self.a2, y)

    def main(self):
        batch_size = 128
        lr = 1e-2

        train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True)

        for x, y in train_loader:
            x = x.numpy()
            y = y.numpy()
            self.train(x, y, lr)
            print(self.loss_train)


if __name__ == '__main__':
    net = HandmadeFC()
    x = np.random.randn(3, 1, 28, 28)
    y = np.array([3, 8, 2])
    lr = 1e-5
    net.train(x, y, lr)

    net.main()
