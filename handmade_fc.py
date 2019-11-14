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

    def init(self):
        self.vw1 = 0
        self.vb1 = 0
        self.vw2 = 0
        self.vb2 = 0

    def update(self, lr, rho):
        self.vb2 = rho * self.vb2 + self.grad_b2
        self.b2 -= lr * self.vb2
        self.vw2 = rho * self.vw2 + self.grad_w2
        self.w2 -= lr * self.vw2
        self.vb1 = rho * self.vb1 + self.grad_b1
        self.b1 -= lr * self.vb1
        self.vw1 = rho * self.vw1 + self.grad_w1
        self.w1 -= lr * self.vw1

    def train(self, x, y, lr, rho):
        self.init()
        self.forward(x)
        self.loss_train = self.loss(self.a2, y)
        self.backward(y)
        self.update(lr, rho)

    def test(self, x, y):
        self.forward(x)
        self.loss_test = self.loss(self.a2, y)


def main():
    net = HandmadeFC()
    batch_size = 128
    lr = 1e-2
    rho = 0.9
    epochs = 10
    log_interval = 10

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

    for epoch in range(1, epochs + 1):
        # train
        for batch_idx, (x, y) in enumerate(train_loader):
            x = x.numpy()
            y = y.numpy()
            net.train(x, y, lr, rho)
            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(x), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), net.loss_train))

        # test
        test_loss = 0
        correct = 0
        for x, y in test_loader:
            x = x.numpy()
            y = y.numpy()
            net.test(x, y)
            pred = net.a2.argmax(axis=1)
            test_loss += net.loss_test * batch_size
            correct += sum(pred == y)

        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

if __name__ == '__main__':
    main()
