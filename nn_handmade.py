import numpy as np
import math
from scipy.special import logsumexp, softmax
from scipy.signal import correlate
from torchvision import datasets, transforms
import torch

class MyTensor():
    def __init__(self, val):
        self.val = val
        self.grad = 0
        self.velocity = 0

class MyNN():
    def __init__(self):
        self.size_in_side = 28
        self.size_in = self.size_in_side * self.size_in_side
        self.size_out = 10
        self.params = {}

    def xavier_init(self, size_in, size):
        bd = 1 / math.sqrt(self.size_in)
        return np.random.uniform(-bd, bd, size)

    def add(self, name, size_in, size):
        self.params[name] = MyTensor(self.xavier_init(size_in, size))

    def add_linear(self, name, size):
        self.add('w' + name, size[0], size)
        self.add('b' + name, size[0], size[1])

    def fw_linear(self, x, w, b):
        return x.dot(w) + b

    def add_conv2d(self, name, size):
        self.add('cw' + name, size[1], size)
        self.add('cb' + name, size[1], size[0])

    def fw_conv2d(self, x, w, b):
        sheet = [correlate(x, np.expand_dims(w[i], axis=0), 'valid') for i in range(w.shape[0])]
        y = np.concatenate(sheet, axis=1)
        b_rep = np.expand_dims(np.expand_dims(np.tile(b, (y.shape[0], 1)), -1), -1)
        y += b_rep
        return y

    def fw_maxpool2d(self, x, k):
        kh = k[0]
        kw = k[1]
        h_out = x.shape[2] // kh
        w_out = x.shape[3] // kw
        y = [np.max(x[:, :, i*kh:(i+1)*kh, :], axis=2) for i in range(h_out)]
        y = np.stack(y, axis=2)
        y = [np.max(y[:, :, :, j*kw:(j+1)*kw], axis=3) for j in range(w_out)]
        y = np.stack(y, axis=3)
        return y

    def forward(self, x):
        pass

    def loss(self, out, y):
        n = y.shape[0]
        l = 0
        for i in range(n):
            l -= out[i][y[i]]
        l = l / n
        return l

    def backward(self, y):
        pass

    def init_velocity(self):
        for par in self.params.values():
            par.velocity = 0

    def update(self, lr, rho):
        for par in self.params.values():
            par.velocity = rho * par.velocity + par.grad
            par.val -= lr * par.velocity

    def train(self, x, y, lr, rho):
        self.init_velocity()
        out = self.forward(x)
        self.loss_train = self.loss(out, y)
        self.backward(y)
        self.update(lr, rho)

    def test(self, x, y):
        out = self.forward(x)
        self.loss_test = self.loss(out, y)


class MyFC(MyNN):
    def __init__(self):
        super().__init__()

        size_hidden = 100
        self.add_linear('1', (self.size_in, size_hidden))
        self.add_linear('2', (size_hidden, self.size_out))

    def forward(self, x):
        self.x = x.reshape(-1, self.size_in)
        self.z1 = self.fw_linear(self.x, self.params['w1'].val, self.params['b1'].val)
        self.a1 = np.maximum(self.z1, 0)
        self.z2 = self.fw_linear(self.a1, self.params['w2'].val, self.params['b2'].val)
        self.a2 = self.z2 - logsumexp(self.z2, axis=1).reshape(-1, 1)
        return self.a2

    def backward(self, y):
        n = y.shape[0]

        self.grad_a2 = np.zeros(self.a2.shape)
        for i in range(n):
            self.grad_a2[i][y[i]] -= 1 / n
        self.grad_z2 = softmax(self.z2, axis=1) / n + self.grad_a2

        self.params['b2'].grad = self.grad_z2.sum(axis=0)
        self.params['w2'].grad = np.transpose(self.a1).dot(self.grad_z2)

        self.grad_a1 = self.grad_z2.dot(np.transpose(self.params['w2'].val))
        self.grad_z1 = (self.a1 > 0) * self.grad_a1

        self.params['b1'].grad = self.grad_z1.sum(axis=0)
        self.params['w1'].grad = np.transpose(self.x).dot(self.grad_z1)

        self.grad_x = self.grad_z1.dot(np.transpose(self.params['w1'].val))


def main():
    net = MyFC()
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
