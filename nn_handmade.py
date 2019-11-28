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
        self.linear_count = 0
        self.conv_count = 0

    def xavier_init(self, size_in, size):
        bd = 1 / math.sqrt(self.size_in)
        return np.random.uniform(-bd, bd, size)

    def add(self, name, size_in, size):
        self.params[name] = MyTensor(self.xavier_init(size_in, size))

    def add_linear(self, size_in, size_out):
        self.linear_count += 1
        self.add(f'w{self.linear_count}', size_in, (size_in, size_out))
        self.add(f'b{self.linear_count}', size_in, size_out)

    def fw_linear(self, i, x):
        w = self.params[f'w{i}'].val
        b = self.params[f'b{i}'].val
        y = x @ w + b
        return y

    def bw_linear(self, i, x, dy):
        self.params[f'b{i}'].grad = dy.sum(axis=0)
        self.params[f'w{i}'].grad = x.T @ dy
        dx = dy @ self.params[f'w{i}'].val.T
        return dx

    def fw_relu(self, x):
        return np.maximum(x, 0)

    def bw_relu(self, x, dy):
        return (x > 0) * dy

    def add_conv2d(self, ch_out, ch_in, kw, kh):
        self.conv_count += 1
        self.add(f'cw{self.conv_count}', ch_in, (ch_out, ch_in, kw, kh))
        self.add(f'cb{self.conv_count}', ch_in, ch_out)

    def fw_conv2d(self, i, x):
        w = self.params[f'cw{i}'].val
        b = self.params[f'cb{i}'].val
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
        y = np.stack([
            np.max(x[:, :, i*kh:(i+1)*kh, j*kw:(j+1)*kw], axis=(2, 3))
            for i in range(h_out) for j in range(w_out)
        ], axis=-1).reshape(x.shape[0], x.shape[1], h_out, w_out)
        return y

    def bw_maxpool2d(self, x, dy):
        n, c, h, w = dy.shape
        size_out = h * w
        kh = x.shape[2] // h
        kw = x.shape[3] // w

        index = np.unravel_index(
            x.reshape(n, c, h, kh, w, kw)\
             .transpose([0, 1, 2, 4, 3, 5])\
             .reshape(n, c, -1, kh * kw)\
             .argmax(axis=-1),
            (kh, kw)
        )
        offset = np.unravel_index(np.arange(size_out), (h, w))
        i, j = index[0] + offset[0]*kh, index[1] + offset[1]*kw

        u, v = np.unravel_index(np.arange(n * c), (n, c))
        u = np.repeat(u, size_out)
        v = np.repeat(v, size_out)

        dx = np.zeros(x.shape)
        dx[u, v, i.ravel(), j.ravel()] = dy.ravel()

        return dx

    def fw_logsoftmax(self, x):
        return x - logsumexp(x, axis=1).reshape(-1, 1)

    def bw_logsoftmax(self, x, dy):
        return dy - softmax(x, axis=1) * dy.sum(axis=1).reshape(-1, 1)

    def fw_nll_loss(self, x, y):
        n = y.shape[0]
        l = -x[np.arange(n), y].sum() / n
        return l

    def bw_nll_loss(self, x, y):
        n = y.shape[0]
        dx = np.zeros(x.shape)
        dx[np.arange(n), y] = -1 / n
        return dx

    def forward(self, x):
        pass

    def backward(self, y):
        pass

    def loss(self, x, y):
        out = self.forward(x)
        return self.fw_nll_loss(out, y)

    def init_velocity(self):
        for par in self.params.values():
            par.velocity = 0

    def update(self, lr, rho):
        for par in self.params.values():
            par.velocity = rho * par.velocity + par.grad
            par.val -= lr * par.velocity

    def train(self, x, y, lr, rho):
        self.init_velocity()
        self.loss_train = self.loss(x, y)
        self.backward(y)
        self.update(lr, rho)

    def test(self, x, y):
        self.loss_test = self.loss(x, y)


class MyFC(MyNN):
    def __init__(self):
        super().__init__()

        size_hidden = 100
        self.add_linear(self.size_in, size_hidden)
        self.add_linear(size_hidden, self.size_out)

    def forward(self, x):
        self.x = x.reshape(-1, self.size_in)
        self.z1 = self.fw_linear(1, self.x)
        self.a1 = self.fw_relu(self.z1)
        self.z2 = self.fw_linear(2, self.a1)
        self.a2 = self.fw_logsoftmax(self.z2)
        return self.a2

    def backward(self, y):
        da2 = self.bw_nll_loss(self.a2, y)
        dz2 = self.bw_logsoftmax(self.z2, da2)
        da1 = self.bw_linear(2, self.a1, dz2)
        dz1 = self.bw_relu(self.z1, da1)
        dx = self.bw_linear(1, self.x, dz1)


class MyCNN(MyNN):
    def __init__(self):
        super().__init__()

        k = 5
        ch1 = 20
        ch2 = 50
        self.add_conv2d(ch1, 1, k, k)
        self.add_conv2d(ch2, ch1, k, k)

        size_hidden = 500
        self.size_fc_in = 800
        self.add_linear(self.size_fc_in, size_hidden)
        self.add_linear(size_hidden, self.size_out)

    def forward(self, x):
        self.x = x.reshape(-1, 1, self.size_in_side, self.size_in_side)
        self.z1 = self.fw_conv2d(1, self.x)
        self.a1 = self.fw_relu(self.z1)
        self.m1 = self.fw_maxpool2d(self.a1, (2, 2))
        self.z2 = self.fw_conv2d(2, self.m1)
        self.a2 = self.fw_relu(self.z2)
        self.m2 = self.fw_maxpool2d(self.a2, (2, 2))

        self.f = self.m2.reshape(-1, self.size_fc_in)
        self.z3 = self.fw_linear(1, self.f)
        self.a3 = self.fw_relu(self.z3)
        self.z4 = self.fw_linear(2, self.a3)
        self.a4 = self.fw_logsoftmax(self.z4)

        return self.a4



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
