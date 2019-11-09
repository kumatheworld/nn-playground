import numpy as np
import math
from scipy.special import logsumexp

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
        x = x.reshape(-1, self.size_in)
        z1 = x.dot(self.w1) + self.b1
        a1 = np.maximum(z1, 0)
        z2 = a1.dot(self.w2) + self.b2
        a2 = z2 - logsumexp(z2)

if __name__ == '__main__':
    net = HandmadeFC()
    x = np.random.randn(10, 1, 28, 28)
    net.forward(x)
