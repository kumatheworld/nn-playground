import argparse
import numpy as np
import torch
import torch.optim as optim
from torch.nn.functional import nll_loss
from torchvision import datasets, transforms
from visdom import Visdom


class VisdomLinePlotter(object):
    """Plots to Visdom"""
    def __init__(self, env='main', title='', xlabel='Epoch', ylabel='Loss'):
        self.viz = Visdom()
        self.plots = {}
        self.env = env
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel

    def plot(self, var_name, split_name, x, y):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(
                X=np.array([x,x]),
                Y=np.array([y,y]),
                env=self.env,
                opts=dict(
                    legend=[split_name],
                    title=self.title,
                    xlabel=self.xlabel,
                    ylabel=self.ylabel
                )
            )
        else:
            self.viz.line(
                X=np.array([x]),
                Y=np.array([y]),
                env=self.env,
                win=self.plots[var_name],
                name=split_name,
                update = 'append'
            )


def stats(dataset):
    return {
        'MNIST': ((0.1307,), (0.3081,)),
        'CIFAR10': ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    }[dataset]

def loader(dataset, train, batch_size, kwargs):
    mean, std = stats(dataset)
    return torch.utils.data.DataLoader(
        getattr(datasets, dataset)(
            root='data',
            train=train,
            download=train,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        ),
        batch_size=batch_size,
        shuffle=True,
        **kwargs
    )

def loaders(dataset, args, kwargs):
    train_loader = loader(dataset, True, args.batch_size, kwargs)
    test_loader = loader(dataset, False, args.test_batch_size, kwargs)

    return train_loader, test_loader


def train(args, model, device, train_loader, optimizer, epoch):
    kinda_loss = 0
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = nll_loss(output, target)
        kinda_loss += loss.item() * len(data)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

    kinda_loss /= len(train_loader.dataset)
    plotter.plot('nn_torch', 'Train', epoch, kinda_loss)

def test(args, model, device, test_loader, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    plotter.plot('nn_torch', 'Test', epoch, test_loss)


def babysit(net):
    dataset = net.dataset
    # Training settings
    parser = argparse.ArgumentParser(description=f'PyTorch {dataset} Example')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disable CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='save the current model')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    torch.manual_seed(args.seed)
    train_loader, test_loader = loaders(dataset, args, kwargs)

    model = net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    global plotter
    plotter = VisdomLinePlotter(title=net.__name__)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader, epoch)

    if args.save_model:
        torch.save(model.state_dict(), "model.pt")
