import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.autograd import Variable
import args
import numpy as np
import os

import ipdb

class MLP(nn.Module):
    def __init__(self, layer_dims):
        """
        :param layer_dims: a list of the size of data after each layer.
        the first layer is nn.Linear(layer_dims[0], layer_dims[1])
        """
        super(MLP, self).__init__()
        assert(len(layer_dims) > 1)
        self.ninp = layer_dims[0]
        self.nclasses = layer_dims[-1]
        self.num_layers = len(layer_dims)-1

        self.nonlinears = nn.ModuleList([ nn.ReLU() for _ in range(self.num_layers-1)])
        self.nonlinears.append(nn.LogSoftmax())

        layer_list = [ nn.Linear(layer_dims[i], layer_dims[i+1]) for i in range(len(layer_dims)-1)]
        self.layers = nn.ModuleList(layer_list)


    def forward(self, inputs):
        """
        :param inputs: [bsz, ninp]
        :return: [bsz, nout]
        """
        out = inputs
        for i in range(self.num_layers):
            layer = self.layers[i]
            nonlinear = self.nonlinears[i]
            out = nonlinear(layer(out))
        return out


# PREPARE DATA
if args.use_mnist is True:
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(args.data_path, train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(args.data_path, train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
else:
    raise NotImplementedError()

# LOAD MODEL
model = MLP(args.layers_dims)
if args.cuda:
    model.cuda()
opt = torch.optim.SGD(
    model.parameters(), lr=args.lr, momentum=args.mom
)

def train(epoch):
    model.train()
    num_corrects = 0
    num_train = 0
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        opt.zero_grad()
        output = model(data.view(args.batch_size, -1))
        loss = torch.nn.functional.nll_loss(output, target)
        loss.backward()

        # l2 reg
        l2 = 0
        for p in model.parameters():
            l2 += torch.sum(p.pow(2))
        (args.l2_weight * l2).backward()

        opt.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))

        # stats
        _, preds = torch.max(output, -1)
        num_corrects += (preds == target).sum().data[0]
        num_train += target.size(0)
        total_loss += loss.data[0]*args.batch_size
    train_acc = float(num_corrects / num_train * 100)
    train_loss = total_loss / num_train
    print('Train set: Average loss: {:.4f}, Accuracy: ({:.2f}%)'.format(
        train_loss, train_acc
    ))
    return train_acc, train_loss

def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data.view(args.batch_size, -1))
        test_loss += nn.functional.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    test_acc = 100. * correct / len(test_loader.dataset)
    print('Test set: Average loss: {:.4f}, Accuracy: ({:.2f}%)'.format(
        test_loss, test_acc))
    return test_acc, test_loss

# TRAIN
if not os.path.isdir(args.logdir):
    os.mkdir(args.logdir)

best_val_loss = np.inf
for epoch in range(args.epochs):
    train(epoch)
    _, val_loss = test()
    if val_loss > best_val_loss:
        print("Checkpoint!")
        best_val_loss = val_loss
        torch.save(model.state_dict(), os.path.join(args.logdir, "best_model.pkl"))

