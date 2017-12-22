import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.autograd import Variable
import args
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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

        self.nonlinears = nn.ModuleList([nn.ReLU() for _ in range(self.num_layers-1)])
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

def get_iterator(is_train):
    if args.dataset == "mnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))]
        )
        dataset = datasets.MNIST(root='./mnist_data', download=False, train=is_train, transform=transform)
    elif args.dataset == "cifar":
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )
        dataset = datasets.CIFAR10(root="/data/lisa/data/cifar10", download=False, train=is_train, transform=transform)

    return torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=4, shuffle=is_train)



# PREPARE DATA
train_loader = get_iterator(True)
test_loader = get_iterator(False)

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
        output = model(data.view(data.size(0), -1))
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
        output = model(data.view(data.size(0), -1))
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
train_accs = []
val_accs = []
for epoch in range(args.epochs):
    train_acc, train_loss = train(epoch)
    val_acc, val_loss = test()
    if val_loss < best_val_loss:
        print("Checkpoint!")
        best_val_loss = val_loss
        torch.save(model.state_dict(), os.path.join(args.logdir, "best_model.pkl"))

    train_accs.append(train_acc)
    val_accs.append(val_acc)

plt.figure()
plt.plot(train_accs)
plt.plot(val_accs)
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.savefig(os.path.join(args.logdir, "acc.png"))
plt.close()


