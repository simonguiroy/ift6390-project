from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

import os

from models import *
from utils import progress_bar

#Training settings
parser = argparse.ArgumentParser(description='ConvNet for MNIST')
parser.add_argument('--batch-size', type=int, default=64, metavar='N', help='input training batch size (default=64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N')
parser.add_argument('--epochs', type=int, default=10, metavar='N')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M')
parser.add_argument('--no-cuda', action='store_true', default=False)
parser.add_argument('--seed', type=int, default=1, metavar='S')
parser.add_argument('--log-interval', type=int,default=200, metavar='N')
parser.add_argument('--save_mode', default='none')
parser.add_argument('--model', default='LeNet', help='LeNet | LeNet2 | LeNetDropout')
parser.add_argument('--checkpoint', default='./checkpoint/check1')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

args.dataset = 'mnist'

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

'''	
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
trainloader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=True, download=True,
                  transform=transforms.Compose([
                      transforms.ToTensor(),
                      transforms.Normalize((0.5,), (0.5,))
                  ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)
testloader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.5,), (0.5,))
                   ])),
    batch_size=args.test_batch_size, shuffle=True, **kwargs)
'''

def get_iterator(mode):

    if args.dataset == "mnist":
        transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))])
        dataset = datasets.MNIST('./data', download=True, train=mode,transform=transform)
    elif args.dataset == "cifar":
        transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset = datasets.CIFAR10("./data",download=False,train=mode, transform=transform)

    return torch.utils.data.DataLoader(dataset,batch_size=args.batch_size, num_workers=4, shuffle=mode)

trainloader = get_iterator(True)
testloader = get_iterator(False)

if args.model == 'LeNet':
    model = LeNet()
elif args.model == 'LeNet2':
    model = LeNet2()

if args.cuda:
    model.cuda()
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
criterion = nn.CrossEntropyLoss()



def train(epoch):
    print('\nEpoch: %d' % epoch)
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if args.cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))



def test(epoch):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if args.cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))



for epoch in range(1, args.epochs + 1):
    train(epoch)
    test(epoch)
