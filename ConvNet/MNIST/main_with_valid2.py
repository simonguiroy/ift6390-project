from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torch.utils.data.sampler import SubsetRandomSampler

from utils import progress_bar

import os

from models import *

#Training settings
parser = argparse.ArgumentParser(description='ConvNet for MNIST')
parser.add_argument('--batch-size', type=int, default=64, metavar='N', help='input training batch size (default=64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N')
parser.add_argument('--epochs', type=int, default=10, metavar='N')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M')
parser.add_argument('--no-cuda', action='store_true', default=False)
parser.add_argument('--seed', type=int, default=1, metavar='S')
parser.add_argument('--log-interval', type=int,default=200, metavar='N')
parser.add_argument('--save_mode', default='none')
parser.add_argument('--checkpoint', default='./checkpoint/check1')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)



transform_data = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))])


trainset = torchvision.datasets.MNIST('./data', train=True, download=True, transform=transform_data)
testset = torchvision.datasets.MNIST('./data', train=False, download=True, transform=transform_data)

train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

valid_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, sampler=SubsetRandomSampler(range(0, 5000)), num_workers=2)

test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, sampler=SubsetRandomSampler(range(5000, 10000)), num_workers=2)


'''
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=True, download=True,
                  transform=transforms.Compose([
                      transforms.ToTensor(),
                      transforms.Normalize((0.1307,), (0.3081,))
                  ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.test_batch_size, shuffle=True, **kwargs)
'''


best_acc = 0
final_acc = 0


model = LeNet()
if args.cuda:
    model.cuda()
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))

def test(epoch):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        #summing up the loss at each batch
        test_loss += F.nll_loss(output, target, size_average=False).data[0]
        #the prediction is the output with max probability
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        test_loss, correct,len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    acc = 100.*correct/ len(test_loader.dataset)



def valid(epoch):
    global best_acc
    global final_acc
    model.eval()
    valid_loss = 0
    correct = 0
    for data, target in valid_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        #summing up the loss at each batch
        valid_loss += F.nll_loss(output, target, size_average=False).data[0]
        #the prediction is the output with max probability
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        
    valid_loss /= len(valid_loader.dataset)
    print('\nValid set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        valid_loss, correct,len(valid_loader.dataset),
        100. * correct / len(valid_loader.dataset)))

    acc = 100.*correct/ len(valid_loader.dataset)
    # Save checkpoint.
    if acc > best_acc:
        best_acc = acc
        if args.save_mode == 'best_acc':
            print('Saving..')
            state = {
                'net': model.module if args.cuda else model,
                'acc': acc,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, args.checkpoint)
    final_acc = acc





for epoch in range(1, args.epochs + 1):
    train(epoch)
    valid(epoch)
    test(epoch)
if args.save_mode == 'full_train':
    print('Saving..')
    state = {
        'net': model.module if args.cuda else model,
        'acc': final_acc,
        'epoch': args.epochs,
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, args.checkpoint)

