"""
Evan Racah
Adapted Version of PyTorch implementation by Kenta Iwasaki @ Gram.AI.
Thanks Kenta!
"""

import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from subprocess import Popen, PIPE
from tensorboardX import SummaryWriter
import argparse
import os
parser = argparse.ArgumentParser()
parser.add_argument("--dataset",type=str,default="mnist")
parser.add_argument("--output_folder",type=str,default="CapsNet")
parser.add_argument("--weights_file",type=str,default="None")
parser.add_argument("--test",action='store_true', default=False)
parser.add_argument("--weight_decay",type=float, default=0.0)
parser.add_argument("--lr",type=float, default=0.001)
parser.add_argument("--conv_filters",default=256,type=int)
parser.add_argument("--primary_capsule_filters",default=32,type=int)
parser.add_argument("--primary_capsule_length",default=8,type=int)
parser.add_argument("--digit_capsule_length",default=16,type=int)
parser.add_argument("--kernel_size",default=9,type=int)
parser.add_argument("--batch_size",default=100,type=int)
parser.add_argument("--rout_iter",default=3,type=int)
parser.add_argument("--no_reconstruction",action='store_true', default=False)
parser.add_argument("--lenet",action='store_true', default=False)
parser.add_argument("--rec_coeff",type=float, default=0.0005)
args = parser.parse_args()
if args.lenet:
    args.conv_filters=7
    args.primary_capsule_filters=6
    args.primary_capsule_length=4
    args.digit_capsule_length=3
    args.kernel_size=9
    args.no_reconstruction = True

args.epoch=0
args.output_folder += '-{0}'.format(args.dataset)
if args.test:
    args.output_folder += "-test"
    args.output_folder += os.path.basename(os.path.dirname(args.weights_file))
else:
    if 'SLURM_JOB_ID' in os.environ:
        args.output_folder += '-{0}'.format(os.environ['SLURM_JOB_ID'])
    args.output_folder += "bs-%i-wd%s-lr%s-pclen%i-rout_iter%i-kern%i-norec%s-lenet%sreccoef%s"%(args.batch_size,
                            str(args.weight_decay),
                            str(args.lr),
                            args.primary_capsule_length,
                            args.rout_iter,
                            args.kernel_size, str(args.no_reconstruction), str(args.lenet),str(args.rec_coeff))

saved_model_dir = './epochs/{0}'.format(args.output_folder)
if not os.path.exists(saved_model_dir):
    os.makedirs(saved_model_dir)




def check_for_gpus():
    try:
        p = Popen(["nvidia-smi"],stdout=PIPE)
        cuda=True
    except:
        cuda=False
    return cuda




CUDA = check_for_gpus()

NUM_CLASSES = 10
if args.test:
    NUM_EPOCHS=1
else:
    NUM_EPOCHS = 500
TR_PROP = 0.8
VAL_PROP = 0.2


def augmentation(x, max_shift=2):

    _, _, height, width = x.size()

    h_shift, w_shift = np.random.randint(-max_shift, max_shift + 1, size=2)
    source_height_slice = slice(max(0, h_shift), h_shift + height)
    source_width_slice = slice(max(0, w_shift), w_shift + width)
    target_height_slice = slice(max(0, -h_shift), -h_shift + height)
    target_width_slice = slice(max(0, -w_shift), -w_shift + width)

    shifted_image = torch.zeros(*x.size())
    shifted_image[:, :, source_height_slice, source_width_slice] = x[:, :, target_height_slice, target_width_slice]
    return shifted_image.float()


class CapsuleLayer(nn.Module):
    def __init__(self, num_capsules, num_route_nodes, in_channels, out_channels, kernel_size=None, stride=None,
                 num_iterations=args.rout_iter):
        super(CapsuleLayer, self).__init__()

        self.num_route_nodes = num_route_nodes
        self.num_iterations = num_iterations

        self.num_capsules = num_capsules

        if num_route_nodes != -1:
            self.route_weights = nn.Parameter(torch.randn(num_capsules, num_route_nodes, in_channels, out_channels))
        else:
            self.capsules = nn.ModuleList(
                [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=0) for _ in
                 range(num_capsules)])

    def squash(self, tensor, dim=-1):
        squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * tensor / torch.sqrt(squared_norm)

    def forward(self, x):
        if self.num_route_nodes != -1:
            priors = x[None, :, :, None, :] @ self.route_weights[:, None, :, :, :]

            logits = Variable(torch.zeros(*priors.size()))

            if CUDA:
                logits = logits.cuda()
            for i in range(self.num_iterations):
                probs = F.softmax(logits, dim=2)
                outputs = self.squash((probs * priors).sum(dim=2, keepdim=True))

                if i != self.num_iterations - 1:
                    delta_logits = (priors * outputs).sum(dim=-1, keepdim=True)
                    logits = logits + delta_logits

        else:
            outputs = [capsule(x).view(x.size(0), -1, 1) for capsule in self.capsules]
            outputs = torch.cat(outputs, dim=-1)
            outputs = self.squash(outputs)

        return outputs


class CapsuleNet(nn.Module):
    def __init__(self,dataset,conv_filters,
                 primary_capsule_filters,
                 primary_capsule_length,
                 digit_capsule_length,kernel_size):
        super(CapsuleNet, self).__init__()
        if dataset=="cifar":
                num_channels = 3
                fmap_size = int(((32 - kernel_size + 1) - kernel_size + 1) / 2)
                num_route_nodes = primary_capsule_filters * fmap_size * fmap_size
                total_pixels = 32*32*3
                num_classes = 10
                activation_function = nn.Linear(total_pixels,total_pixels)

        else:
                num_channels = 1
                num_route_nodes =32*6*6
                total_pixels = 28*28*1
                num_classes = 10
                activation_function = nn.Sigmoid()

        self.conv1 = nn.Conv2d(in_channels=num_channels, out_channels=conv_filters, kernel_size=kernel_size, stride=1)
        self.primary_capsules = CapsuleLayer(num_capsules=primary_capsule_length, num_route_nodes=-1, in_channels=conv_filters,
                                             out_channels=primary_capsule_filters,
                                             kernel_size=kernel_size, stride=2)
        self.digit_capsules = CapsuleLayer(num_capsules=NUM_CLASSES, num_route_nodes=num_route_nodes, in_channels=primary_capsule_length,
                                           out_channels=digit_capsule_length)
        #
        if not args.no_reconstruction:
            self.decoder = nn.Sequential(
                nn.Linear(16 * num_classes, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, 1024),
                nn.ReLU(inplace=True),
                nn.Linear(1024, total_pixels),
                activation_function

            )

    def forward(self, x, y=None):
        x = F.relu(self.conv1(x), inplace=True)
        x = self.primary_capsules(x)
        x = self.digit_capsules(x).squeeze().transpose(0, 1)

        classes = (x ** 2).sum(dim=-1) ** 0.5
        classes = F.softmax(classes,dim=1)

        if y is None:
            # In all batches, get the most active capsule.
            _, max_length_indices = classes.max(dim=1)
            if CUDA:
                y = Variable(torch.sparse.torch.eye(NUM_CLASSES)).cuda().index_select(dim=0, index=max_length_indices)
            else:
                y = Variable(torch.sparse.torch.eye(NUM_CLASSES)).index_select(dim=0, index=max_length_indices)

        if not args.no_reconstruction:
            reconstructions = self.decoder((x * y[:, :, None]).view(x.size(0), -1))
        else:
            reconstructions = None
        return classes, reconstructions


class CapsuleLoss(nn.Module):
    def __init__(self):
        super(CapsuleLoss, self).__init__()

        self.reconstruction_loss = nn.MSELoss(size_average=False)



    def forward(self, images, labels, classes, reconstructions):
        left = F.relu(0.9 - classes, inplace=True) ** 2
        right = F.relu(classes - 0.1, inplace=True) ** 2

        margin_loss = labels * left + 0.5 * (1. - labels) * right
        margin_loss = margin_loss.sum()
        loss = margin_loss
        if not args.no_reconstruction:
            reconstruction_loss = self.reconstruction_loss(reconstructions, images)
            loss += args.rec_coeff * reconstruction_loss
        return loss / images.size(0)


if __name__ == "__main__":
    from torch.autograd import Variable
    from torch.optim import Adam
    from torchnet.engine import Engine
    from torchvision.utils import make_grid
    from torchvision.datasets.mnist import MNIST
    from torchvision.datasets.cifar import CIFAR10
    import torchvision.transforms as transforms
    from tqdm import tqdm
    import torchnet as tnt
    from torch.utils.data.sampler import SubsetRandomSampler

    writer = SummaryWriter('./.logs/{0}'.format(args.output_folder))


    model = CapsuleNet(args.dataset,
                               args.conv_filters,
                               args.primary_capsule_filters,
                               args.primary_capsule_length,
                               args.digit_capsule_length,
                               args.kernel_size)
    if args.weights_file != "None":
        model.load_state_dict(torch.load(args.weights_file))
    if CUDA:
        model.cuda()

    print("# parameters:", sum(param.numel() for param in model.parameters()))

    optimizer = Adam(model.parameters(),
                     weight_decay=args.weight_decay,
                     lr=args.lr)

    engine = Engine()
    meter_loss = tnt.meter.AverageValueMeter()
    meter_accuracy = tnt.meter.ClassErrorMeter(accuracy=True)
    confusion_meter = tnt.meter.ConfusionMeter(NUM_CLASSES, normalized=True)

    capsule_loss = CapsuleLoss()

    def get_tr_val_sampler(trainset,tr_prop=0.8):

        try:
            indices = np.arange(len(trainset.train_data))
        except:
            indices = np.arange(len(trainset.data))

        np.random.RandomState(5)
        np.random.shuffle(indices)

        cutoff_ind = int(tr_prop*len(indices))

        tr_ind = indices[:cutoff_ind]
        val_ind = indices[cutoff_ind:]
        tr_sampler = SubsetRandomSampler(indices=tr_ind)
        val_sampler = SubsetRandomSampler(indices=val_ind)
        return tr_sampler, val_sampler

    def get_dataset(mode):
        is_train = True if mode=="train" or mode=="val" else False
        if args.dataset == "mnist":
            transform=transforms.Compose([
                    transforms.ToTensor()])
            dataset = MNIST(root='./data', download=True, train=is_train,transform=transform)


        elif args.dataset == "cifar":
            transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            dataset = CIFAR10(root="/data/lisa/data/cifar10",download=True,train=is_train, transform=transform)
        return dataset


    def setup_iterators():
        dataset = get_dataset("train")
        tr_sampler, val_sampler = get_tr_val_sampler(dataset,tr_prop=TR_PROP)
        return tr_sampler, val_sampler


    tr_sampler, val_sampler = setup_iterators()
    def get_iterator(mode):
            dataset = get_dataset(mode)
            if mode == "val":
                sampler = val_sampler
            elif mode =="train":
                sampler = tr_sampler
            else:
                sampler=None
            return torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                              shuffle=False, num_workers=4,sampler=sampler)




    # def get_iterator(mode):
    #
    #     if args.dataset == "mnist":
    #         transform=transforms.Compose([
    #                 transforms.ToTensor()])
    #         dataset = MNIST(root='./data', download=True, train=mode,transform=transform)
    #     elif args.dataset == "cifar":
    #         transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    #         dataset = CIFAR10(root="/data/lisa/data/cifar10",download=False,train=mode, transform=transform)
    #
    #     return torch.utils.data.DataLoader(dataset,batch_size=BATCH_SIZE, num_workers=4, shuffle=mode)


    def processor(sample):
        data, labels, training = sample

        labels = torch.LongTensor(labels)
        labels = torch.sparse.torch.eye(NUM_CLASSES).index_select(dim=0, index=labels)
        data = Variable(data)#.cuda()
        labels = Variable(labels)#.cuda()

        if CUDA:
            data = data.cuda()
            labels = labels.cuda()


        if training:
            classes, reconstructions = model(data, labels)
        else:
            classes, reconstructions = model(data)

        loss = capsule_loss(data, labels, classes, reconstructions)

        return loss, classes
    def test_processor(sample):
            data, labels = sample

            labels = torch.LongTensor(labels)
            labels = torch.sparse.torch.eye(NUM_CLASSES).index_select(dim=0, index=labels)
            data = Variable(data)#.cuda()
            labels = Variable(labels)#.cuda()

            if CUDA:
                data = data.cuda()
                labels = labels.cuda()


            classes, reconstructions = model(data)

            loss = capsule_loss(data, labels, classes, reconstructions)

            return loss, classes


    def reset_meters():
        meter_accuracy.reset()
        meter_loss.reset()
        confusion_meter.reset()


    def on_sample(state):
        state['sample'].append(state['train'])


    def on_forward(state):
        meter_accuracy.add(state['output'].data, torch.LongTensor(state['sample'][1]))
        confusion_meter.add(state['output'].data, torch.LongTensor(state['sample'][1]))
        meter_loss.add(state['loss'].data[0])
        # print(state["iterator"])
        if not args.test:
            if state["train"]:
                cur_it = state["t"]
            else:
                cur_it = args.epoch * num_val_iterations + state["t"]
        else:
            cur_it = state["t"]
        writer.add_scalar(("train" if state["train"] else "val") + "_iteration_loss",state['loss'].data[0], cur_it )


    def on_start_epoch(state):
        reset_meters()


        state['iterator'] = tqdm(state['iterator'])




    def on_end_epoch(state):
        print('[Epoch %d] Training Loss: %.4f (Accuracy: %.2f%%)' % (
            state['epoch'], meter_loss.value()[0], meter_accuracy.value()[0]))

        writer.add_scalar('train/loss', meter_loss.value()[0], state['epoch'])
        writer.add_scalar('train/accuracy', meter_accuracy.value()[0], state['epoch'])


        reset_meters()

        args.epoch = state["epoch"]
        test_mode = "test" if args.test else "val"
        engine.test(processor, get_iterator(test_mode))
        writer.add_scalar(test_mode + '/loss', meter_loss.value()[0], state['epoch'])
        writer.add_scalar(test_mode + '/accuracy', meter_accuracy.value()[0], state['epoch'])
        #writer.add_image('val/confusion',confusion_meter.value(),state['epoch'])

        print('[Epoch %d] Testing Loss: %.4f (Accuracy: %.2f%%)' % (
            state['epoch'], meter_loss.value()[0], meter_accuracy.value()[0]))

        torch.save(model.state_dict(), saved_model_dir+'/epoch_%d.pt' % state['epoch'])

        # Reconstruction visualization.
        #
        # test_sample = next(iter(get_iterator(False)))
        #
        # ground_truth = test_sample[0]
        # if CUDA:
        #     _, reconstructions = model(Variable(ground_truth).cuda())
        # else:
        #     _, reconstructions = model(Variable(ground_truth))

        #reconstruction = reconstructions.cpu().view_as(ground_truth).data

        #writer.add_image("val/ground_truth", make_grid(ground_truth, nrow=int(BATCH_SIZE ** 0.5),
        #                normalize=True, range=(0, 1)).numpy(), state["epoch"])
        #writer.add_image("val/reconstructions",make_grid(reconstruction, nrow=int(BATCH_SIZE ** 0.5), normalize=True, range=(0, 1)).numpy(),state["epoch"])

    # def on_start(state):
    #     state['epoch'] = 327
    #
    # engine.hooks['on_start'] = on_start
    # if args.test:
    #         reset_meters()
    #         engine.test(test_processor, get_iterator("test"))
    #         writer.add_scalar('test/loss', meter_loss.value()[0], 0)
    #         writer.add_scalar('test/accuracy', meter_accuracy.value()[0], 0)
    #         print(meter_accuracy.value()[0])
    #
    # else:

    engine.hooks['on_sample'] = on_sample
    engine.hooks['on_forward'] = on_forward
    engine.hooks['on_start_epoch'] = on_start_epoch
    engine.hooks['on_end_epoch'] = on_end_epoch
    iterator = get_iterator("train")
    num_val_iterations = int(VAL_PROP*len(iterator.dataset) / args.batch_size)
    if args.test:
        engine.test(processor, get_iterator("test"))
        writer.add_scalar('test/loss', meter_loss.value()[0], 0)
        writer.add_scalar('test/accuracy', meter_accuracy.value()[0], 0)
        print('Testing Loss: %.4f (Accuracy: %.2f%%)' % (meter_loss.value()[0], meter_accuracy.value()[0]))
    else:
        engine.train(processor, iterator, maxepoch=NUM_EPOCHS, optimizer=optimizer)
