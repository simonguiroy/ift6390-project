import torch

dataset = "mnist"
#dataset = "cifar"

batch_size = 100
epochs = 10
cuda = torch.cuda.is_available()

layers_dims = [784, 400, 200, 100, 10]
lr = 0.1
mom = 0.5
l2_weight = 0.0001

log_interval = 50

logdir = "./mnist_result/"
