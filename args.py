import torch

use_mnist = True
batch_size = 100
epochs = 10
data_path = "./data"
cuda = torch.cuda.is_available()

layers_dims = [784, 400, 200, 100, 10]
lr = 0.1
mom = 0.5

log_interval = 50

logdir = "./mnist_result/"
