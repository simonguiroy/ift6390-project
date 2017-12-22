import torch

######################MNIST####################
dataset = "mnist"
batch_size = 100
epochs = 5
cuda = torch.cuda.is_available()
layers_dims = [784, 400, 200, 100, 10]
lr = 0.1
mom = 0.5
l2_weight = 0.0001
log_interval = 50
logdir = "./mnist_result/"
print("cuda: {}".format(cuda))

###################CIFAR#######################
#dataset = "cifar"
#batch_size = 64
#epochs = 300
#cuda = torch.cuda.is_available()
#layers_dims = [3072, 1000, 500, 100, 10]
#lr = 0.1
#mom = 0.5
#l2_weight = 0.0001
#log_interval = 50
#logdir = "./cifar_result/"
#print("cuda: {}".format(cuda))
#

