


dataset = datasets.MNIST(path + 'mnist', train=True, download=True, transform=transforms.ToTensor())
data_loader_train = DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(range(5000)))
data_loader_valid = DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(range(5000, 5500)))
data_loader = DataLoader(datasets.MNIST(path + 'mnist', train=False, download=True, transform=transforms.Compose( [transforms.ToTensor()]))
