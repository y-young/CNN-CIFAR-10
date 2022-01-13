import torch

torch.manual_seed(2021)

DATASET_PATH = "./data"
OUTPUT_PATH = "./cifar_net"

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse',
           'ship', 'truck')

batchSize = 128
epochs = 100

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)