from PIL import Image
from torch.autograd import Variable
import numpy as np
import torch
import torchvision.transforms as transforms
import sys
from train import evaluate

from model import Net
from config import OUTPUT_PATH, classes

transform = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

net = Net()
# add map_location=torch.device('cpu') to torch.load if loading a CUDA model on CPU-only machine
net.load_state_dict(torch.load(OUTPUT_PATH))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)

def softmax(x):
    y = np.exp(x - np.max(x))
    f_x = y / np.sum(np.exp(x))
    return f_x

def predictImage(image, model):
    model.eval()
    imageTensor = transform(image).float()
    imageTensor = imageTensor.unsqueeze_(0)
    input = Variable(imageTensor)
    input = input.to(device)
    output = model(input)
    index = output.data.cpu().numpy()
    print(softmax(index))
    index = index.argmax()
    return index

image = Image.open(sys.argv[1])
index = predictImage(image, net)
print(classes[index])