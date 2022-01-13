import torch
import torch.nn as nn
import torch.nn.functional as F

class Layer(nn.Module):
    def __init__(self, inChannel, outChannel, kernelSize=3):
        super().__init__()
        self.conv = nn.Conv2d(inChannel,
                              outChannel,
                              kernelSize,
                              stride=1,
                              padding=1)
        self.bn = nn.BatchNorm2d(outChannel)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))

class Net(nn.Module):
    cfg = [(3, 32, 3), (32, 32, 3), 'M', (32, 64, 3), 'M', (64, 128, 3), 'M',
           (128, 128, 3), 'M']

    def __init__(self):
        super().__init__()
        self.layers = self.buildLayers()
        self.dropout1 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 10)

    def buildLayers(self):
        layers = []
        for l in self.cfg:
            if l == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers.append(Layer(l[0], l[1], l[2]))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x