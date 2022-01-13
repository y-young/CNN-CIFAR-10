from typing import Tuple
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd

from utils import plotGraph
from config import DATASET_PATH, OUTPUT_PATH, classes, device, batchSize, epochs

from model import Net

transformTrain = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(5),
    transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transformTest = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root=DATASET_PATH,
                                        train=True,
                                        download=True,
                                        transform=transformTrain)
trainloader = torch.utils.data.DataLoader(trainset,
                                          batch_size=batchSize,
                                          shuffle=True,
                                          num_workers=0)

testset = torchvision.datasets.CIFAR10(root=DATASET_PATH,
                                       train=False,
                                       download=True,
                                       transform=transformTest)
testloader = torch.utils.data.DataLoader(testset,
                                         batch_size=batchSize,
                                         shuffle=False,
                                         num_workers=0)

def train(dataloader, model, lossFunction, optimizer,
          epoch: int) -> Tuple[float, float]:
    trainingLoss = 0.0
    batches = len(dataloader)
    correctCount = 0
    total = len(dataloader.dataset)
    model.train()

    for batch, (inputs, labels) in enumerate(dataloader, 0):
        inputs, labels = inputs.to(device), labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = lossFunction(outputs, labels)
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs.data, 1)
        trainingLoss += loss.item()
        correctCount += (predicted == labels.data).sum().item()

        if batch % 100 == 99:  # print every 100 mini-batches
            print('[Epoch %3d Batch %3d] Loss: %.3f' %
                  (epoch, batch + 1, loss.item()))

    trainingLoss /= batches
    accuracy = 100.0 * correctCount / total
    print("[Epoch %3d] Training Loss: %0.3f, Accuracy: %0.2f %%" %
          (epoch, trainingLoss, accuracy))
    return (trainingLoss, accuracy)

def test(dataloader, model, lossFunction) -> Tuple[float, float]:
    testLoss = 0.0
    batches = len(dataloader)
    correctCount = 0
    total = len(dataloader.dataset)
    model.eval()

    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            # calculate outputs by running images through the network
            outputs = model(images)
            loss = lossFunction(outputs, labels)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            testLoss += loss.item()
            correctCount += (predicted == labels).sum().item()

    testLoss /= batches
    accuracy = 100.0 * correctCount / total
    print("Test Loss: %0.3f, Accuracy: %0.2f %%" % (testLoss, accuracy))
    return (testLoss, accuracy)

def evaluate(dataloader, model, classes):
    # prepare to count predictions for each class
    correctPred = {classname: 0 for classname in classes}
    totalPred = {classname: 0 for classname in classes}
    yPred = []
    yTrue = []
    model.eval()

    # again no gradients needed
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predictions = torch.max(outputs, 1)
            yPred.extend(predictions.view(-1).detach().cpu().numpy())
            yTrue.extend(labels.view(-1).detach().cpu().numpy())
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correctPred[classes[label]] += 1
                totalPred[classes[label]] += 1

    # print accuracy for each class
    for classname, correct_count in correctPred.items():
        accuracy = 100 * float(correct_count) / totalPred[classname]
        print("Accuracy for class {:5s} is: {:.1f} %".format(
            classname, accuracy))

    confusionMatrix = confusion_matrix(yTrue, yPred)
    print(confusionMatrix)
    confusionMatrix = confusionMatrix / confusionMatrix.sum(axis=1)
    matrix = pd.DataFrame(confusionMatrix, classes, classes)
    plt.figure(figsize=(9, 6))
    sns.heatmap(matrix, annot=True, cmap="Greens")
    plt.title("Confusion Matrix", fontsize=14)
    plt.xlabel("prediction", fontsize=12)
    plt.ylabel("label (ground truth)", fontsize=12)
    plt.show()

net = Net()
net.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

trainingLosses = []
trainingAccuracies = []
testLosses = []
testAccuracies = []

for epoch in range(epochs):  # loop over the dataset multiple times
    loss, accuracy = train(trainloader, net, criterion, optimizer, epoch + 1)
    trainingLosses.append(loss)
    trainingAccuracies.append(accuracy)

    loss, accuracy = test(testloader, net, criterion)
    testLosses.append(loss)
    testAccuracies.append(accuracy)

    scheduler.step()

print("Finished Training")
plotGraph(trainingLosses, testLosses, "Loss")
plotGraph(trainingAccuracies, testAccuracies, "Accuracy")

torch.save(net.state_dict(), OUTPUT_PATH)

print("Evaluation Result on Test:")
evaluate(testloader, net, classes)
