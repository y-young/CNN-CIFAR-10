# CIFAR-10 Image Classification with CNN

A small and simple Convolution Neural Network for CIFAR-10 image classification, with about 0.4M parameters and an accuracy of 89.31%, built with PyTorch.

## Dependencies

-   [PyTorch](https://pytorch.org/)
-   [torchvision](https://pytorch.org/vision/stable/index.html)
-   [seaborn](https://seaborn.pydata.org/)
-   [scikit-learn](https://scikit-learn.org/)
-   [pandas](https://pandas.pydata.org/)
-   [matplotlib](https://matplotlib.org/)
-   [numpy](https://numpy.org/)

## Design

### Preprocessing

-   `transforms.RandomCrop(32, padding=4)`
-   `transforms.RandomHorizontalFlip()`
-   `transforms.RandomRotation(5)`
-   `transforms.RandomAffine(0, shear=10, scale=(0.8,1.2))`
-   `transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)`
-   `transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))`

### Model

| Type        | Parameters                                                                      |
| ----------- | ------------------------------------------------------------------------------- |
| Conv2d      | Input Channels: 3, Output Channels: 32, Kernel Size：3, Padding: 1, Stride: 1    |
| BatchNorm2d |                                                                                 |
| Conv2d      | Input Channels: 32, Output Channels: 32, Kernel Size：3, Padding: 1, Stride: 1   |
| BatchNorm2d |                                                                                 |
| MaxPool2d   | Kernel Size：2, Stride: 2                                                        |
| Conv2d      | Input Channels: 32, Output Channels: 64, Kernel Size：3, Padding: 1, Stride: 1   |
| BatchNorm2d |                                                                                 |
| MaxPool2d   | Kernel Size：2, Stride: 2                                                        |
| Conv2d      | Input Channels: 64, Output Channels: 128, Kernel Size：3, Padding: 1, Stride: 1  |
| BatchNorm2d |                                                                                 |
| MaxPool2d   | Kernel Size：2, Stride: 2                                                        |
| Conv2d      | Input Channels: 128, Output Channels: 128, Kernel Size：3, Padding: 1, Stride: 1 |
| BatchNorm2d |                                                                                 |
| MaxPool2d   | Kernel Size：2, Stride: 2                                                        |
| Linear      | Input: 512, Output: 256                                                         |
| Dropout     | Rate: 0.5                                                                       |
| Linear      | Input: 256, Output: 64                                                          |
| Linear      | Input: 64, Output: 10                                                           |

\# of parameters:

    =================================================================
    Layer (type:depth-idx)                   Param #
    =================================================================
    Net                                      --
    ├─Sequential: 1-1                        --
    │    └─Layer: 2-1                        --
    │    │    └─Conv2d: 3-1                  896
    │    │    └─BatchNorm2d: 3-2             64
    │    └─Layer: 2-2                        --
    │    │    └─Conv2d: 3-3                  9,248
    │    │    └─BatchNorm2d: 3-4             64
    │    └─MaxPool2d: 2-3                    --
    │    └─Layer: 2-4                        --
    │    │    └─Conv2d: 3-5                  18,496
    │    │    └─BatchNorm2d: 3-6             128
    │    └─MaxPool2d: 2-5                    --
    │    └─Layer: 2-6                        --
    │    │    └─Conv2d: 3-7                  73,856
    │    │    └─BatchNorm2d: 3-8             256
    │    └─MaxPool2d: 2-7                    --
    │    └─Layer: 2-8                        --
    │    │    └─Conv2d: 3-9                  147,584
    │    │    └─BatchNorm2d: 3-10            256
    │    └─MaxPool2d: 2-9                    --
    ├─Linear: 1-2                            131,328
    ├─Dropout: 1-3                           --
    ├─Linear: 1-4                            16,448
    ├─Linear: 1-5                            650
    =================================================================
    Total params: 399,274
    Trainable params: 399,274
    Non-trainable params: 0
    Total mult-adds (M): 22.36
    =================================================================
    Input size (MB): 0.01
    Forward/backward pass size (MB): 1.48
    Params size (MB): 1.60
    Estimated Total Size (MB): 3.09
    =================================================================

### Training

-   Loss Function: `CrossEntropyLoss`
-   Optimizer: `Adam`, Learning Rate = 0.001
-   LR Scheduler: `CosineAnnealingLR`

## Result

After 100 epochs:

| Dataset\\Metrics | Loss  | Accuracy |
| ---------------- | ----- | -------- |
| Training         | 0.304 | 89.42%   |
| Test             | 0.336 | 89.16%   |

Confusion Matrix on Test Set:

![](https://github.com/y-young/CNN-CIFAR-10/raw/master/images/Confusion%20Matrix.png)

Training Process:

| ![](https://github.com/y-young/CNN-CIFAR-10/raw/master/images/Loss.png) | ![](https://github.com/y-young/CNN-CIFAR-10/raw/master/images/Accuracy.png) |
| ----------------------------------------------------------------------- | --------------------------------------------------------------------------- |

## Train the model

Set the parameters in `config.py`, run `train.py` with `python train.py`.

## Play with the model

`python play.py cat.jpg`

## About Jupyter Notebook

The Jupyter Notebook has only been tested on [Gradient](https://gradient.run/) using Free-GPU instance with PyTorch image, to train on your own machine with GPU, you may need to figure out how to configure CUDA driver and dependencies.
