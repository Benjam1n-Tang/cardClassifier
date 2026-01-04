import torch
import kaggle
import kagglehub
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import timm

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


class PlayingCardDataset(Dataset):

    def __init__(self, data_dir, transform=None):
        self.data = ImageFolder(data_dir, transform=transform)
        pass

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    @property
    def classes(self):
        return self.data.classes

def show_image(n):
    image, label = dataset[n]
    plt.imshow(image)
    plt.title(dataset.classes[label])
    plt.axis("off")
    plt.savefig("debug.png", bbox_inches="tight")
    plt.close()

path = kagglehub.dataset_download("gpiosenka/cards-image-datasetclassification")

dataset = PlayingCardDataset(data_dir=path)

print(len(dataset))

show_image(500)

