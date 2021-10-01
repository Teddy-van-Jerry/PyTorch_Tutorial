# PyTorch Tutorial
# Chapter 10: Dataset Transforms
# https://youtu.be/c36lUUr864M
# Reimplemented by Teddy van Jerry
# 2021-10-01

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
import os

class WineDataset(Dataset):

    def __init__(self, transform=None):
        # data loading
        xy = np.loadtxt(os.path.dirname(__file__) + '/data/wine.csv', delimiter=",", dtype=np.float32, skiprows=1)
        self.x = xy[:, 1:]
        self.y = xy[:, [0]] # n_samples, 1
        self.n_samples = xy.shape[0]

        self.transform = transform

    def __getitem__(self, index):
        # dataset[0]
        sample = self.x[index], self.y[index]

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        # len(dataset)
        return self.n_samples

class ToTensor:
    def __call__(self, sample):
        inputs, targets = sample
        return torch.from_numpy(inputs), torch.from_numpy(targets)

class MulTransform:
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, sample):
        inputs, target = sample
        inputs *= self.factor
        return inputs, target

dataset = WineDataset(transform=None)
first_data = dataset[0]
features, labels = first_data
print(type(features), type(labels))

dataset = WineDataset(transform=ToTensor())
first_data = dataset[0]
features, labels = first_data
print(features)
print(type(features), type(labels))

composed = torchvision.transforms.Compose([ToTensor(), MulTransform(2)])
dataset = WineDataset(transform=composed)
first_data = dataset[0]
features, labels = first_data
print(features)
print(type(features), type(labels))
