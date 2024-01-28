################################
# DO NOT EDIT THE FOLLOWING CODE
################################
import numpy as np

import torch
import torch.nn as nn

import torchvision
from torchvision.transforms import Compose, ToTensor, Normalize

from matplotlib import pyplot as plt

from MyGenerator import MyGenerator
from MyDiscriminator import MyDiscriminator

from hw4_utils import load_MNIST

np.random.seed(2023)

batch_size = 128

normalize_vals = (0.5, 0.5)

# load MNIST dataset
train_dataset, test_dataset, train_loader, test_loader = load_MNIST(batch_size, normalize_vals)

#####################
# ADD YOUR CODE BELOW
#####################
