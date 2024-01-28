################################
# DO NOT EDIT THE FOLLOWING CODE
################################
import numpy as np

import torch
import torch.nn as nn

import torchvision
from torchvision.transforms import Compose, ToTensor, Normalize

from matplotlib import pyplot as plt

from MyAutoencoder import MyAutoencoder

from hw4_utils import load_MNIST, plot_points, convert_data_to_numpy

np.random.seed(2023)

batch_size = 10

normalize_vals = (0.1307, 0.3081)

# load MNIST dataset
train_dataset, test_dataset, train_loader, test_loader = load_MNIST(batch_size, normalize_vals)

#####################
# ADD YOUR CODE BELOW
#####################

criterion = nn.MSELoss()

X, y = convert_data_to_numpy(train_dataset)

lr = 0.001

import random as rd

choose = rd.sample(range(len(X)), 1000)


maxEpoch = 50

n_input = X.shape[1]


use_autoencoder = MyAutoencoder(n_input, lr, 0.0002, maxEpoch)

optimizer = torch.optim.Adam(use_autoencoder.parameters(), lr)

lossList, epochList = use_autoencoder.fit(train_loader, criterion, optimizer)


from matplotlib import pyplot as plt
plt.plot(epochList,lossList)
plt.savefig('p2_lossTrain.png')

plt.figure()


dataa = X[choose, :]

encoder = use_autoencoder.data_projection(dataa, criterion)

points_x=encoder[:, 0]
points_y=encoder[:, 1]
labels = y[choose]
plot_points(points_x, points_y, labels , 'p2_plot.png')
