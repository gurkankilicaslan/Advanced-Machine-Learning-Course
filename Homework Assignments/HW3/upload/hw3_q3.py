################################
# DO NOT EDIT THE FOLLOWING CODE
################################
import numpy as np

import torch
import torch.nn as nn

import torchvision
from torchvision.transforms import Compose, ToTensor, Normalize

from MyCNN import MyCNN

from hw3_utils import load_MNIST

np.random.seed(2023)

batch_size = 32

normalize_vals = (0.1307, 0.3081)

# load MNIST dataset
train_dataset, test_dataset, train_loader, test_loader = load_MNIST(batch_size, normalize_vals)

#####################
# ADD YOUR CODE BELOW
#####################

l_rate = [1e-5, 1e-4, 1e-3, 0.01, 0.1]
# l_rate = [1e-5, 0.1]

N = 10
losss = nn.CrossEntropyLoss()
err_list = []
for eta in l_rate:
    print('SGD with Learning Rate =',eta,':')
    mycnn = MyCNN(28*28, 10, 3, 2, 2, eta, N)
    mini = torch.optim.SGD(mycnn.parameters(), eta)
    mycnn_fit = mycnn.fit(train_loader, losss, mini)
    pred_false, pred_true, missed_img, ERR_RATE = mycnn.predict(test_loader, losss)
    err_list.append(ERR_RATE)
best_index = err_list.index(min(err_list))
print('\n') 


eta_best = l_rate[best_index]
print("Using the best Learning Rate",eta_best,":")
mycnn_best = MyCNN(28*28, 10, 3, 2, 2, eta_best, N)
mini_best = torch.optim.SGD(mycnn_best.parameters(), eta_best)
mycnn_fit_best = mycnn_best.fit(train_loader, losss, mini_best)
pred_false, pred_true, wrong_pred_image, ERR_RATE = mycnn_best.predict(test_loader, losss)


import random
random_5_numbers = random.sample(range(len(wrong_pred_image)), 5)

false_labels = []
true_labels = []

import matplotlib.pyplot as plt
for i in random_5_numbers:
    image_plottt = wrong_pred_image[i]
    false_labels.append(pred_false[i])
    true_labels.append(pred_true[i])
    plt.imshow(image_plottt[0])
    plt.show()
print("\n Correct labels are:", true_labels)
print(" Network predicted:", false_labels)



