################################
# DO NOT EDIT THE FOLLOWING CODE
################################
import numpy as np

import torch
import torch.nn as nn

import torchvision
from torchvision.transforms import Compose, ToTensor, Normalize

from MyMLP import MyMLP

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
for i in [1,2,3,4]:
    for eta in l_rate:
        mymlp = MyMLP(28*28, 128, 10, eta, N)
        
        if i==1: # SGD
            print('SGD with Learning Rate =',eta,':')
            mini = torch.optim.SGD(mymlp.parameters(), eta)
        elif i==2: # Adagrad
            print('Adagrad with Learning Rate =',eta,':')
            mini = torch.optim.Adagrad(mymlp.parameters(), eta)
        elif i==3: # RMSprop
            print('RMSprop with Learning Rate =',eta,':')
            mini = torch.optim.RMSprop(mymlp.parameters(), eta)
        else: # Adam
            print('Adam with Learning Rate =',eta,':')
            mini = torch.optim.Adam(mymlp.parameters(), eta)
            
        mymlp_fit = mymlp.fit(train_loader, losss, mini)
        mymlp_test = mymlp.predict(test_loader, losss)
    print('\n')
    