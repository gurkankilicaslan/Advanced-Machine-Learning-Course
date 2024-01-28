import numpy as np

import torch
import torch.nn as nn

# Fully connected neural network with one hidden layer
class MyMLP(nn.Module):
    
    def __init__(self, input_size, hidden_size, output_size, learning_rate, max_epochs):
        '''
        input_size: [int], feature dimension 
        hidden_size: number of hidden nodes in the hidden layer
        output_size: number of classes in the dataset, 
        learning_rate: learning rate for gradient descent,
        max_epochs: maximum number of epochs to run gradient descent
        '''
        ### Construct your MLP Here (consider the recommmended functions in homework writeup)  
        super(MyMLP, self).__init__()

    def forward(self, x):
        ''' Function to do the forward pass with images x '''
        ### Use the layers you constructed in __init__ and pass x through the network
        ### and return the output

    def fit(self, train_loader, criterion, optimizer):
        '''
        Function used to train the MLP

        train_loader: includes the feature matrix and class labels corresponding to the training set,
        criterion: the loss function used,
        optimizer: which optimization method to train the model.
        '''

        # Epoch loop
        for i in range(self.max_epochs):

            # Mini batch loop
            for j,(images,labels) in enumerate(train_loader):

                # Forward pass (consider the recommmended functions in homework writeup)
            
                # Backward pass and optimize (consider the recommmended functions in homework writeup)
                # Make sure to zero out the gradients using optimizer.zero_grad() in each loop

                # Track the loss and error rate
                
            # Print/return training loss and error rate in each epoch


    def predict(self, test_loader, criterion):
        '''
        Function used to predict with the MLP

        test_loader: includes the feature matrix and classlabels corresponding to the test set,
        criterion: the loss function used.
        '''

        with torch.no_grad(): # no backprop step so turn off gradients
            for j,(images,labels) in enumerate(test_loader):

                # Compute prediction output and loss

                # Measure loss and error rate and record

        # Print/return test loss and error rate


