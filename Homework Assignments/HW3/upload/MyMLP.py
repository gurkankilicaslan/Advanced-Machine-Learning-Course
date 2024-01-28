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
        
        self.lay_first = nn.Linear(input_size, hidden_size)
        self.lay_second= nn.Linear(hidden_size, output_size)
        self.learning_rate = learning_rate # this is not necessary
        self.max_epochs= max_epochs
        self.RELU= nn.ReLU()

    def forward(self, x):
        ''' Function to do the forward pass with images x '''
        ### Use the layers you constructed in __init__ and pass x through the network
        ### and return the output
        # print(x)
        xOut = self.lay_second(self.RELU(self.lay_first(x)))
        return xOut

    def fit(self, train_loader, criterion, optimizer):
        '''
        Function used to train the MLP

        train_loader: includes the feature matrix and class labels corresponding to the training set,
        criterion: the loss function used,
        optimizer: which optimization method to train the model.
        '''
        
        total=0
        error=0
        # Epoch loop
        for i in range(self.max_epochs):

            # Mini batch loop
            for j,(images,labels) in enumerate(train_loader):
                im_rows = images.size(0)
                images = images.reshape(im_rows, 28*28)

                # Forward pass (consider the recommmended functions in homework writeup)
                y_hat = self.forward(images)
                
                # Backward pass and optimize (consider the recommmended functions in homework writeup)
                # Make sure to zero out the gradients using optimizer.zero_grad() in each loop
                loss = criterion(y_hat, labels)
                loss.backward()
                
                optimizer.step()
                optimizer.zero_grad()
                
                # Track the loss and error rate
                
                ind_max= torch.argmax(y_hat,1)
                total = total + im_rows
                
                for k,l in zip(ind_max, labels):
                    if k != l:
                        error = error + 1
                        
            LOSS = round(loss.item(),5)
            ERR_RATE = round(error/total, 5)
            # Print/return training loss and error rate in each epoch
            print("    Epoch", i+1, "====> ", "loss =", LOSS, "|| error rate =", ERR_RATE)
    
    
    def predict(self, test_loader, criterion):
        '''
        Function used to predict with the MLP

        test_loader: includes the feature matrix and classlabels corresponding to the test set,
        criterion: the loss function used.
        '''
        total=0
        error=0
        with torch.no_grad(): # no backprop step so turn off gradients
            for j,(images,labels) in enumerate(test_loader):
                im_rows = images.size(0)
                images = images.reshape(im_rows, 28*28)
                # Compute prediction output and loss
                y_hat = self.forward(images)

                # Measure loss and error rate and record
                loss = criterion(y_hat, labels)
                
                ind_max= torch.argmax(y_hat,1)
                total = total + im_rows
                
                for k,l in zip(ind_max, labels):
                    if k != l:
                        error = error + 1
                
            LOSS = round(loss.item(),5)
            ERR_RATE = round(error/total, 5)

        # Print/return test loss and error rate
        print("    Test Set Loss/Error Rate:")
        print("    loss =", LOSS, "|| error rate =", ERR_RATE)