import numpy as np

import torch
import torch.nn as nn

class MyCNN(nn.Module):
    def __init__(self, input_size, output_size, kernel_size, stride_size, max_pool_size, learning_rate, max_epochs):
          
        super(MyCNN, self).__init__()
    
        self.lay_first = nn.Linear(3380, 128)
        self.lay_second = nn.Linear(128, output_size)
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.RELU= nn.ReLU()
        
        self.lay_convol = nn.Conv2d(1, 20, kernel_size, padding = 0, bias = True)
        self.max_pool = nn.MaxPool2d(max_pool_size, stride_size)
        # print(max_pool)
        p=0.5
        self.lay_dropout = nn.Dropout(p)
        # print(lay_dropout)
        self.input_size = input_size
        # print(input_size)

    def forward(self, x):
        
        # print(x)
        xOut = self.max_pool(self.RELU(self.lay_convol(x)))
        # print(xOut)
        xOut = self.lay_dropout(xOut)
        # print(xOut)
        self.flatten = nn.Flatten()
        xOut = self.lay_first(self.flatten(xOut))
        # print(xOut)
        xOut = self.lay_second(self.lay_dropout(self.RELU(xOut)))
        # print(xOut)
        
        return xOut

    def fit(self, train_loader, criterion, optimizer):
        
        total= 0
        error = 0

        for i in range(self.max_epochs):

            for j,(images,labels) in enumerate(train_loader, 0):
                
                im_rows = images.size(0)
                y_hat = self.forward(images)

                loss = criterion(y_hat, labels)
                loss.backward()
                
                optimizer.step()
                optimizer.zero_grad()
                
                ind_max= torch.argmax(y_hat,1)
                total = total + im_rows
                
                for k,l in zip(ind_max, labels):
                    if k != l:
                        error = error + 1

            LOSS = round(loss.item(),5)
            ERR_RATE = round(error/total, 5)

            print("    Epoch", i+1, "====> ", "loss =", LOSS, "|| error rate =", ERR_RATE)
            
            
    def predict(self, test_loader, criterion):
        
        total = 0 
        error = 0
        wrong_pred_image = []
        pred_false = []
        pred_true = []
        with torch.no_grad(): 
            for j,(images,labels) in enumerate(test_loader, 0):
                im_rows = images.size(0)

                y_hat = self.forward(images)
                
                loss = criterion(y_hat, labels)
                
                predict= torch.argmax(y_hat,1)
                
                count=0
                for k,l in zip(predict, labels):
                    count=count+1
                    if k != l:
                        error = error + 1
                        pred_false.append(k)
                        pred_true.append(l)
                        wrong_pred_image.append(images[count-1])
                  
                total = total + im_rows
            LOSS = round(loss.item(),5)
            ERR_RATE = round(error/total, 5)

        print("    Test Set Loss/Error Rate:")
        print("    loss =", LOSS, "|| error rate =", ERR_RATE)
        
        return pred_false, pred_true, wrong_pred_image, ERR_RATE
    
    