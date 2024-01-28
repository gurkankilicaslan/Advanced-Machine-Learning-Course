import numpy as np

import torch
import torch.nn as nn



class MyAutoencoder(nn.Module):
    
    '''
    I am using the functions that we used in homework 3:
    '''
    def __init__ (self, n_input, lr, th, epoch_numbers):
        super(MyAutoencoder, self).__init__()
        
        num_int_nodes = 400
        num_out_nodes = 2
        
        self.epoch_numbers = epoch_numbers
        self.lr = lr
        self.th = th
        self.in_function_1 = nn.Linear(n_input, num_int_nodes)
        self.in_function_2 = nn.Linear(num_int_nodes, num_out_nodes)
        self.in_function_3 = nn.Linear(num_out_nodes, num_int_nodes)
        self.in_function_4 = nn.Linear(num_int_nodes, n_input)
        
        
    def forward(self, x):
        self.flatten = nn.Flatten()
        tanhFunc = nn.Tanh()
        sigmoidFunc = nn.Sigmoid()
        
        en_flat = self.flatten(x)
        encodee = tanhFunc(self.in_function_2(tanhFunc(self.in_function_1(en_flat))))
        
        decodee = sigmoidFunc(self.in_function_4(tanhFunc(self.in_function_3(encodee))))

        return encodee, decodee

    def fit(self, train_loader, criterion, optimizer):
        lossList = []
        epochList = []
        # Epoch loop
        
        for i in range(self.epoch_numbers):
            loss = 0
            # Mini batch loop
            for j,(feat,labels) in enumerate(train_loader):
                im_rows = feat.size(0)
                feat = feat.reshape(im_rows, 28*28)

                # Forward pass (consider the recommmended functions in homework writeup)
                enc, dec = self.forward(feat)
                
                # Backward pass and optimize (consider the recommmended functions in homework writeup)
                # Make sure to zero out the gradients using optimizer.zero_grad() in each loop
                loss2 = criterion(dec, feat)
                loss2.backward()
                
                
                optimizer.step()
                optimizer.zero_grad()
                
                
                loss = loss + float(loss2)
            
            
            print()
            
            lossList.append(loss /len(train_loader))
            
            new_loss = loss /len(train_loader)
            
            count = i+1
            epochList.append(count)
            print("Epoch ", count, "---> ", "loss =", new_loss)
            
            
            if i>0:
                if np.abs(lossList[-1] - lossList[-2]) < self.th:
                    break
        
        return lossList, epochList
            
        



    def data_projection(self, feat, criterion):
        
        with torch.no_grad(): # no backprop step so turn off gradients
        
            feat = torch.tensor(feat)
            
            enc, dec = self.forward(feat)

        return enc




