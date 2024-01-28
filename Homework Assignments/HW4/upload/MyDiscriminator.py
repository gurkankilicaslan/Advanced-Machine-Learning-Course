import numpy as np

import torch
import torch.nn as nn

class MyDiscriminator(nn.Module):
    def __init__ (self, input_size):
        super(MyDiscriminator, self).__init__()
        
        out1 = 1024
        
        in2 = 1024
        out2 = 512
        
        in3 = 512
        out3 = 256
        
        in4 = 256
        out4 = 1
        
        self.in_function_1 = nn.Linear(input_size, out1)
        self.in_function_2 = nn.Linear(in2, out2)
        self.in_function_3 = nn.Linear(in3, out3)
        self.in_function_4 = nn.Linear(in4, out4)
        
        
    def forward(self, x):
        
        self.flatten = nn.Flatten()
        self.drop = nn.Dropout(0.30)
                
        sigmoid = nn.Sigmoid()
        relu = nn.ReLU()
        
        xout1 = self.flatten(x)
        
        xout2 = self.in_function_2(self.drop(relu(self.in_function_1(xout1))))
       
        xout3 = self.in_function_3(self.drop(relu(xout2)))
        
        xout4 = sigmoid(self.in_function_4(self.drop(relu(xout3))))

        return xout4
    
