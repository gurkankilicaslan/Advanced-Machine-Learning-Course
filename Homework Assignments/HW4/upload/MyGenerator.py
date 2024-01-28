import numpy as np

import torch
import torch.nn as nn

class MyGenerator(nn.Module):
    def __init__ (self):
        super(MyGenerator, self).__init__()
        
        out1 = 256
        
        in2 = 256
        out2 = 512
        
        in3 = 512
        out3 = 1024
        
        in4 = 1024
        out4 = 784
        
        self.in_function_1 = nn.Linear(128, out1)
        self.in_function_2 = nn.Linear(in2, out2)
        self.in_function_3 = nn.Linear(in3, out3)
        self.in_function_4 = nn.Linear(in4, out4)
        

    def forward(self, x):
        tanh = nn.Tanh()
        relu = nn.ReLU()
        xout1 = self.in_function_1(x)
       
        xout2 = self.in_function_2(relu(xout1))
 
        xout3 = self.in_function_3(relu(xout2))
        
        xout4 = tanh(self.in_function_4(relu(xout3)))

        return xout4
