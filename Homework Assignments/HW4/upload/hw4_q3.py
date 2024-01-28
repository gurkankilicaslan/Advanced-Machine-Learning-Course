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



import random as rd
class GAN(nn.Module):
    def __init__(self, gennnn, dissss):
        super(GAN, self).__init__()
        self.gennnn = gennnn
        self.dissss = dissss

    def forward(self, x):
        
        
        temp = torch.normal(0, 1, size=(batch_size, batch_size))
        wrong = self.gennnn.forward(temp)
        
        outWrong = self.dissss.forward(wrong)

        outTrue = self.dissss.forward(x)
        
        return outWrong, outTrue, wrong

    def fit(self, train_loader, th , Epoch):
        OptimizeGen = torch.optim.Adam(self.gennnn.parameters(), 0.0002)
        OptimizeDis = torch.optim.Adam(self.dissss.parameters(), 0.0002)
        
        emptyLossDis,emptyLossGen,listEpoch,ImsRandom = [],[],[],[]
        
        ImWrong = torch.empty((1, 784))
        for i in range(Epoch):
            lossDis,lossGen = 0,0
            
            
            
            for j, (IMAGES, useless1) in enumerate(train_loader):
                criterion = nn.BCELoss()
                
                useIMAGES = IMAGES.view(-1, 784)
                outWrong, outTrue, wrongIms = self.forward(useIMAGES) 

                self.dissss.zero_grad()
                lossDis = criterion(outTrue, torch.ones(outTrue.shape, dtype=outTrue.dtype))
                
                lossDis = lossDis + criterion(outWrong, torch.zeros_like(outWrong))
                
                
                lossDis.backward()
                OptimizeDis.step()
                
                lossDis = lossDis + float(lossDis)

                self.gennnn.zero_grad()
                outWrongr, useless2, useless3 = self.forward(useIMAGES)
                lossGen = criterion(outWrongr, torch.ones(outWrong.shape, dtype=outWrong.dtype))
                
                
                lossGen.backward()
                OptimizeGen.step()
                
                lossGen = lossGen + float(lossGen)

                # image_s = wrongIms
            ImWrong = torch.cat((ImWrong, wrongIms), dim=0)
            chooseRandom = rd.choice(range(len(ImWrong)))
            img_from_epoch = ImWrong[chooseRandom]
            ImsRandom.append(img_from_epoch)
            self.ImsRandom = ImsRandom

            lossDis = lossDis.item()
            lossGen = lossGen.item()
            emptyLossDis.append(lossDis)
            emptyLossGen.append(lossGen)
            epochhh= i+1
            listEpoch.append(epochhh)
            print("Epoch ", epochhh, "---> ", "Disc. Loss =", round(lossDis,5), " |||||  Gen. Loss =", round(lossGen,5))
            
            
            if i > 39:
                if abs(emptyLossDis[-1] - sum(emptyLossDis[-40:])/40) < th and abs(emptyLossGen[-1] - sum(emptyLossGen[-40:])/40) < th:
                    plt.plot(listEpoch, emptyLossDis)
                    plt.plot(listEpoch, emptyLossGen)
                    plt.legend(['Discriminator', 'Generator'])
                    plt.savefig('p3_lossTrain.png')
                    break
                else:
                    if i == Epoch-1:
                        plt.plot(listEpoch, emptyLossDis)
                        plt.plot(listEpoch, emptyLossGen)
                        plt.legend(['Discriminator', 'Generator'])
                        plt.savefig('p3_lossTrain.png')
                        
        return self.ImsRandom
     

gen = MyGenerator()
disc = MyDiscriminator(784)
useGAN = GAN(gen, disc)

threshh = 0.03
maxEpochh = 150
use_for_images = useGAN.fit(train_loader, threshh, maxEpochh)


plt.figure()

    
for i in range(len(use_for_images)):
    plt.figure()
    a = use_for_images[i]
    a = a.detach().numpy()
    
    a = np.reshape(a, (28, 28))
    plt.imshow(a)
    plt.title('Epoch: ' + str(i+1))
    plt.show()
    plt.savefig('p3_Image_at_Epoch_' + str(i+1))
    


