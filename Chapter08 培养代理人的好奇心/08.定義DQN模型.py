import torch
import matplotlib.pyplot as plt
import numpy as np
from torch import nn
from torch.nn import functional as F

class Qnetwork(nn.Module):

    def __init__(self):
        super(Qnetwork,self).__init__()
        self.conv1 = nn.Conv2d( in_channels=3,out_channels=32,kernel_size=(3,3),stride=2,padding=1)
        self.conv2 = nn.Conv2d(in_channels=32,out_channels=32,kernel_size=(3,3),stride=2,padding=1)
        self.conv3 = nn.Conv2d(in_channels=32,out_channels=32,kernel_size=(3,3),stride=2,padding=1)
        self.conv4 = nn.Conv2d(in_channels=32,out_channels=32,kernel_size=(3,3),stride=2,padding=1)
        self.linear1 = nn.Linear(288,100) # 1*32*3*3
        self.linear2 = nn.Linear(100,12)

    def forward(self,x):
        x = F.normalize(x)
        y = F.elu(self.conv1(x))
        y = F.elu(self.conv2(y))
        y = F.elu(self.conv3(y))
        y = F.elu(self.conv4(y))
        y = y.flatten(start_dim=2)
        y = F.elu(self.linear1(y))
        y = self.linear2(y)
        return y