import torch
import matplotlib.pyplot as plt
import numpy as np
from torch import nn
from torch.nn import functional as F

class Phi(nn.Module): # Phi代表編碼器網路

    def __init__(self):
        super(Phi,self).__init__()
        self.conv1 = nn.Conv2d(3,32,kernel_size=(3,3),stride=2,padding=1)
        self.conv2 = nn.Conv2d(32,32,kernel_size=(3,3),stride=2,padding=1)
        self.conv3 = nn.Conv2d(32,32,kernel_size=(3,3),stride=2,padding=1)
        self.conv4 = nn.Conv2d(32,32,kernel_size=(3,3),stride=2,padding=1)

    def forward(self,x):
        x = F.normalize(x)
        y = F.elu(self.conv1(x))
        y = F.elu(self.conv2(y))
        y = F.elu(self.conv3(y))
        y = F.elu(self.conv4(y)) # 輸出的shape為[1,32,3,3]
        y = y.flatten(start_dim=1) # shape扁平化成N,288
        return y


class Gnet(nn.Module): # Gnet代表反向模型

    def __init__(self):
        super(Gnet,self).__init__()
        self.linear1 = nn.Linear(575,256)
        self.linear2 = nn.Linear(256,12)

    def forward(self,state1,state2):
        x = torch.cat((state1,state2),dim=1)
        y = F.relu((self.linear1(x)))
        y = self.linear2(y)
        y = F.softmax(y,dim=1)
        return y

class Fnet(nn.Module): # Fnet代表正向模型

    def __init__(self):
        super(Fnet,self).__init__()
        self.linear1 = nn.Linear(300,256)
        self.linear2 = nn.Linear(256,288)

    def forward(self,state,action):
        action_ = torch.zeros(action.shape[0],12) # 將執行的動作批次編碼成one-hot向量
        indices = torch.stack((torch.arange(action.shape[0]),action.squeeze()),dim=0)
        indices = indices.tolist()
        action_[indices] = 1.
        x = torch.cat((state,action_),dim=1)
        y = F.relu(self.linear1(x))
        y = self.linear2(y)
        return y