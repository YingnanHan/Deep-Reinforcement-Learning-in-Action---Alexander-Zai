import torch
import numpy as np
import matplotlib.pyplot as plt

def lossfn(x,y): # 计算预测分布x和目标分布y之间的损失
    loss = torch.Tensor([0.])
    loss.requires_grad = True
    for i in range(x.shape[0]): # 走访批次中的每一个元素
        loss_ = -1*torch.log(x[i].flatten(start_dim=0)) @ y[i].flatten(start_dim=0) # 将个别资料的损失存放进一个阵列当中
        loss = loss + loss_ # 将损失阵列中的元素加总，得到该批次的总损失
    return loss
