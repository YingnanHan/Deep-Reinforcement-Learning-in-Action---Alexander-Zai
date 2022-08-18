import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

def policy(qvalues,eps=None): # 策略函数接受动作价值向量与ε参数(eps)
    if esp is not None: # 若有指定一个eps值，则使用ε-贪婪策略
        if torch.rand(1)<eps:
            return torch.randint(low=0,high=11,size=(1,)) # 从12个动作中选择一个来执行
        else:
            return torch.argmax(qvalues)
    else:
        # 若未指定一个eps值，则不使用ε-贪婪策略
        return torch.multinomial(F.softmax(F.normalize(qvalues)),num_samples=1) # 选择一个要执行的动作