import torch
import numpy as np
from matplotlib import pyplot as plt

vmin,vmax = -10,10 # 设定支撑集的最小值与最大值
nsup = 51 # 设定支撑集元素数量
support = np.linspace(vmin,vmax,nsup) # support 里面的元素有51个，范围介于-10到10之间，且元素间隔相等
probs = np.ones(nsup) # 设定每个元素的几率皆为1/51
probs /= probs.sum()
z3 = torch.from_numpy(probs).float() # 将几率分布存在z3中
plt.bar(support,probs) # 将分布画成长条图
plt.show()
print(support) # 印出支撑集中的元素


