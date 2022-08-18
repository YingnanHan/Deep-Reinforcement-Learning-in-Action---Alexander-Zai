import torch
import numpy as np
from matplotlib import pyplot as plt

def update_dist(r,support,probs,lim=(-10,10),gamma=0.8):
    nsup = probs.shape[0] # 取得支撑集内的元素数量
    vmin,vmax = lim[0],lim[1] # 取得支撑集内元素值得上下限
    dz = (vmax-vmin)/(nsup-1) # 计算支撑集中元素间隔的大小
    bj = np.round((r-vmin)/dz) # 计算支撑集中，与回馈值对应的元素索引
    bj = int(np.clip(bj,0,nsup-1)) # 将输出索引限制在[0,nsup-1]的范围内并对其进行做进位处理
    m = probs.clone() # 复制一份当前的几率分布
    j = 1
    for i in range(bj,1,-1): # 从左侧的元素拿走一部分几率质量
        m[i] += np.power(gamma,j)*m[i-1]
        j += 1
    j = 1
    for i in range(bj,nsup-1,1): # 从右侧的元素拿走一部分几率质量
        m[i] += np.power(gamma,j)*m[i+1]
        j += 1
    m /= m.sum() # 将m阵列中的几率值进行正规化，确保所有几率质量相加后等于1
    return m

## 测验

vmin,vmax = -10,10 # 设定支撑集的最小值与最大值
nsup = 51 # 设定支撑集元素数量
support = np.linspace(vmin,vmax,nsup) # support 里面的元素有51个，范围介于-10到10之间，且元素间隔相等
probs = np.ones(nsup) # 设定每个元素的几率皆为1/51
probs /= probs.sum()
z3 = torch.from_numpy(probs).float() # 将几率分布存在z3中
plt.bar(support,probs) # 将分布画成长条图
plt.show()
print(support) # 印出支撑集中的元素
print(probs)

## 根据单一观测结果重新分配几率质量
ob_reward = -1 # 假设观测到的回馈值为-1
Z = torch.from_numpy(probs).float()
Z = update_dist(ob_reward,torch.from_numpy(support).float(),Z,lim=(vmin,vmax),gamma=0.1) # 更新几率分布
plt.bar(support,Z)
plt.show()
print(Z)