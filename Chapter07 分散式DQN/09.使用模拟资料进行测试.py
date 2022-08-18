import torch
import numpy as np
import matplotlib.pyplot as plt

def dist_dqn(x,theta,aspace=3): # x是128个元素的状态向量，theta是参数向量，aspace则是动作空间的大小
    dim0,dim1,dim2,dim3 = 128,100,25,51 # 51代表元素支撑集的数量 可认为是每一个动作概率分布中自变量所映射到离散空间的模
    t1 = dim0 * dim1 # 第一层网路参数的shape 元素这样摆放方便批处理
    t2 = dim1 * dim2 # 第二层网路参数的shape
    theta1 = theta[0:t1].reshape(dim0,dim1) # 将theta中的[0:t1]的参数分配到第一层网路的参数矩阵中
    theta2 = theta[t1:t1+t2].reshape(dim1,dim2) # 将theta中的[t1:t2]的参数分配到第一层网路的参数矩阵中
    l1 = x @ theta1 # 输入资料的shape是BX128，theta1的shape是128X100，因此l1(第一层网路)的shape是BX100 (B是代表批次的大小)
    l1 = torch.selu(l1) # 以scaled expotential linear units(SELUs)作为激活函数
    l2 = l1 @ theta2 # l1的shape是BX100 l2的shape是100X25 因此l2(第二层网路)的shape为BX25
    l2 = torch.selu(l2)
    l3 = []
    for i in range(aspace): # 利用循环走访每一个动作 并产生各个动作的价值分布
        step = dim2 * dim3 # 这里需要结合网络结构来进行思考
        theta5_dim = t1 + t2 + i*step
        theta5 = theta[theta5_dim:theta5_dim+step].reshape(dim2,dim3) # 将参数分配到不同的矩阵当中
        l3_ = l2 @ theta5 # l2 的shape是BX25，theta5的shape是25X51，因此该计算结果的shape为BX51
        l3.append(l3_) # 目前l3的大小是BX51 表示一共有B条数据/批次 每一个批次的输出中有51中状态
    l3 = torch.stack(l3,dim=1) # 最后一层的网路的shape BX3X51 也就是说 一共有B条数据 每一条数据由上左右三个状态的几率分布向量堆叠而成
    l3 = torch.nn.functional.softmax(l3,dim=2) # 对每一个状态的每一个输出预测做归一化操作
    return l3.squeeze() # 得到神经网路的输出

def get_target_dist(dist_batch,action_batch,reward_batch,support,lim=(-10,10),gamma=0.8):
    nsup =support.shape[0]
    vmin,vmax = lim[0],lim[1]
    dz = (vmax-vmin)/(nsup-1)
    target_dist_batch = dist_batch.clone() # 建立一个目标分布
    for i in range(dist_batch.shape[0]): # 使用循环走访整个分布
        dist_full = dist_batch[i]
        action = int(action_batch[i].item())
        dist = dist_full[action]
        r = reward_batch[i]
        if r != -1 : # 如果回馈值不是-1，代表已经到达最终状态，目标分布为退化分布(所有几率值集中在回馈值的位置)
            target_dist = torch.zeros(nsup)
            bj = np.round((r-vmin)/dz)
            bj = int(np.clip(bj,0,nsup-1))
            target_dist[bj] = 1
        else:
            # 若目前状态为非最终状态，则根据回馈值，按照贝叶斯方法来更新先验分布
            target_dist = update_dist(r,support,dist,lim=lim,gamma=gamma)
        target_dist_batch[i,action,:] = target_dist # 变更与执行动作相关的分布
    return target_dist_batch

aspace = 3 # 动作空间大小为3
tot_params = 128*100 + 25*100 + aspace*25*51 # 根据神经网络的大小来定义参数的总数
theta = torch.randn(tot_params) / 10 #随机生成Dist-DQN的初始参数向量
theta.requires_grad = True
theta_2 = theta.detach().clone() # 复制theta，将其用于目标网络
vmin,vmax = -10,10
gamma = 0.9
lr = 0.00001
update_rate = 75 # 每隔75步同步一次和主要目标DistDQN网路
support = torch.linspace(-10,10,51)
state = torch.randn(2,128)/10 # 随机产生两个测用使用的初始状态
action_batch = torch.Tensor([0,2]) # 产生动作资料
reward_batch = torch.Tensor([0,8]) # 产生回馈值资料
losses = []
pred_batch = dist_dqn(state,theta,aspace=aspace) # 初始化预测分布的批次
target_dist = get_target_dist(pred_batch,action_batch,reward_batch,support,lim=(vmin,vmax),gamma=gamma) # 初始化目标分布的批次
plt.plot((target_dist.flatten(start_dim=1)[0].data.numpy()),color="red",label="target")
plt.plot((pred_batch.flatten(start_dim=1)[0].data.numpy()),color="green",label="pred")
plt.legend()
plt.show()