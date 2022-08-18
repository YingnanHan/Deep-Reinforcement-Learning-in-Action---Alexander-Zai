import gym
import torch
from collections import deque
import numpy as np
import matplotlib.pyplot as plt

env = gym.make("Freeway-ram-v0")
aspace = 3

vmin,vmax = -10,10
replay_size = 200
batch_size = 50
nsup = 51
dz = (vmax-vmin)/(nsup-1)
support = torch.linspace(vmin,vmax,nsup)

replay = deque(maxlen=replay_size) # 利用deque的资料结构建立经验池
lr = 0.0001 # 学习率
gamma = 0.1 # 折扣系数
epochs = 3000
eps = 0.20 # ε-贪婪策略中的初始ε值
eps_min = 0.05 # ε的最小值
priority_level = 5 # 优先回放：将重要的经验复制5次
update_freq = 25 # 每隔25步同步一次目标网路

# 初始化DQN参数向量
tot_params = 128*100 + 25*100 + aspace*25*51 # Dist-DQN的总参数数量
theta = torch.randn(tot_params)/10.0 # 随机产生Dist-DQN的初始参数
theta.requires_grad = True
theta_2 = theta.detach().clone() # 初始化目标网路的参数

losses = []
cum_rewards = [] # 每一次赢的游戏(成功越过公路)便在该串列中记录1
renders = []

def preproc_state(state): # 资料预处理函数
    p_state = torch.from_numpy(state).unsqueeze(dim=0).float()
    p_state = torch.nn.functional.normalize(p_state,dim=1) # 将状态中的数值正规化至0-1之间
    return p_state

def get_action(dist,support): # 动作选择策略函数
    actions = []
    for b in range(dist.shape[0]): # 以回圈的形式批次走访分布维度上的资料
        expectations = [support @ dist[b,a,:] for a in range(dist.shape[1])] # 计算每个动作价值分布的期望值
        action = int(np.argmax(expectations))
        actions.append(action)
    actions = torch.Tensor(actions).int()
    return actions

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

def lossfn(x,y): # 计算预测分布x和目标分布y之间的损失
    loss = torch.Tensor([0.])
    loss.requires_grad = True
    for i in range(x.shape[0]): # 走访批次中的每一个元素
        loss_ = -1*torch.log(x[i].flatten(start_dim=0)) @ y[i].flatten(start_dim=0) # 将个别资料的损失存放进一个阵列当中
        loss = loss + loss_ # 将损失阵列中的元素加总，得到该批次的总损失
    return loss

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

state = preproc_state(env.reset())

## 主要训练回圈
from random import shuffle

for i in range(epochs):
    print(i)
    pred = dist_dqn(state,theta,aspace=aspace)
    if i<replay_size or np.random.rand(1)<eps: # 利用ε-贪婪策略来选择动作
        action = np.random.randint(aspace)
    else:
        action = get_action(pred.unsqueeze(dim=0).detach(),support).item()

    state2,reward,done,info = env.step(action) # 在环境中执行选择的动作
    state2 = preproc_state(state2)

    if reward == 1:
        cum_rewards.append(1)

    if reward == 1: # 若成功穿越公路，将回馈值改为+10
        reward = 10
    else:
        reward = reward

    if done: # 若游戏以失败告终(很长一段时间后，仍未穿越公路)将回馈值改为-10
        reward = -100
    else:
        reward = reward

    if reward == 0: # 若游戏尚未有结果，将当前的回馈值修改为-1
        reward = -1
    else:
        reward = reward

    exp = (state,action,reward,state2) # 将得到的资讯打包成tuple的资料形态，作为训练资料
    replay.append(exp) # 将训练资料加入经验池之中
    if reward == 10: # 如果回馈值为10，那么代表该经验是重要的，要复制5分，重新加入到缓冲区
        for e in range(priority_level):
            replay.append(exp)
    shuffle(replay)
    state = state2

    if len(replay) == replay_size: # 当经验池的经验放满之后，开始进行训练
        indx = np.random.randint(low=0,high=len(replay),size=batch_size) # 随机从经验池中选取训练批次

        exps = [replay[j] for j in indx]
        state_batch = torch.stack([ex[0] for ex in exps]).squeeze()
        action_batch = torch.Tensor([ex[1] for ex in exps])
        reward_batch = torch.Tensor([ex[2] for ex in exps])
        state2_batch = torch.stack([ex[3] for ex in exps]).squeeze()

        pred_batch = dist_dqn(state_batch.detach(),theta,aspace=aspace)
        pred2_batch = dist_dqn(state2_batch.detach(),theta_2,aspace=aspace)

        target_dist = get_target_dist(pred2_batch,action_batch,reward_batch,support,lim=(vmin,vmax),gamma=gamma)

        loss = lossfn(pred_batch,target_dist.detach())
        losses.append(loss.item())
        loss.backward()
        with torch.no_grad():
            theta -= lr*theta.grad
        theta.requires_grad = True

    if i%update_freq == 0: # 同步目标网路与主模型的参数
        theta_2 = theta.detach().clone()

    if i>100 and eps > eps_min: # ε会随着训练的次数增加而下降，除非已经达到最小值
        dec = 1./np.log2(i)
        dec /= 1e3
        eps -= dec

    if done: # 当游戏结束时，重置游戏环境
        state = preproc_state(env.reset())
        done = False


plt.figure(figsize=(20,15))
plt.plot(losses)
plt.show()


## 测试
import time
env = gym.make("Freeway-ram-v0",render_mode="human") #创建出租车游戏环境
state = preproc_state(env.reset()) #初始化环境
for _ in range(5000):
    env.render()
    time.sleep(0.02)
    pred = dist_dqn(state,theta_2,3)
    action = get_action(pred.unsqueeze(dim=0).detach(),support).item()
    env.step(action) # take a selected action
env.close()