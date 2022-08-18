# 導入必要的模組
import gym
import gym_super_mario_bros
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from collections import deque
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT,COMPLEX_MOVEMENT # 我们可以汇入的动作包括两种：一种只包括5个动作，另一种只包括10个动作
from nes_py.wrappers import JoypadSpace # 此wrapper模组透过结合不同动作来缩小动作空间
from random import shuffle
from skimage.transform import resize # 该函式库中的内建函数可调整画面的大小
from torch import nn
from torch import optim

# 編寫主要的類模塊
class ExperienceReplay:

    def __init__(self,N=500,batch_size=100):
        self.N = N # N為記憶串列的最大長度
        self.batch_size = batch_size # 訓練批次的長度
        self.memory = []
        self.counter = 0

    def add_memory(self,state1,action,reward,state2):
        self.counter += 1
        if self.counter % 500 == 0 : # 記憶串列每新增500筆資料，便對記憶串列的內容進行洗牌，以更加隨機的方式選取訓練批次
            self.shuffle_memory()
        if len(self.memory)<self.N: # 若記憶串列未滿則將資料新增到記憶串列當中，否則隨機將一筆資料替換為新資料
            self.memory.append((state1,action,reward,state2))
        else:
            rand_indx = np.random.randint(0,self.N-1) # 隨機產生需要被替換掉的經驗索引
            self.memory[rand_indx] = (state1,action,reward,state2)

    def shuffle_memory(self): # 使用Python內建的shuffle函式來對記憶串列的內容進行洗牌
        shuffle(self.memory)

    def get_batch(self): # 從記憶串列中個隨機選取資料出來組成小批次
        if len(self.memory)<self.batch_size:
            batch_size = len(self.memory)
        else:
            batch_size = self.batch_size
        if len(self.memory) < 1:
            print("Error : No data in memory!")
            return None
        ind = np.random.choice(np.arange(len(self.memory)),size=batch_size,replace=False) # 隨機選擇出來要組成訓練批次的經驗索引
        batch = [self.memory[i] for i in ind]

        state1_batch = torch.stack([x[0].squeeze(dim=0) for x in batch],dim=0)
        action_batch = torch.Tensor([x[1] for x in batch]).long()
        reward_batch = torch.Tensor([x[2] for x in batch])
        state2_batch = torch.stack([x[3].squeeze(dim=0) for x in batch],dim=0)
        return state1_batch,action_batch,reward_batch,state2_batch

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
        self.linear1 = nn.Linear(576,256)
        self.linear2 = nn.Linear(256,12)

    def forward(self,state1,state2):
        x = torch.cat((state1,state2),dim=1)
        y = F.relu(self.linear1(x))
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
        y = y.view(y.shape[0],-1,32)
        y = y.flatten(start_dim=1)
        y = F.elu(self.linear1(y))
        y = self.linear2(y)
        return y

# 編寫的預處理函數
def downscale_obs(obs,new_size=(42,42),to_gray=True):
    # 原始状态资料  调整后的shape 转换成灰阶
    if to_gray:
        return resize(obs,new_size,anti_aliasing=True).max(axis=2) # 为了将图片转换成灰阶，我们取出obs的第二阶颜色通道中的最大值
    else:
        return resize(obs,new_size,anti_aliasing=True)

def prepare_state(state):
    # 将Numpy阵列转换成PyTorch张量 在第0阶加入一个批次阶(batch)
    return torch.from_numpy(downscale_obs(state,to_gray=True)).float().unsqueeze(dim=0)

def prepare_multi_state(state1,state2):
    # 更新最近三个游戏画面state1是包含3个游戏画面的状态资料，state2是最近的游戏画面资料
    tmp = torch.from_numpy(downscale_obs(state2,to_gray=True)).float() # 调整state2的shape
    state1[0][0] = state1[0][1] # 更新state1的3个游戏画面，最久的画面被淘汰，加入最新的游戏画面
    state1[0][1] = state1[0][2]
    state1[0][2] = tmp
    return state1

def prepare_initial_state(state,N=3):
    state_ = torch.from_numpy(downscale_obs(state,to_gray=True)).float()
    tmp = state_.repeat((N,1,1)) # 依照通道的排布顺序将图片信息(三维数组)复制三分
    return tmp.unsqueeze(dim=0) # 在第0阶加入批次阶

def policy(qvalues,eps=None): # 策略函数接受动作价值向量与ε参数(eps)
    if eps is not None: # 若有指定一个eps值，则使用ε-贪婪策略
        if torch.rand(1)<eps:
            return torch.randint(low=0,high=11,size=(1,)) # 从12个动作中选择一个来执行
        else:
            return torch.argmax(qvalues)
    else:
        # 若未指定一个eps值，则不使用ε-贪婪策略
        return torch.multinomial(F.softmax(F.normalize(qvalues)),num_samples=1) # 选择一个要执行的动作

def loss_fn(q_loss,inverse_loss,forward_loss):
    loss_ = (1-params['beta'])*inverse_loss
    loss_ += params['beta']*forward_loss
    loss_ = loss_.sum() / loss_.flatten().shape[0]
    loss = loss_ + params['lambda'] * q_loss
    return loss

def reset_env():
    env.reset()
    state1 = prepare_initial_state(env.render('rgb_array'))
    return state1

def ICM(state1,action,state2,forward_scale=1.,inverse_scale=1e4): # 計算ICM的預測誤差
    state1_hat = encoder(state1) # 使用編碼器將狀態1和2編碼
    state2_hat = encoder(state2)
    state2_hat_pred = forward_model(state1_hat.detach(),action.detach()) # 利用正向模型預測新的狀態
    forward_pred_err = forward_scale*forward_loss(state2_hat_pred,state2_hat.detach()).sum(dim=1).unsqueeze(dim=1)
    pred_action = inverse_model(state1_hat,state2_hat) # 反向模型傳回個動作的幾率分佈
    inverse_pred_err = inverse_scale*inverse_loss(pred_action,action.detach().flatten()).unsqueeze(dim=1)
    return forward_pred_err,inverse_pred_err

def minibatch_train(use_extrinsic=True):
    state1_batch,action_batch,reward_batch,state2_batch = replay.get_batch()
    action_batch = action_batch.view(action_batch.shape[0],1) # 在次張量中增加一個維度，以符合模型的需求
    reward_batch = reward_batch.view(reward_batch.shape[0],1)
    forward_pred_err,inverse_pred_err = ICM(state1_batch,action_batch,state2_batch) # 運行ICM
    i_reward = (1./params["eta"])*forward_pred_err # 使用eta參數來調整預測誤差的權重
    reward = i_reward.detach() # 把i_reward張量從運算圖中分離，並開始計算總回饋值
    if use_extrinsic: # 決定演算法是否要使用外在回饋值
        reward += reward_batch
    qvals = Qmodel(state2_batch) # 計算新狀態的動作價值
    reward += params["gamma"] * torch.max(qvals)
    reward_pred = Qmodel(state1_batch)
    reward_target = reward_pred.clone()
    indices = torch.stack((torch.arange(action_batch.shape[0]),action_batch.squeeze()),dim=0) # action_batch是由整個數組組成的張量，且每一個整數代表一個動作，這裡將他們轉換成有多個one-hot編碼向量組成的張量
    indices = indices.tolist()
    reward_target[indices] = reward.squeeze()
    q_loss = 1e5*qloss(F.normalize(reward_pred),F.normalize(reward_target.detach()))
    return forward_pred_err,inverse_pred_err,q_loss

if __name__ == '__main__':

    # 定義全局環境
    env = gym_super_mario_bros.make("SuperMarioBros-v0")
    env = JoypadSpace(env, COMPLEX_MOVEMENT)  # 选择complex的动作空间

    # 定義關鍵的參數
    params = {
        'batch_size': 150,
        'beta': 0.2,
        'lambda': 0.1,
        'eta': 1.0,
        'gamma': 0.2,
        'max_episode_len': 100,
        'min_progress':15,
        'action_repeats': 6,
        'frames_per_state': 3
    }
    epochs = 500
    done = True
    eps = 0.15
    losses = []
    episode_length = 0
    switch_to_eps_greedy = 1000
    state_deque = deque(maxlen=params['frames_per_state'])
    e_reward = 0
    last_x_pos = env.env.env._x_position # 用來追蹤瑪麗歐是否有在前進，若很久沒前進則重置遊戲
    ep_lengths = []
    use_explicit = False # 不使用外在回饋值

    # 定義必要的類物件
    replay = ExperienceReplay(N=1000, batch_size=params['batch_size'])
    Qmodel = Qnetwork()
    encoder = Phi()
    forward_model = Fnet()
    inverse_model = Gnet()
    forward_loss = nn.MSELoss(reduction='none')
    inverse_loss = nn.CrossEntropyLoss(reduction='none')
    qloss = nn.MSELoss()
    all_model_params = list(Qmodel.parameters()) + list(encoder.parameters()) + list(forward_model.parameters()) + list(inverse_model.parameters())
    opt = optim.Adam(lr=0.001, params=all_model_params)

    # 訓練前的準備
    env.reset()
    state1 = prepare_initial_state(env.render("rgb_array"))

    # 主要訓練迴圈
    for i in range(epochs):
        print("Epochs:",i," Game progress:"," ",last_x_pos)
        opt.zero_grad()
        episode_length += 1

        q_val_pred = Qmodel(state1) # 運行DQN並預測一個動作的價值向量
        if  i>switch_to_eps_greedy: # 在1000次訓練過後，換成ε-貪婪策略
            action = int(policy(q_val_pred,eps))
        else:
            action = int(policy(q_val_pred))
        for j in range(params['action_repeats']): # 為加速學習，將選擇的動作重複6次
            state2,e_reward_,done,info = env.step(action)
            last_x_pos = info["x_pos"]
            if done:
                state1 = reset_env()
                break
            e_reward += e_reward
            state_deque.append(prepare_state(state2))
        state2 = torch.stack(list(state_deque),dim=1) # 將state_deque串列轉換成向量
        replay.add_memory(state1,action,e_reward,state2) # 將一筆經驗資料存入經驗池
        e_reward = 0
        if episode_length > params['max_episode_len']: # 若瑪麗歐未在遊戲中前進足夠多的距離，則將遊戲重做，再試一次
            if (info['x_pos']-last_x_pos)<params['min_progress']:
                done = True
            else:
                last_x_pos = info['x_pos']
        if done:
            ep_lengths.append(info['x_pos'])
            state1 = reset_env()
            last_x_pos=  env.env.env._x_position
            episode_length = 0
        else:
            state1 = state2
        if len(replay.memory)<params['batch_size']:
            continue
        forward_pred_err,inverse_pred_err,q_loss = minibatch_train(use_extrinsic=False) # 根據從經驗池選出的一小批次資料，計算各類模型的誤差
        loss = loss_fn(q_loss,forward_pred_err,inverse_pred_err) # 計算整體損失
        loss_list = (q_loss.mean(),forward_pred_err.flatten().mean(),inverse_pred_err.flatten().mean())
        losses.append(loss_list)
        loss.backward()
        opt.step()

    ## 模式轉換
    losses_ = []
    for i in range(len(losses)):
        l = []
        for j in range(len(losses[0])):
            l.append(float(losses[i][j].detach().numpy()))
        losses_.append(l)


    ## 繪製圖像
    losses_ = np.array(losses_)
    plt.figure(figsize=(8,6))
    plt.plot(np.log(losses_[:, 0]), label="Q Loss")
    plt.plot(np.log(losses_[:, 1]), label="Forward Loss")
    plt.plot(np.log(losses_[:, 2]), label="Inverse Loss")
    plt.legend()
    plt.show()

    ## 測試模型
    import time
    env = gym_super_mario_bros.make("SuperMarioBros-v0")
    env = JoypadSpace(env, COMPLEX_MOVEMENT)  # 选择complex的动作空间
    eps = 0.1
    done = True
    state_deque = deque(maxlen=params['frames_per_state'])
    for step in range(5000):
        print(step,env.env.env._x_position)
        env.render()
        time.sleep(0.05)
        if done:
            env.reset()
            state1 = prepare_initial_state(env.render("rgb_array"))
        q_val_pred = Qmodel(state1)
        action = int(policy(q_val_pred,eps))
        state2,reward,done,info = env.step(action)
        state2 = prepare_multi_state(state1,state2)
        state1 = state2