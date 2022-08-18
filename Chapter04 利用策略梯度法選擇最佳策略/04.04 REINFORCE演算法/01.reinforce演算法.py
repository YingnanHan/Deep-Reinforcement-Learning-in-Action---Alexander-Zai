import gym
import numpy as np
import torch
from gym import envs


L1 = 4 # 輸入資料的向量長度
L2 = 150 # 隱藏層會輸出長度為150的向量
L3 = 2 # 策略網路所輸出的向量長度

env = gym.make("CartPole-v0")

model = torch.nn.Sequential(
    torch.nn.Linear(L1,L2), # 隱藏層的shape為(L1,L2)
    torch.nn.LeakyReLU(),
    torch.nn.Linear(L2,L3), # 輸出層的shape為(L2,L3)
    torch.nn.Softmax(dim=0) # 使用softmax()將動作價值轉化為幾率
)
learning_rate = 0.009
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)

# pred = model(torch.from_numpy(state1).float()) # 呼叫策略網路模型以產生各動作的幾率分佈向量(命名為pred)
# action = np.random.choice(np.array([0,1]),p=pred.data.numpy()) # 根據策略網路的輸出的幾率分佈pred來隨機選取動作
# state2,reward,done,info = env.step(action) # 執行動作並記錄不同資料

# 計算折扣回報值
def discount_rewards(rewards,gamma=0.99):
    lenr = len(rewards) # 遊戲進行了多少步
    disc_return = torch.pow(gamma,torch.arange(lenr).float())*rewards # 計算呈指數方式下降的折扣回報值陣列
    disc_return /= disc_return.max() # 正規化以上陣列，使其元素的值落在[0,1]
    return disc_return

# 定義損失函數
def loss_fn(preds,r): # 輸入參數為:幾率陣列以及折扣回報值陣列
    return -1*torch.sum(r*torch.log(preds))

# REINFOPRCE算法的訓練迴圈
MAX_DUR = 200 # 每場遊戲最大步數
MAX_EPISODES = 5000 # 訓練遊戲的次數
gamma = 0.99
score = [] # 創立一個串列來記錄每場遊戲的長度

for episode in range(MAX_EPISODES):
    cur_state = env.reset()
    done = False
    transition = [] # 使用串列記錄狀態 動作 回饋值(即經驗)
    for t in range(MAX_DUR):
        act_prob = model(torch.from_numpy(cur_state).float()) # 用模型預測個動作的幾率分佈
        action = np.random.choice(np.array([0,1]),p=act_prob.data.numpy()) # 參照幾率分佈隨機選擇一個動作
        prev_state = cur_state
        cur_state,_,done,info = env.step(action) # 在環境中執行動作，並取得新的狀態以及是否結束的信息
        transition.append((prev_state,action,t+1)) # 將當前的狀態記錄下來(每一輪遊戲結束後，會把回饋值按照時間步遞增並存進串列，接下來會將這個串列翻轉(有遞增改為遞減排列)為匯報值張量)
        if done: #如果輸掉了遊戲 就跳出迴圈
            break
    ep_len = len((transition)) # 取得整場遊戲的長度
    score.append(ep_len)

    reward_batch = torch.Tensor([r for (s,a,r) in transition]).flip(dims=(0,)) # 將整場遊戲中的所有回饋值記錄到一個張量中，flip()可以將指定階(第0階)中的元素進行翻轉(前後順序對調)
    disc_returns = discount_rewards(reward_batch) # 計算折扣匯報值陣列
    state_batch = torch.Tensor([s for (s,a,r) in transition]) # 將整場遊戲中的所有狀態記錄到一個張量當中去
    action_batch = torch.Tensor([a for (s,a,r) in transition]) # 將整場遊戲中的所有动作記錄到一個張量當中去
    pred_batch = model(state_batch) # 重新計算所有狀態下個動作的幾率分佈
    prob_batch = pred_batch.gather(dim=1,index=action_batch.long().unsqueeze(dim=1)).squeeze() # 找出所執行動作對應的幾率值
    loss = loss_fn(prob_batch,disc_returns)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# 計算移動平均值
def running_mean(x,N=50):
    kernel = np.ones(N)
    conv_len = x.shape[0]-N
    y = np.zeros(conv_len)
    for i in range(conv_len):
        y[i] = kernel@x[i:i+N]
        y[i]/=N
    return y

import matplotlib.pyplot as plt

score = np.array(score)
avg_score = running_mean(score,50)

plt.figure(figsize=(10,7))
plt.ylabel("Episode Duration",fontsize=22)
plt.xlabel("Training Epochs",fontsize=22)
plt.plot(avg_score,color="green")
plt.show()

# 測試過程
import time
for i in range(10000):
    cur_state = env.reset()
    count = 0
    while True:
        # 采样动作，探索环境
        env.render()
        act_prob = model(torch.from_numpy(cur_state).float())
        action = np.random.choice(np.array([0, 1]), p=act_prob.data.numpy())
        cur_state, _, done, info = env.step(action)
        if done:
            print(count)
            break

        count+=1
        time.sleep(0.001)
    print(count)