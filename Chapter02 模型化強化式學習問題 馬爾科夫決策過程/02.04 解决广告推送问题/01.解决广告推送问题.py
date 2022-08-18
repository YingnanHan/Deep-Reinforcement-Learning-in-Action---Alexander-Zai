import numpy as np
import random
import torch
import matplotlib.pyplot as plt

class ContextBandit: # 拉霸機環境類別

 def __init__(self,arms=10):
     self.arms = arms # 這裡的arms代表廣告
     self.init_distributions(arms)
     self.update_state()

 def init_distributions(self,arms):
     states = arms # 讓狀態數=廣告數以方便處理
     self.bandit_matrix = np.random.rand(states,arms) # 隨機產生10種狀態下，10個arms的幾率分佈(10X10種幾率)

 def reward(self,prob): # 用途與前面的程式2.3相同
     reward = 0
     for i in range(self.arms):
         if random.random() < prob:
             reward += 1
     return reward

 def update_state(self):
     self.state = np.random.randint(0,self.arms) # 隨機產生一個新的狀態

 def get_state(self): # 取得當前狀態 回傳當前的狀態值
     return self.state

 def get_reward(self,arm): # 根據當前狀態以及選擇的arm傳回回饋值
     return self.reward(self.bandit_matrix[self.get_state()][arm])

 def choose_arm(self,arm): # 推送一個廣告，並回傳一個回饋值
     reward = self.get_reward(arm)
     self.update_state() # 產生下一個狀態
     return reward # 傳回回饋值

def one_hot(N,pos,val=1): # pos代表當前狀態號碼(即客戶所處在的網站編號)
    one_hot_vec = np.zeros(N)
    one_hot_vec[pos] = val
    return one_hot_vec

def softmax(av,tau=1.12): # av即動作的價值陣列，tau即溫度參數(預設值1.12)
    softm = (np.exp(av/tau))/np.sum(np.exp(av/tau))

def train(env,epochs=10000,learning_rate=1e-2): # 執行10000次訓練
    cur_state = torch.Tensor(one_hot(arms,env.get_state())) # 取得當前環境狀態，並將其編碼為one-hot張量
    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
    rewards = []
    for  i in range(epochs):
        y_pred = model(cur_state) # 執行神經網絡 並預測回饋值
        av_softmax = softmax(y_pred.data.numpy(),tau=1.12) # 利用softmax()將預測結果轉化為幾率分佈向量
        choice = np.random.choice(arms,p=av_softmax) # 依照softmax輸出的幾率分佈來選取新的動作
        cur_reward = env.choose_arm(choice) # 執行所選擇的動作，並得到一個回饋值
        one_hot_reward = y_pred.data.numpy().copy()
        one_hot_reward[choice] = cur_reward # 更新one_hot陣列的值，把他當作標籤(即實際的回饋值)
        reward = torch.Tensor(one_hot_reward)
        rewards.append(cur_reward) # 將回饋值存入rewards之中，以便稍後繪製線圖
        loss = loss_fn(y_pred,reward) # y_pred是預測的回饋值 reward是實際的回饋值 將預測出的回饋值與實際的回饋值作比較 計算出損失

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        cur_state = torch.Tensor(one_hot(arms,env.get_state())) # 更新目前的環境狀態
    return np.array(rewards)

def runnning_mean(x,N): # 定義一個可以算出移動平均回饋值的函式
    c = x.shape[0] - N
    y = np.zeros(c)
    conv = np.ones(N)
    for i in range(c):
        y[i] = (x[i:i+N]@conv)/N
    return y

if __name__ == '__main__':

    env = ContextBandit(arms=10) # 創建一個環境
    state = env.get_state() # 取得當前的狀態
    reward = env.choose_arm(1) # 在目前的狀態下選擇推送1號網站的廣告，並計算其回饋值
    print(state,reward)

    # 下列程式碼匯入以後所需的函式庫
    arms = 10
    N,D_in,H,D_out, = 1,arms,100,arms

    # 建構神經網路
    model = torch.nn.Sequential(
        torch.nn.Linear(D_in,H), # 隱藏層
        torch.nn.ReLU(), #
        torch.nn.Linear(H,2*H),
        torch.nn.Linear(2*H,3*H),
        torch.nn.Linear(3*H,2*H),
        torch.nn.Linear(2*H,H),
        torch.nn.Linear(H,D_out), # 輸出層
        torch.nn.ReLU(),
    )
    loss_fn = torch.nn.MSELoss() # 以均方誤差作為損失函數
    env = ContextBandit(arms)

    # 測試獨熱編碼
    A = one_hot(10,4)
    print(A)

    # 開始訓練1000次
    rewards = train(env)

    plt.figure(figsize=(20,7))
    plt.ylabel("Average reward",fontsize=14)
    plt.xlabel("Training Epochs",fontsize=14)
    plt.plot(runnning_mean(rewards,N=50)) # 計算最近50次的平均回饋值，並將結果畫出來
    plt.show()