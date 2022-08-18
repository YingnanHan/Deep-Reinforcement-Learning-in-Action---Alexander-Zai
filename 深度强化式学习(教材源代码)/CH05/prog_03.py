# gamma = 0.9
# for i in epochs: #重複訓練 epochs 次
#  state = environment.get_state() #取得環境目前的狀態
#  value = critic(state) #價值網路預測目前狀態的價值
#  policy = actor(state) #策略網路預測目前狀態下各動作的機率分佈
#  action = policy.sample() #根據策略網路輸出的機率分佈選擇動作
#  next_state, reward = environment.take_action(action) #執行動作，產生新狀態及回饋值
#  value_next = critic(next_state) #預測新狀態的價值
#  advantage = (reward + gamma * value_next) - value #優勢值函數
#  loss =-1 * policy.logprob(action) * advantage #根據動作的優勢值來強化（或弱化）該動作
#  minimize(loss) #想辦法最小化損失

import torch
from torch import nn
from torch import optim
import numpy as np
from torch.nn import functional as F
import gym
import torch.multiprocessing as mp
import matplotlib
import matplotlib.pyplot as plt

buffer = mp.Queue() #編註：小編在這裡創建了一個buffer，用來儲存每一場的遊戲長度（多少回合）

## CartPole-演员 评论家模型
class ActorCritic(nn.Module): #定義演員—評論家模型
  def __init__(self):
    super(ActorCritic, self).__init__()
    self.l1 = nn.Linear(4,25) #定義模型中各神經層的shape，參考圖5.10
    self.l2 = nn.Linear(25,50)
    self.actor_lin1 = nn.Linear(50,2)
    self.l3 = nn.Linear(50,25)
    self.critic_lin1 = nn.Linear(25,1)
  def forward(self,x):
    x = F.normalize(x,dim=0) #正規化輸入資料
    y = F.relu(self.l1(x))
    y = F.relu(self.l2(y))
    actor = F.log_softmax(self.actor_lin1(y),dim=0) #演員端輸出遊戲中兩種可能動作的對數化機率值
    c = F.relu(self.l3(y.detach())) #先將評論家段的節點分離，再經過ReLU的處理
    critic = torch.tanh(self.critic_lin1(c)) #評論家端輸出一個範圍在–1到+1之間的純量
    return actor, critic #使用tuple傳回演員和評論家的輸出結果

## 主要训练回圈
from IPython.display import clear_output
def worker(t, worker_model, counter, params):
  worker_env = gym.make("CartPole-v1")
  worker_env.reset()
  worker_opt = optim.Adam(lr=1e-4,params=worker_model.parameters()) #每條程序有獨立的運行環境和優化器，但共享模型參數
  worker_opt.zero_grad()
  for i in range(params['epochs']):
    worker_opt.zero_grad()
    values, logprobs, rewards, length = run_episode(worker_env,worker_model) #呼叫run_episode( )來執行一場遊戲並收集資料（譯註：該函式的定義見程式5.7）
    actor_loss,critic_loss,eplen = update_params(worker_opt,values,logprobs,rewards) #使用所收集的資料來更新神經網路參數（譯註：update_params( )的定義見程式5.8）
    counter.value = counter.value + 1 # counter是一個全域計數器，被所有程序共享
    if(i%10 == 0): #印出當前的訓練進度
      print(i)
      print(len(rewards))
      clear_output(wait=True)
    buffer.put(length) #將遊戲長度存進buffer中

## 执行一场游戏
def run_episode(worker_env, worker_model):
  state = torch.from_numpy(worker_env.env.state).float() #將環境狀態的資料型態從NumPy陣列轉換為PyTorch張量
  values, logprobs, rewards = [],[],[] #建立三個串列，分別用來儲存狀態價值（評論家）、對數化機率分佈（演員）、以及回饋值
  done = False
  j=0
  while (done == False): #除非滿足結束條件，否則遊戲繼續進行
    j+=1
    policy, value = worker_model(state) #計算狀態價值以及各種可能動作的對數化機率
    values.append(value)
    logits = policy.view(-1) #呼叫.view(-1) 將對數化機率轉成向量形式
    action_dist = torch.distributions.Categorical(logits=logits)
    action = action_dist.sample() #參考演員所提供的對數化機率來選擇動作
    logprob_ = policy.view(-1)[action]
    logprobs.append(logprob_)
    state_, _, done, info = worker_env.step(action.detach().numpy())
    state = torch.from_numpy(state_).float()
    if done: #{3}若某動作造成遊戲結束，則將回饋值設為–10，並且重置環境
      reward = -10
      worker_env.reset()
    else:
      reward = 1.0
    rewards.append(reward)
  return values, logprobs, rewards, len(rewards)

## 执行并最小化损失
def update_params(worker_opt,values,logprobs,rewards,clc=0.1,gamma=0.95):
  rewards = torch.Tensor(rewards).flip(dims=(0,)).view(-1) #{3}將rewards、logprobs及values陣列中的元素順序顛倒
  logprobs = torch.stack(logprobs).flip(dims=(0,)).view(-1)
  values = torch.stack(values).flip(dims=(0,)).view(-1)
  Returns = []
  ret_ = torch.Tensor([0])
  for r in range(rewards.shape[0]): #使用順序巔倒後的回饋值來計算每一步的回報，並將結果存入Returns陣列中
    ret_ = rewards[r] + gamma * ret_
    Returns.append(ret_)
  Returns = torch.stack(Returns).view(-1)
  Returns = F.normalize(Returns,dim=0) #將Returns陣列中的值做正規化處理
  actor_loss = -1*logprobs * (Returns - values.detach()) #將values張量的節點從運算圖中分離，並計算演員的損失，以避免其反向傳播到評論家端
  critic_loss = torch.pow(values - Returns,2) #計算評論家的損失
  loss = actor_loss.sum() + clc*critic_loss.sum() #將演員和評論家的損失加起來，變成總損失。注意，我們使用clc參數來降低評論家損失的影響
  loss.backward()
  worker_opt.step()
  return actor_loss, critic_loss, len(rewards)

if __name__ == '__main__':

    MasterNode = ActorCritic() #建立一個共享的全域演員—評論家實例
    MasterNode.share_memory() # share_memory()允許不同程序共用同一組模型參數（無須複製參數，節省空間）
    processes = [] #用來儲存不同程序實例的串列
    params = {
        'epochs':500, #進行500次訓練
        'n_workers':7, #設定程序數目為7
    }
    counter = mp.Value('i',0) #使用multiprocessing函式庫創建一個全域計數器，參數『i』代表其資料型態為整數

    for i in range(params['n_workers']):
      p = mp.Process(target=worker, args=(i,MasterNode,counter,params)) #啟動新的程序來運行worker函式（譯註：該函式的定義見程式5.6）
      p.start()
      processes.append(p)
    for p in processes: #{2}利用join讓每條程序皆完成運算後，再將結果傳回
      p.join()
    for p in processes: #{2}終止各程序
      p.terminate()

    #小編補充：原文沒有畫出平均遊戲長度的程式，因此小編加上了以下程式：
    n = params['n_workers']
    score = []
    running_mean = []
    total = torch.Tensor([0])
    mean = torch.Tensor([0])
    while not buffer.empty():
      score.append(buffer.get()) #將buffer中的資料存入score
    print(len(score))
    for i in range (params['epochs']):
      if (i>=50): #若訓練次數已超過50，則計算過去50場遊戲的平均長度
        total = total - sum(score[n*(i-50) : n*(i-50)+n])/n
        total = total + sum(score[n*i : n*i+n])/n
        mean = int(total/50)
      else: #若訓練次數未超過50次，則計算到目前為止的平均遊戲長度
        total = total + sum(score[n*i : n*i+n])/n
        mean = int(total/(i+1))
      running_mean.append(mean)
    plt.figure(figsize=(17,12))
    plt.ylabel("Mean Episode Length",fontsize=17)
    plt.xlabel("Training Epochs",fontsize=17)
    plt.plot(running_mean)
    print(counter.value, processes[0].exitcode) #列印全域計數器的值、以及第一個程序的退出碼（exit code，此值應為0）
    plt.show()