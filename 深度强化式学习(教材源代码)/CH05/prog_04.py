import torch
from torch import nn
from torch import optim
import numpy as np
from torch.nn import functional as F
import gym
import torch.multiprocessing as mp
import matplotlib
import matplotlib.pyplot as plt

class ActorCritic(nn.Module): #定義演員—評論家模型
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.l1 = nn.Linear(4,25)
        self.l2 = nn.Linear(25,50)
        self.actor_lin1 = nn.Linear(50,2)
        self.l3 = nn.Linear(50,25)
        self.critic_lin1 = nn.Linear(25,1)
    def forward(self,x):
        x = F.normalize(x,dim=0)
        y = F.relu(self.l1(x))
        y = F.relu(self.l2(y))
        actor = F.log_softmax(self.actor_lin1(y),dim=0) #模型的演員端輸出遊戲中兩種可能動作的對數化機率值
        c = F.relu(self.l3(y.detach()))
        critic = torch.tanh(self.critic_lin1(c)) #模型的評論家端輸出一個範圍在–1到+1之間的純量
        return actor, critic #使用tuple資料型態傳回演員和評論家的輸出結果

def run_episode(worker_env, worker_model, N_steps=100):
    raw_state = np.array(worker_env.env.state)
    state = torch.from_numpy(raw_state).float()
    values, logprobs, rewards = [],[],[]
    done = False
    j=0
    check = 1
    G=torch.Tensor([0]) #變數G代表回報，它的初始值為0
    while (j < N_steps and done == False): #持續進行遊戲，直到執行了N個動作、或者遊戲結束
        j+=1
        policy, value = worker_model(state)
        values.append(value)
        logits = policy.view(-1)
        action_dist = torch.distributions.Categorical(logits=logits)
        action = action_dist.sample()
        logprob_ = policy.view(-1)[action]
        logprobs.append(logprob_)
        state_, _, done, info = worker_env.step(action.detach().numpy())
        state = torch.from_numpy(state_).float()
        if done:
            reward = -10
            worker_env.reset()
            check = 1
        else: #{3}若遊戲並未結束，令回報等於最新的狀態價值
            reward = 1.0
            G = value.detach()
            check = 0
        rewards.append(reward)
    return values, logprobs, rewards, G, check

def update_params(worker_opt,values,logprobs,rewards,G,clc=0.1,gamma=0.95):
        rewards = torch.Tensor(rewards).flip(dims=(0,)).view(-1) #A
        logprobs = torch.stack(logprobs).flip(dims=(0,)).view(-1)
        values = torch.stack(values).flip(dims=(0,)).view(-1)
        Returns = []
        ret_ = G
        for r in range(rewards.shape[0]): #B
            ret_ = rewards[r] + gamma * ret_
            Returns.append(ret_)
        Returns = torch.stack(Returns).view(-1)
        Returns = F.normalize(Returns,dim=0)
        actor_loss = -1*logprobs * (Returns - values.detach()) #C
        critic_loss = torch.pow(values - Returns,2) #D
        loss = actor_loss.sum() + clc*critic_loss.sum() #E
        loss.backward()
        worker_opt.step()
        return actor_loss, critic_loss, len(rewards)

# def worker(t, worker_model, counter, params, buffer):
def worker(t, worker_model, counter, params):
    worker_env = gym.make("CartPole-v1")
    worker_env.reset()
    worker_opt = optim.Adam(lr=1e-4,params=worker_model.parameters()) #A
    worker_opt.zero_grad()
    tot_rew = torch.Tensor([0])
    for i in range(params['epochs']):
        worker_opt.zero_grad()
        values, logprobs, rewards, G, check = run_episode(worker_env,worker_model) #B
        actor_loss,critic_loss,tot_rew = update_params(worker_opt,values,logprobs,rewards,G) #C
        while(check == 0):
          worker_opt.zero_grad()
          values, logprobs, rewards, G, check = run_episode(worker_env,worker_model) #B
          actor_loss,critic_loss,eplen = update_params(worker_opt,values,logprobs,rewards,G) #C
          tot_rew += eplen
        counter.value = counter.value + 1 #D
        #print("Training ", i,": ",len(rewards))
        if(i%10 == 0):
          print(i,end="\t")
          print(tot_rew)
          #clear_output(wait=True)
        # buffer.put(tot_rew)

if __name__ == '__main__':

    MasterNode = ActorCritic() #A
    MasterNode.share_memory() #B
    processes = [] #C
    params = {
        'epochs':500,
        'n_workers':7,
    }
    # buffer = mp.Queue()
    counter = mp.Value('i',0) #D
    for i in range(params['n_workers']):
        # p = mp.Process(target=worker, args=(i, MasterNode, counter, params, buffer))  # E
        p = mp.Process(target=worker, args=(i,MasterNode,counter,params)) #E
        p.start()
        processes.append(p)
    for p in processes: #F
        p.join()
    for p in processes: #G
        p.terminate()
    # ###小編補充：
    # n = params['n_workers']
    # score = []
    # running_mean = []
    # total = torch.Tensor([0])
    # mean = torch.Tensor([0])
    # while not buffer.empty():
    #     score.append(buffer.get()) #將buffer中的資料存入score
    # print(len(score))
    # for i in range (params['epochs']):
    #   #print("Epochs ", i, ": ", sum(score[n*i : n*i+n])/n)
    #   if (i>=50): #{4}若訓練次數已超過50，則計算過去50場遊戲的平均長度
    #     total = total - sum(score[n*(i-50) : n*(i-50)+n])/n
    #     total = total + sum(score[n*i : n*i+n])/n
    #     mean = int(total/50)
    #   else: #{3}若訓練次數未超過50次，則計算到目前為止的遊戲長度
    #     total = total + sum(score[n*i : n*i+n])/n
    #     mean = int(total/(i+1))
    #   #mean = sum(score[n*i : n*i+n])/n
    #   running_mean.append(mean)
    # plt.figure(figsize=(20,12))
    # plt.ylabel("Mean Episode Length",fontsize=15)
    # plt.xlabel("Training Epochs",fontsize=15)
    # plt.plot(running_mean)
    # #
    print(counter.value,processes[0].exitcode)