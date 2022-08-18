from random import shuffle
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import numpy as np

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