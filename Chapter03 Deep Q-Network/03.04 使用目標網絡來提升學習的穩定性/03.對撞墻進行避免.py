import torch
import copy
from collections import deque
from Gridworld import Gridworld
import numpy as np
import random
import matplotlib.pyplot as plt

L1 = 64
L2 = 150
L3 = 100
L4 = 4

model = torch.nn.Sequential(
    torch.nn.Linear(L1,L2),
    torch.nn.ReLU(),
    torch.nn.Linear(L2,L3),
    torch.nn.ReLU(),
    torch.nn.Linear(L3,L4)
)

model2 = copy.deepcopy(model) # 完整複製主要Q網絡的架構，產生目標網路
model2.load_state_dict(model.state_dict()) # 將主要Q網絡的參數賦值給目標網絡
loss_fn = torch.nn.MSELoss()
learning_rate = 1e-3
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)

gamma = 0.9
epsilon = 1.0

epochs = 5000
losses = []
mem_size = 1000 # 設定記憶串列的大小
batch_size = 200 # 設定批次的大小
move_pos = [(-1,0),(1,0),(0,-1),(0,1)] # 移動方向 u l r d 的實際移動向量
replay = deque(maxlen=mem_size)
max_moves = 50
sync_freq = 500 # 設定Q網路和目標網絡的參數同步頻率(每500steps就同步一次參數)
j = 0 # 記錄當前訓練次數
action_set = {
    0: 'u',
    1: 'd',
    2: 'l',
    3: 'r'
}

for i in range(epochs):
    game = Gridworld(size=4,mode="random")
    state1_ = game.board.render_np().reshape(1,64) + np.random.rand(1,64)/100.0
    state1 = torch.from_numpy(state1_).float()
    status = 1
    mov = 0
    while(status == 1):
        j += 1 # 將訓練次數加一
        mov += 1
        qval = model(state1)
        qval_ = qval.data.numpy()
        if (random.random() < epsilon):
            action_ = np.random.randint(0,4)
        else:
            action_ = np.argmax(qval_)

        hit_wall = game.validateMove("Player",move_pos[action_]) == 1 # 若有撞墻的動作，hit_wall就設置為1

        action = action_set[action_]
        game.makeMove(action)

        state2_ = game.board.render_np().reshape(1,64) + np.random.rand(1,64)/100.0
        state2 = torch.from_numpy(state2_).float()
        reward = game.reward()

        # 針對撞墻情況作出處理
        if hit_wall == True:
            reward -= 5

        done = True if reward > 0 else False
        exp = (state1,action_,reward,state2,done)
        replay.append(exp)
        state1 = state2

        if len(replay) > batch_size:
            mini_batch = random.sample(replay,batch_size)

            state1_batch = torch.cat([s1 for (s1,a,r,s2,d) in mini_batch])
            action_batch = torch.Tensor([a for (s1,a,r,s2,d) in mini_batch])
            reward_batch = torch.Tensor([r for (s1,a,r,s2,d) in mini_batch])
            state2_batch = torch.cat([s2 for (s1,a,r,s2,d) in mini_batch])
            done_batch = torch.Tensor([d for (s1,a,r,s2,d) in mini_batch])

            Q1 = model(state1_batch)
            with torch.no_grad(): # 用目標網絡計算目標Q值，但不要優化計算模型的參數
                Q2 = model2(state2_batch)
            Y = reward_batch + gamma*((1-done_batch)*torch.max(Q2,dim=1)[0])
            X = Q1.gather(dim=1,index=action_batch.long().unsqueeze(dim=1)).squeeze()

            loss = loss_fn(X,Y.detach())
            # print(i,"\t",loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            if j % sync_freq == 0:
                model2.load_state_dict(model.state_dict()) # 每500步，就將Q網路當前的參數複製一份給目標網路
        if reward != -1 or mov > max_moves:
            status = 0
            mov = 0
    if epsilon > 0.1:
        epsilon -= (1/epochs) # 讓ε的值隨著訓練的進行而慢慢下降，直到0.1(還是要保留探索的動作)
losses = np.array(losses)

plt.figure(figsize=(10,7))
plt.plot(losses)
plt.xlabel("Epochs",fontsize=11)
plt.ylabel("Loss",fontsize=11)
plt.show()

def test_model(model,mode="static",display=True):
    i = 0
    game = Gridworld(size=4,mode = mode) # 產生一場測試遊戲
    state_ = game.board.render_np().reshape(1,64)+np.random.rand(1,64)/10.0
    state = torch.from_numpy(state_).float()
    if display:
        print("Initial state:")
        print(game.display())
    status = 1
    while(status == 1): # 如果遊戲仍在進行
        qval = model(state)
        qval_ = qval.data.numpy()
        action_ = np.argmax(qval_)
        action = action_set[action_]
        if display:
            print("Move #: %s; Taking action: %s" % (i,action))
        game.makeMove(action)
        state_ = game.board.render_np().reshape(1,64) + np.random.rand(1,64)/10.0
        state = torch.from_numpy(state_).float()
        if display:
            print(game.display())
        reward = game.reward()
        if reward != -1: # 代表勝利(抵達終點)或落敗(掉入陷阱)
            if reward > 0: # reward>0,代表成功抵達終點
                status = 2 # 將狀態設置為2，跳出迴圈
                if display:
                    print("Game won! Reward is :%s"%(reward))
            else: # reward≮0,代表落入陷阱
                status = 0 # 將狀態設置為2，跳出迴圈
                if display:
                    print("Game LOST. Reward: %s"% reward)
        i+=1 # 每移動一步，i就加1
        if (i>15):
            if display:
                print("Game lost .Too many moves!")
            break # 若移動了15步仍未取得勝利，則一樣視為落敗
    win = True if status==2 else False
    return win

def test():
    count = 0
    for i in range(1000):
        count += test_model(model,mode="random")
    print("最終勝率為:",count/1000)

test()   # 0.92