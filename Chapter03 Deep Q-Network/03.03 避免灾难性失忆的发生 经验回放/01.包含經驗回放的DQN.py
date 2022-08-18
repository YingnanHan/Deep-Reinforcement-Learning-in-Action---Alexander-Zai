import numpy as np
import torch
from Gridworld import Gridworld
import random
from matplotlib import pyplot as plt
from collections import deque

L1 = 64  # 輸入層的寬度
L2 = 150  # 第一隱藏層的寬度
L3 = 100  # 第二隱藏層的寬度
L4 = 4  # 輸出層的寬度

model = torch.nn.Sequential(
    torch.nn.Linear(L1, L2),  # 第一隐藏层的shape
    torch.nn.ReLU(),
    torch.nn.Linear(L2, L3),  # 第二隐藏层的shape
    torch.nn.ReLU(),
    torch.nn.Linear(L3, L4),  # 输出层的shape
)
loss_fn = torch.nn.MSELoss()  # 制定损失函数为MSE(均方误差)
learning_rate = 1e-3  # 设定学习率
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # 指定優化器為Adam，其中model.parameters()會回傳所有要優化的權重參數

gamma = 0.9  # 折扣因子
epsilon = 0.3

action_set = {
    0: 'u',
    1: 'd',
    2: 'l',
    3: 'r'
}
epochs = 5000 # 訓練5000次
losses = []
mem_size = 1000 # 設定記憶串列的大小
batch_size = 200 # 設定單一小批次(mini-batch)的大小
replay = deque(maxlen=mem_size) # 產生一個記憶串列(資料型別的deque)來存儲經驗回放的資料，將其命名為replay
max_moves = 50 # 設定每場遊戲最多可以走幾步
for i in range(epochs):
    game = Gridworld(size=4,mode="random")
    state1_ = game.board.render_np().reshape(1,64)+np.random.rand(1,64).reshape(1,64)
    state1 = torch.from_numpy(state1_).float()
    status = 1
    mov = 0 # 記錄移動的步數，初始化為0
    while(status == 1):
        mov += 1
        qval = model(state1) # 輸出各個動作的Q值
        qval_ = qval.data.numpy()
        if (random.random()<epsilon):
            action_ = np.random.randint(0,4)
        else:
            action_ = np.argmax(qval_)
        action = action_set[action_]
        game.makeMove(action)

        state2_ = game.board.render_np().reshape(1,64)+np.random.rand(1,64)/100.0
        state2 = torch.from_numpy(state2_).float()
        reward = game.reward()

        done = True if reward > 0 else False # 在rewards不等於-1時設定done=True，代表遊戲已經結束了(分出勝負時，reward會等於10或者-10)
        exp = (state1,action_,reward,state2,done) # 產生一筆經驗，其中包含當前的狀態，動作，新狀態，回饋值以及done值
        replay.append(exp) # 將該經驗加入名為replay中

        state1 = state2 # 產生的新狀態會變成下一次訓練的輸入狀態
        if len(replay) > batch_size: # 當replay長度的大小大於批次量(mini-batch_size),啟動小批次訓練
            minibatch = random.sample(replay,batch_size) # 隨機選擇replay中的資料來組成子集，將經驗中的不同元素分別存儲到對應的小批次張量中

            state1_batch = torch.cat([s1 for (s1,a,r,s2,d) in minibatch])
            action_batch = torch.Tensor([a for (s1,a,r,s2,d) in minibatch])
            reward_batch = torch.Tensor([r for (s1,a,r,s2,d) in minibatch])
            state2_batch = torch.cat(([s2 for (s1,a,r,s2,d) in minibatch]))
            done_batch = torch.Tensor([d for (s1,a,r,s2,d) in minibatch])

            Q1 = model(state1_batch) # 利用小批次資料中的目前狀態批次來計算Q值
            with torch.no_grad():
                Q2 = model(state2_batch) # 利用小批次資料中的新狀態批次來計算Q值，但設定為不需要計算梯度
            Y = reward_batch + gamma*((1-done_batch)*torch.max(Q2,dim=1)[0]) # 計算我們希望DQN學習的目標Q值
            X = Q1.gather(dim=1,index=action_batch.long().unsqueeze(dim=1)).squeeze()
            loss = loss_fn(X,Y.detach())
            print(i,loss.item())
            optimizer.zero_grad()
            loss.backward()
            losses.append(loss.item())
            optimizer.step()
        if abs(reward) == 10 or mov > max_moves:
            status = 0
            mov = 0 # 若遊戲結束 則重設status的值和mov變數的值

    if epsilon > 0.1:
        epsilon -= (1/epsilon) # 讓ε的值隨著訓練的進行而慢慢下降，直到0.1(還是要保留探索的動作)
losses = np.array(losses)
plt.figure(figsize=(10,7))
plt.plot(losses)
plt.xlabel("Epochs",fontsize=11)
plt.ylabel("Loss",fontsize=11)
plt.show()