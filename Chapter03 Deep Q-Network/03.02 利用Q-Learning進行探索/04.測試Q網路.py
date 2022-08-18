import numpy as np
import torch
from Gridworld import Gridworld
import random
from matplotlib import pyplot as plt

# # 未整理部分
# L1 = 64  # 輸入層的寬度
# L2 = 150  # 第一隱藏層的寬度
# L3 = 100  # 第二隱藏層的寬度
# L4 = 4  # 輸出層的寬度
#
# model = torch.nn.Sequential(
#     torch.nn.Linear(L1, L2),  # 第一隐藏层的shape
#     torch.nn.ReLU(),
#     torch.nn.Linear(L2, L3),  # 第二隐藏层的shape
#     torch.nn.ReLU(),
#     torch.nn.Linear(L3, L4),  # 输出层的shape
# )
# loss_fn = torch.nn.MSELoss()  # 制定损失函数为MSE(均方误差)
# learning_rate = 1e-3  # 设定学习率
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # 指定優化器為Adam，其中model.parameters()會回傳所有要優化的權重參數
#
# gamma = 0.9  # 折扣因子
# epsilon = 1.0
#
# action_set = {
#     0: 'u',
#     1: 'd',
#     2: 'l',
#     3: 'r'
# }
#
# ## 主要訓練迴圈
# epochs = 1000
# epsilon = 1.0
# losses = [] # 使用串列將每一次的loss記錄下來，方便之後將loss的變化畫成趨勢圖
# for i in range(epochs):
#     game = Gridworld(size=4,mode='static') # 建立遊戲，設定方格邊長為4，物體初始位置模式為static
#     state_ = game.board.render_np().reshape(1,64) + np.random.rand(1,64)/10.0 # 將三階狀態陣列(4x4x4)轉換成向量(長度為64)，並將每個值都上一些隨機雜訊(很小的數值)
#     state1 = torch.from_numpy(state_).float() # 將Numpy陣列轉換成Pytorch張量，並存於state1中
#     status = 1 # 用來追蹤遊戲是否仍在繼續(1代表仍在繼續)
#     while (status == 1):
#         '''計算當前狀態的Q值'''
#         qval = model(state1) # 執行Q網路，取得所有動作的預測Q值
#         qval_ = qval.data.numpy() # 將qval轉化成Numpy陣列
#         # 依照ε-greedy策略選擇動作
#         if (random.random() < epsilon):
#             action_ = np.random.randint(0,4) # 隨機選擇一個動作(探索)
#         # 選擇Q值最大的動作 (利用)
#         else:
#             action_ = np.argmax(qval_)
#         action = action_set[action_] # 將代表某動作的數字對應到makeMove()的英文字母
#         game.makeMove(action) # 執行之前按照ε-greedy策略所做出選擇的動作
#         '''計算下一狀態的Q值'''
#         state2_ = game.board.render_np().reshape(1,64) + np.random.rand(1,64)/10.0
#         state2  = torch.from_numpy(state2_).float() # 執行動作完畢,取得遊戲的新狀態，並轉換成張量
#         reward = game.reward()
#         with torch.no_grad(): # 該程式的作用會在稍後說明
#             newQ = model(state2.reshape(1,64))
#         maxQ = torch.max(newQ) # 將新狀態下輸出的Q值向量中的最大Q值記錄下來
#
#         if reward == -1: # 計算訓練所用的目標Q值
#             Y = reward + (gamma*maxQ)
#         else:
#             # 若reward不等於-1，則代表遊戲已經結束，也就沒有下一個狀態了，因此目標Q值就等於回饋值
#             Y = reward
#         Y = torch.Tensor([Y]).detach() # 該程式的作用會在稍後說明
#         X = qval.squeeze()[action_] # 將演算法對執行的動作所預測的Q值存進X，並使用squeeze()將qval中維度為1的階去掉
#         loss = loss_fn(X,Y) # 計算目標Q值與預測Q值之間的誤差
#         # print(i,loss.item())
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         state1 = state2
#         # if abs(reward)==10:
#         #     status = 0 # 若reward的值已經為10，那麼則代表遊戲結束，所以設置status=0
#         if reward != -1:
#             status = 0 # 如果掉入陷阱或者遊戲結束，則本episode終止
#     losses.append(loss.item())
#     if epsilon > 0.1:
#         epsilon -= (1/epochs) # 讓ε的值隨著訓練的進行而慢慢下降直到0.1
#
# plt.figure(figsize=(10,7))
# plt.plot(losses)
# plt.xlabel("Epochs",fontsize=11)
# plt.ylabel("Loss",fontsize=11)
# plt.show()
#
# def test_model(model,mode="static",display=True):
#     i = 0
#     game = Gridworld(size=4,mode = mode) # 產生一場測試遊戲
#     state_ = game.board.render_np().reshape(1,64)+np.random.rand(1,64)/10.0
#     state = torch.from_numpy(state_).float()
#     if display:
#         print("Initial state:")
#         print(game.display())
#     status = 1
#     while(status == 1): # 如果遊戲仍在進行
#         qval = model(state)
#         qval_ = qval.data.numpy()
#         action_ = np.argmax(qval_)
#         action = action_set[action_]
#         if display:
#             print("Move #: %s; Taking action: %s" % (i,action))
#         game.makeMove(action)
#         state_ = game.board.render_np().reshape(1,64) + np.random.rand(1,64)/10.0
#         state = torch.from_numpy(state_).float()
#         if display:
#             print(game.display())
#         reward = game.reward()
#         if reward != -1: # 代表勝利(抵達終點)或落敗(掉入陷阱)
#             if reward > 0: # reward>0,代表成功抵達終點
#                 status = 2 # 將狀態設置為2，跳出迴圈
#                 if display:
#                     print("Game won! Reward is :%s"%(reward))
#             else: # reward≮0,代表落入陷阱
#                 status = 0 # 將狀態設置為2，跳出迴圈
#                 if display:
#                     print("Game LOST. Reward: %s"% reward)
#         i+=1 # 每移動一步，i就加1
#         if (i>15):
#             if display:
#                 print("Game lost .Too many moves!")
#             break # 若移動了15步仍未取得勝利，則一樣視為落敗
#     win = True if status==2 else False
#     return win
#
# for i in range(10):
#     test_model(model)
#     print("-------------------------")
#
# print("=========================")
# print("-------------------------")
# print("=========================")
#
# for i in range(10):
#     test_model(model,mode="random")
#     print("-------------------------")


def train():
    ## 主要訓練迴圈
    epochs = 1000
    epsilon = 1.0
    losses = [] # 使用串列將每一次的loss記錄下來，方便之後將loss的變化畫成趨勢圖
    for i in range(epochs):
        game = Gridworld(size=4,mode='static') # 建立遊戲，設定方格邊長為4，物體初始位置模式為static
        state_ = game.board.render_np().reshape(1,64) + np.random.rand(1,64)/10.0 # 將三階狀態陣列(4x4x4)轉換成向量(長度為64)，並將每個值都上一些隨機雜訊(很小的數值)
        state1 = torch.from_numpy(state_).float() # 將Numpy陣列轉換成Pytorch張量，並存於state1中
        status = 1 # 用來追蹤遊戲是否仍在繼續(1代表仍在繼續)
        while (status == 1):
            '''計算當前狀態的Q值'''
            qval = model(state1) # 執行Q網路，取得所有動作的預測Q值
            qval_ = qval.data.numpy() # 將qval轉化成Numpy陣列
            # 依照ε-greedy策略選擇動作
            if (random.random() < epsilon):
                action_ = np.random.randint(0,4) # 隨機選擇一個動作(探索)
            # 選擇Q值最大的動作 (利用)
            else:
                action_ = np.argmax(qval_)
            action = action_set[action_] # 將代表某動作的數字對應到makeMove()的英文字母
            game.makeMove(action) # 執行之前按照ε-greedy策略所做出選擇的動作
            '''計算下一狀態的Q值'''
            state2_ = game.board.render_np().reshape(1,64) + np.random.rand(1,64)/10.0
            state2  = torch.from_numpy(state2_).float() # 執行動作完畢,取得遊戲的新狀態，並轉換成張量
            reward = game.reward()
            with torch.no_grad(): # 該程式的作用會在稍後說明
                newQ = model(state2.reshape(1,64))
            maxQ = torch.max(newQ) # 將新狀態下輸出的Q值向量中的最大Q值記錄下來

            if reward == -1: # 計算訓練所用的目標Q值
                Y = reward + (gamma*maxQ)
            else:
                # 若reward不等於-1，則代表遊戲已經結束，也就沒有下一個狀態了，因此目標Q值就等於回饋值
                Y = reward
            Y = torch.Tensor([Y]).detach() # 該程式的作用會在稍後說明
            X = qval.squeeze()[action_] # 將演算法對執行的動作所預測的Q值存進X，並使用squeeze()將qval中維度為1的階去掉
            loss = loss_fn(X,Y) # 計算目標Q值與預測Q值之間的誤差
            # print(i,loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            state1 = state2
            # if abs(reward)==10:
            #     status = 0 # 若reward的值已經為10，那麼則代表遊戲結束，所以設置status=0
            if reward != -1:
                status = 0 # 如果掉入陷阱或者遊戲結束，則本episode終止
        losses.append(loss.item())
        if epsilon > 0.1:
            epsilon -= (1/epochs) # 讓ε的值隨著訓練的進行而慢慢下降直到0.1

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
    '''標準情況'''
    for i in range(10):
        test_model(model)
        print("-------------------------")
    '''隨機情況'''
    for i in range(10):
        test_model(model,mode="random")
        print("-------------------------")

if __name__ == '__main__':

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
    epsilon = 1.0

    action_set = {
        0: 'u',
        1: 'd',
        2: 'l',
        3: 'r'
    }

    train()
    test()