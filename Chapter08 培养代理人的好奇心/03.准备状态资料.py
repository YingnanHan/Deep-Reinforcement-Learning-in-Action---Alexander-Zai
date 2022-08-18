import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from collections import deque

def downscale_obs(obs,new_size=(42,42),to_gray=True):
    # 原始状态资料  调整后的shape 转换成灰阶
    if to_gray:
        return resize(obs,new_size,anti_aliasing=True).max(axis=2) # 为了将图片转换成灰阶，我们取出obs的第二阶颜色通道中的最大值
    else:
        return resize(obs,new_size,anti_aliasing=True)

def prepare_state(state):
    # 将Numpy阵列转换成PyTorch张量 在第0阶加入一个批次阶(batch)
    return torch.from_numpy(downscale_obs(state,to_gray=True)).float().unsqueeze()

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
