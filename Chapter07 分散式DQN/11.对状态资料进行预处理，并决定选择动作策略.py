import torch
import numpy as np

def preproc_state(state): # 资料预处理函数
    p_state = torch.from_numpy(state).unsqueeze(dim=0).float()
    p_state = torch.nn.functional.normalize(p_state,dim=1) # 将状态中的数值正规化至0-1之间
    return p_state

def get_action(dist,support): # 动作选择策略函数
    actions = []
    for b in range(dist.shape[0]): # 以回圈的形式批次走访分布维度上的资料
        expectations = [support @ dist[b,a:] for a in range(dist.shape[1])] # 计算每个动作价值分布的期望值
        action = int(np.argmax(expectations))
        actions.append(action)
    actions = torch.Tensor(actions).int()
    return actions
