import torch
import numpy as np

def get_target_dist(dist_batch,action_batch,reward_batch,support,lim=(-10,10),gamma=0.8):
    nsup =support.shape[0]
    vmin,vmax = lim[0],lim[1]
    dz = (vmax-vmin)/(nsup-1)
    target_dist_batch = dist_batch.clone() # 建立一个目标分布
    for i in range(dist_batch.shape[0]): # 使用循环走访整个分布
        dist_full = dist_batch[i]
        action = int(action_batch[i].item())
        dist = dist_full[action]
        r = reward_batch[i]
        if r != -1 : # 如果回馈值不是-1，代表已经到达最终状态，目标分布为退化分布(所有几率值集中在回馈值的位置)
            target_dist = torch.zeros(nsup)
            bj = np.round((r-vmin)/dz)
            bj = int(np.clip(bj,0,nsup-1))
            target_dist[bj] = 1
        else:
            # 若目前状态为非最终状态，则根据回馈值，按照贝叶斯方法来更新先验分布
            target_dist = update_dist(r,support,dist,lim=lim,gamma=gamma)
        target_dist_batch[i,action,:] = target_dist # 变更与执行动作相关的分布
    return target_dist_batch