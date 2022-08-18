
def minibatch_train(use_extrinsic=True):
    state1_batch,action_batch,reward_batch,state2_batch = replay.get_batch()
    action_batch = action_batch.view(action_batch.shape[0],1) # 在次張量中增加一個維度，以符合模型的需求
    reward_batch = reward_batch.view(reward_batch.shape[0],1)
    forward_pred_err,inverse_pred_err = ICM(state1_batch,action_batch,state2_batch) # 運行ICM
    i_reward = (1./params["eta"])*forward_pred_err # 使用eta參數來調整預測誤差的權重
    reward = i_reward.detach() # 把i_reward張量從運算圖中分離，並開始計算總回饋值
    if use_extrinsic: # 決定演算法是否要使用外在回饋值
        reward += reward_batch
    qvals = Qmodel(state2_batch) # 計算新狀態的動作價值
    reward += params["gamma"] * torch.max(qvals)
    reward_pred = Qmodel(state1_batch)
    reward_target = reward_pred.clone()
    indices = torch.stack((torch.arange(action_batch.shape[0]),action_batch.squeeze()),dim=0) # action_batch是由整個數組組成的張量，且每一個整數代表一個動作，這裡將他們轉換成有多個one-hot編碼向量組成的張量
    indices = indices.tolist()
    reward_target[indices] = reward.squeeze()
    q_loss = 1e5*q_loss(F.normalize(reward_pred),F.normalize(reward_target.detach()))
    return forward_pred_err,inverse_pred_err,q_loss