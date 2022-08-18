def loss_fn(q_loss,inverse_loss,forward_loss):
    loss_ = (1-params['beta'])*inverse_loss
    loss_ += params['beta']*forward_loss
    loss_ = loss_.sum() / loss_.flatten().shape[0]
    loss = loss_ + params['lambda'] * q_loss
    return loss

def reset_env():
    env.reset()
    state1 = prepare_initial_state(env.render('rgb_array'))
    return state1

# 計算ICM的預測誤差
def ICM(state1,action,state2,forward_scale=1.,inverse_scale=1e4):
    state1_hat = encoder(state1) # 使用編碼器將狀態1和2編碼
    state2_hat = encoder(state2)
    state2_hat_pred = forward_model(state1_hat.detach(),action.detach()) # 利用正向模型預測新的狀態
    forward_pred_err = forward_scale*forward_loss(state2_hat_pred,state2_hat.detach()).sum(dim=1).unsqueeze(dim=1)
    pred_action = inverse_model(state1_hat,state2_hat) # 反向模型傳回個動作的幾率分佈
    inverse_pred_err = inverse_scale*inverse_loss(pred_action,action.detach().flatten()).unsqueeze(dim=1)
    return forward_pred_err,inverse_pred_err