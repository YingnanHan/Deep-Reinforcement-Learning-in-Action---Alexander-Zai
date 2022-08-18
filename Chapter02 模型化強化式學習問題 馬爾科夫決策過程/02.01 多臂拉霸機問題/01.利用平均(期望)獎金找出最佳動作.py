
def get_best_action(actions,history):
    '''這個函式可以用NumPy的argmax()來取代，請參考程式2.5'''
    best_action = 0
    max_action_value = 0
    for i in range(len(actions)):
        cur_action_value = exp_reward(actions[i],history) # history及exp_reward()的定義可以參考表2.1
        if cur_action_value > max_action_value:
            best_action = i # 若cur_action_value比較大，即更新索引best_action的值
            max_action_value = cur_action_value
    return best_action
