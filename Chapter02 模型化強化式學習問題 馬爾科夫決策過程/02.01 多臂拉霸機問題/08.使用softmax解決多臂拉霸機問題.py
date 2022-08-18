
import numpy as np
from scipy import stats
import random
import matplotlib.pyplot as plt


def get_reward(prob): # prob為某台拉霸機的中獎率，注意prob和之前的probs不同，probs是所有拉霸機的中獎概率構成的陣列
    reward = 0
    for i in range(10):
        if random.random() < prob: # 注意：因為random會產生均勻分佈的亂數，所以在10次迴圈中產生的亂數值小於prob的次數正比於prob的大小
            reward += 1 # 若隨機產生的數字小於中獎概率，就把reward+1
    return reward # 傳回reward(存有本次遊戲中開出的獎金)

def update_record(record,action,r):
    r_ave = (record[action,0]*record[action,1]+r)/(record[action,0]+1) # 算出新的平均值
    record[action,0] += 1 # action號機台的拉桿次數加1
    record[action,1] = r_ave # 更新該機台的平均獎金
    return record

def exp_reward(a,history):
    rewards_for_a = history[a]
    return np.sum(rewards_for_a)/len(rewards_for_a)

def get_best_arm(record):
    arm_index = np.argmax(record[:,1]) # 找出record第一行的元素中值最大的元素索引
    return arm_index

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

def softmax(av,tau=0.83): # av即動作的價值陣列，tau即溫度參數(預設值1.12)
    softm = (np.exp(av/tau))/np.sum(np.exp(av/tau))

if __name__ == '__main__':
    # 這裡有個問題 就是中獎率和中獎金額是不一樣的，但是在這裡把中獎率以成正比的方式轉化成中獎金額
    n = 10  # 設定拉霸機的數量
    probs = np.random.rand(n)  # 隨機設定不同的拉霸機中獎幾率(0-1之間)
    eps = 0.2  # 設定ε為0.2
    '''解決多臂拉霸機問題'''
    fig,ax = plt.subplots(1,1)
    ax.set_xlabel("Plays")
    ax.set_ylabel("Average Rewards")
    fig.set_size_inches(9,5)
    record = np.zeros((n,2)) # 先產生一個初始值全為0的record陣列
    probs = np.random.rand(n) # 隨機設定每台拉霸機的中獎率
    rewards = [0] # 記錄每次拉桿後，計算出的總平均獎金
    for i in range(500):
        p = softmax(record[:,1],tau=0.7) # record[:,1]存儲了各拉霸機的價值(即平均獎金) 根據各個動作的價值計算出相應的幾率值
        choice = np.random.choice(np.arange(n),p=p) # 根據p陣列中的幾率分佈來隨機選擇一個動作
        r = get_reward(probs[choice]) # 取得此次的獎金
        record = update_record(record,choice,r) # 更新record陣列中與該拉霸機號碼對應的遊戲次數的平均獎金
        mean_reward = ((i+1)*rewards[-1]+r)/(i+2) # 計算最新的總體平均獎金
        rewards.append(mean_reward) # 記錄到rewards串列
    ax.scatter(np.arange(len(rewards)),rewards) # 畫出散佈圖
    plt.show()