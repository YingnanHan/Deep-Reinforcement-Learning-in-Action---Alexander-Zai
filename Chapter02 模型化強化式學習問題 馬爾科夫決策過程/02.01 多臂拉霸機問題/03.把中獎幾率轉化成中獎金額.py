import random
import numpy as np

def get_reward(prob): # prob為某台拉霸機的中獎率，注意prob和之前的probs不同，probs是所有拉霸機的中獎概率構成的陣列
    reward = 0
    for i in range(10):
        if random.random() < prob: # 注意：因為random會產生均勻分佈的亂數，所以在10次迴圈中產生的亂數值小於prob的次數正比於prob的大小
            reward += 1 # 若隨機產生的數字小於中獎概率，就把reward+1
    return reward # 傳回reward(存有本次遊戲中開出的獎金)

print(np.mean([get_reward(0.7) for _ in range(2000)])) # 執行2000次get_reward(),並取結果的平均值
