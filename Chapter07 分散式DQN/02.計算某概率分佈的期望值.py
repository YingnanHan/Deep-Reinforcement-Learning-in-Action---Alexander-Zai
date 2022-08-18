import numpy as np

outcomes = np.array([18,21,17,17,21]) # 所有可能的結果
probs = np.array([0.6,0.1,0.1,0.1,0.1]) # 各個結果發生的幾率
expected_value = 0.0 # 初始化期望值為0
for i in range(probs.shape[0]):
    expected_value += probs[i]*outcomes[i] # 計算期望值
print(expected_value)

# 利用@运算符来进行内积运算
expected_value = probs@outcomes
print(expected_value)