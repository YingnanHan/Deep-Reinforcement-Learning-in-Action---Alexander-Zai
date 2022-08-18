
import numpy as np
from scipy import stats
import random
import matplotlib.pyplot as plt

# 這裡有個問題 就是中獎率和中獎金額是不一樣的，但是在這裡把中獎率以成正比的方式轉化成中獎金額
n = 10 # 設定拉霸機的數量
probs = np.random.rand(n) # 隨機設定不同的拉霸機中獎幾率(0-1之間)
eps = 0.2 # 設定ε為0.2
