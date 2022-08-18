import numpy as np

x = np.array([1,2,3,4,5,6])
p = np.array([0.1,0.1,0.1,0.1,0.2,0.4])

def expected_value(x,p):
    return x@p # 將這兩個陣列進行內積計算

print(expected_value(x,p))