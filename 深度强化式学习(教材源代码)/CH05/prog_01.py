import multiprocessing as mp
from multiprocess import queues
import numpy as np

def square(x): #將陣列輸入此函數後，該函數會將陣列中的數字分別進行平方
  return np.square(x)

if __name__ == '__main__':
    # 1
    x = np.arange(64) #生成內有數字序列的陣列
    print(x) #印出陣列中的數字
    # 2
    print(mp.cpu_count()) #輸出CPU數量，結果因電腦而異
    # 3
    pool = mp.Pool(8) #建立內含8個程序的多程序池（processor pool)
    squared = pool.map(square, [x[8*i:8*i+8] for i in range(8)]) #使用多程序池的.map()，對陣列中的每個數字呼叫square()，並將結果存入串列中傳回
    print(squared) #印出結果串列
