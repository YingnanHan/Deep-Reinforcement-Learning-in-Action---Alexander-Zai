import multiprocessing
import os
import numpy as np

def square(n):
    print("Worker proccess id for {0} : {1}".format(n,os.getpid()))
    return n*n

if __name__ == '__main__':
    mylist = np.array([1,2,3,4,5])

    # creating a pool object
    p = multiprocessing.Pool(2) # 這裡的參數用於指定進程分配的數量

    # map list to target function
    result = p.map(square,mylist)

    print(result)