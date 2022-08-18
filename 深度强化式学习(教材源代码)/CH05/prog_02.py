import multiprocessing as mp
from multiprocess import queues
import numpy as np

def square(i, x, queue):
    print("In process {}".format(i, ))
    queue.put(np.square(x))  # 將輸出結果存進queue

if __name__ == '__main__':
    processes = []  # 建立用來儲存不同程序的串列
    queue = mp.Queue()  # 建立多程序處理的Queue；該資料結構可以被不同的程序共享
    x = np.arange(64)  # 生成一個數列做為目標數列（內含0～63的整數）
    for i in range(8):  # 開啟8條程序，並讓它們利用square函式分別處理目標數列中的一部份資料
        start_index = 8 * i
        proc = mp.Process(target=square, args=(i, x[start_index:start_index + 8], queue))
        proc.start()
        processes.append(proc)

    for proc in processes:  # {待所有程序皆執行完畢後，再將結果傳回主執行緒
        proc.join()

    for proc in processes:  # 終止各程序
        proc.terminate()

    results = []
    while not queue.empty():  # 將queue內的資料存進results串列，至到資料已清空
        results.append(queue.get())

    print(results)