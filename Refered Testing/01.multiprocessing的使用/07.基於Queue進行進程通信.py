from multiprocessing import Process,Queue
import os , time , random

# 寫數據執行緒執行的代碼
def write(q):
    print("Process to write :%s" % os.getpid())
    for value in ['A','B','C']:
        print("Put %s to queue...")
        q.put(value)
        time.sleep(random.random())

# 讀數據執行緒執行的代碼
def read(q):
    print("Process to read :%s" % os.getpid())
    while True:
        value = q.get(True)
        print('Get %s from queue.' % value)

if __name__ == '__main__':
    # 父執行緒創建Queue
    q = Queue()
    pw = Process(target=write,args=(q,))
    pr = Process(target=read,args=(q,))
    # 啟動子執行緒pw 寫入
    pw.start()
    # 啟動子執行緒pr 讀取
    pr.start()
    # 等待pw結束
    pw.join()
    pr.terminate()