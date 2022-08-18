# import multiprocessing module
import multiprocessing
import os

def worker1():
    # printing process id
    print("ID of process running worker1:{}".format(os.getpid()))

def worker2():
    # printing process id
    print("ID of process running worker1:{}".format(os.getpid()))

if __name__ == '__main__':
    # printing main program proccess id
    print("ID of main proccess:{}".format(os.getpid()))

    # creating proccesses
    p1 = multiprocessing.Process(target=worker1)
    p2 = multiprocessing.Process(target=worker2)

    # starting proccesses
    p1.start()
    p2.start()

    # process IDs
    print("ID of proccess p1:{}".format(p1.pid))
    print("ID of proccess p2:{}".format(p2.pid))

    # wait until proccess are finished
    p1.join()
    p2.join()

    # both procceses finished
    print("Both proccesses finished executing!")

    # checking if progresses are alive
    print("Progress p1 is alive?:{}".format(p1.is_alive()))
    print("Progress p1 is alive?:{}".format(p1.is_alive()))
    