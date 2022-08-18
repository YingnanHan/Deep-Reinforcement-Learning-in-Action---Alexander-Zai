# Python program to illustrate
# the concept of race condition
# in multiproccesing
import multiprocessing

# function to withdraw from account
def withdraw(balance,lock):
    for _ in range(10000):
        lock.acquire()
        balance.value = balance.value - 1
        lock.release()

# function to deposite to account
def deposite(balance,lock):
    for _ in range(10000):
        lock.acquire()
        # initial balance (in shared memory)
        balance.value = balance.value + 1
        lock.release()

def perform_transactions():
    # initial balance in shared memory
    balance = multiprocessing.Value('i',100)

    # creating a lock object
    lock = multiprocessing.Lock()

    # creating new proccesses
    p1 = multiprocessing.Process(target=withdraw,args=(balance,lock))
    p2 = multiprocessing.Process(target=deposite,args=(balance,lock))

    # starting proccess
    p1.start()
    p2.start()

    # wait until proccess are finished
    p1.join()
    p2.join()

    # print final balance
    print("Finished balance = {}".format(balance.value))

if __name__ == '__main__':
    for _ in range(10):
        # perform same transaction proccess 10 times
        perform_transactions()