# 再多執行緒當中，所有的執行緒都會有這兩個特征 獨立運行 有自己的內存空間

import multiprocessing

# empty list with global scope
result = []

def square_list(mylist):
    global result
    # append squares of my list to global list result
    for num in mylist:
        result.append(num*num)
    # print global list result
    print("Result(in procces p1):{}".format(result))

if __name__ == '__main__':
    # input list
    mylist = [1,2,3,4]

    # creating new process
    p1 = multiprocessing.Process(target=square_list,args=(mylist,))
    # starting process
    p1.start()
    # wait until proccess is finished
    p1.join()

    # print global result list
    print("Result(in main program): {}".format(result))