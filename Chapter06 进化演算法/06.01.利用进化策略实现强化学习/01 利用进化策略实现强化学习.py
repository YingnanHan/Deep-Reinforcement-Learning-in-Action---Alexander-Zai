# 程式 6.1 生成隨機字串
import random
import numpy as np
from matplotlib import pyplot as plt
from difflib import SequenceMatcher

class Individual: # 建立類別來存儲族群中每一個個體的資訊

    def __init__(self,string,fitness = 0):
        self.string = string
        self.fitness = fitness

def spawn_population(length=26,size=100): # 兩個參數分別表示要生成字串的長度 族群中字串(個體的數量) ; 生成初始族群中的隨機字串
    pop = []
    for i in range(size):
        string = ''.join(random.choices(alphabet,k=length)) # 將選出的字元拼在一起
        individual = Individual(string)
        pop.append(individual)
    return pop

def similar(a,b): # 計算兩個字符串的相似度，並回傳適應度分數
    return SequenceMatcher(None,a,b).ratio() # 傳回值介於0~1 1代表完全符合 0代表完全不符合

def recombine(p1_,p2_): # 將兩個親代字串重組，並產生兩個後代字串
    p1 = p1_.string
    p2 = p2_.string
    child1 = []
    child2 = []
    cross_pt = random.randint(0,len(p1)) # 隨機設定重組的切割位置，會切成兩段以進行重組
    child1.extend(p1[0:cross_pt]) # 重組兩個親代字串
    child1.extend(p2[cross_pt:])
    child2.extend(p2[0:cross_pt])
    child2.extend(p1[cross_pt:])
    c1 = Individual(''.join(child1))
    c2 = Individual(''.join(child2))
    return c1,c2

def mutate(x,mut_rate=0.01): # 透過隨機改變字串中的元素來達到突變的目的
    # mut_rate: 突變率
    new_x_ = []
    for char in x.string:
        if random.random() < mut_rate:
            new_x_.extend(random.choices(alphabet,k=1)) # 從之前的字元清單中隨機選出一個字元替換原有字元
        else:
            new_x_.append(char) # 保留原有字元
    new_x = Individual(''.join(new_x_))
    return new_x

def evaluate_population(pop,target): # 計算族群中每一個個體的適應度分數
    avg_fit = 0 # 用來存儲族群內個體的平均適應度
    for i in range(len(pop)):
        fit = similar(pop[i].string,target) # 利用程式6.1中的similar()函式計算適應度
        pop[i].fitness = fit
        avg_fit += fit
    avg_fit /= len(pop) # 计算整个族群的平均适应度
    return pop,avg_fit # 返回整個族群的種群本身以及平均適應度

def next_generation(pop,size=100,length=26,mut_rate=0.01): # size為初始種群內個體數 ；透過重組和突變來產生新一代字串族群
    new_pop = []
    while(len(new_pop))<size: # 當新一代的個體數少於族群內的個體數時
        parents = random.choices(pop,k=2,weights=[x.fitness for x in pop]) # 根據適應度隨機選出2個親代個體
        offspring_ = recombine(parents[0],parents[1]) # 利用程式6.2中的recombione()函式進行重組
        child1 = mutate(offspring_[0],mut_rate=mut_rate)
        child2 = mutate(offspring_[1],mut_rate=mut_rate)
        offspring =[child1,child2]
        new_pop.extend(offspring)
    return new_pop


if __name__ == '__main__':
    alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ,.! "  # 我們用來組成字串的字元清單
    target = "Hello World!"  # 目標子串

    # 完整的進化過程
    num_generations = 100 # 總世代數
    population_size = 3000 # 族群大小
    str_len = len(target) # 取得目標子串的長度
    mutation_rate = 0.001 # 將突變率設置為 0.1%

    pop_fit = [] # 建立一個存儲族群適應度的串列
    pop = spawn_population(str_len,population_size) # 建立初始族群
    done = False # 用來記錄是否以繁衍出目標子串，若已生成則為True

    for gen in range(num_generations):
        print(gen)
        pop,avg_fit = evaluate_population(pop,target)
        pop_fit.append(avg_fit) # 将训练过程中，每个世代的族群的平均适应值记录下来
        new_pop = next_generation(pop,size=population_size,length=str_len,mut_rate=mutation_rate) # 产生新时代的族群
        pop = new_pop
        for x in pop:
            if x.string == target:
                print("Target found!")
                done = True
        if done: # 只要繁衍出目标子串，就提前结束进化过程
            break
    # 输出适应度最高的个体
    pop.sort(key=lambda x:x.fitness,reverse=True)
    print(pop[0].string)

    plt.plot(np.arange(len(pop_fit)),np.array(pop_fit))
    plt.show()