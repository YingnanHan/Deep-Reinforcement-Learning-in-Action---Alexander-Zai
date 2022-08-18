import matplotlib.pyplot as plt
import random
import numpy as np
import torch
import gym

# 定義模型
def model(x,unpacked_params): # 狀態資料 參數向量
    l1,b1,l2,b2,l3,b3 = unpacked_params # 對參數向量進行拆解，將不同的層的參數矩陣獨立出來
    y = torch.nn.functional.linear(x,l1,b1) # 加入含有偏置值的簡單線性單元
    y = torch.relu(y) # 以ReLU函數作為激活函數
    y = torch.nn.functional.linear(y,l2,b2)
    y = torch.relu(y)
    y = torch.nn.functional.linear(y,l3,b3)
    y = torch.log_softmax(y,dim=0) # 輸出各個動作的對數化幾率
    return y

# 拆解參數向量
def unpack_params(params,layers=[(25,4),(10,25),(2,10)]): # 定義每一層網路矩陣的形狀
    unpacked_params = [] # 存儲每一層網路的權重以及偏置
    e = 0
    for i,layer in enumerate(layers): # 逐一走訪網路中的每一層
        s,e = e,e+np.prod(layer) # 計算目前層權重資料的索引位置(由s到e)，例如第一層，s=0,e=25*4=100
        weights = params[s:e].view(layer) # 取出目前層權重的參數並轉換成矩陣形式，例如第一層會取出params[0:100]並轉換成25X4的權重矩陣
        s,e = e,e+layer[0] # 計算目前層偏置資料的索引位置(由s到e)，例如第一層，s=100,e=100+25=125
        bias = params[s:e] # 取出目前層的偏置參數並轉換成矩陣形式，例如第一層回去出params[100:125]作為偏置向量
        unpacked_params.extend([weights,bias]) # 將獨立出來的兩個張量存進串列中
    return unpacked_params

# 產生代理人族群
def spawn_population(N,size): # N代表族群中的個體數量，size則是參數向量的參數總數
    pop = []
    for i in range(N):
        vec = torch.randn(size)/2.0 # 隨機生成代理人的初始參數向量
        fit = 0
        p = {'params':vec,'fitness':fit} # 將參數向量和適應度分數存入字典中，代表一個代理人的資訊
        pop.append(p)
    return pop

# 基因重組
def recombine(x1,x2): # x1和x2代表親代代理人，資料型別分別為字典
    x1 = x1['params'] # 將代理人的參數向量抽取出來
    x2 = x2['params']
    n = x1.shape[0] # 取得參數向量的長度
    split_pt = np.random.randint(n) # 隨機產生一個數字，代表重組時的切割位置索引
    child1 = torch.zeros(n)
    child2 = torch.zeros(n)
    child1[0:split_pt] = x1[0:split_pt] # 第一個後代是由x1的前端和x2的後端組成
    child1[split_pt:] = x2[split_pt:]
    child2[0:split_pt] = x1[0:split_pt]
    child2[split_pt:] = x2[split_pt:]
    c1 = {'params':child1,'fitness':0.0} # 將新產生的兩個參數向量分別存入字典中，產生兩個後代代理人
    c2 = {'params':child2,'fitness':0.0}
    return c1,c2

# 基因突變
def mutate(x,rate):# x代表突變率
    x_ = x['params'] # 取出參數向量
    num_to_change = int(rate*x_.shape[0]) # 使用突變率來決定參數向量中有多少參數可以發生突變
    idx = np.random.randint(low=0,high=x_.shape[0],size=(num_to_change))
    x_[idx] = torch.randn(num_to_change) / 10.0 # 將參數向量中指定位置的參數，替換成標準常態分佈的隨機值
    x['params'] = x_
    return x

def test_model(agent):
    done = False
    state = torch.from_numpy(env.reset()).float()
    score = 0 # 追蹤遊戲進行了多少步，並以此作為代理人的得分
    while not done: # 只要遊戲還沒有結束便持續進行迴圈
        params = unpack_params(agent['params'])
        probs = model(state,params) # 將代理人的參數套入模型中，並產生各個動作的分佈幾率
        action = torch.distributions.Categorical(probs=probs).sample() # 依照各個動作的分佈幾率選擇一個動作
        state_,reward,done,info = env.step(action.item())
        state = torch.from_numpy(state_).float()
        score += 1
    return score

def evaluate_population(pop):
    tot_fit = 0  # 存儲族群的適應度，可用來計算族群的平均適應度
    for agent in pop:  # 測試族群中的每一位代理人
        score = test_model(agent)  # 在環境中執行代理人，評價其適應度
        agent['fitness'] = score  # 將代理人的適應度存儲起來
        tot_fit += score
    avg_fit = tot_fit / len(pop)  # 計算族群的平均適應度
    return pop, avg_fit

def next_generation(pop,mut_rate,tournament_size): # tournament_size介於0 1之間，用來決定競賽人數
    new_pop = []
    lp = len(pop)
    while len(new_pop) < len(pop): # 若後代族群尚未被填滿 則持續進行迴圈
        rids = np.random.randint(low=0,high=lp,size=int(tournament_size*lp)) # 隨機選擇一定的比例的族群個體組成子集(將他們的索引存到rids)
        batch = np.array([[i,x['fitness']] for (i,x) in enumerate(pop) if i in rids]) # 從族群中挑選代理人組成批次，並記錄這些代理人在原始族群中的索引值，以及他們的適應度
        scores = batch[batch[:,1].argsort()] # 將批次中的代理人依照適應度由低到高排序
        i0,i1 = int(scores[-1][0]),int(scores[-2][0]) # 順序位於最下方的代理人具有最高的適應度
        parent0,parent1 = pop[i0],pop[i1] # 此處選擇最末端的兩個代理人作為親代
        offspring_ = recombine(parent0,parent1) # 將親代重組成後代
        child1 = mutate(offspring_[0],rate=mut_rate)
        child2 = mutate(offspring_[1],rate=mut_rate) # 再放入後代族群前，對新代理人進行突變處理
        offspring = [child1,child2]
        new_pop.extend(offspring)
    return new_pop

def running_mean(x,n=5):
    conv = np.ones(n)
    y = np.zeros(x.shape[0]-n)
    for i in range(x.shape[0]-n):
        y[i] = (conv @ x[i:i+n]) / n
    return y

if __name__ == '__main__':
    env = gym.make("CartPole-v0")
    num_generation = 20 # 進化過程中的世代數
    population_size = 500 # 每一代族群中的個體數
    mutation_rate = 0.01 # 突變率
    pop_fit = []
    pop = spawn_population(N=population_size,size=407) # 產生初始種群
    for i in range(num_generation):
        # print(i)
        pop,avg_fit = evaluate_population(pop) # 評估族群中每一個代理人的適應度
        pop_fit.append(avg_fit)
        pop = next_generation(pop,mut_rate=mutation_rate,tournament_size=0.2) # 產生後代族群
    plt.figure(figsize=(12, 7))
    plt.xlabel("Generations", fontsize=22)
    plt.ylabel("Score", fontsize=22)
    plt.plot(running_mean(np.array(pop_fit), 3))
    plt.show()