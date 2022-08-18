import gym
import numpy as np

env = gym.make("Freeway-ram-v0") #创建出租车游戏环境
state = env.reset() #初始化环境

env.reset()
for _ in range(1000):
    env.render()
    env.step(env.action_space.sample()) # take a random action
env.close()

envspace = env.observation_space.shape[0] #状态空间的大小
actspace = env.action_space.n #动作空间的大小

print(envspace)
print(actspace)