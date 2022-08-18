from gym import envs
import gym

'''列出OpenAI Gym中所有的環境'''
print(envs.registry.all())
print(len(envs.registry.all()))

'''在Cart-Pole中創建環境'''
env = gym.make("CartPole-v0")

'''在Cart-Pole中執行行動'''
state1 = env.reset() # 初始化環境
action = env.action_space.sample() # 利用sample()隨即從動作空間中取得一個動作
state,reward,done,info = env.step(action) # 利用step()執行所選擇的操作，並傳回狀態資料
print(action,"\n",state,"\n",reward,"\n",done,"\n",info)