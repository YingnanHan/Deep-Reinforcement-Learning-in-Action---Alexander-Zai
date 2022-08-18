import gym
from nes_py.wrappers import JoypadSpace # 此wrapper模组透过结合不同动作来缩小动作空间
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT,COMPLEX_MOVEMENT # 我们可以汇入的动作包括两种：一种只包括5个动作，另一种只包括10个动作
import matplotlib.pyplot as plt
from skimage.transform import resize # 该函式库中的内建函数可调整画面的大小

env = gym_super_mario_bros.make("SuperMarioBros-v0")
env = JoypadSpace(env,COMPLEX_MOVEMENT) # 选择complex的动作空间
done = True

# ## 原始游戏画面
# for step in range(10):
#     if done:
#         state = env.reset() # 如果游戏结束就重置环境
#     state,reward,done,info = env.step(env.action_space.sample()) # 从动作空间中随机选择动作来执行
#
#     plt.imshow(env.render('rgb_array')) # 显示游戏画面
#     plt.pause(0.05)
#     plt.show()

def downscale_obs(obs,new_size=(42,42),to_gray=True):
    # 原始状态资料  调整后的shape 转换成灰阶
    if to_gray:
        return resize(obs,new_size,anti_aliasing=True).max(axis=2) # 为了将图片转换成灰阶，我们取出obs的第二阶颜色通道中的最大值
    else:
        return resize(obs,new_size,anti_aliasing=True)

plt.figure(figsize=(12,12))
plt.imshow(env.render("rgb_array")) # 呈现游戏原始画面
plt.figure(figsize=(12,12))
plt.imshow(downscale_obs(env.render("rgb_array"))) # 呈现经过处理的游画面

plt.show()