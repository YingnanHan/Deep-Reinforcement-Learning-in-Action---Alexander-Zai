import torch

def dist_dqn(x,theta,aspace=3): # x是128个元素的状态向量，theta是参数向量，aspace则是动作空间的大小
    dim0,dim1,dim2,dim3 = 128,100,25,51 # 51代表元素支撑集的数量 可认为是每一个动作概率分布中自变量所映射到离散空间的模
    t1 = dim0 * dim1 # 第一层网路参数的shape 元素这样摆放方便批处理
    t2 = dim1 * dim2 # 第二层网路参数的shape
    theta1 = theta[0:t1].reshape(dim0,dim1) # 将theta中的[0:t1]的参数分配到第一层网路的参数矩阵中
    theta2 = theta[t1:t1+t2].reshape(dim1,dim2) # 将theta中的[t1:t2]的参数分配到第一层网路的参数矩阵中
    l1 = x @ theta1 # 输入资料的shape是BX128，theta1的shape是128X100，因此l1(第一层网路)的shape是BX100 (B是代表批次的大小)
    l1 = torch.selu(l1) # 以scaled expotential linear units(SELUs)作为激活函数
    l2 = l1 @ theta2 # l1的shape是BX100 l2的shape是100X25 因此l2(第二层网路)的shape为BX25
    l2 = torch.selu(l2)
    l3 = []
    for i in range(aspace): # 利用循环走访每一个动作 并产生各个动作的价值分布
        step = dim2 * dim3 # 这里需要结合网络结构来进行思考
        theta5_dim = t1 + t2 + i*step
        theta5 = theta[theta5_dim:theta5_dim+step].reshape(dim2,dim3) # 将参数分配到不同的矩阵当中
        l3_ = l2 @ theta5 # l2 的shape是BX25，theta5的shape是25X51，因此该计算结果的shape为BX51
        l3.append(l3_) # 目前l3的大小是BX51 表示一共有B条数据/批次 每一个批次的输出中有51中状态
    l3 = torch.stack(l3,dim=1) # 最后一层的网路的shape BX3X51 也就是说 一共有B条数据 每一条数据由上左右三个状态的几率分布向量堆叠而成
    l3 = torch.nn.functional.softmax(l3,dim=2) # 对每一个状态的每一个输出预测做归一化操作
    return l3.squeeze() # 得到神经网路的输出

