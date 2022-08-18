import torch

x = torch.Tensor([2,4]) # 建立值為[2,4]的張量作為輸入張量
m = torch.randn(2,requires_grad=True) # 隨機產生一個"斜率"的張量
b = torch.randn(1,requires_grad=True) # 隨機產生一個"截距"的張量

y = m*x + b # 線性模型
y_known = torch.Tensor([5,9]) # 建立值為[5,9]的張量作為標籤張量(label)
loss = (torch.sum(y_known-y))**2 # 建立損失函數(這裡選擇的是簡單的平方誤差函數)
loss.backward() # 執行反向傳播計算梯度
print(m.grad) # 利用grad屬性既可以得到m張量的梯度

# model = torch.nn.Sequential(
#     torch.nn.Linear(input_size_1,output_size_1),
#     torch.nn.ReLU(),
#     torch.nn.Linear(input_size_2,output_size_2),
#     torch.nn.ReLU()
# )
#
# loss_fn = torch.nn.MSELoss()
# optimizer = torch.optim.Adam(model.parameters(),lr=0.01)
#
# for step in range(100):
#     y_pred = model(x) # 将x输入模型，然后模型会输出预测值
#     loss = loss_fn(y_pred,y) # 计算预测值与标签值的误差 标签y即正确值
#     optimizer.zero_grad() # 将梯度重设为0，否则上一次反向传播所算出的梯度会累加进来
    loss.backward()
    optimizer.step()