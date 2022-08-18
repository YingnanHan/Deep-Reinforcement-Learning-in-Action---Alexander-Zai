import torch

m = torch.Tensor([2.0])
m.requires_grad = True
b = torch.Tensor([1.0])
b.requires_grad = True

def linear_model(x,m,b):
    y = m*x + b
    return y

y = linear_model(torch.Tensor([4.]),m,b)
print(y)
print(y.grad_fn)

y = linear_model(torch.Tensor([4.0]),m,b)
y.backward()
print(m.grad)
print(b.grad)