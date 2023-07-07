import  torch
# 矩阵到矩阵之间求梯度
a = torch.randn(2,3,requires_grad=True)
b = torch.randn(3,2,requires_grad=True)
y = a@b


def fun(x):
    return x@b

print(fun(a[0]))