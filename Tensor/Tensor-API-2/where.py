import torch

x = torch.randn([3, 2]) #使用高斯分布进行随机
print(x)
y = torch.ones([3,2])
print(y)

z = torch.where(x>0,x,y)
print(z)

a = torch.randn(2,2,dtype=torch.double)
print(a)
# b = torch.where(a>0,x,2.)
b = torch.where(a > 0, a, 2.)
print(b)
