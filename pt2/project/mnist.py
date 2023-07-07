import torch
from torch import  nn
from torch.nn import  functional as F
from torch import optim

import torchvision

print("------------------")
a = torch.randn(3)
print(a)
print(a.type())
print(isinstance(a,torch.FloatTensor))
print("------------------")


# pytorch中的标量(dimension为0)
print(torch.tensor(1.3))
print(type(torch.tensor(1.3)))

print("----------")
a = torch.tensor(2.2)
print(a)
print(a.shape)
print(len(a.shape))
print(a.size())








