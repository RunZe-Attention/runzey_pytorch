import torch
import numpy as np

a =torch.randn(3,3)
print(a)

a = torch.full([2,3],7)
print(a)

a = torch.full([],7)
print(a)

print(torch.arange(0, 10))
print(torch.arange(0, 10, 2))

print(torch.linspace(0,10,10))

print(torch.eye([3,4]))

