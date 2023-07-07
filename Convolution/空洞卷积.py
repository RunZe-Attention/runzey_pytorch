import  torch
import torch.nn as nn
import torch.nn.functional as F

a = torch.randn(7,7)
print(a)

print(a[0:3,0:3])

print(a[0:3:3,0:3:3])
