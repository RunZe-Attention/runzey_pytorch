import torch

a = torch.rand([4,3])

print(a)

b = torch.transpose(a,0,1)
print(b)