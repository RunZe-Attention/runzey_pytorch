import torch

a = torch.rand([2,4,3])

print(a)
b = torch.unbind(a,dim=0)
print(b)



