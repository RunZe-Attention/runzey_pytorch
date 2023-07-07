import torch

x = torch.tensor([1,2,3,4])

a = torch.unsqueeze(x,dim = 0)
print(a)

b = torch.unsqueeze(x,dim=1)
print(b)