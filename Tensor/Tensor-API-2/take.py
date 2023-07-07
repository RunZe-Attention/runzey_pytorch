import torch

src = torch.arange(1, 11).reshape(2, 5)
print(src)
print(torch.take(src,torch.tensor([0, 2, 5])))


