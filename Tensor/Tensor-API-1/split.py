import torch

split_tensor = torch.arange(10).reshape(5,2)

print("original tensor is :",split_tensor)

a = torch.split(split_tensor,1)
print(a)
b = torch.split(split_tensor,[2,3])

print(b)