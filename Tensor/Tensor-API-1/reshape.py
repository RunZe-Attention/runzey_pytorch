import torch

shape_tensor = torch.arange(4)
a = torch.reshape(shape_tensor,(2,2))
print(a)
b = torch.reshape(shape_tensor,(-1,))
print(b)