import torch


y = torch.tensor([[1, 2], [3, 4]])

print(y)
y = torch.tile(y, [2, 1]) #行多一倍 列不变
print(y)

y1 = torch.tensor([[1, 2], [3, 4]])
y1 = torch.tile(y1, [1, 3]) #行多一倍 列不变
print(y1)





