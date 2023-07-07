import torch
src = torch.ones((2, 5))
index = torch.tensor([[0, 1, 2, 0, 0]])
r1 = torch.zeros(3, 5, dtype=src.dtype).scatter_add_(0, index, src)

index = torch.tensor([[0, 1, 2, 0, 0], [0, 1, 2, 2, 2]])
r2 = torch.zeros(3, 5, dtype=src.dtype).scatter_add_(0, index, src)

print(r1)
print("-----\n")
print(r2)

