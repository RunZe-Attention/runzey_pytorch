import torch

a = torch.rand(3,2)
b = torch.rand(3,2)

print("a:",a)
print("b:",b)
print("-----/n")
stack_tensor = torch.stack([a,b],dim=0)

print(stack_tensor)
print(stack_tensor.shape)

print("-----/n")
stack_tensor = torch.stack([a,b],dim=1)
print(stack_tensor)
print(stack_tensor.shape)

