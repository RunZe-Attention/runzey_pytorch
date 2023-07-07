import torch

Mean = torch.arange(1., 11.)
Std = torch.arange(1, 0, -0.1)

print(Mean)
print(Std)
Gaussian = torch.normal(mean=Mean, std=Std)
print(Gaussian)

