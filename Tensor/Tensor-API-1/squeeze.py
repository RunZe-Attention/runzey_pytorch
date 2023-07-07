import torch

# 移除dim = 1
squeeze_tensor = torch.rand([3,2])
print(squeeze_tensor)

a = torch.reshape(squeeze_tensor,[3,1,2])
print(a)

b = torch.squeeze(a)
print(b)