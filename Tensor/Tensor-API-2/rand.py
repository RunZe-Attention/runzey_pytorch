# rand是从[0,1]之间生成一个均匀分布
# randn对一个标准高斯分布进行采样
#


import torch
# 传入上届 进行随机shuffle
print(torch.randperm(4))
# tensor([1, 3, 2, 0])
