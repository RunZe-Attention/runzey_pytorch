import torch
import torch.nn as nn
import torch.nn.functional as nn

batch_size = 2
time_step = 3
embedding_dim = 4
inputx = torch.randn(batch_size,time_step,embedding_dim)

ins_norm_op = torch.nn.InstanceNorm1d(embedding_dim)
ins_y = ins_norm_op(inputx.transpose(-1,-2)).transpose(-1,-2)


# 风格迁移
ins_mean  = inputx.mean(dim=(1),keepdim = True)
ins_std  = inputx.std(dim=(1),keepdim = True,unbiased = False)
print(ins_mean)
verify_ins_y = (inputx-ins_mean)/((ins_std+1e-5))
print(ins_y)
print(verify_ins_y)
