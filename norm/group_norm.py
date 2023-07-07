import torch
import torch.nn as nn
import torch.nn.functional as nn

batch_size = 2
time_step = 3
embedding_dim = 4
inputx = torch.randn(batch_size,time_step,embedding_dim)

num_groups = 2
gr_norm_op = torch.nn.GroupNorm(num_groups,embedding_dim,affine=False)
gr_y = gr_norm_op(inputx.transpose(-1,-2)).transpose(-1,-2)

group_inputx = torch.split(inputx,split_size_or_sections=embedding_dim//num_groups,dim=-1)
result = []
for g_inputx in group_inputx:
    gn_mean = torch.mean(g_inputx,dim=(1,2),keepdim=True)
    gn_std = torch.std(g_inputx,dim=(1,2),keepdim=True,unbiased = False)
    gn_result = (g_inputx - gn_mean)/(gn_std + 1e-5)
    result.append(gn_result)

verify_gr_y = torch.cat(result,dim=-1)
print(gr_y)
print(verify_gr_y)






