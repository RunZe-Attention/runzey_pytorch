import torch
import torch.nn as nn
import torch.nn.functional as nn

batch_size = 2
time_step = 3
embedding_dim = 4
inputx = torch.randn(batch_size,time_step,embedding_dim)

layer_norm_op = torch.nn.LayerNorm(embedding_dim,elementwise_affine=False)
ln_y = layer_norm_op(inputx)

import  math

ln_mean  = inputx.mean(dim=(-1),keepdim = True)
ln_std  = inputx.std(dim=(-1),keepdim = True,unbiased = False)
verify_ln_y = (inputx-ln_mean)/((ln_std+1e-5))
print(ln_mean)
print(ln_std)

print(ln_y)
print(verify_ln_y)
