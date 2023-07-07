import torch
import torch.nn as nn
import torch.nn.functional as nn

# batch_norm1_1d 官方api
batch_size = 2
time_step = 3
embedding_dim = 4
inputx = torch.randn(batch_size,time_step,embedding_dim)
batch_norm_op = torch.nn.BatchNorm1d(embedding_dim,affine=False)
bn_y = batch_norm_op(inputx.transpose(-1,-2)).transpose(-1,-2)
#print(bn_y)

bn_mean  = inputx.mean(dim=(0,1),keepdim = True)
print(inputx)
print("\n")
print(bn_mean)

print("\n")
print(inputx-bn_mean)

bn_std  = inputx.std(dim=(0,1),keepdim = True,unbiased = False)
#print(bn_std)
verify_bn_y = (inputx-bn_mean)/(bn_std+1e-5)

#print(verify_bn_y)


