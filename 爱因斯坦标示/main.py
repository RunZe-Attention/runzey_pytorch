import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce, repeat

x = torch.randn(2,3,4,5) #[bs,i,h,w]

# 1.transpose
out1 = x.transpose(1,2)
out2 = rearrange(x,"b i h w -> b h i w")
print(x.shape)
flag = torch.allclose(out1,out2)
print(flag)

# 2.reshape
out1 = x.reshape(6,4,5)
out2 = rearrange(x,"b i h w -> (b i) h w")
out3 = rearrange(out2,"(b i) h w -> b i h w",b=2)
flag = torch.allclose(out1,out2)
print(flag)
print(out3.shape)

# 3.image2patch
x_image = torch.randn(2,3,4,4)
out1 = rearrange(x_image,"b ic (h1 p1) (w1 p2) -> b ic (h1 w1) (p1 p2)",p1=2,p2=2)
out2 = rearrange(out1,"b ic n p -> b n (ic p)")
print(out2.shape)

# 4.求平均池化
out1 = reduce(x,"b i h w -> b i h","mean")
print(x.shape)
print(out1.shape)

out2 = reduce(x,"b i h w -> b i h","sum")
print(out2.shape)

out3 = reduce(x,"b i h w -> b i","max")
print(out3.shape)


# 5.堆叠张量
tensor_list = [x,x,x]
out1 = rearrange(tensor_list,"n b i h w -> n b i h w")
print(out1.shape)

# 6.扩维
out1 = rearrange(x,"b i h w -> b i h w 1") # 类似于torch.unsqueece
print(out1.shape)

# 7.复制
out2 = repeat(out1,"b i h w 1 -> b i h w 2") # 类似于torch.tile
print(out2.shape)

# 8.对后面的两个维度复制两份
out3 = repeat(x_image,"b i h w -> b i (h 2) (w 2)")
print(out3.shape)










