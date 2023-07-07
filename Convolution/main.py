import torch
import torch.nn.functional as F
import torch.nn as nn
import time

in_channels = 2
out_channels = 2
kernel_size = 3
w = 9
h = 9

x = torch.ones(1, in_channels, w, h)

# 方法一:原生写法
t1 = time.time()
conv_2D_layer = nn.Conv2d(in_channels,out_channels,kernel_size,padding='same')
conv_point_wise_layer = nn.Conv2d(in_channels,out_channels,1,padding='valid')
result1 = conv_2D_layer(x) + conv_point_wise_layer(x)+x
print(result1.size())
t2 = time.time()

# 方法2:算子融合
 # 将原生中的1*1的point-wise kernel扩充至3*3
 # 此时conv_2D_for_pointwise使用的卷积参数就是以原生的pointwise为核心的3*3的kernel
point_wise_to_conv_weight = F.pad(conv_point_wise_layer.weight,[1,1,1,1,0,0,0,0])
print(point_wise_to_conv_weight)
conv_2D_for_pointwise = nn.Conv2d(in_channels,out_channels,kernel_size,padding='same')
conv_2D_for_pointwise.weight = nn.Parameter(point_wise_to_conv_weight)
conv_2D_for_pointwise.bias = nn.Parameter(conv_point_wise_layer.bias)

zeros = torch.unsqueeze(torch.zeros(kernel_size,kernel_size),0)
stars = torch.unsqueeze(F.pad(torch.ones(1,1),[1,1,1,1]),0)
stars_zeros = torch.unsqueeze(torch.cat([stars,zeros]),0)
zeros_stars = torch.unsqueeze(torch.cat([zeros,stars]),0)
identity_to_conv_weight = torch.cat([stars_zeros,zeros_stars],0)
identity_to_conv_bias = torch.zeros([out_channels])
conv_2d_for_identity = nn.Conv2d(in_channels,out_channels,kernel_size,padding='same')
conv_2d_for_identity.weight = nn.Parameter(identity_to_conv_weight)
conv_2d_for_identity.bias = nn.Parameter(identity_to_conv_bias)
result2 = conv_2D_layer(x) + conv_2D_for_pointwise(x) + conv_2d_for_identity(x)

print(torch.all(torch.isclose(result1,result2)))


# 融合
t3 = time.time()
conv_2D_for_fusion = nn.Conv2d(in_channels,out_channels,kernel_size,padding='same')
conv_2D_for_fusion.weight = nn.Parameter(conv_point_wise_layer.weight.data + conv_2D_for_pointwise.weight.data + conv_2d_for_identity.weight.data)
conv_2D_for_fusion.bias = nn.Parameter(conv_point_wise_layer.bias.data + conv_2D_for_pointwise.bias.data + conv_2d_for_identity.bias.data)
result3 = conv_2D_for_fusion(x)
t4 = time.time()

print("原生算法计算时间:",t2-t1)
print("算子融合计算时间:",t4-t3)
















