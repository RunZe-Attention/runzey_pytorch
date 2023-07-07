import torch
import torch.nn as nn
import torch.nn.functional as F


in_channels=1
out_channels=1
kernel_size=3
bias=False
batch_size=1
input_size = [batch_size, in_channels, 4, 4]

# 类的方式定义卷积层 推荐 其实内部还是调用了nn.functional的conv2d
conv_layer = nn.Conv2d(in_channels,out_channels,kernel_size,bias=bias)
input_feature_map = torch.randn(input_size)
output_feature_map = conv_layer(input_feature_map)
# 直接调用函数方式定义卷积层
output_feature_map1 = F.conv2d(input_feature_map,conv_layer.weight)

# 输出一致
print(output_feature_map)
print(output_feature_map1)


