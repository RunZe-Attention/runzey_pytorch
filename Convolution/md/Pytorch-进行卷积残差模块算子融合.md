# <font color = 'red'>卷积残差模块算子融合</font>

## 卷积两个假设

- 局部关联性
- 平移不变性(各种旋转卷积不变)

## torch.nn.Conv2D

![卷积直观动态图](./pic/卷积直观动态图.gif)

## 按照上图举例

- 输入channel=3

- 输出channel=2

- kernel_size = 3(3*3)

- 代码

  ```PYTHON
  import torch
  conv_layer = torch.nn.Conv2d(3,2,3,padding="same")
  print(conv_layer.weight.size())
  ```

- 输出结果

  ```python
  # torch.Size([2, 3, 3, 3])
  ```

  

## 为什么conv_layer的weight是这个shape

- 1.首先我们拥有了一个3*3的kernel,如上图W0中的W0[:,:,0]
- 2.因为输入channel=3,因此完整的单个kernel为3X3X3,也就是三个3*3的卷积核对应输入channel=3的图片卷积运算后相加,这样就能得到1/输出channels的image分量
- 3.因为定义了输出为两个channel,每个channel由3X3X3的卷积核运算得来,因此一共有2X3X3X3个参数
- 4,如果其他其他保持不变,但是输出的channel变为10,那么这一层的weight的性转就是(10,3,3,3)
- 5.可以简单的总结一下输出weight输出shape的含义(**input_channel**,**output_channel**,**kernel_size_width**,**kernel_size_height**)

- 6.**bias**的shape比较好理解,就是output_channel,也就是说输出image的每个channel对应一个bias



## point-wise convolution(不再考虑局部pixel之间的关联

## )

- 相当于一个MLP,kernel_size=1

## depth-wise convolution(不再考虑channel之间的关联)

- 增加了一个groups参数
- groups的目的就是减少计算量
- groups引入的前提假设不需要所有的channel进行混合

- 代码

  ```python
  import torch
  conv_layer = torch.nn.Conv2d(2,4,3,padding="same",groups=2)
  print(conv_layer.weight.size())
  ```

- 输出

  ```PYTHON
  # torch.Size([4, 1, 3, 3])
  ```

- 实际上groups=2的操作就是将input_channel=2，output_channel=4的输入分成**两个**input-channel=1,output_channel=2分别运算,完事后简单做一个拼接

- 也就是说在这个例子中，输入得两个channel在做卷积的时候根本没有融合





# <font color = 'red'>CNN的算子融合</font>

- 代码

```python
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
```



- 输出

```python
原生算法计算时间: 0.0011911392211914062
算子融合计算时间: 0.00030612945556640625

# 算子融合将大幅度节省计算时间
```



