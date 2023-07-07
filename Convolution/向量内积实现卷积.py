import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
in_channels=1
out_channels=1
kernel_size=3
bias=False
batch_size=1
input_size = [batch_size, in_channels, 4, 4]

conv_layer = nn.Conv2d(in_channels,out_channels,kernel_size,bias=bias)
input_feature_map = torch.randn(input_size)
output_feature_map = conv_layer(input_feature_map)
output_feature_map1 = F.conv2d(input_feature_map,conv_layer.weight)


input=torch.randn(5,5)  # 卷积输入特征图
kernel = torch.randn(3,3) # 卷积核
bias = torch.randn([1])

# step1 DIY convolution
def matrix_multiplication_for_convolution2D_fatten(input,kernel,bias=0,stride=1,padding=0):
    if padding > 0:
        input = F.pad(input,[padding,padding,padding,padding])

    (input_h,input_w) = input.shape
    (kernel_h,kernel_w) = kernel.shape

    output_h = math.floor((input_h-kernel_h)/stride)+1
    output_w = math.floor((input_w - kernel_w) / stride) + 1
    output = torch.zeros(output_h,output_w)

    region_matrix = torch.zeros(output.numel(),kernel.numel())
    kernel_matrix = kernel.reshape((kernel.numel(),1))
    row_index = 0
    for i in range(0,input_h-kernel_h+1,stride):
        for j in range(0,input_w-kernel_w+1,stride):
            region = input[i:i+kernel_h,j:j+kernel_w]
            region_vector = torch.flatten(region)
            region_matrix[row_index] = region_vector
            row_index = row_index+1
    output_matrix = region_matrix @ kernel_matrix
    output=output_matrix.reshape((output_h,output_w)) + bias

    return output

diy = matrix_multiplication_for_convolution2D_fatten(input=input,\
                                              kernel=kernel,\
                                              bias=bias,\
                                              padding=1)

api = F.conv2d(input.reshape((1,1,input.shape[0],input.shape[1])),\
               kernel.reshape((1,1,kernel.shape[0],kernel.shape[1])),\
               padding=1,\
               bias=bias).squeeze(0).squeeze(0)

flag = torch.allclose(diy,api)
print(flag)


def matrix_multiplication_for_convolution2D_all(input,kernel,bias=0,stride=1,padding=0):
    if padding > 0:
        input = F.pad(input,[padding,padding,padding,padding])

    (bs,in_channel,input_h,input_w,) = input.shape
    (out_channel,in_channel,kernel_h,kernel_w) = kernel.shape

    if bias == None:
        bias = torch.zeros(out_channel)

    output_h = math.floor((input_h-kernel_h)/stride)+1
    output_w = math.floor((input_w - kernel_w) / stride) + 1
    output = torch.zeros(bs,out_channel,output_h,output_w)

    region_matrix = torch.zeros(output.numel(),kernel.numel())
    kernel_matrix = kernel.reshape((kernel.numel(),1))
    row_index = 0

    for ind in range(bs):
        for oc in range(out_channel):
            for ic in range(in_channel):
                for i in range(0,input_h-kernel_h+1,stride):
                    for j in range(0,input_w-kernel_w+1,stride):
                        region = input[ind,ic,i:i+kernel_h,j:j+kernel_w]
                        output[ind,oc,int(i/stride),int(j/stride)] += torch.sum(region * kernel[oc,ic])
            output[ind, oc] += bias[oc]

    return output



input = torch.randn(2,2,5,5)
kernel = torch.randn(3,2,3,3)
bias = torch.randn(3)

api = F.conv2d(input,kernel,bias=bias,padding=1,stride=2)
diy = matrix_multiplication_for_convolution2D_all(input=input,kernel=kernel,bias=bias,padding=1,stride=2)

print(api)
print(diy)

flag = torch.allclose(api,diy)

print(flag)



#
def get_kernel_matrix(kernel,input_size):
    kernel_h,kernel_w = kernel.shape
    input_h,input_w = input_size
    num_out_feature_map = (input_h-kernel_h+1) * (input_w-kernel_w+1)
    result = torch.zeros((num_out_feature_map,input_h * input_w))

    row_index = 0
    for i in range(0,input_h - kernel_h + 1 , 1):
        for j in range(0,input_w - kernel_w + 1,1):
            kernel_pad = F.pad(kernel,(i,input_h-kernel_h-i,j,input_w-kernel_w-j))
            result[row_index] = kernel_pad.flatten()
            row_index+=1

    return result

kernel = torch.randn(3,3)
input = torch.randn(4,4)

kernel_matrix= get_kernel_matrix(kernel,input.shape)
diy = kernel_matrix@input.reshape((-1,1))
api = F.conv2d(input.unsqueeze(0).unsqueeze(0),kernel.unsqueeze(0).unsqueeze(0))

print(diy)
print(api)




my_transpose_conv2d_output=kernel_matrix.transpose(-1,-2) @ diy
pt_transpose_conv2d_output=F.conv_transpose2d(api,kernel.unsqueeze(0).unsqueeze(0))

print(my_transpose_conv2d_output.reshape(4,4))
print(pt_transpose_conv2d_output)









