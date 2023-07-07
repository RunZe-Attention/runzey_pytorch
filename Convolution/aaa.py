import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


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


input=torch.randn(5,5)  # 卷积输入特征图
kernel = torch.randn(3,3) # 卷积核
bias = torch.randn([1])

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