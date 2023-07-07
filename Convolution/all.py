import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def matrix_multiplication_for_convolution2D_full(input,kernel,bias=0,stride=1,padding=0):
    if padding > 0:
        input = F.pad(input,[padding,padding,padding,padding])

    (bs,in_channel,input_h,input_w,) = input.shape
    (out_channel,in_channel,kernel_h,kernel_w) = kernel.shape

    if bias == None:
        bias = torch.zeros(out_channel)

    output_h = math.floor((input_h-kernel_h)/stride)+1
    output_w = math.floor((input_w - kernel_w) / stride) + 1
    output = torch.zeros(bs,out_channel,output_h,output_w)


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
kernel = torch.randn(3,2,5,5)
bias = torch.randn(3)

api = F.conv2d(input,kernel,bias=bias,padding=1,stride=2)
diy = matrix_multiplication_for_convolution2D_full(input=input,kernel=kernel,bias=bias,padding=1,stride=2)


print(api)
print(diy)

print(torch.allclose(api,diy))

