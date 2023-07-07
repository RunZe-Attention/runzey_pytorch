import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def get_kernel_matrix(kernel, input_size):
    kernel_h, kernel_w = kernel.shape
    input_h, input_w = input_size
    num_out_feature_map = (input_h - kernel_h + 1) * (input_w - kernel_w + 1)
    result = torch.zeros((num_out_feature_map, input_h * input_w))

    row_index = 0
    for i in range(0, input_h - kernel_h + 1, 1):
        for j in range(0, input_w - kernel_w + 1, 1):
            kernel_pad = F.pad(kernel, (i, input_h - kernel_h - i, j, input_w - kernel_w - j))
            result[row_index] = kernel_pad.flatten()
            row_index += 1

    return result


kernel = torch.randn(3, 3)
input = torch.randn(4, 4)

kernel_matrix = get_kernel_matrix(kernel, input.shape)
diy = kernel_matrix @ input.reshape((-1, 1))
api = F.conv2d(input.unsqueeze(0).unsqueeze(0), kernel.unsqueeze(0).unsqueeze(0))

print(diy)
print(api)
