import torch
import torch.nn as nn


in_channels = 3
out_channels = 3
kernel_size = 3

conv_standard = nn.Conv2d(in_channels, out_channels, kernel_size, padding="same")

for p in conv_standard.parameters():
    print(torch.numel(p))

conv_depth = nn.Conv2d(in_channels, out_channels, kernel_size, groups = 3,padding="same")
for p in conv_depth.parameters():
    print(torch.numel(p))

    

conv_point = nn.Conv2d(in_channels, out_channels, 1)
for p in conv_point.parameters():
    print(torch.numel(p))