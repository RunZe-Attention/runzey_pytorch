import torch
import torch.nn as nn
import torch.nn.functional as F

# 1.单向 单层RNN
single_rnn = nn.RNN(input_size=4,hidden_size=3,num_layers=1,batch_first=True)
input = torch.randn(1,2,4)
output,h_n = single_rnn(input)
print(output.shape)
print(h_n.shape)

# 2.双向 单层RNN
bidirectional_rnn = nn.RNN(input_size=4,hidden_size=3,num_layers=1,batch_first=True,bidirectional=True)
input = torch.randn(1,2,4)
bi_output,bi_h_n = bidirectional_rnn(input)
print(bi_output.shape) # 在双向RNN的单个输出中 ,会把forward layer和backward layer的结果concat到一起
print(bi_h_n.shape)

# 矩阵展示
bs = 2
T = 3
input_size = 2
hidden_size = 3

input = torch.randn(bs,T,input_size)
h_prev = torch.zeros(bs,hidden_size) # 初始隐藏状态

# 1.调用 torch RNN API

rnn = nn.RNN(input_size=input_size,hidden_size=hidden_size,batch_first=True)
output,final_output = rnn(input,h_prev.unsqueeze(0))
print(output.shape)
for k,v in rnn.named_parameters():
    print(k,v.shape)

# 2.手写RNN forward函数
def rnn_forward(input,weight_ih,bias_ih,weight_hh,bias_hh,h_prev):
    bs,T,input_size = input.shape
    h_dim=weight_ih.shape[0]





