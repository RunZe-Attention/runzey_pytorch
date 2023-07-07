import torch
import torch.nn as nn
import torch.nn.functional as F

# 1.比较一下GRU与LSTM的参数量
lstm_layer = nn.LSTM(input_size=3,hidden_size=5)
print(sum(p.numel() for p in lstm_layer.parameters()))
gru_layer = nn.GRU(input_size=3,hidden_size=5)
print(sum(p.numel() for p in gru_layer.parameters()))

def gru_forward(input,initial_states,w_ih,w_hh,b_ih,b_hh):
    prev_h = initial_states
    bs,T,i_size = input.shape
    h_size = w_ih.shape[0] // 3
    batch_w_ih = w_ih.unsqueeze(0).tile(bs,1,1)
    batch_w_hh = w_hh.unsqueeze(0).tile(bs,1,1)

    output = torch.zeros(bs,T,h_size)

    for t in range(T):
        x = input[:,t,:] # t时刻GRU的输入特征向量
        w_times_x =  torch.bmm(batch_w_ih,x.unsqueeze(-1))
        w_times_x = w_times_x.squeeze(-1) # [bs , 3 * hidden_size]

        w_times_h_prev = torch.bmm(batch_w_hh, prev_h.unsqueeze(-1))
        w_times_h_prev = w_times_h_prev.squeeze(-1)  # [bs , 3 * hidden_size]

        r_t = torch.sigmoid(w_times_x[:, :h_size] + w_times_h_prev[:, :h_size]+b_ih[:h_size]+b_hh[:h_size])
        z_t = torch.sigmoid(w_times_x[:, h_size:2*h_size] + w_times_h_prev[:, h_size:2*h_size] + b_ih[h_size:2*h_size]+b_hh[h_size:2*h_size])

        a = w_times_x[:, 2*h_size:3*h_size]+b_ih[2*h_size:3*h_size]
        b = r_t*(w_times_h_prev[:, 2*h_size:3*h_size]+b_hh[2*h_size:3*h_size])
        tmp = a + b
        n_t = torch.tanh(tmp) # 候选状态

        prev_h = (1-z_t)*n_t + z_t * prev_h
        output[:,t,:] = prev_h

    return output,prev_h



def test():
    bs, T, i_size, h_size = 2, 3, 4, 5
    input = torch.randn(bs, T, i_size)
    h0 = torch.randn(bs, h_size) # 初始值 不需要训练

    gru_layer_official = nn.GRU(input_size=i_size,hidden_size=h_size,batch_first=True)
    output,h_final= gru_layer_official(input,h0.unsqueeze(0))
    print(output)

    for k,v in gru_layer_official.named_parameters():
        print(k,v.shape)

    output_custom,h_final_custom = gru_forward(input,h0,gru_layer_official.weight_ih_l0,gru_layer_official.weight_hh_l0,
                                               gru_layer_official.bias_ih_l0,gru_layer_official.bias_hh_l0)
    print(output_custom)

if __name__ == '__main__':
    test()








