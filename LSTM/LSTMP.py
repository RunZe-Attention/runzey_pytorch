import torch
import torch.nn as nn
import torch.nn.functional as F


# 1.官方api
bs, T, i_size, h_size = 2, 3, 4, 5
proj_size = 3
input = torch.randn(bs,T,i_size)
c0 = torch.randn(bs,h_size)
h0 = torch.randn(bs,proj_size)

lstm_layer = nn.LSTM(input_size=i_size,hidden_size=h_size,batch_first=True,proj_size = proj_size)
output,(h_final,c_final) = lstm_layer(input,(h0.unsqueeze(0),c0.unsqueeze(0)))






# 2.自己写一个LSTM模型
def lstm_forward(input,initial_states,w_ih,w_hh,b_ih,b_hh,w_hr = None):
    (h0,c0) = initial_states
    bs,T,input_size = input.shape
    h_size = int(w_ih.shape[0]//4)

    prev_h = h0
    prev_c = c0




    batch_w_ih = w_ih.unsqueeze(0).tile(bs,1,1) #[bs, 4*h_size, i_size]
    batch_w_hh = w_hh.unsqueeze(0).tile(bs,1,1) #[bs, 4*h_size, h_size]

    if w_hr is not None:
        p_size = w_hr.shape[0]
        output_size = p_size
        batch_w_hr = w_hr.unsqueeze(0).tile(bs, 1, 1)
    else:
        output_size = h_size

    output = torch.zeros(bs, T, output_size)

    for t in range(T):
        x = input[:, t, :] # [batch_size,i_size]
        w_times_x = torch.bmm(batch_w_ih, x.unsqueeze(-1))
        w_times_x = w_times_x.squeeze(-1) # [bs 4*h_size]

        w_times_h_prev = torch.bmm(batch_w_hh, prev_h.unsqueeze(-1))
        w_times_h_prev = w_times_h_prev.squeeze(-1) # [bs 4*h_size]

        i_t = torch.sigmoid(w_times_x[:,:h_size]+w_times_h_prev[:,:h_size]+b_ih[:h_size]+b_hh[:h_size])
        f_t = torch.sigmoid(w_times_x[:, h_size:2*h_size] + w_times_h_prev[:, h_size:2*h_size] + b_ih[h_size:2*h_size] + b_hh[h_size:2*h_size])
        g_t = torch.tanh(w_times_x[:, 2*h_size:3*h_size] + w_times_h_prev[:, 2*h_size:3*h_size] + b_ih[2*h_size:3*h_size] + b_hh[2*h_size:3 *h_size])
        o_t = torch.sigmoid(w_times_x[:, 3 * h_size : 4 * h_size] + w_times_h_prev[:, 3 * h_size:4 * h_size] + b_ih[3 * h_size:4 * h_size] + b_hh[3*h_size:4 * h_size])

        prev_c = f_t * prev_c + i_t * g_t
        prev_h = o_t * torch.tanh(prev_c)

        if w_hr is not None:
            prev_h = torch.bmm(batch_w_hr,prev_h.unsqueeze(-1))
            prev_h = prev_h.squeeze(-1)

        output[:,t,:] = prev_h

    return output,(prev_h,prev_c)


output_custom,(h_custom,c_custom) = lstm_forward(input,(h0,c0),lstm_layer.weight_ih_l0,lstm_layer.weight_hh_l0,lstm_layer.bias_ih_l0,lstm_layer.bias_hh_l0,lstm_layer.weight_hr_l0)

def test():
    print(output)
    print(output_custom)
    print("------------------------")

    print(h_final)
    print(h_custom)

    print("------------------------")
    print(c_final)
    print(c_custom)


if __name__ == '__main__':
    for k,v in lstm_layer.named_parameters():
        print(k,v.shape)

    test()




