import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# 优化了复杂度, 效果升级,
# 使用了transformer热点
# transformer多模态:CV+NLP+语音

# shifted windows + transformer
# 可以作为基础模型
# Tran用于CV的难点,image中的pixel太多(爆炸)，直接使用pixel会面临高计算复杂度的问题

# 1.基于图片生成patch embbedding
def image2emb_naive(image,patch_size,weight):
    patch = F.unfold(image,kernel_size=(patch_size,patch_size),stride = (patch_size,patch_size)).transpose(-1,-2)
    patch_embbeding = patch @ weight
    return patch_embbeding

def image2emb_convolution(image,kernel,stride):
    conv_output = F.conv2d(image, kernel, stride=stride)
    bs,oc,oh,ow = conv_output.shape
    patch_embbeding = conv_output.reshape((bs,oc,oh*ow)).transpose(-1,-2)
    return patch_embbeding


class MultiHeadSelfAttention(nn.Module):
    def __init__(self,model_dim,num_head):
        super(MultiHeadSelfAttention,self).__init__()
        self.num_head = num_head

        self.proj_linear_layer = nn.Linear(model_dim, 3*model_dim)
        self.final_linear_layer = nn.Linear(model_dim, model_dim)

    def forward(self, input, additive_mask=None):
        bs,seq_len,model_dim = input.shape
        num_head = self.num_head
        head_dim = model_dim // num_head
        proj_output = self.proj_linear_layer(input)
        q, k, v = proj_output.chunk(3,dim=-1)
        # q:[bs,seq_len,model_dim]
        q = q.reshape(bs, seq_len, num_head, head_dim).transpose(1, 2).reshape(bs * num_head, seq_len, head_dim)
        k = k.reshape(bs, seq_len, num_head, head_dim).transpose(1, 2).reshape(bs * num_head, seq_len, head_dim)
        v = v.reshape(bs, seq_len, num_head, head_dim).transpose(1, 2).reshape(bs * num_head, seq_len, head_dim)
        if additive_mask is None:
            attention_prob = F.softmax(torch.bmm(q,k.transpose(-1,-2))/math.sqrt(head_dim),dim=-1)
        else:
            additive_mask = additive_mask.tile(num_head,1,1)
            attention_prob = F.softmax(torch.bmm(q, k.transpose(-1, -2)) / math.sqrt(head_dim) + additive_mask, dim=-1)
        output = torch.bmm(attention_prob, v) # [bs*num_head,seq_len,head_dim]
        output = output.reshape(bs, num_head, seq_len, head_dim).transpose(1,2)
        output = output.reshape(bs, seq_len, model_dim)
        output = self.final_linear_layer(output)
        return attention_prob, output


def windows_multi_head_self_attention(patch_embbeding,mhsa,window_size=4,num_head=2):
    num_patch_in_window = window_size*window_size
    bs, num_patch, patch_depth = patch_embbeding.shape
    image_height = image_width = int(math.sqrt(num_patch))

    patch_embbeding = patch_embbeding.transpose(-1, -2)
    patch = patch_embbeding.reshape(bs,patch_depth, image_height, image_width)
    window = F.unfold(patch,kernel_size=(window_size,window_size),stride=(window_size,window_size),).transpose(-1,-2)


    bs,num_windows,patch_depth_times_num_patch_in_window = window.shape
    window = window.reshape(bs*num_windows,patch_depth,num_patch_in_window).transpose(-1,-2)

    _, output = mhsa(window)# [bs*num_window,num_patch_in_window,PATCH_depth]

    output = output.reshape(bs,num_windows,num_patch_in_window,patch_depth)
    return output


# 开始书写S-MHSA
## window转成image
def window2image(msa_output):
    bs, num_windows, num_patch_in_windows, patch_depth = msa_output.shape
    window_size = int(math.sqrt(num_patch_in_windows))
    image_height = image_width = int(math.sqrt(num_windows)) * window_size
    msa_output = msa_output.reshape(bs,int(math.sqrt(num_windows)),int(math.sqrt(num_windows)),window_size,window_size,patch_depth)
    msa_output = msa_output.transpose(2,3)
    image = msa_output.reshape(bs,image_height*image_width,patch_depth)
    image = image.transpose(-2,-1)
    image = image.reshape(bs,patch_depth,image_height,image_width)
    return image

def shift_window(w_msa_output,window_size,shift_size,generate_mask=False):
    bs, num_window, num_patch_in_window,patch_depth = w_msa_output.shape
    w_msa_output = window2image(w_msa_output)

    bs,patch_depth,image_height,image_weight = w_msa_output.shape

    rolled_w_msa_output = torch.roll(w_msa_output,shifts=(shift_size,shift_size),dims=(2,3))

    shifted_w_msa_output = rolled_w_msa_output.reshape(bs,patch_depth,int(math.sqrt(num_window)),window_size,int(math.sqrt(num_window)),window_size)

    shifted_w_msa_output = shifted_w_msa_output.transpose(3,4)
    shifted_w_msa_output = shifted_w_msa_output.reshape(bs,patch_depth,num_window*num_patch_in_window)
    shifted_w_msa_output = shifted_w_msa_output.transpose(-1,-2)
    shifted_window = shifted_w_msa_output.reshape(bs,num_window,num_patch_in_window,patch_depth)

    if generate_mask:
        additive_mask = build_mask_for_shifted_wmsa(bs,image_height,image_weight,window_size)
    else:
        additive_mask = None

    return shifted_window,additive_mask


def build_mask_for_shifted_wmsa(batch_size,image_height,image_width,window_size):
    index_matrix = torch.zeros(image_height,image_width)
    for i in range(image_height):
        for j in range(image_width):
            row_times = (i+window_size//2) // window_size
            col_times = (j+window_size//2) // window_size
            value = row_times * (image_height // window_size) + col_times + 1
            index_matrix[i,j] = value

    #print(index_matrix)
    roll_index_matrix = torch.roll(index_matrix,shifts=(-window_size//2,-window_size//2),dims=(0,1))
    roll_index_matrix = roll_index_matrix.unsqueeze(0).unsqueeze(0) #[bs,ch,h,w]

    c = F.unfold(roll_index_matrix,kernel_size=(window_size,window_size),stride=(window_size,window_size)).transpose(-1,-2)

    #c = c.tile(batch_size,1,1)

    bs,num_window,num_patch_in_window = c.shape

    c1 = c.unsqueeze(-1)
    c2 = (c1-c1.transpose(-1,-2))== 0
    valid_matrix = c2.to(torch.float32)

    unlimit_min = -1e-9
    additive_mask = (1-valid_matrix) * unlimit_min

    additive_mask = additive_mask.reshape(bs * num_window,num_patch_in_window,num_patch_in_window)
    return additive_mask

def shift_window_multi_head_self_attention(w_msa_output,mhsa,window_size=4,num_head=2):
    bs,num_window,num_patch_in_window,patch_depth = w_msa_output.shape
    shifted_w_msa_output ,additive_mask = shift_window(w_msa_output,window_size,
                                                       shift_size=-window_size//2,generate_mask=True)

    shifted_w_msa_output = shifted_w_msa_output.reshape(bs*num_window,num_patch_in_window,patch_depth)

    _, output = mhsa(shifted_w_msa_output,additive_mask)
    output = output.reshape(bs, num_window,num_patch_in_window,patch_depth)

    output,_ = shift_window(output,window_size,shift_size=window_size//2,generate_mask=False)
    return output

class PatchMerging(nn.Module):
    def  __init__(self, model_dim, merge_size, output_depth_scale = 0.5):
        super(PatchMerging,self).__init__()
        self.merge_size = merge_size
        self.proj_layer = nn.Linear(
            model_dim*merge_size*merge_size,
            int(model_dim*merge_size*merge_size*output_depth_scale))

    def forward(self,input):
        bs,num_window,num_patch_in_window,patch_depth = input.shape
        window_size = int(math.sqrt(num_patch_in_window))
        input = window2image(input)  # [bs,patch_depth,image_h,image_w ]
        merged_window = F.unfold(input,kernel_size=(self.merge_size,self.merge_size),
                                 stride=(self.merge_size,self.merge_size)).transpose(-1,-2)

        merged_window = self.proj_layer(merged_window)

        return merged_window


# 构建Swin_Transformer block
class SwinTransformerBlock(nn.Module):
    def __init__(self,model_dim,window_size,num_head):
        super(SwinTransformerBlock,self).__init__()
        self.layer_norm1 = nn.LayerNorm(model_dim)
        self.layer_norm2 = nn.LayerNorm(model_dim)
        self.layer_norm3 = nn.LayerNorm(model_dim)
        self.layer_norm4 = nn.LayerNorm(model_dim)

        self.wsma_mlp1 = nn.Linear(model_dim,4*model_dim)
        self.wsma_mlp2 = nn.Linear(4*model_dim,model_dim)
        self.swsma_mlp1 = nn.Linear(model_dim, 4 * model_dim)
        self.swsma_mlp2 = nn.Linear(4 * model_dim, model_dim)

        self.mhsa1 = MultiHeadSelfAttention(model_dim,num_head)
        self.mhsa2 = MultiHeadSelfAttention(model_dim,num_head)

    def forward(self, input):
        bs,num_patch,patch_depth = input.shape
        input1 = self.layer_norm1(input)
        w_msa_output = windows_multi_head_self_attention(input1,self.mhsa1,window_size=4,num_head=2)
        bs,num_window, num_patch_in_window, patch_depth = w_msa_output.shape
        w_msa_output = input + w_msa_output.reshape(bs,num_patch,patch_depth)
        output1 = self.wsma_mlp2(self.wsma_mlp1(self.layer_norm2(w_msa_output)))
        output1 += w_msa_output

        input2 = self.layer_norm3(output1)
        input2 = input2.reshape(bs,num_window,num_patch_in_window,patch_depth)
        sw_msa_output = shift_window_multi_head_self_attention(input2, self.mhsa2,window_size=4,num_head=2)
        sw_msa_output = output1+sw_msa_output.reshape(bs,num_patch,patch_depth)
        output2 = self.swsma_mlp2(self.swsma_mlp1(self.layer_norm4(sw_msa_output)))
        output2 += sw_msa_output

        output2 = output2.reshape(bs,num_window,num_patch_in_window,patch_depth)
        return output2


class SwinTransformerModel(nn.Module):
    def __init__(self,input_image_channel = 3,patch_size = 4,model_dim_C = 8,num_classes = 10,window_size = 4,num_head=2,merge_size = 2):
        super(SwinTransformerModel,self).__init__()

        patch_depth = patch_size*patch_size*input_image_channel
        self.patch_size = patch_size
        self.model_dim_C = model_dim_C
        self.num_classes = num_classes

        self.patch_embbeding_weight = nn.Parameter(torch.randn(patch_depth,model_dim_C))

        self.block1 = SwinTransformerBlock(model_dim_C,window_size,num_head)
        self.block2 = SwinTransformerBlock(model_dim_C*2, window_size, num_head)
        self.block3 = SwinTransformerBlock(model_dim_C*4, window_size, num_head)
        self.block4 = SwinTransformerBlock(model_dim_C*8, window_size, num_head)

        self.patch_merging1 = PatchMerging(model_dim_C,merge_size)
        self.patch_merging2 = PatchMerging(model_dim_C*2, merge_size)
        self.patch_merging3 = PatchMerging(model_dim_C*4, merge_size)

        self.final_linear = nn.Linear(model_dim_C*8,num_classes)


    def forward(self,image):
        patch_embbeding_naive = image2emb_naive(image,self.patch_size,self.patch_embbeding_weight)

        # stage1
        patch_embbeding = patch_embbeding_naive
        print(patch_embbeding.shape)
        sw_msa_output = self.block1(patch_embbeding)
        print("stage1_output:",sw_msa_output.shape)

        # stage2
        merged_patch1 = self.patch_merging1(sw_msa_output)
        sw_msa_output_1 = self.block2(merged_patch1)
        print("stage2_output:", sw_msa_output_1.shape)

        # stage3
        merged_patch2 = self.patch_merging2(sw_msa_output_1)
        sw_msa_output_2 = self.block3(merged_patch2)
        print("stage3_output:", sw_msa_output_2.shape)

        # stage4
        merged_patch3 = self.patch_merging3(sw_msa_output_2)
        sw_msa_output_3 = self.block4(merged_patch3)
        print("stage4_output:", sw_msa_output_3.shape)

        # logits
        bs,num_window,num_patch_in_window,patch_depth = sw_msa_output_3.shape
        sw_msa_output_3 = sw_msa_output_3.reshape(bs,-1,patch_depth)
        pool_output = torch.mean(sw_msa_output_3,dim=1)
        logits = self.final_linear(pool_output)

        print("logits",logits.shape)


def test():
    bs,ic,image_h,image_w = 1,3,256,256
    patch_size = 4
    model_dim_C = 8
    num_classes = 10
    window_size = 4
    num_head = 2
    merge_size = 2

    patch_depth = patch_size*patch_size*ic
    image = torch.randn(bs,ic,image_h,image_w)
    model = SwinTransformerModel(ic,patch_size,model_dim_C,num_classes,window_size,num_head,merge_size)

    model(image)

if __name__ == '__main__':
    test()























































































