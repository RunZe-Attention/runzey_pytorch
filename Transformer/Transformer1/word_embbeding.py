import torch
import numpy
import torch.nn as nn
import torch.nn.functional as F

# 关于word embbeding
batch_size = 2

# 词表大小
max_num_src_words = 8
max_num_tgt_words = 8
model_dim = 8


# 最长的squence_len
max_src_seq_len = 5
max_tgt_seq_len = 5
max_position_len = 5

src_len = torch.Tensor([2,4]).to(torch.int32)
src_pos = torch.cat([torch.unsqueeze(torch.arange(0,max(src_len)),0) for _ in src_len]).to(torch.int32)
tgt_len = torch.Tensor([4,3]).to(torch.int32)
tgt_pos = torch.cat([torch.unsqueeze(torch.arange(0,max(tgt_len)),0) for _ in tgt_len]).to(torch.int32)

src_seq = torch.cat([torch.unsqueeze(F.pad(torch.randint(1,max_num_src_words,(L ,)),(0,max_src_seq_len-L)),0) for L in src_len])
tgt_seq = torch.cat([torch.unsqueeze(F.pad(torch.randint(1,max_num_tgt_words,(L ,)),(0,max_tgt_seq_len-L)),0) for L in tgt_len])

# 构造embedding table
src_embbeding_table = nn.Embedding(max_num_src_words+1, model_dim)
tgt_embbeding_table = nn.Embedding(max_num_tgt_words+1, model_dim)

# 构造word embbeding
src_embbeding = src_embbeding_table(src_seq)
tgt_embbeding = tgt_embbeding_table(tgt_seq)

# 构造position embbeding
pos_mat = torch.arange(max_position_len).reshape(-1,1)
#print(pos_mat)
i_mat = torch.pow(10000,torch.arange(0, 8, 2).reshape((1,-1)) / model_dim)
#print(i_mat)
#print(pos_mat/i_mat)
pe_embedding_table = torch.zeros(max_position_len,model_dim)
print(torch.sin(pos_mat/i_mat))
print(pe_embedding_table[:, 0::2])
pe_embedding_table[:, 0::2] = torch.sin(pos_mat/i_mat)
print(pe_embedding_table[:, 0::2])
pe_embedding_table[:, 1::2] = torch.cos(pos_mat/i_mat)
#print(pe_embedding_table)

#print(torch.arange(0,8,2))

pe_embedding = nn.Embedding(max_position_len,model_dim)
pe_embedding.weight = nn.Parameter(pe_embedding_table,requires_grad=False)
src_pe_embedding = pe_embedding(src_pos)
tgt_pe_embedding = pe_embedding(tgt_pos)



# 构造encoder mask
valid_encoder_pos = torch.unsqueeze(torch.cat([torch.unsqueeze(F.pad(torch.ones(L),(0,max(src_len)-L)),0) for L in src_len]),2)
valid_encoder_pos_matrix = torch.bmm(valid_encoder_pos,valid_encoder_pos.transpose(2,1))
invalid_encoder_pos_matrix = 1 - valid_encoder_pos_matrix
mask_encoder_self_attention = invalid_encoder_pos_matrix.to(torch.bool)
score = torch.randn(batch_size,max(src_len),max(src_len))
maskd_score = score.masked_fill(mask_encoder_self_attention,-1e9)
prob = F.softmax(maskd_score,-1)

valid_encoder_pos = torch.unsqueeze(torch.cat([torch.unsqueeze(F.pad(torch.ones(L),(0,max(src_len)-L)),0) for L in src_len]),2)
valid_decoder_pos = torch.unsqueeze(torch.cat([torch.unsqueeze(F.pad(torch.ones(L),(0,max(tgt_len)-L)),0) for L in tgt_len]),2)

#print(valid_encoder_pos)
print('--------------------\n')
#print(valid_decoder_pos)
print('--------------------\n')
valid_cross_pos_matrix = torch.bmm(valid_decoder_pos,valid_encoder_pos.transpose(1,2))
#print(valid_cross_pos_matrix)
invalid_cross_pos_matrix = 1-valid_cross_pos_matrix
mask_score_attention = invalid_cross_pos_matrix.to(torch.bool)
print(mask_score_attention)


cross_score = torch.randn(batch_size,max(tgt_len),max(tgt_len))
maskd_cross_score = score.masked_fill(mask_score_attention,-1e9)
cross_prob = F.softmax(maskd_cross_score,-1)
print(cross_prob)










