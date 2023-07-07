import torch
import torch.nn as nn
import torch.nn.functional as F

"""以离散符号的分类任务为例,实现基于注意力机制的seq2seq模型"""

class Seq2SeqEncoder(nn.Module):
    def __init__(self,embedding_dim,hidden_size,source_vocab_size):
        super(Seq2SeqEncoder,self).__init__()
        self.lstm_layer = nn.LSTM(input_size=embedding_dim,hidden_size=hidden_size,batch_first=True)
        self.embedding_table = nn.Embedding(source_vocab_size,embedding_dim)

    def forward(self,input_ids):
        input_sequence = self.embedding_table(input_ids)
        output_states,(h_final,c_final) = self.lstm_layer(input_sequence)

        return output_states,h_final


class Seq2SeqAttentionMechanism(nn.Module):
    def __init__(self):
        super(Seq2SeqAttentionMechanism,self).__init__()

    def forward(self,decoder_state_t,encoder_state):
        bs,source_length,hidden_size = encoder_state.shape
        decoder_state_t = decoder_state_t.unsqueeze(-1)
        decoder_state_t = torch.tile(decoder_state_t,dims=(1,source_length,1))
        score = torch.sum(decoder_state_t * encoder_state ,-1)
        attn_prob = F.softmax(score,-1) # [bs,source_length]
        context = torch.sum(attn_prob.unsqueeze(-1) * encoder_state,dim=1) # [bs,hidden_size]

        return context,attn_prob


class Seq2SeqDecoder(nn.Module):
    def __init__(self,embedding_dim,hidden_size,num_classes,target_vocab_size,start_id,end_id):
        super(Seq2SeqDecoder,self).__init__()

        self.lstm_cell = nn.LSTMCell(embedding_dim,hidden_size)
        self.proj_layer = nn.Linear(2*hidden_size,num_classes)
        self.attention_mechanism = Seq2SeqAttentionMechanism()
        self.embedding_table = torch.nn.Embedding(target_vocab_size, embedding_dim)
        self.num_classes = num_classes
        self.start_id = start_id
        self.end_id = end_id


    def forward(self,shifted_target_ids,encoder_states):
        shift_target = self.embedding_table(shifted_target_ids)
        bs , target_length , embedding_dim = shift_target.shape
        bs, source_length ,hidden_size = encoder_states.shape
        logits = torch.zeros(bs,target_length,self.num_classes)
        probs = torch.zeros(bs,target_length,source_length)

        for t in range(target_length):
            decoder_input_t = shift_target[:,t,:]
            if t == 0:
                h_t,h_c = self.lstm_cell(decoder_input_t)
            else:
                h_t, h_c = self.lstm_cell(decoder_input_t,(h_t,h_c))

            attn_prob, context = self.attention_mechanism(h_t, encoder_states)
            decoder_output = torch.cat((context,h_t),dim=-1)
            logits[:,t,:] = self.proj_layer(decoder_output)
            probs[:,t,:] = attn_prob

        return probs,logits

    def inference(self,encoder_states):
        target_id = self.start_id
        h_t = None
        result = []

        while True:
            decoder_input_t = self.embedding_table(target_id)
            if h_t is None:
                h_t,h_c = self.lstm_cell(decoder_input_t)
            else:
                h_t, h_c = self.lstm_cell(decoder_input_t,(h_t,h_c))
            attn_prob, context = self.attention_mechanism(h_t, encoder_states)
            decoder_output = torch.cat((context,h_t),dim=-1)
            logits = self.proj_layer(decoder_output)
            target_id = torch.argmax(logits,dim=-1)
            result.append(target_id)
            if torch.any(target_id == self.end_id):
                print("stop decoding")
                break

        predicted_ids = torch.stack(result,dim=0)

        return predicted_ids


class Model(nn.Module):
    def __init__(self,embedding_size,hidden_size,num_classes,source_vocab_size,target_vocab_size,start_id,end_id):
        super(Model,self).__init__()
        self.encoder = Seq2SeqEncoder(embedding_size,hidden_size,source_vocab_size)
        self.decoder = Seq2SeqDecoder(embedding_size,hidden_size,num_classes,target_vocab_size,start_id,end_id)

    def forward(self,input_squence_ids,shifted_target_ids):
        encoder_states,final_h = self.encoder(input_squence_ids)
        probs,logits = self.decoder(shifted_target_ids,encoder_states)

        return probs,logits

if __name__ == '__main__':
    source_length = 3
    target_length = 5
    embedding_dim = 8
    hidden_size = 16
    num_classes = 10
    bs = 2
    start_id = end_id = 0

    source_vocab_size = 100
    target_vocab_size = 100

    input_squence_ids = torch.randint(0,source_vocab_size,(bs,source_length))
    target_ids = torch.randint(0,target_length,(bs,target_length))
    target_ids = torch.cat((target_ids,end_id * torch.ones(bs,1)),dim=1).to(torch.int32)

    model = Model(embedding_dim,hidden_size,num_classes,source_vocab_size,target_vocab_size,start_id,end_id)

    probs,logits = model(input_squence_ids,target_ids)

    print(probs)
    print(logits)



































