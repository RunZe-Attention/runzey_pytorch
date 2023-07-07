import torch


t = torch.tensor([[1,2],[3,4]])
#print(t)

b = torch.gather(input=t,dim=0,index=torch.tensor([[0,1],[1,0]]))


x = torch.randn(5,9,2)
print(torch.numel(x))


def create_1d_absolute_sincos_embeddings(n_pos_vec, dim):
    assert dim % 2 == 0, "wrong dimension"
    positional_embedding = torch.zeros(torch.numel(n_pos_vec), dim, dtype=torch.float)
    omega = torch.arange(dim//2,dtype=torch.float)
    omega /= dim/2
    omega = 1./ (10000**omega)
    out1 = n_pos_vec[:, None]
    out2 = omega[None, :]
    out = out1@out2
    emb_sin = torch.sin(out)
    emb_cos = torch.cos(out)
    positional_embedding[:, 0::2] = emb_sin
    positional_embedding[:, 1::2] = emb_cos
    return positional_embedding

def create_1d_absolute_trainable_embeddings(n_pos_vec, dim):
    positional_embedding = torch.nn.Embedding(torch.numel(n_pos_vec), dim)
    torch.nn.init.constant(positional_embedding.weight,0)
    return positional_embedding






if __name__ == '__main__':
    n_pos = 8
    dim = 6
    n_pos_vec = torch.arange(n_pos,dtype=torch.float)

    y = create_1d_absolute_sincos_embeddings(n_pos_vec,dim)

    print(y)



