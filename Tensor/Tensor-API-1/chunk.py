import torch

# chunk
chunk_tesor = torch.rand([3,2])
print(chunk_tesor)
a = torch.chunk(chunk_tesor,chunks=2)
print(a)
b = torch.chunk(chunk_tesor,chunks=2,dim=1)
print(b)

'''
tensor([[0.7789, 0.4336],
        [0.5792, 0.7371],
        [0.9690, 0.2642]])
        
(tensor([[0.7789, 0.4336],
        [0.5792, 0.7371]]),
 tensor([[0.9690, 0.2642]]))
        
(tensor([[0.7789],
        [0.5792],
        [0.9690]]), 
 tensor([[0.4336],
        [0.7371],
        [0.2642]]))
'''

