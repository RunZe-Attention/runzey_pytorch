import torch
import torch
x = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8,9,10,11,12,13,14,15,16]).view(4, 4)

print(x)



#print(torch.roll(x,1,0))

#print(torch.roll(x,3,0))

a1 = torch.roll(x,shifts=(1),dims=(1))
print(a1)

a2 = torch.roll(a1,shifts=(-1),dims=())






