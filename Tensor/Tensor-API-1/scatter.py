import torch

src = torch.arange(1,11).reshape((2,5))

index = torch.tensor([[0,1,2,0]])



re = torch.zeros((3,5),dtype=src.dtype)
re = re.scatter(dim=1,index=index,src=src)
print(re)



r1 = torch.full((2,4),2.)
r1 = r1.scatter_(dim=1,index = torch.tensor([[2],[3]]),value = 1.23,reduce='multiply')
print(r1)

r1 = torch.full((2,4),2.)
r1 = r1.scatter_(dim=1,index = torch.tensor([[2],[3]]),value = 1.23,reduce='add')
print(r1)


'''
tensor([[4, 2, 3, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]])
tensor([[2.0000, 2.0000, 2.4600, 2.0000],
        [2.0000, 2.0000, 2.0000, 2.4600]])
tensor([[2.0000, 2.0000, 3.2300, 2.0000],
        [2.0000, 2.0000, 2.0000, 3.2300]])
'''