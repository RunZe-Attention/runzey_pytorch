import torch
a = torch.rand(4,3,28,28)

print("示例1:",a.index_select(0,torch.tensor([0,2])).shape)
print("示例2:",a.index_select(2,torch.arange(0,8)).shape)
print("示例3:",a[...].shape)
print("示例4:",a[0,...].shape)
print("示例5:",a[:,1,...].shape)


print("------------------------------------------------------")

x = torch.randn(3,4)
print(x)
mask = x.ge(0.5)
print(mask)
print(torch.masked_select(x,mask))
print(torch.masked_select(x,mask).shape)





