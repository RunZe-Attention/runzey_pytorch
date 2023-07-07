import torch
import numpy as np

torch.set_default_tensor_type(torch.DoubleTensor)

np_array = np.random.normal((2,3))
print(np_array)
tensor_array = torch.tensor(np_array)
print(tensor_array)


data = [1,2,3.]
print(type(data))
print(data)
x_data = torch.tensor(data)
print(type(x_data))
print(x_data)
print(x_data.dtype)


ones = torch.ones_like(tensor_array)
print(ones)

zeros = torch.zeros_like(tensor_array)
print(zeros)

rand = torch.rand_like(tensor_array,dtype=torch.float32)
print(rand)

# 随机生成一个2*2的tensor
rand_gen = torch.rand([2,2])
print(rand_gen)
print(rand_gen.dtype)
print(rand_gen.shape)
print(rand_gen.device)

bCuda = torch.cuda.is_available()
print(bCuda)

print(torch.is_tensor(rand_gen))
print(torch.is_tensor(data))

np_array_int = [4,5,6]
tensor_array_int = torch.tensor(np_array_int)
print(tensor_array_int)
print(torch.is_floating_point(tensor_array_int))

print(torch.is_floating_point(rand_gen))

print("--------\n")
rand_gen = torch.rand([10,2,3])
print(rand_gen)
print(torch.numel(rand_gen))

print(torch.zeros(2,10))
print(torch.ones(3,10))
print(torch.ones_like(tensor_array))

a = torch.zeros([9,9],dtype=torch.int32)
print(a.dtype)
print(torch.ones_like(a).dtype)

torch.set_default_tensor_type(torch.DoubleTensor )

print("--------\n")
print(torch.arange(start=4,end=10,step=2))
#print(torch.range(start=0,end=5,step=1,dtype=torch.int32))

for i in torch.arange(10):
    print("epcho:",i)

#for i in torch.range(start=0,end=9,dtype=torch.int32):
#    print("epcho:",i)

print("-------\n")
print(torch.eye(10,20))
print(torch.full([2,2],10))

a = torch.rand([10,5])
b = torch.rand([3,5])
c = torch.cat([a,b],dim=0)
print(c)


