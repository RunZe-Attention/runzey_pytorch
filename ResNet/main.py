import timm
resnet50 = timm.create_model("resnet50")
#print(resnet50)




import torch
x = torch.randn(4,5,6)

m = torch.nn.BatchNorm1d(5)

print(m.weight)
print(m.bias)

output= m(x)
print(x)
print(output)

a = [2.0995,  1.9884,  0.3236, -0.8745,  0.2130, -0.3656,1.0391, -0.0386, -1.5195, -0.9822, -1.5191,  0.3380,0.0687,  0.0036, -1.1233,  1.1001, -0.4856,  0.5439,-1.5955,  1.3951, -0.2389, -0.4062,  0.2302, -0.1940]

sum = 0
lenn = len(a)
for i in range(lenn):
    sum = sum + a[i]

print(sum)

print(output.mean(dim=1))
