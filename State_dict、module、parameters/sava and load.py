import torch
import torchvision.models as models

# 保存模型的的权重
model = models.vgg16(pretrained=True)
torch.save(model.state_dict(),'model_weights.pth')

# 从pth文件中加载模型再加载模型的weights
model = models.vgg16()
checkpoint = torch.load("model_weights.pth")
model.load_state_dict(checkpoint)
model.eval()

class Test(torch.nn.Module):
    def __init__(self):
        super(Test,self).__init__()
        self.linear1 = torch.nn.Linear(2,3)
        self.linear2 = torch.nn.Linear(3,4)
        self.batch_norm = torch.nn.BatchNorm2d(4)


test_module = Test()
leaner1_weight = test_module._modules['linear1'].weight
print(leaner1_weight)
print(leaner1_weight.dtype)

test_module.to(torch.double)
print(leaner1_weight)
print(leaner1_weight.dtype)

test_module.to(torch.float32)
print(leaner1_weight)
print(leaner1_weight.dtype)
print("------------------------------\n")


print(test_module._parameters)
print(test_module._buffers)


print(test_module.state_dict()['linear1.weight'])

for p in test_module.parameters(): #还是使用parameters()吧 别使用 _parameters
    print(p)


for p in test_module.named_parameters(): #还是使用parameters()吧 别使用 _parameters
    print(p)

print("----------------------------------~~~~\n")
for i  in test_module.children():
    print(i)

for i  in test_module.named_children():
    print(i)


print("~~~~~~~~~~~~~~~named_modules~~~~~~~~~~~~~~~~\n")
for i in test_module.named_modules():
    print(i)
    print("\n")
print("~~~~~~~~~~~~~~~modules~~~~~~~~~~~~~~~~\n")
for i in test_module.modules():
    print(i)
    print("\n")







