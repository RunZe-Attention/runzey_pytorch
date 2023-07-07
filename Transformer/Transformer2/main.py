import  torch
import torch.nn as nn
import torch.nn.functional as F


logits = torch.rand(2,3,4)
print(logits)

labels = torch.randint(0,4,(2,3))
print(labels)