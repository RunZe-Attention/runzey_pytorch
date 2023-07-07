import torch
import torch.nn as nn


class GaussianModel(nn.Module):
    def __init__(self):
        super(GaussianModel, self).__init__()
        self.register_parameter('mean',nn.parameter(torch.zeros(1),requires_grad=True))
        self.pdf = torch.distributions.Normal(self.state_dict()['mean'],torch.tensor([1.0]))

    def forward(self,x):
        return -self.pdf.log_prob(x)

model = GaussianModel()


