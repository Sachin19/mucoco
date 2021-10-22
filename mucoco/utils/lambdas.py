import torch

class Lambda(torch.nn.Module): #multipliers for the constraints
    def __init__(self, count=1):
        super(Lambda, self).__init__()
        self.lambda_ = torch.nn.Parameter(torch.Tensor([0. for _ in range(count)]))

    def forward(self):
        return self.lambda_

    def get_loss(self, i, damp, loss):
        return (self.lambda_[i] - damp) * loss
    
    def make_positive(self):
        posmask = self.lambda_.detach() > 0.
        self.lambda_.data.copy_(self.lambda_.data * posmask.float())