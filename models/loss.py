import torch
from torch import nn

class LCMLoss(nn.Module):
    def __init__(self, lambda_: float = 1):
        super().__init__()
        self.lambda_ = lambda_

    def forward(
        self, 
        noise_target: torch.Tensor, 
        noise_pred1: torch.Tensor, 
        pred1: torch.Tensor, 
        noise_pred2: torch.Tensor, 
        pred2: torch.Tensor
    ):
        mseloss = ((noise_pred1-noise_target)**2).mean() + ((noise_pred2-noise_target)**2).mean()
        consistency_loss = ((pred1-pred2)**2).mean()
        
        return mseloss + self.lambda_ * consistency_loss