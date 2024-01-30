from torch import nn

class LCMLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, target, noise_pred1, x_0_pred1, noise_pred2, x_0_pred2):
        mseloss = ((noise_pred1-target)**2).mean() + ((noise_pred2-target)**2).mean()
        consistency_loss = ((x_0_pred1-x_0_pred2)**2).mean()
        # print(f"mseloss: {mseloss}, consistency_loss: {consistency_loss}")
        return mseloss + consistency_loss