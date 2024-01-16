import torch
from torch import nn
from diffusers.models.lora import LoRACompatibleConv
from diffusers.models.resnet import ResnetBlock2D
from einops import rearrange, repeat

class FlattenConv(nn.Module):
    def __init__(self, conv: nn.Conv2d | LoRACompatibleConv):
        super().__init__()
        self.conv = conv
    
    def forward(self, x, scale=None):
        video_length = x.shape[2]

        x = rearrange(x, "b c f h w -> (b f) c h w")
        x = self.conv(x) if scale is None else self.conv(x, scale)
        x = rearrange(x, "(b f) c h w -> b c f h w", f=video_length)

        return x
    
    
class FlattenResnetBlock2D(ResnetBlock2D):
    def __init__(self, resnet: ResnetBlock2D):
        self.__dict__.update(resnet.__dict__)
        self.load_state_dict(resnet.state_dict())
        
    def forward(
        self,
        input_tensor: torch.FloatTensor,
        temb: torch.FloatTensor,
        scale: float = 1.0,
    ) -> torch.FloatTensor:
        video_length = input_tensor.shape[2]
        input_tensor = rearrange(input_tensor, "b c f h w -> (b f) c h w")
        temb = repeat(temb, 'b c -> (b f) c', f=video_length)
        output_tensor = super().forward(input_tensor, temb, scale)
        output_tensor = rearrange(output_tensor, "(b f) c h w -> b c f h w", f=video_length)
        return output_tensor