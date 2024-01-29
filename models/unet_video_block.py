import torch
from torch import nn
from diffusers.models.unet_2d_blocks import Downsample2D, Upsample2D, CrossAttnDownBlock2D, CrossAttnUpBlock2D, UNetMidBlock2DCrossAttn
from diffusers.utils import is_torch_version
from diffusers.utils.torch_utils import apply_freeu
from einops import rearrange

from typing import Optional, Dict, Tuple, Any

from models.attention import CrossFrameTransformer2DModel

class FlattenDownsample2D(Downsample2D):
    def __init__(self, downsample: Downsample2D):
        self.__dict__.update(downsample.__dict__)
        self.load_state_dict(downsample.state_dict())
        
    def forward(self, hidden_states: torch.FloatTensor, scale: float = 1.0) -> torch.FloatTensor:
        video_length = hidden_states.shape[2]
        hidden_states = rearrange(hidden_states, "b c f h w -> (b f) c h w")
        hidden_states = super().forward(hidden_states, scale=scale)
        hidden_states = rearrange(hidden_states, "(b f) c h w -> b c f h w", f=video_length)
        return hidden_states
    
    
class FlattenUpsample2D(Upsample2D):
    def __init__(self, upsample: Upsample2D):
        self.__dict__.update(upsample.__dict__)
        self.load_state_dict(upsample.state_dict())
        
    def forward(
        self,
        hidden_states: torch.FloatTensor,
        output_size: Optional[int] = None,
        scale: float = 1.0,
    ) -> torch.FloatTensor:
        video_length = hidden_states.shape[2]
        hidden_states = rearrange(hidden_states, "b c f h w -> (b f) c h w")
        hidden_states = super().forward(hidden_states, output_size=output_size, scale=scale)
        hidden_states = rearrange(hidden_states, "(b f) c h w -> b c f h w", f=video_length)
        return hidden_states
    
    
class CrossAttnVideoDownBlock2D(CrossAttnDownBlock2D):
    def __init__(
        self, 
        cross_attn: CrossAttnDownBlock2D, 
        cross_attention_dim: int = 768, 
        attention_head_dim: int = None, 
        max_num_frames: int = 32,
    ):
        self.__dict__.update(cross_attn.__dict__)
        self.load_state_dict(cross_attn.state_dict())
        self.frame_attentions = nn.ModuleList()
        for i in range(len(self.attentions)):
            self.frame_attentions.append(CrossFrameTransformer2DModel(
                in_channels=self.attentions[i].in_channels,
                encoder_hidden_states_dim=cross_attention_dim,
                num_attention_heads=self.attentions[i].num_attention_heads,
                attention_head_dim=attention_head_dim or self.attentions[i].attention_head_dim,
                num_positional_embeddings=max_num_frames,
            ))
            
    def forward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor, 
        num_frames: int,
        temb: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        additional_residuals: Optional[torch.FloatTensor] = None,
    ) -> Tuple[torch.FloatTensor, Tuple[torch.FloatTensor, ...]]:
        output_states = ()

        lora_scale = cross_attention_kwargs.get("scale", 1.0) if cross_attention_kwargs is not None else 1.0

        blocks = list(zip(self.resnets, self.attentions, self.frame_attentions))

        for i, (resnet, attn, frame_attn) in enumerate(blocks):
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(resnet),
                    hidden_states,
                    temb,
                    **ckpt_kwargs,
                )
                hidden_states = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                    return_dict=False,
                )[0]
            else:
                hidden_states = resnet(hidden_states, temb, scale=lora_scale)
                hidden_states = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                    return_dict=False,
                )[0]
            
            hidden_states = frame_attn(
                hidden_states, 
                encoder_hidden_states=encoder_hidden_states,
                num_frames=num_frames, 
                cross_attention_kwargs=cross_attention_kwargs,
            )

            # apply additional residuals to the output of the last pair of resnet and attention blocks
            if i == len(blocks) - 1 and additional_residuals is not None:
                hidden_states = hidden_states + additional_residuals

            output_states = output_states + (hidden_states,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states, scale=lora_scale)

            output_states = output_states + (hidden_states,)

        return hidden_states, output_states
        

class CrossAttnVideoUpBlock2D(CrossAttnUpBlock2D):
    def __init__(
        self, 
        cross_attn: CrossAttnUpBlock2D, 
        cross_attention_dim: int = 768, 
        attention_head_dim: int = None, 
        max_num_frames: int = 32,
    ):
        self.__dict__.update(cross_attn.__dict__)
        self.load_state_dict(cross_attn.state_dict())
        self.frame_attentions = nn.ModuleList()
        for i in range(len(self.attentions)):
            self.frame_attentions.append(CrossFrameTransformer2DModel(
                in_channels=self.attentions[i].in_channels,
                encoder_hidden_states_dim=cross_attention_dim,
                num_attention_heads=self.attentions[i].num_attention_heads,
                attention_head_dim=attention_head_dim or self.attentions[i].attention_head_dim,
                num_positional_embeddings=max_num_frames,
            ))
    
    def forward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor,
        res_hidden_states_tuple: Tuple[torch.FloatTensor, ...],
        num_frames: int,
        temb: Optional[torch.FloatTensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        upsample_size: Optional[int] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        lora_scale = cross_attention_kwargs.get("scale", 1.0) if cross_attention_kwargs is not None else 1.0
        is_freeu_enabled = (
            getattr(self, "s1", None)
            and getattr(self, "s2", None)
            and getattr(self, "b1", None)
            and getattr(self, "b2", None)
        )

        for resnet, attn, frame_attn in zip(self.resnets, self.attentions, self.frame_attentions):
            # pop res hidden states
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]

            # FreeU: Only operate on the first two stages
            if is_freeu_enabled:
                hidden_states, res_hidden_states = apply_freeu(
                    self.resolution_idx,
                    hidden_states,
                    res_hidden_states,
                    s1=self.s1,
                    s2=self.s2,
                    b1=self.b1,
                    b2=self.b2,
                )

            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(resnet),
                    hidden_states,
                    temb,
                    **ckpt_kwargs,
                )
                hidden_states = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                    return_dict=False,
                )[0]
            else:
                hidden_states = resnet(hidden_states, temb, scale=lora_scale)
                hidden_states = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                    return_dict=False,
                )[0]
                
            hidden_states = frame_attn(
                hidden_states, 
                encoder_hidden_states=encoder_hidden_states,
                num_frames=num_frames, 
                cross_attention_kwargs=cross_attention_kwargs,
            )

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, upsample_size, scale=lora_scale)

        return hidden_states
        
        
class CrossAttnVideoMidBlock2D(UNetMidBlock2DCrossAttn):
    def __init__(
        self, 
        cross_attn: UNetMidBlock2DCrossAttn, 
        cross_attention_dim: int = 768, 
        attention_head_dim: int = None, 
        max_num_frames: int = 32,
    ):
        self.__dict__.update(cross_attn.__dict__)
        self.load_state_dict(cross_attn.state_dict())
        self.frame_attentions = nn.ModuleList()
        for i in range(len(self.attentions)):
            self.frame_attentions.append(CrossFrameTransformer2DModel(
                in_channels=self.attentions[i].in_channels,
                encoder_hidden_states_dim=cross_attention_dim,
                num_attention_heads=self.attentions[i].num_attention_heads,
                attention_head_dim=attention_head_dim or self.attentions[i].attention_head_dim,
                num_positional_embeddings=max_num_frames,
            ))
            
    def forward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor,
        num_frames: int,
        temb: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        lora_scale = cross_attention_kwargs.get("scale", 1.0) if cross_attention_kwargs is not None else 1.0
        hidden_states = self.resnets[0](hidden_states, temb, scale=lora_scale)
        for attn, resnet, frame_attn in zip(self.attentions, self.resnets[1:], self.frame_attentions):
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                hidden_states = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                    return_dict=False,
                )[0]
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(resnet),
                    hidden_states,
                    temb,
                    **ckpt_kwargs,
                )
            else:
                hidden_states = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                    return_dict=False,
                )[0]
                hidden_states = resnet(hidden_states, temb, scale=lora_scale)
            
            hidden_states = frame_attn(
                hidden_states, 
                encoder_hidden_states=encoder_hidden_states,
                num_frames=num_frames, 
                cross_attention_kwargs=cross_attention_kwargs,
            )

        return hidden_states
        
        