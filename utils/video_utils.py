import torch
import torchvision
import numpy as np
from einops import rearrange
import imageio

import os

def output2video(videos: torch.Tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    mean = torch.tensor(mean, device=videos.device).reshape(1, -1, 1, 1, 1)
    std = torch.tensor(std, device=videos.device).reshape(1, -1, 1, 1, 1)
    videos = videos.mul_(std).add_(mean)
    videos.clamp_(0, 1)
    return videos


def save_videos(videos: torch.Tensor, path: str, rescale=False, n_rows=6, fps=8):
    videos = rearrange(videos, "b c f h w -> f b c h w")
    outputs = []
    for x in videos:
        x = torchvision.utils.make_grid(x, nrow=n_rows)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
        if rescale:
            x = (x + 1.0) / 2.0  # -1,1 -> 0,1
        x = (x * 255).numpy().astype(np.uint8)
        outputs.append(x)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.mimsave(path, outputs, fps=fps)