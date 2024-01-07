import torch
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
from einops import rearrange
import numpy as np

import os, io, csv, math, random
from decord import VideoReader


class WebVid(Dataset):
    def __init__(
            self,
            csv_path, video_folder,
            sample_size=256, sample_stride=4, sample_n_frames=16,
            is_image=False,
        ):
        print(f"loading annotations from {csv_path} ...")
        with open(csv_path, 'r') as csvfile:
            self.dataset = list(csv.DictReader(csvfile))[:2]
        self.length = len(self.dataset)
        print(f"data scale: {self.length}")

        self.video_folder    = video_folder
        self.sample_stride   = sample_stride
        self.sample_n_frames = sample_n_frames
        self.is_image        = is_image
        
        sample_size = tuple(sample_size) if not isinstance(sample_size, int) else (sample_size, sample_size)
        self.pixel_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize(sample_size[0]),
            transforms.CenterCrop(sample_size),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])
    
    def get_batch(self, idx):
        video_dict = self.dataset[idx]
        videoid, text, page_dir = video_dict['videoid'], video_dict['name'], video_dict['page_dir']
        
        video_dir    = os.path.join(self.video_folder, f"{videoid}.mp4")
        video_reader = VideoReader(video_dir)
        video_length = len(video_reader)
        
        if not self.is_image:
            clip_length = min(video_length, (self.sample_n_frames - 1) * self.sample_stride + 1)
            start_idx   = random.randint(0, video_length - clip_length)
            batch_index = np.linspace(start_idx, start_idx + clip_length - 1, self.sample_n_frames, dtype=int)
        else:
            batch_index = [random.randint(0, video_length - 1)]

        videos = torch.from_numpy(video_reader.get_batch(batch_index).asnumpy()).permute(0, 3, 1, 2).contiguous()
        videos = videos / 255.
        del video_reader

        if self.is_image:
            videos = videos[0]
        
        return videos, text

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        while True:
            try:
                videos, text = self.get_batch(idx)
                break

            except Exception as e:
                idx = random.randint(0, self.length-1)

        videos = self.pixel_transforms(videos)
        sample = {
            "videos": videos,
            "text": text,
        }
        return sample
