import torch
from diffusers import DiffusionPipeline, TextToVideoSDPipeline
from diffusers.schedulers import LCMScheduler
from omegaconf import OmegaConf

import argparse

from models.unet_video import UNetVideo
from utils.video_utils import save_videos, output2video

def main(args):
    config = OmegaConf.load(args.config)
    
    # Initialize pipeline
    pipe = DiffusionPipeline.from_pretrained(config.pretrains.sd_pretrains_folder)
    pipe.load_lora_weights(config.pretrains.lcm_pretrains_path)
    pipe = TextToVideoSDPipeline(
        vae=pipe.vae,
        text_encoder=pipe.text_encoder,
        tokenizer=pipe.tokenizer,
        unet=UNetVideo(
            pipe.unet, 
            cross_attention_dim=pipe.text_encoder.config.hidden_size,
            max_num_frames=config.video.num_frames, 
        ),
        scheduler=LCMScheduler.from_config(pipe.scheduler.config),
    )
    pipe.unet.load_state_dict(torch.load(config.pretrains.lcvideo_checkpoint_path), strict=False)

    # Efficient settings
    pipe.enable_vae_slicing()
    pipe.enable_vae_tiling()
    pipe.enable_model_cpu_offload()
    pipe.enable_xformers_memory_efficient_attention()


    video = pipe(
        prompt=config.video.prompt, 
        height=config.video.height,
        width=config.video.width,
        num_frames=config.video.num_frames,
        num_inference_steps=config.video.num_inference_steps, 
        guidance_scale=0.0, 
        output_type="pt"
    ).frames

    video = output2video(video)

    video = video.cpu()

    save_videos(video, config.video.output_path, fps=config.video.video_fps)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/inference/inference_config.yaml")
    args = parser.parse_args()
    
    main(args)