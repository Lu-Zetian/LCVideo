import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from accelerate import Accelerator
from diffusers import DiffusionPipeline
from diffusers.schedulers import LCMScheduler
from diffusers.optimization import get_scheduler
from einops import rearrange
from omegaconf import OmegaConf
from tqdm.auto import tqdm

import os
import argparse
from multiprocessing import cpu_count

from data.dataset import WebVid
from models.unet_video import UNetVideo
from models.loss import LCMLoss
from utils.denoise_utils import predicted_original


def main(args):
    config = OmegaConf.load(args.config)
    
    # Data
    train_dataset = WebVid(
        csv_path=config.data.csv_path, 
        video_folder=config.data.video_folder,
        sample_size=(config.data.height, config.data.width), 
        sample_stride=config.data.frame_stride, 
        sample_n_frames=config.data.num_frames,
        is_image=False,
    )
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True, 
        num_workers=config.data.dataloader_num_worker if hasattr(config.data, "dataloader_num_worker") else cpu_count(),
        drop_last=True,
    )
    
    # Model initialization
    pipe = DiffusionPipeline.from_pretrained(config.pretrains.sd_pretrains_folder)
    pipe.load_lora_weights(config.pretrains.lcm_pretrains_path)
    unet = UNetVideo(
        pipe.unet, 
        cross_attention_dim=pipe.text_encoder.config.hidden_size, 
        max_num_frames=config.data.max_num_frames
    )
    vae = pipe.vae
    text_encoder = pipe.text_encoder
    tokenizer = pipe.tokenizer
    noise_scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    
    alpha_schedule = torch.sqrt(noise_scheduler.alphas_cumprod)
    sigma_schedule = torch.sqrt(1 - noise_scheduler.alphas_cumprod)
    
    del pipe
    
    vae.enable_slicing()
    vae.enable_tiling()
    
    if config.load_checkpoint:
        unet.load_state_dict(torch.load(config.checkpoint_path), strict=False)
        
    if config.enable_xformers_memory_efficient_attention:
        unet.enable_xformers_memory_efficient_attention()
    
    # Freeze parameters
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    
    if config.enable_gradient_checkpointing:
        unet.enable_gradient_checkpointing()
    
    # Trainable parameters
    for j in range(len(unet.down_blocks)):
        if hasattr(unet.down_blocks[j], "frame_attentions"):
            for i in range(len(unet.down_blocks[j].frame_attentions)):
                unet.down_blocks[j].frame_attentions[i].requires_grad_(True)
                
    for i in range(len(unet.mid_block.frame_attentions)):
        unet.mid_block.frame_attentions[i].requires_grad_(True)
        
    for j in range(len(unet.up_blocks)):
        if hasattr(unet.up_blocks[j], "frame_attentions"):
            for i in range(len(unet.up_blocks[j].frame_attentions)):
                unet.up_blocks[j].frame_attentions[i].requires_grad_(True)
                
    # Move to device
    vae = vae.to(config.device)
    text_encoder = text_encoder.to(config.device)
    alpha_schedule = alpha_schedule.to(config.device)
    sigma_schedule = sigma_schedule.to(config.device)

    # Training initialization
    trainable_params = list(filter(lambda p: p.requires_grad, unet.parameters()))
    optimizer = AdamW(
        trainable_params,
        lr=config.optimizer.learning_rate,
        betas=(config.optimizer.adam_beta1, config.optimizer.adam_beta2),
        weight_decay=config.optimizer.adam_weight_decay,
        eps=config.optimizer.adam_epsilon,
    )
    lr_scheduler = get_scheduler(
        name=config.lr_scheduler_name, 
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=(len(train_dataloader) * config.num_epochs),
    )
    loss_fn = LCMLoss()
    
    # Training Loop
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision, 
        gradient_accumulation_steps=config.gradient_accumulation_steps, 
    )
    
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(unet, optimizer, train_dataloader, lr_scheduler)
    
    global_step = 0
    
    for epoch in range(config.num_epochs):
        unet.train()
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)

        progress_bar.set_description(f"Epoch {epoch}")
        
        for step, batch in enumerate(train_dataloader):
            videos, text = batch["videos"], batch["text"]
            video_length = videos.shape[1]
            with torch.no_grad():
                videos = rearrange(videos, "b f c h w -> (b f) c h w")
                latent_videos = vae.encode(videos).latent_dist
                latent_videos = latent_videos.sample()
                latent_videos = latent_videos * vae.config.scaling_factor
                latent_videos = rearrange(latent_videos, "(b f) c h w -> b c f h w", f=video_length)
                
                prompt_ids = tokenizer(
                    text, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
                ).input_ids.to(config.device)
                encoder_hidden_states = text_encoder(prompt_ids)[0]
                
            noise = torch.randn_like(latent_videos)
            timesteps1 = torch.randint(
                0, 
                noise_scheduler.config.num_train_timesteps, 
                (config.batch_size,), 
                device=latent_videos.device,
                dtype=torch.int64, 
            )
            timesteps2 = torch.randint(
                0, 
                noise_scheduler.config.num_train_timesteps, 
                (config.batch_size,), 
                device=latent_videos.device,
                dtype=torch.int64, 
            )
            noisy_latent_videos1 = noise_scheduler.add_noise(latent_videos, noise, timesteps1)
            noisy_latent_videos2 = noise_scheduler.add_noise(latent_videos, noise, timesteps2)
        
            with accelerator.accumulate(unet):
                noise_pred1 = unet(noisy_latent_videos1, timesteps1, encoder_hidden_states, return_dict=False)[0]
                noise_pred2 = unet(noisy_latent_videos2, timesteps2, encoder_hidden_states, return_dict=False)[0]
                x_0_pred1 = predicted_original(noise_pred1, timesteps1, noisy_latent_videos1, alpha_schedule, sigma_schedule)
                x_0_pred2 = predicted_original(noise_pred2, timesteps2, noisy_latent_videos2, alpha_schedule, sigma_schedule)
                
                loss = loss_fn(noise, noise_pred1, x_0_pred1, noise_pred2, x_0_pred2)
                accelerator.backward(loss)
                
                accelerator.clip_grad_norm_(unet.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            global_step += 1
            
        # Save checkpoint
        if accelerator.is_main_process:
            if config.output_dir is not None:
                os.makedirs(config.output_dir, exist_ok=True)
            if (epoch+1) % config.checkpoint_epochs == 0:
                save_path = os.path.join(config.output_dir, f"checkpoint-epoch-{epoch+1}.pth")
                state_dict = unet.state_dict()
                for param_tensor in unet.state_dict():
                    if "frame_attentions" not in param_tensor:
                        del state_dict[param_tensor]
                torch.save(state_dict, save_path)
    
    # Save model after training
    save_path = os.path.join(config.output_dir, config.output_name)
    state_dict = unet.state_dict()
    for param_tensor in unet.state_dict():
        if "frame_attentions" not in param_tensor:
            del state_dict[param_tensor]
        if "pos_embed" in param_tensor:
            del state_dict[param_tensor]
    torch.save(state_dict, save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/training/training_config.yaml")
    args = parser.parse_args()
    
    main(args)