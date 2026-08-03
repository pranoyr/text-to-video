import os
from pathlib import Path

import torch
import torchvision.utils as tv_utils
from PIL import Image
from diffusers import AutoencoderKLCosmos
from einops import rearrange

from lapflow_pytorch.lapflow import LapFlow, LapFlowDiT, Trainer
from lapflow_pytorch.dataloader import MSRVTTDataset, worker_init_fn
from lapflow_pytorch.xm_wrapper import XMWrapper

import argparse
import wandb

use_vae = True

if use_vae:
    IMG_SIZE = 256
    kwargs = dict(
        base_image_size = IMG_SIZE // 8,
        channels = 16,
        num_scales = 2
    )
else:
    IMG_SIZE = 64
    kwargs = dict(
        base_image_size = 64,
        channels = 3,
        num_scales = 2
    )

is_cuda_available = torch.cuda.is_available()
device = torch.device('cuda' if is_cuda_available else 'cpu')

dataset = MSRVTTDataset(
    video_dir="/mnt/h200_disk/pranoy/datasets/msrvtt_videos",
    metadata_path="/mnt/h200_disk/pranoy/datasets/msrvtt_metadata.csv",
    num_frames=16,
    frame_stride=4,
    image_size=IMG_SIZE
)

vae = AutoencoderKLCosmos.from_pretrained(
    "nvidia/Cosmos-1.0-Tokenizer-CV8x8x8",
    subfolder="vae",
    torch_dtype=torch.float32,
)

vae.eval()

for p in vae.parameters():
    p.requires_grad = False

from config.models import MODEL_CONFIGS

#  "small", "medium", "large", "1B"
selected_model_size = "small" 
model_kwargs = MODEL_CONFIGS[selected_model_size]

model = LapFlowDiT(
    **kwargs,
    **model_kwargs
)

lap_flow = LapFlow(
    model=model,
    normalize_data_fn=lambda t: (t * 2) - 1, # Normalizes dataset's [0,1] -> [-1,1] for VAE
    unnormalize_data_fn=lambda t: (t + 1) * 0.5,
    cfg_scale=3,
    vae=vae
).to(device)

def save_video(tensor, path):
    frames = []
    num_temporal_frames = tensor.shape[2]
    
    for t in range(num_temporal_frames):
        frame_t = tensor[:, :, t, :, :]
        grid_t = tv_utils.make_grid(frame_t, nrow=4)
        ndarr = rearrange(grid_t.mul(255).add_(0.5).clamp_(0, 255), 'c h w -> h w c').to('cpu', torch.uint8).numpy()
        im = Image.fromarray(ndarr)
        frames.append(im)
    
    gif_path = str(path).replace('.png', '.gif')
    frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=100, loop=0)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train LapFlow DiT")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from (e.g. checkpoint.pt)")
    parser.add_argument("--pretrain", type=str, default=None, help="Path to checkpoint to initialize weights from (starts from scratch)")
    parser.add_argument("--candidates", type=int, default=1, help="Number of candidates for explorative modeling")
    args = parser.parse_args()

    if args.candidates > 1:
        lap_flow = XMWrapper(lap_flow, candidates=args.candidates).to(device)

    wandb.init(project="world-2-video-lapflow")

    trainer = Trainer(
        lap_flow,
        dataset=dataset,
        batch_size=16,
        num_workers=8,
        learning_rate=1e-4,
        num_train_steps=2000000,
        save_results_every=1000,
        checkpoint_every=50000,
        grad_accum_every=1,
        use_ema=True,
        ema_kwargs={'beta': 0.9999},
        save_sample_fn=save_video,
        sample_kwargs={'steps': 80}
    )
    if args.resume:
        print(f"Resuming training from checkpoint: {args.resume}")
        trainer.load(args.resume, load_training_state=True)
    elif args.pretrain:
        print(f"Pretraining from weights in: {args.pretrain}. Starting from scratch.")
        trainer.load(args.pretrain, load_training_state=False)

    trainer()