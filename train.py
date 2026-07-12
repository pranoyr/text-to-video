import os
from pathlib import Path

# Configure all dataset/model downloads to use the server target directory
SERVER_CACHE_DIR = "/mnt/h200_disk/pranoy/datasets/vid"
os.environ.setdefault("HF_HOME", SERVER_CACHE_DIR)
os.environ.setdefault("HF_DATASETS_CACHE", SERVER_CACHE_DIR)

import torch
import torchvision.transforms as T
from torch.utils.data import Dataset
from diffusers import AutoencoderKLCosmos
from lapflow import LapFlow, LapFlowDiT, Trainer

from PIL import Image
import torchvision.utils as tv_utils

from datasets import load_dataset

import torch.nn.functional as F
from transformers import CLIPTokenizer, CLIPTextModel

from dataloader import WebVid2MDataset, WebVidDataset


use_vae = True

if use_vae:
    IMG_SIZE = 256
    kwargs = dict(
        base_image_size = IMG_SIZE // 8,
        channels = 16, # Cosmos VAE produces 16-channel latents
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

# Active Dataset: Local folder of MP4 files with hardcoded OpenVid auto-download links:
dataset = WebVid2MDataset(
    image_size=IMG_SIZE,
    video_folder="/mnt/h200_disk/pranoy/datasets/vid/videos",
    metadata_file="/mnt/h200_disk/pranoy/datasets/vid/OpenVid-1M.csv",
    cache_dir="/mnt/h200_disk/pranoy/datasets/vid",
    zip_url="https://huggingface.co/datasets/nkp37/OpenVid-1M/resolve/main/OpenVid_part0.zip",
    csv_url="https://huggingface.co/datasets/nkp37/OpenVid-1M/resolve/main/data/train/OpenVid-1M.csv",
)



vae = AutoencoderKLCosmos.from_pretrained(
    "nvidia/Cosmos-1.0-Tokenizer-CV8x8x8",
    subfolder="vae",
    torch_dtype=torch.float32,
).to(device)
vae.eval()
for p in vae.parameters():
    p.requires_grad = False


model = LapFlowDiT(
    **kwargs,
    patch_size=2,
    dim=640,
    depth=16,
    heads=10,
    dim_head=64,
    mlp_dim=2560,
    cond_as_labels=False, # Now accepting text embeddings
    dim_cond=512          # Text embedding dimension
)

lap_flow = LapFlow(
    model=model,
    normalize_data_fn=lambda t: (t * 2) - 1,
    unnormalize_data_fn=lambda t: (t + 1) * 0.5,
    cfg_scale=3,
    vae=vae
).to(device)


def save_video(tensor, path):
    frames = []
    for t in range(tensor.shape[2]):
        frame_t = tensor[:, :, t, :, :] # shape: (B, C, H, W)
        grid_t = tv_utils.make_grid(frame_t, nrow=4)
        ndarr = grid_t.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        im = Image.fromarray(ndarr)
        frames.append(im)
    
    gif_path = str(path).replace('.png', '.gif')
    frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=100, loop=0)
    print(f"Saved sample video to {gif_path}")


if __name__ == '__main__':

    trainer = Trainer(
        lap_flow,
        dataset=dataset,
        batch_size=16,
        learning_rate=1e-4,
        num_train_steps=10000000,
        save_results_every=1000,
        checkpoint_every=500000000000,
        grad_accum_every = 1,
        use_ema=True,
        ema_kwargs={'beta': 0.9999},
        save_sample_fn=save_video
    )

    trainer()


