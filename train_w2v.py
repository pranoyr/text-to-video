import torch
import torchvision.transforms as T
from torch.utils.data import Dataset
from diffusers import AutoencoderKLCosmos

from lapflow import LapFlow, LapFlowDiT

from rectified_flow import Trainer


from datasets import load_dataset

ds = load_dataset("Max-Ploter/detection-moving-mnist-easy")


import torch.nn.functional as F

class MovingMNISTDataset(Dataset):
    def __init__(self, image_size, frames=17, cond_dim=512):
        self.image_size = image_size
        self.frames = frames
        self.cond_dim = cond_dim
        self.dataset = ds["train"]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        # video is a list of frames, each frame is a 2D list of shape (128, 128)
        video_val = item["video"][:self.frames]
        
        # Convert list of lists of lists to a PyTorch tensor (T, H, W) in [0., 1.] range
        video_tensor = torch.tensor(video_val, dtype=torch.float32) / 255.0
        
        # Add channel dimension to get (T, 1, H, W) and repeat to get 3 channels: (T, 3, H, W)
        video_tensor = video_tensor.unsqueeze(1).repeat(1, 3, 1, 1)
        
        # Resize to (self.image_size, self.image_size)
        video_tensor = F.interpolate(video_tensor, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False)
        
        # Permute to (C, T, H, W) shape
        video = video_tensor.permute(1, 0, 2, 3)
        
        # Pad with black frames if there are fewer than self.frames frames
        if video.shape[1] < self.frames:
            padding = torch.zeros(3, self.frames - video.shape[1], self.image_size, self.image_size)
            video = torch.cat([video, padding], dim=1)
            
        # Return video and a dummy/zero text embedding of size cond_dim
        text_embed = torch.zeros(self.cond_dim)
        return video, text_embed




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

dataset = MovingMNISTDataset(image_size=IMG_SIZE)

if use_vae:
    config = AutoencoderKLCosmos.load_config("nvidia/Cosmos3-Nano", subfolder="vae")
    config["in_channels"] = 3
    config["out_channels"] = 3
    vae = AutoencoderKLCosmos.from_config(config).to(device, dtype=torch.float32)
    for param in vae.parameters():
        param.requires_grad = False
else:
    vae = None

model = LapFlowDiT(
    **kwargs,
    patch_size=2,
    dim=256,
    depth=6,
    heads=8,
    mlp_dim=1024,
    accept_cond=True,
    cond_as_labels=False, # Now accepting text embeddings
    dim_cond=512          # Text embedding dimension
)

lap_flow = LapFlow(
    model=model,
    normalize_data_fn=lambda t: (t * 2) - 1,
    unnormalize_data_fn=lambda t: (t + 1) * 0.5,
    cfg_scale=3.0,
    vae=vae,
    vae_scale_factor=0.18215
).to(device)


if __name__ == '__main__':

    trainer = Trainer(
        lap_flow,
        dataset=dataset,
        batch_size=8,
        learning_rate=1e-4,
        num_train_steps=100000,
        save_results_every=1000,
        checkpoint_every=5000,
        grad_accum_every = 4,
        use_ema=True,
        ema_kwargs={'beta': 0.9999}
    )

    trainer()

