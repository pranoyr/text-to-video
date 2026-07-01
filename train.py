import torch
import torchvision.transforms as T
from torch.utils.data import Dataset
from diffusers import AutoencoderKLCosmos

from lapflow import LapFlow, LapFlowDiT

from rectified_flow import Trainer


from datasets import load_dataset

ds = load_dataset("Max-Ploter/detection-moving-mnist-easy")


import torch.nn.functional as F
from transformers import CLIPTokenizer, CLIPTextModel

class MovingMNISTDataset(Dataset):
    def __init__(self, image_size, frames=17, cond_dim=512):
        self.image_size = image_size
        self.frames = frames
        self.cond_dim = cond_dim
        self.dataset = ds["train"]
        
        # Initialize CLIP tokenizer and text model for text-to-video conditioning
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        self.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").eval()
        for p in self.text_encoder.parameters():
            p.requires_grad = False

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        video_val = item["video"][:self.frames]
        
        video_tensor = torch.tensor(video_val, dtype=torch.float32) / 255.0
        
        video_tensor = video_tensor.unsqueeze(1).repeat(1, 3, 1, 1)
        
        video_tensor = F.interpolate(video_tensor, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False)
    
        video = video_tensor.permute(1, 0, 2, 3)
        
        # Pad with black frames if there are fewer than self.frames frames
        if video.shape[1] < self.frames:
            padding = torch.zeros(3, self.frames - video.shape[1], self.image_size, self.image_size)
            video = torch.cat([video, padding], dim=1)
            
        # Extract unique digits present in this video from dataset labels
        raw_labels = item.get("labels", [])
        unique_digits = sorted(list(set([
            int(d) for frame_labels in raw_labels 
            if isinstance(frame_labels, (list, tuple)) 
            for d in frame_labels
        ] + [
            int(d) for d in raw_labels 
            if isinstance(d, (int, float))
        ])))
        
        if len(unique_digits) > 0:
            prompt_str = f"A video showing handwritten digits {', '.join(map(str, unique_digits))} moving and bouncing."
        else:
            prompt_str = "A video showing handwritten digits moving and bouncing."
            
        # Encode text prompt into embedding tensor of shape (sequence_length, cond_dim)
        inputs = self.tokenizer(prompt_str, padding="max_length", max_length=16, truncation=True, return_tensors="pt")
        with torch.no_grad():
            text_embed = self.text_encoder(inputs.input_ids)[0].squeeze(0).to(torch.float32)  # Shape: (16, 512)
            
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


vae = AutoencoderKLCosmos.from_pretrained(
    "nvidia/Cosmos-1.0-Tokenizer-CV8x8x8",
    subfolder="vae",
    torch_dtype=torch.float32,
).to(device)
vae.eval()
for p in vae.parameters():
    p.requires_grad = False


vae_scale_factor = vae.config.scaling_factor   # ~1.0

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
    cfg_scale=3,
    vae=vae,
    vae_scale_factor=vae_scale_factor
).to(device)


from PIL import Image
import torchvision.utils as tv_utils

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
        batch_size=8,
        learning_rate=1e-4,
        num_train_steps=10000000,
        save_results_every=1000,
        checkpoint_every=500000000000,
        grad_accum_every = 4,
        use_ema=True,
        ema_kwargs={'beta': 0.9999},
        save_sample_fn=save_video
    )

    trainer()


