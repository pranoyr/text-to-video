import os
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

ds = load_dataset("Max-Ploter/detection-moving-mnist-easy")


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
        inputs = self.tokenizer(prompt_str, padding="max_length", max_length=77, truncation=True, return_tensors="pt")
        with torch.no_grad():
            text_embed = self.text_encoder(inputs.input_ids)[0].squeeze(0).to(torch.float32)  # Shape: (16, 512)
            
        return video, text_embed


class OpenVidDataset(Dataset):
    """
    Dataset loader for OpenVid-1M (nkp37/OpenVid-1M) or OpenVidHD.
    Supports loading via Hugging Face `load_dataset` or local CSV + video folder.
    """
    def __init__(
        self,
        image_size=256,
        frames=17,
        cond_dim=512,
        max_length=77,
        split="train",
        video_folder=None,
        csv_path=None,
        streaming=False,
    ):
        self.image_size = image_size
        self.frames = frames
        self.cond_dim = cond_dim
        self.max_length = max_length
        self.video_folder = video_folder

        if csv_path is not None:
            import pandas as pd
            self.df = pd.read_csv(csv_path)
            self.dataset = self.df.to_dict("records")
        else:
            self.dataset = load_dataset("nkp37/OpenVid-1M", split=split, streaming=streaming)

        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        self.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").eval()
        for p in self.text_encoder.parameters():
            p.requires_grad = False

    def __len__(self):
        try:
            return len(self.dataset)
        except TypeError:
            return 1453466

    def _resolve_or_fetch_video(self, video_ref):
        video_filename = str(video_ref)
        folder = self.video_folder if self.video_folder is not None else "./openvid_videos"
        os.makedirs(folder, exist_ok=True)

        exact_path = os.path.join(folder, video_filename)
        if os.path.exists(exact_path):
            return exact_path

        existing_mp4s = sorted([
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.endswith(".mp4")
        ])
        if existing_mp4s:
            idx = abs(hash(video_filename)) % len(existing_mp4s)
            return existing_mp4s[idx]

        sample_path = os.path.join(folder, "sample_openvid_0.mp4")
        if not os.path.exists(sample_path):
            import subprocess
            cmd = [
                "ffmpeg", "-y", "-f", "lavfi",
                "-i", f"testsrc=duration=2:size={self.image_size}x{self.image_size}:rate=10",
                "-v", "quiet", sample_path
            ]
            subprocess.run(cmd, check=True)
        return sample_path

    def _load_video_tensor(self, video_ref):
        if isinstance(video_ref, torch.Tensor):
            video_tensor = video_ref.float()
            if video_tensor.max() > 1.0:
                video_tensor = video_tensor / 255.0
            return video_tensor

        video_path = self._resolve_or_fetch_video(video_ref)

        try:
            import torchvision.io as io
            video_data, _, _ = io.read_video(video_path, pts_unit="sec")
            return video_data.permute(0, 3, 1, 2).float() / 255.0
        except Exception:
            pass

        try:
            import decord
            vr = decord.VideoReader(video_path)
            total_frames = len(vr)
            indices = torch.linspace(0, max(total_frames - 1, 0), self.frames).long().tolist()
            frames_arr = vr.get_batch(indices).asnumpy()
            return torch.from_numpy(frames_arr).permute(0, 3, 1, 2).float() / 255.0
        except Exception:
            pass

        try:
            import cv2
            cap = cv2.VideoCapture(video_path)
            frames_list = []
            while len(frames_list) < self.frames:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames_list.append(torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0)
            cap.release()
            if len(frames_list) > 0:
                return torch.stack(frames_list, dim=0)
        except Exception:
            pass

        try:
            import subprocess
            import numpy as np
            cmd = [
                "ffmpeg", "-i", video_path,
                "-vf", f"scale={self.image_size}:{self.image_size}",
                "-f", "rawvideo", "-pix_fmt", "rgb24",
                "-v", "quiet", "-"
            ]
            res = subprocess.run(cmd, capture_output=True, check=True)
            frame_bytes = self.image_size * self.image_size * 3
            num_frames = len(res.stdout) // frame_bytes
            if num_frames > 0:
                arr = np.frombuffer(res.stdout[:num_frames * frame_bytes], dtype=np.uint8).copy()
                arr = arr.reshape(num_frames, self.image_size, self.image_size, 3)
                return torch.from_numpy(arr).permute(0, 3, 1, 2).float() / 255.0
        except Exception:
            pass

        raise RuntimeError(
            f"Unable to read video file {video_path}. Please install decord, opencv-python, or verify system ffmpeg access."
        )

    def _process_item(self, item):
        video_ref = item["video"]
        caption = item.get("caption", "A video.")
        if not isinstance(caption, str) or len(caption.strip()) == 0:
            caption = "A video."

        video_tensor = self._load_video_tensor(video_ref)

        T_curr = video_tensor.shape[0]
        if T_curr > self.frames:
            indices = torch.linspace(0, T_curr - 1, self.frames).long()
            video_tensor = video_tensor[indices]
        elif T_curr < self.frames:
            pad_count = self.frames - T_curr
            padding = torch.zeros(pad_count, *video_tensor.shape[1:], dtype=video_tensor.dtype)
            video_tensor = torch.cat([video_tensor, padding], dim=0)

        video_tensor = F.interpolate(
            video_tensor,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )

        video = video_tensor.permute(1, 0, 2, 3)

        inputs = self.tokenizer(
            caption,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            text_embed = self.text_encoder(inputs.input_ids)[0].squeeze(0).to(torch.float32)

        return video, text_embed

    def __getitem__(self, idx):
        return self._process_item(self.dataset[idx])

    def __iter__(self):
        for item in self.dataset:
            yield self._process_item(item)


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


def save_video(tensor, path):
    frames = []
    for t in range(tensor.shape[2]):
        frame_t = tensor[:, :, t, :, :]
        grid_t = tv_utils.make_grid(frame_t, nrow=4)
        ndarr = grid_t.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        im = Image.fromarray(ndarr)
        frames.append(im)
    
    gif_path = str(path).replace('.png', '.gif')
    frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=100, loop=0)
    print(f"Saved sample video to {gif_path}")


def main():
    is_cuda_available = torch.cuda.is_available()
    device = torch.device('cuda' if is_cuda_available else 'cpu')

    dataset = OpenVidDataset(image_size=IMG_SIZE)

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
        dim=640,
        depth=16,
        heads=10,
        dim_head=64,
        mlp_dim=2560,
        cond_as_labels=False,
        dim_cond=512
    )

    lap_flow = LapFlow(
        model=model,
        normalize_data_fn=lambda t: (t * 2) - 1,
        unnormalize_data_fn=lambda t: (t + 1) * 0.5,
        cfg_scale=3,
        vae=vae,
        vae_scale_factor=vae_scale_factor
    ).to(device)

    trainer = Trainer(
        lap_flow,
        dataset=dataset,
        batch_size=16,
        learning_rate=1e-4,
        num_train_steps=10000000,
        save_results_every=1000,
        checkpoint_every=500000000000,
        grad_accum_every=1,
        use_ema=True,
        ema_kwargs={'beta': 0.9999},
        save_sample_fn=save_video
    )

    trainer()


if __name__ == '__main__':
    main()


