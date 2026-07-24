import os
import pandas as pd
import torch
import numpy as np
import random
from torch.utils.data import Dataset
import torchvision.transforms as T
from decord import VideoReader, cpu
from einops import rearrange
from datasets import load_dataset
from einops import repeat

def worker_init_fn(worker_id: int):
    """Ensures deterministic yet distinct seeding across DataLoader worker processes."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

class MSRVTTDataset(Dataset):
    """
    MSR-VTT Dataset for Text-to-Video Training.
    Returns:
        frames_tensor: Tensor of shape [C, T, H, W] scaled to [0.0, 1.0].
        caption: Raw text caption string.
    """
    def __init__(
        self, 
        video_dir: str = "/mnt/h200_disk/pranoy/datasets/msrvtt_videos", 
        metadata_path: str = "/mnt/h200_disk/pranoy/datasets/msrvtt_metadata.csv", 
        num_frames: int = 16, 
        frame_stride: int = 4, 
        image_size: int = 256,
        max_retries: int = 5
    ):
        self.num_frames = num_frames
        self.frame_stride = frame_stride
        self.max_retries = max_retries
        self.video_dir = self._resolve_video_dir(video_dir)
        
        if metadata_path.endswith('.csv'):
            df = pd.read_csv(metadata_path)
        else:
            df = pd.read_json(metadata_path)

        # Filter out missing video files during initialization
        valid_rows = []
        for _, row in df.iterrows():
            v_id = str(row['video_id'])
            v_name = v_id if v_id.endswith('.mp4') else f"{v_id}.mp4"
            if os.path.isfile(os.path.join(self.video_dir, v_name)):
                valid_rows.append(row)

        self.metadata = pd.DataFrame(valid_rows).reset_index(drop=True)
        print(f"Loaded {len(self.metadata)} valid video-caption samples from {self.video_dir}")
            
        self.transform = T.Compose([
            T.Resize(image_size, antialias=True),
            T.CenterCrop(image_size)
        ])

    def _resolve_video_dir(self, target_dir: str) -> str:
        if not os.path.exists(target_dir):
            return target_dir
            
        for root, _, files in os.walk(target_dir):
            if any(f.endswith('.mp4') for f in files):
                return root
        return target_dir

    def __len__(self):
        return len(self.metadata)

    def _get_random_idx(self) -> int:
        return torch.randint(0, len(self.metadata), (1,)).item()

    def __getitem__(self, idx: int):
        current_idx = idx
        
        for attempt in range(self.max_retries):
            try:
                row = self.metadata.iloc[current_idx]
                
                video_id = str(row['video_id'])
                video_filename = video_id if video_id.endswith('.mp4') else f"{video_id}.mp4"
                video_path = os.path.join(self.video_dir, video_filename)
                caption = str(row['caption'])
                
                vr = VideoReader(video_path, ctx=cpu(0))
                total_frames = len(vr)

                required_span = (self.num_frames - 1) * self.frame_stride + 1
                
                if total_frames >= required_span:
                    max_start = total_frames - required_span
                    start_idx = torch.randint(0, max_start + 1, (1,)).item()
                    frame_indices = range(start_idx, start_idx + required_span, self.frame_stride)
                elif total_frames >= self.num_frames:
                    dynamic_stride = total_frames // self.num_frames
                    required_span = (self.num_frames - 1) * dynamic_stride + 1
                    max_start = total_frames - required_span
                    start_idx = torch.randint(0, max_start + 1, (1,)).item()
                    frame_indices = range(start_idx, start_idx + required_span, dynamic_stride)
                else:
                    frame_indices = [i % total_frames for i in range(self.num_frames)]

                frames = vr.get_batch(list(frame_indices)).asnumpy()
                
                frames_tensor = rearrange(torch.from_numpy(frames).float() / 255.0, 't h w c -> t c h w')
                frames_tensor = self.transform(frames_tensor)
                frames_tensor = rearrange(frames_tensor, 't c h w -> c t h w')

                return frames_tensor, caption

            except Exception as e:
                current_idx = self._get_random_idx()

        raise RuntimeError(f"Failed to load a valid video sample after {self.max_retries} attempts.")




class HFImageAsVideoDataset(Dataset):
    def __init__(self, image_size: int = 256, num_frames: int = 16):
        self.num_frames = num_frames
        
    
        print("Loading COCO dataset from disk cache (or downloading if first time)...")
        self.hf_dataset = load_dataset(
            "lmms-lab/COCO-Caption", 
            split="val",
            trust_remote_code=True
        )
        
        self.transform = T.Compose([
            T.Resize(image_size, antialias=True),
            T.CenterCrop(image_size),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        item = self.hf_dataset[idx]
    
        img = item['image'].convert("RGB")
        img_tensor = self.transform(img)
        
        video_tensor = repeat(img_tensor, 'c h w -> c t h w', t=self.num_frames)
        
        caption = item['answer'][0] if 'answer' in item else "a high quality detailed photograph"
        
        return video_tensor, caption