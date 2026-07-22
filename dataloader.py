import os
import random
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from decord import VideoReader, cpu
from einops import rearrange

class MSRVTTDataset(Dataset):
    def __init__(
        self, 
        video_dir: str = "", 
        metadata_path: str = "", 
        num_frames: int = 16, 
        frame_stride: int = 4, 
        image_size: int = 256
    ):
        self.video_dir = video_dir
        self.num_frames = num_frames
        self.frame_stride = frame_stride
        
        if metadata_path.endswith('.csv'):
            self.metadata = pd.read_csv(metadata_path)
        else:
            self.metadata = pd.read_json(metadata_path)
            
        self.transform = T.Compose([
            T.Resize(image_size, antialias=True),
            T.CenterCrop(image_size)
        ])

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        
        video_id = str(row['video_id'])
        video_filename = video_id if video_id.endswith('.mp4') else f"{video_id}.mp4"
        video_path = os.path.join(self.video_dir, video_filename)
        caption = str(row['caption'])
        
        try:
            vr = VideoReader(video_path, ctx=cpu(0))
            total_frames = len(vr)
        except Exception:
            return self.__getitem__(random.randint(0, len(self.metadata) - 1))

        required_span = (self.num_frames - 1) * self.frame_stride + 1
        
        if total_frames >= required_span:
            max_start_idx = total_frames - required_span
            start_idx = random.randint(0, max_start_idx)
            frame_indices = range(start_idx, start_idx + required_span, self.frame_stride)
            
        elif total_frames >= self.num_frames:
            dynamic_stride = total_frames // self.num_frames
            required_span = (self.num_frames - 1) * dynamic_stride + 1
            max_start_idx = total_frames - required_span
            start_idx = random.randint(0, max_start_idx)
            frame_indices = range(start_idx, start_idx + required_span, dynamic_stride)
        else:
            frame_indices = [i % total_frames for i in range(self.num_frames)]

        frames = vr.get_batch(list(frame_indices)).asnumpy()
        
        frames_tensor = rearrange(torch.from_numpy(frames).float() / 255.0, 't h w c -> t c h w')
        frames_tensor = self.transform(frames_tensor)
        frames_tensor = rearrange(frames_tensor, 't c h w -> c t h w')

        return frames_tensor, caption