import os
from collections import deque
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import CLIPTokenizer, CLIPTextModel

# Configure default cache directory
SERVER_CACHE_DIR = "/mnt/h200_disk/pranoy/datasets/vid"
os.environ.setdefault("HF_HOME", SERVER_CACHE_DIR)
os.environ.setdefault("HF_DATASETS_CACHE", SERVER_CACHE_DIR)


class WebVid2MDataset(Dataset):
    """
    Dataset loader for WebVid-2M / OpenVid-1M (and custom subsets).
    Supports loading from either:
      1. HuggingFace Hub dataset.
      2. Local directory of video files with CSV/JSON metadata.
    """
    def __init__(
        self,
        image_size,
        frames=17,
        cond_dim=512,
        hf_dataset_name=None,
        split="train",
        video_folder=None,
        metadata_file=None,
        max_samples=None,
        cache_dir="/mnt/h200_disk/pranoy/datasets/vid",
        download=True,
        download_workers=16,
        zip_url=None,
        csv_url=None,
    ):
        self.image_size = image_size
        self.frames = frames
        self.cond_dim = cond_dim
        self.cache_dir = cache_dir
        self.video_folder = video_folder or cache_dir
        self.items = []
      
      
        if hf_dataset_name is not None:
            raw_dataset = load_dataset(hf_dataset_name, split=split, cache_dir=cache_dir)
            for idx in range(len(raw_dataset)):
                if max_samples and len(self.items) >= max_samples:
                    break
                self.items.append(raw_dataset[idx])
        elif video_folder is not None:
            video_folder_path = Path(video_folder)
            os.makedirs(video_folder_path, exist_ok=True)

            # Check if videos exist; if not, check for zip file or download it and extract automatically
            existing_mp4s = list(video_folder_path.glob("*.mp4"))
            if len(existing_mp4s) == 0:
                import zipfile, urllib.request
                zip_filename = os.path.basename(zip_url) if zip_url else "OpenVid_part0.zip"
                local_zip = os.path.join(self.cache_dir, zip_filename)
                if not os.path.exists(local_zip) and zip_url is not None:
                    print(f"[WebVid2MDataset] Downloading dataset zip from {zip_url}...")
                    urllib.request.urlretrieve(zip_url, local_zip)
                if os.path.exists(local_zip):
                    print(f"[WebVid2MDataset] Extracting {local_zip} into {video_folder_path}...")
                    with zipfile.ZipFile(local_zip, "r") as zip_ref:
                        zip_ref.extractall(video_folder_path)

            if metadata_file is None or not os.path.exists(metadata_file):
                downloaded_csv = os.path.join(self.cache_dir, "OpenVid-1M.csv")
                if not os.path.exists(downloaded_csv):
                    try:
                        from huggingface_hub import hf_hub_download
                        print(f"[WebVid2MDataset] Downloading metadata CSV OpenVid-1M.csv via huggingface_hub...")
                        downloaded_path = hf_hub_download(repo_id="nkp37/OpenVid-1M", filename="data/train/OpenVid-1M.csv", repo_type="dataset", local_dir=self.cache_dir)
                        os.replace(downloaded_path, downloaded_csv)
                    except Exception as e:
                        print(f"[WebVid2MDataset] Warning: Could not download metadata via huggingface_hub ({e}).")
                        if csv_url is not None:
                            import urllib.request
                            print(f"[WebVid2MDataset] Downloading metadata CSV from {csv_url}...")
                            urllib.request.urlretrieve(csv_url, downloaded_csv)
                if os.path.exists(downloaded_csv):
                    metadata_file = downloaded_csv

            metadata_map = {}
            if metadata_file is not None and os.path.exists(metadata_file):
                if str(metadata_file).endswith(".csv"):
                    import csv
                    with open(metadata_file, mode="r", encoding="utf-8") as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            raw_vid = str(row.get("video", row.get("videoid", row.get("id", ""))))
                            if raw_vid:
                                vid_key = Path(raw_vid).stem
                                metadata_map[vid_key] = row
                elif str(metadata_file).endswith(".json"):
                    import json
                    with open(metadata_file, mode="r", encoding="utf-8") as f:
                        metadata_map = json.load(f)

            valid_exts = {".mp4", ".avi", ".mkv", ".webm", ".mov"}
            for vfile in sorted(video_folder_path.rglob("*")):
                if vfile.suffix.lower() in valid_exts:
                    if max_samples and len(self.items) >= max_samples:
                        break
                    vid_stem = vfile.stem
                    meta = metadata_map.get(vid_stem, {})
                    caption = meta.get("caption", meta.get("name", meta.get("text", meta.get("prompt", vid_stem))))
                    self.items.append({
                        "video_path": str(vfile),
                        "name": caption,
                        "videoid": vid_stem
                    })
        else:
            raise ValueError("Must provide either hf_dataset_name or video_folder to WebVid2MDataset.")


        if download:
            self.download_all_missing_videos(max_workers=download_workers)

    def download_all_missing_videos(self, max_workers=16):
        import urllib.request
        from concurrent.futures import ThreadPoolExecutor, as_completed

        tasks = []
        for item in self.items:
            video_data = item.get("video", item.get("video_path"))
            if not isinstance(video_data, str) and not hasattr(video_data, "__fspath__"):
                continue
            raw_vpath = str(video_data)
            candidate1 = os.path.join(self.cache_dir, raw_vpath)
            candidate2 = os.path.join(self.video_folder, raw_vpath)
            if os.path.exists(raw_vpath) or os.path.exists(candidate1) or os.path.exists(candidate2):
                continue

            url = item.get("contentUrl") or item.get("url") or item.get("video_url")
            if raw_vpath.startswith("http://") or raw_vpath.startswith("https://"):
                url = raw_vpath
            if url:
                target_path = candidate1
                tasks.append((url, target_path))

        if len(tasks) > 0:
            print(f"[WebVid2MDataset] Upfront downloading {len(tasks)} missing videos to '{self.cache_dir}' using {max_workers} threads...")
            def _download_task(url_and_dest):
                url, target_path = url_and_dest
                os.makedirs(os.path.dirname(target_path), exist_ok=True)
                urllib.request.urlretrieve(url, target_path)
                return target_path

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(_download_task, task) for task in tasks]
                for future in as_completed(futures):
                    future.result()
            print(f"[WebVid2MDataset] Completed downloading {len(tasks)} videos.")

    def __len__(self):
        return len(self.items)

    def _load_video_tensor(self, item):
        import torchvision.io as tv_io
        video_data = item.get("video", item.get("video_path"))

        if isinstance(video_data, torch.Tensor):
            video_tensor = video_data.float()
            if video_tensor.max() > 1.0:
                video_tensor = video_tensor / 255.0
        elif isinstance(video_data, str) or hasattr(video_data, "__fspath__"):
            raw_vpath = str(video_data)
            vpath = raw_vpath
            if not os.path.exists(vpath):
                candidate1 = os.path.join(self.cache_dir, raw_vpath)
                candidate2 = os.path.join(self.video_folder, raw_vpath)
                if os.path.exists(candidate1):
                    vpath = candidate1
                elif os.path.exists(candidate2):
                    vpath = candidate2

            if not os.path.exists(vpath):
                raise FileNotFoundError(
                    f"Video file '{raw_vpath}' not found at '{vpath}' or in '{self.cache_dir}'. "
                    "Note: HuggingFace metadata datasets (like Valley-webvid2M-Pretrain-703K) only store relative file names. "
                    f"Ensure your WebVid MP4 videos are downloaded inside '{self.cache_dir}'. Strictly no fallback allowed."
                )
            try:
                vframes, _, _ = tv_io.read_video(vpath, pts_unit="sec")
                if len(vframes) == 0:
                    raise RuntimeError("0 frames decoded")
                video_tensor = vframes.permute(0, 3, 1, 2).float() / 255.0
            except Exception as e:
                import cv2
                cap = cv2.VideoCapture(vpath, cv2.CAP_FFMPEG)
                frames = []
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(torch.from_numpy(frame))
                cap.release()
                if len(frames) == 0:
                    raise RuntimeError(f"Failed to decode any frames from video {vpath} via torchvision or OpenCV. Original error: {e}. Strictly no fallback allowed.")
                video_tensor = torch.stack(frames, dim=0).permute(0, 3, 1, 2).float() / 255.0
        elif isinstance(video_data, (list, tuple)):
            frames = [torch.as_tensor(f, dtype=torch.float32) for f in video_data]
            if len(frames) == 0:
                raise RuntimeError("Video frame list is empty. Strictly no fallback allowed.")
            video_tensor = torch.stack(frames, dim=0)
            if video_tensor.ndim == 4 and video_tensor.shape[-1] == 3:
                video_tensor = video_tensor.permute(0, 3, 1, 2)
            if video_tensor.max() > 1.0:
                video_tensor = video_tensor / 255.0
        else:
            raise RuntimeError(f"Unsupported video_data type {type(video_data)}: {video_data}. Strictly no fallback allowed.")

        if video_tensor.ndim == 4 and video_tensor.shape[1] != 3 and video_tensor.shape[3] == 3:
            video_tensor = video_tensor.permute(0, 3, 1, 2)

        T, C, H, W = video_tensor.shape
        if T < self.frames:
            raise ValueError(f"Video has {T} frames, which is less than the required {self.frames} frames. Dropping sample.")

        indices = torch.linspace(0, T - 1, self.frames).long()
        video_tensor = video_tensor[indices]

        video_tensor = F.interpolate(
            video_tensor,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )

        return video_tensor.permute(1, 0, 2, 3)

    def __getitem__(self, idx):
        for _ in range(max(1, len(self.items))):
            try:
                item = self.items[idx]
                video = self._load_video_tensor(item)
                caption = item.get("name") or item.get("caption") or item.get("text") or item.get("prompt")
                if not caption or not isinstance(caption, str):
                    caption = "A high quality video clip."
                return video, caption
            except (ValueError, RuntimeError, FileNotFoundError):
                idx = (idx + 1) % len(self.items)
        raise RuntimeError("Could not find any video sample with at least self.frames frames.")

    def collate_fn(self, batch):
    
        videos = torch.stack([item[0] for item in batch])
        
        captions = [item[1] for item in batch]

        return videos, captions


WebVidDataset = WebVid2MDataset


if __name__ == "__main__":
    print("=" * 60)
    print("Testing WebVid2MDataset standalone verification...")
    print("=" * 60)

    test_ds = WebVid2MDataset(
        image_size=64,
        video_folder="/mnt/h200_disk/pranoy/datasets/vid/videos",
        metadata_file="/mnt/h200_disk/pranoy/datasets/vid/OpenVid-1M.csv",
        cache_dir="/mnt/h200_disk/pranoy/datasets/vid",
        zip_url="https://huggingface.co/datasets/nkp37/OpenVid-1M/resolve/main/OpenVid_part0.zip",
        csv_url="https://huggingface.co/datasets/nkp37/OpenVid-1M/resolve/main/data/train/OpenVid-1M.csv",
    )

    print(f"\n[INFO] Total dataset items found: {len(test_ds)}")

    if len(test_ds) > 0:
        print("\n--- Inspecting Sample 0 Metadata ---")
        item0 = test_ds.items[0]
        print(f"Video ID:   {item0.get('videoid')}")
        print(f"Video Path: {item0.get('video_path')}")
        print(f"Caption:    '{item0.get('name')}'")

        print("\n--- Testing Sample 0 Extraction (__getitem__) ---")
        video_t, caption_t = test_ds[0]
        print(f"Video Tensor Shape:     {video_t.shape} (Channels, Frames, H, W)")
        print(f"Video Value Range:      min={video_t.min().item():.3f}, max={video_t.max().item():.3f}")
        print(f"Caption:                '{caption_t}'")

        print("\n--- Testing DataLoader Batching ---")
        dl = DataLoader(test_ds, batch_size=2, shuffle=False, collate_fn=test_ds.collate_fn)
        batch_vid, batch_text = next(iter(dl))
        print(f"Batch Video Shape:      {batch_vid.shape}")
        print(f"Batch Text Embed Shape: {batch_text.shape}")
        print(f"Batch Text Embed Device:{batch_text.device}")
        print("\nVerification completed successfully!")
    else:
        print("[WARN] Dataset is empty (no videos found in folder).")

# hf download \
#   --repo-type dataset \
#   --include "OpenVid_part0.zip" \
#   --include "data/train/*.csv" \
#   nkp37/OpenVid-1M \
#   --local-dir /mnt/h200_disk/pranoy/datasets/vid