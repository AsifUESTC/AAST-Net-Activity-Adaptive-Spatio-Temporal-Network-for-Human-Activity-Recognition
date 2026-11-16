"""
HMDB51 Video Dataset for action recognition.
Extracts fixed number of frames from each video and applies transforms.
"""

import os
from typing import Optional, List, Dict

import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from transformers import ViTImageProcessor


class HMDB51Dataset(Dataset):
    VIDEO_EXTENSION = ".avi"

    def __init__(
        self,
        df: pd.DataFrame,
        transform: Optional[transforms.Compose] = None,
        frames_per_video: int = 32,
        image_size: int = 224,
        vit_model_name: str = "google/vit-base-patch16-224-in21k",
    ):
        if "filename" not in df.columns or "label" not in df.columns:
            raise ValueError("DataFrame must contain 'filename' and 'label' columns.")

        valid_videos = []
        for _, row in df.iterrows():
            video_path = str(row["filename"]).replace("\\", "/")
            if self._is_valid_video(video_path):
                valid_videos.append({"filename": video_path, "label": row["label"]})

        if not valid_videos:
            raise ValueError("No valid videos found. Please check your dataset paths.")

        self.dataset = pd.DataFrame(valid_videos)
        self.video_paths: List[str] = self.dataset["filename"].tolist()
        self.labels: List[str] = self.dataset["label"].tolist()
        self.transform = transform
        self.frames_per_video = frames_per_video
        self.label_to_index: Dict[str, int] = {
            label: idx for idx, label in enumerate(sorted(set(self.labels)))
        }
        self.index_to_label: Dict[int, str] = {v: k for k, v in self.label_to_index.items()}

        # ViT image processor (do_rescale=False because we apply ToTensor ourselves)
        self.vit_image_processor = ViTImageProcessor.from_pretrained(
            vit_model_name, do_rescale=False
        )

    def _is_valid_video(self, video_path: str) -> bool:
        if not os.path.isfile(video_path):
            return False
        if not video_path.endswith(self.VIDEO_EXTENSION):
            return False
        return True

    def __len__(self) -> int:
        return len(self.video_paths)

    def __getitem__(self, idx: int):
        video_path = self.video_paths[idx]
        label = self.labels[idx]

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Error opening video file: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = self._get_frame_indices(total_frames)
        frames = []

        for i in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)

            if self.transform:
                frame = self.transform(frame)

            # ViT processor expects tensor or PIL; we already applied ToTensor
            processed = self.vit_image_processor(
                images=frame, return_tensors="pt"
            )["pixel_values"].squeeze(0)
            frames.append(processed)

        cap.release()

        if len(frames) == 0:
            # Return dummy tensor to avoid crashing the loader
            return (
                torch.zeros((self.frames_per_video, 3, 224, 224)),
                0,
            )

        # Pad if we got fewer frames than expected
        while len(frames) < self.frames_per_video:
            frames.append(frames[-1].clone())

        frames = torch.stack(frames[: self.frames_per_video])
        label_index = self.label_to_index[label]
        return frames, label_index

    def _get_frame_indices(self, total_frames: int) -> List[int]:
        if total_frames <= 0:
            return [0] * self.frames_per_video
        if total_frames < self.frames_per_video:
            return [i % total_frames for i in range(self.frames_per_video)]
        step = total_frames // self.frames_per_video
        return [i * step for i in range(self.frames_per_video)]


def get_train_transforms(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
            ),
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.ToTensor(),
        ]
    )


def get_val_transforms(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )


def collate_fn(batch):
    """Simple collate – all videos already have fixed number of frames."""
    frames = torch.stack([item[0] for item in batch])
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    return frames, labels
