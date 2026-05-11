"""
Flickr8k dataset loader for training the captioning model.
"""

import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

import config
from src.vocabulary import Vocabulary


class Flickr8kDataset(Dataset):
    """
    Loads Flickr8k images and their captions.
    Each image has 5 captions; we treat each (image, caption) pair
    as a separate sample to increase training data.
    """

    def __init__(self, caption_file, image_dir, vocabulary, max_len=52,
                 transform=None, split="train", split_ratio=0.85):
        self.image_dir = image_dir
        self.vocabulary = vocabulary
        self.max_len = max_len

        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
                transforms.RandomHorizontalFlip(p=0.3),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD),
            ])
        else:
            self.transform = transform

        # read captions CSV
        df = pd.read_csv(caption_file)
        df.columns = [c.strip() for c in df.columns]

        # get unique images for splitting
        unique_images = df["image"].unique()
        split_idx = int(len(unique_images) * split_ratio)

        if split == "train":
            split_images = set(unique_images[:split_idx])
        else:
            split_images = set(unique_images[split_idx:])

        df = df[df["image"].isin(split_images)].reset_index(drop=True)

        self.images = df["image"].tolist()
        self.captions = df["caption"].tolist()

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        caption = self.captions[idx]

        # load and transform image
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        # encode caption to indices
        encoded = self.vocabulary.encode_caption(caption, self.max_len)
        caption_tensor = torch.tensor(encoded, dtype=torch.long)
        length = sum(1 for idx in encoded if idx != self.vocabulary.word2idx[self.vocabulary.pad_token])

        return image, caption_tensor, torch.tensor(length, dtype=torch.long)


def get_eval_transform():
    """Transform for validation/test images (no augmentation)."""
    return transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD),
    ])
