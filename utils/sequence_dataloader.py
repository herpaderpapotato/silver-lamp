import os
import json
import sys
import random
import cv2
import numpy as np
import pickle
from glob import glob
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from collections import defaultdict
from pathlib import Path
import time


class OvercomplicatedDataset(Dataset):
    def __init__(self, dataset_folder, duration=60, croppct=0.2, augment=False, image_size=236, max_sequences=100000, random_seed=42, cache_images=False):
        self.max_sequences = max_sequences
        self.random_seed = random_seed
        self.image_size = image_size
        self.duration = duration
        self.croppct = croppct
        self.augment = augment
        self.dataset_folder = dataset_folder
        self.cache_images = cache_images
        # each subfolder is a sequence containing 120 images
        # provided the duration is equal to or less than 120, then the number of sequences will be 120 - self.duration + 1

        self.image_cache = defaultdict(list)
        self.label_cache = defaultdict(list)
        self.sequences = []
        self.load_sequences()
        

    def load_sequences(self):
        subfolders = sorted(glob(os.path.join(self.dataset_folder, '*')))
        subfolders = [f for f in subfolders if os.path.isdir(f)]
        
        for subfolder in tqdm(subfolders, desc="Loading sequences"):
            if subfolder == "rejected":
                continue
            images = sorted(glob(os.path.join(subfolder, '*.jpg')))
            images = [img for img in images if os.path.isfile(img)]
            # remove images that are not in the format of 0.jpg, 1.jpg, ..., n.jpg
            images = [img for img in images if os.path.basename(img).split('.')[0].isdigit()]
            if not images:
                print(f"No valid images found in {subfolder}, skipping...")
                continue

            # sort images by name integer
            images.sort(key=lambda x: int(os.path.basename(x).split('.')[0]))


            if len(images) < self.duration:
                continue
            for start in range(len(images) - self.duration):
                sequence = images[start:start + self.duration]
                if len(sequence) == self.duration:
                    self.sequences.append(sequence)
        random.seed(self.random_seed)
        random.shuffle(self.sequences)
        self.sequences = self.sequences[:self.max_sequences]
        print(f"Loaded {len(self.sequences)} sequences from {self.dataset_folder}")




    def __len__(self):
        return len(self.sequences)
    def __getitem__(self, idx):
        sequence = self.sequences[idx]

        images = torch.empty((self.duration, 3, self.image_size, self.image_size), dtype=torch.float32)
        labels = []
        for i, img_path in enumerate(sequence):
            if img_path not in self.image_cache:
                image = cv2.imread(img_path)
                if image is None:
                    continue
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                if image.shape[0] != self.image_size or image.shape[1] != self.image_size:
                    image = cv2.resize(image, (self.image_size, self.image_size))
                if self.cache_images:
                    self.image_cache[img_path] = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0
                else:
                    images[i] = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0
            if self.cache_images:
                images[i] = self.image_cache[img_path]

            label_path = img_path.replace('.jpg', '.txt')
            if label_path not in self.label_cache:
                if os.path.exists(label_path):
                    # label is a text file with a single line containing a label between 0 and 1
                    with open(label_path, 'r') as f:
                        label = f.read().strip()
                        self.label_cache[label_path] = float(label) if label else 0.0

            labels.append(self.label_cache[label_path])


        labels = np.array(labels)
        labels = torch.from_numpy(labels).float()

        if self.augment:
            # all images in the sequence are augmented the same way so looping and then calculating the augmentations
            # would result in different augmentations for each image in the sequence
            horizontal_flip = random.random() < 0.5
            vertical_flip = random.random() < 0.5
            rotation_angle = random.randint(-30, 30)  # degrees
            saturation_factor = random.uniform(0.8, 1.2)
            brightness_factor = random.uniform(0.8, 1.2)
            reversed = random.random() < 0.5
            if reversed:
                images = images.flip(dims=[0])
                labels = labels.flip(dims=[0])
            if horizontal_flip:
                images = images.flip(dims=[3])
            if vertical_flip:
                images = images.flip(dims=[2])
            # Apply rotation
            if rotation_angle != 0:
                images = images.rot90(k=rotation_angle // 90, dims=[2, 3])
            # Apply color jitter
            images = torch.nn.functional.adjust_saturation(images, saturation_factor)
            images = torch.nn.functional.adjust_brightness(images, brightness_factor)


        return images, labels

def collate_fn(batch):
    images, labels = zip(*batch)
    images = torch.stack(images, dim=0)  # Shape: (batch_size, duration, 3, image_size, image_size)
    labels = torch.stack(labels, dim=0)  # Shape: (batch_size, duration)
    return images, labels

def get_dataloader(dataset_folder, duration=60, croppct=0.2, augment=False, image_size=236, max_sequences=10000000, random_seed=42, batch_size=16):
    dataset = OvercomplicatedDataset(
        dataset_folder=dataset_folder,
        duration=duration,
        croppct=croppct,
        augment=augment,
        image_size=image_size,
        max_sequences=max_sequences,
        random_seed=random_seed
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=os.cpu_count() // 2 or 1
    )
    
    return dataloader

if __name__ == "__main__":
    dataset_folder = r"E:\ml\silver-lamp\utils\combined_output"
    dataloader = get_dataloader(dataset_folder, max_sequences=1000000)
    cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Image', 800, 600)
    print("Starting to iterate over the dataset...")
    for images, labels in dataloader:

        for i in range(images.shape[0]):
            print(f"Processing sequence {i + 1}/{images.shape[0]}")
            for j in range(images.shape[1]):
                display_image = cv2.cvtColor(images[i][j].permute(1, 2, 0).numpy(), cv2.COLOR_RGB2BGR)
                display_image = (display_image * 255).astype(np.uint8)
                cv2.putText(display_image, f"Label: {labels[i][j].item():.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow('Image', display_image)


                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    cv2.destroyAllWindows()
    print("Done.")


