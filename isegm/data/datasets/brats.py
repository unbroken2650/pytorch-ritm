import torch
import cv2
import random
import numpy as np
from isegm.data.base import ISDataset
from pathlib import Path
from isegm.data.sample import DSample
from tqdm import tqdm


class BraTSDataset(ISDataset):
    def __init__(self, dataset_path, split, temp=False, stuff_prob=0.0, **kwargs):
        super(BraTSDataset, self).__init__(**kwargs)
        if temp:
            self.dataset_path = Path(dataset_path) / f'temp_{split}'
        else:
            self.dataset_path = Path(dataset_path) / f'selected_slices_{split}'
        self.split = split
        self.stuff_prob = stuff_prob
        self.file_paths = sorted(list(self.dataset_path.glob('*.npy')))

        if not self.file_paths:
            raise FileNotFoundError(
                f"No data files found in {self.dataset_path}. Please check the directory and split name.")

        self.load_samples()

    def __len__(self):
        return len(self.file_paths)

    def load_samples(self):
        file_paths = self.file_paths
        self.dataset_samples = []

        for file_path in tqdm(file_paths, desc='Loading files'):
            tensor_data = np.load(file_path, allow_pickle=True).item()
            image = np.array(tensor_data['image'])
            min_val, max_val = np.min(image), np.max(image)
            image = (image - min_val) / (max_val - min_val) * 255
            image = image.astype(np.uint8)
            image = np.array([image, image, image])

            if self.split == 'train':
                label = np.array(tensor_data['label'])
                label = label.astype(np.uint8)
                self.dataset_samples.append({"image": image, "label": label})
            else:
                self.dataset_samples.append({"image": image})

    def get_sample(self, index) -> DSample:
        dataset_sample = self.dataset_samples[index]

        image = dataset_sample['image']
        label = dataset_sample['label']

        instance_map = np.zeros(label.shape[:2], dtype=np.int32) if self.split == 'train' else np.zeros(
            label.shape[:2], dtype=np.int32)

        return DSample(image, instance_map, objects_ids=[0])
