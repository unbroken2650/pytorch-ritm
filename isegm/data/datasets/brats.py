import torch
import cv2
import random
import numpy as np
from isegm.data.base import ISDataset
from pathlib import Path
from isegm.data.sample import DSample
from tqdm import tqdm


class BraTSDataset(ISDataset):
    def __init__(self, dataset_path, split, stuff_prob=0.0, **kwargs):
        super(BraTSDataset, self).__init__(**kwargs)
        self.dataset_path = Path(dataset_path) / f'selected_tensors_{split}'
        self.split = split
        self.stuff_prob = stuff_prob
        self.file_paths = sorted(list(self.dataset_path.glob('*.pth')))

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
            tensor_data = torch.load(file_path)
            image = np.array(tensor_data['image'])
            image = image.squeeze()
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

            if self.split == 'train':
                label = np.array(tensor_data['label'])
                self.dataset_samples.append({"image": image, "label": label})
            else:
                self.dataset_samples.append({"image": image})

            print(f'{len(self.dataset_samples)} images loaded from {len(file_paths)} files')

    def get_sample(self, index) -> DSample:
        dataset_sample = self.dataset_samples[index]

        image = dataset_sample['image']
        label = dataset_sample['label']

        instance_map = np.zeros(label.shape[:2], dtype=np.int32) if self.split == 'train' else np.zeros(
            label.shape[:2], dtype=np.int32)

        return DSample(image, instance_map, objects_ids=[0])
