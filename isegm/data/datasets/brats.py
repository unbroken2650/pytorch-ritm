import os
import torch
import numpy as np
import cv2
import json
import random
from pathlib import Path
from isegm.data.base import ISDataset
from isegm.data.sample import DSample


class BraTSDataset(ISDataset):
    def __init__(self, dataset_path, split='train', stuff_prob=0.0, max_samples=None, ** kwargs):
        self.dataset_path = os.path.join(dataset_path, f'preprocessed_tensors_{split}/')
        self.split = split
        self.stuff_prob = stuff_prob
        self.max_samples = max_samples
        
        self.file_paths = [os.path.join(self.dataset_path, f) for f in os.listdir(self.dataset_path) if f.endswith('.pth')]

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        data = torch.load(self.file_paths[idx])
        return data['flair'], data['t1'], data['t1ce'], data['t2'], data['seg']  # 예시로 모든 채널을 반환
