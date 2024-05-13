import random
import cv2
import math
import pickle
import numpy as np
import torch
from .points_sampler import MultiPointSampler
from pathlib import Path


class ISDataset(torch.utils.data.dataset.Dataset):
    def __init__(self,
                 points_sampler=MultiPointSampler(max_num_points=12),
                 min_object_area=0,
                 keep_background_prob=0.0,
                 with_image_info=False,
                 samples_scores_path=None,
                 samples_scores_gamma=1.0,
                 epoch_len=-1):
        super(ISDataset, self).__init__()
        self.epoch_len = epoch_len
        self.min_object_area = min_object_area
        self.keep_background_prob = keep_background_prob
        self.points_sampler = points_sampler
        self.with_image_info = with_image_info
        self.samples_precomputed_scores = self._load_samples_scores(samples_scores_path, samples_scores_gamma)

        self.selected_mask = None
        self._selected_masks = None
        self.expand_ratio = 0.1
        self.max_num_points = 12
        self.neg_strategies = ['bg', 'other', 'border']
        self.neg_strategies_prob = [0.1, 0.4, 0.5]  # background, other, border

        self.vis_base_dir = Path("./experiments/saved_base_images")
        self.vis_base_dir.mkdir(parents=True, exist_ok=True)

        self.dataset_samples = None

    def __getitem__(self, index):
        if self.samples_precomputed_scores is not None:
            index = np.random.choice(self.samples_precomputed_scores['indices'],
                                     p=self.samples_precomputed_scores['probs'])
        else:
            if self.epoch_len > 0:
                index = random.randrange(0, len(self.dataset_samples))

        sample = self.get_sample(index)

        self.sample_object(sample)
        points = np.array(self.sample_points())
        mask = self.selected_mask

        output = {
            'images': torch.tensor(sample['image']),
            'points': np.array(points, dtype=np.float32),
            'instances': mask
        }
        self._visualize_data(sample['image'], mask, 99)

        if self.with_image_info:
            output['image_info'] = sample['id']

        return output

    def get_sample(self, index):
        raise NotImplementedError

    def __len__(self):
        if self.epoch_len > 0:
            return self.epoch_len
        else:
            return self.get_samples_number()

    def get_samples_number(self):
        return len(self.dataset_samples)

    @staticmethod
    def _load_samples_scores(samples_scores_path, samples_scores_gamma):
        if samples_scores_path is None:
            return None

        with open(samples_scores_path, 'rb') as f:
            images_scores = pickle.load(f)

        probs = np.array([(1.0 - x[2]) ** samples_scores_gamma for x in images_scores])
        probs /= probs.sum()
        samples_scores = {
            'indices': [x[0] for x in images_scores],
            'probs': probs
        }
        print(f'Loaded {len(probs)} weights with gamma={samples_scores_gamma}')
        return samples_scores

    def sample_object(self, sample):
        if len(sample) == 0:
            self._selected_masks = [[]]
            self._neg_masks = {strategy: 0 for strategy in self.neg_strategies}
            self._neg_masks['required'] = []
            return

        gt_mask, pos_masks, neg_masks = sample['label'], sample['label'] > 0, sample['label'] == 0

        self.selected_mask = gt_mask
        self._selected_masks = pos_masks

        neg_mask_bg = np.logical_not(gt_mask)
        neg_mask_border = self._get_border_mask(gt_mask)

        if len(sample) <= len(pos_masks):
            neg_mask_other = neg_mask_bg
        else:
            neg_mask_other = np.logical_and(np.logical_not(neg_mask_bg), np.logical_not(gt_mask))

        self._neg_masks = {
            'bg': neg_mask_bg,
            'other': neg_mask_other,
            'border': neg_mask_border,
            'required': neg_masks
        }

    def sample_points(self):
        assert hasattr(self, '_selected_masks') and self._selected_masks is not None, "Selected masks not initialized"
        # Sampling positive points from the positive masks
        pos_points = self._multi_mask_sample_points(self._selected_masks,
                                                    is_negative=[False] * len(self._selected_masks),
                                                    with_first_click=False)

        # Defining the negative sampling strategy
        neg_strategy = [(self._neg_masks[k], prob)
                        for k, prob in zip(self.neg_strategies, self.neg_strategies_prob)]
        neg_masks = self._neg_masks['required'] + [mask for mask, _ in neg_strategy]
        neg_points = self._multi_mask_sample_points(neg_masks,
                                                    is_negative=[True] * (len(self._neg_masks['required']) + len(neg_strategy)))

        # Combine positive and negative points
        points = pos_points + neg_points

        return points

    def _multi_mask_sample_points(self, masks, is_negative, with_first_click=False):
        points = []
        # Iterate over each mask
        for i, mask in enumerate(masks):
            if isinstance(mask, (list, tuple)):
                mask, _ = mask  # Unpacking the mask and its associated probability
            # Sample individual points from the current mask
            points.extend(self._sample_points(
                mask, is_negative=is_negative[i], with_first_click=with_first_click and i == 0))

        if len(points) > self.max_num_points:
            points = points[:self.max_num_points]  # Limiting the number of points if necessary

        return points

    def _sample_points(self, mask, is_negative=False, with_first_click=False):
        num_points = np.random.choice(range(1, self.max_num_points + 1))  # Random choice of number of points to sample

        indices = np.argwhere(mask).astype(int)  # All indices where the mask is True
        sampled_points = []

        for _ in range(num_points):
            if indices.size == 0:
                break  # No points to sample if mask is empty
            idx = np.random.choice(len(indices))  # Random index from available points
            point = indices[idx].tolist()  # The chosen point as a list
            sampled_points.append(point)

        return sampled_points

    def _get_border_mask(self, mask):
        expand_r = int(np.ceil(self.expand_ratio * np.sqrt(mask.sum())))
        kernel = np.ones((3, 3), np.uint8)
        expanded_mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=expand_r)
        expanded_mask[mask.astype(bool)] = 0
        return expanded_mask

    def _visualize_data(self, image, label, idx):
        image = image.transpose(1, 2, 0)
        image_path = self.vis_base_dir / f'image_{idx}.png'
        cv2.imwrite(str(image_path), image)
        label_path = self.vis_base_dir / f'label_{idx}.png'
        cv2.imwrite(str(label_path), label[0, :, :])
