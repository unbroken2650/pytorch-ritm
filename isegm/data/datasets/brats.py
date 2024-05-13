import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import random
import cv2
import math
import pickle
import numpy as np
import torch
from ..points_sampler import MultiPointSampler
from pathlib import Path
from isegm.data.transforms import remove_image_only_transforms
from albumentations import ReplayCompose


class BraTSDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, split, augmentator=None, temp=False,
                 stuff_prob=0.0, points_sampler=MultiPointSampler(max_num_points=12),
                 min_object_area=0,
                 keep_background_prob=0.0,
                 with_image_info=False,
                 samples_scores_path=None,
                 samples_scores_gamma=1.0,
                 epoch_len=-1, **kwargs):
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
        self.vis_dir = Path("./experiments/saved_images")
        self.vis_dir.mkdir(parents=True, exist_ok=True)
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
        self.sfc_inner_k = 1.7
        self.sfc_full_inner_prob = 0.0
        self.augmentator = augmentator

        self.dataset_samples = None
        self._augmented = False

        self.load_samples()

    def __getitem__(self, index):
        if self.samples_precomputed_scores is not None:
            index = np.random.choice(self.samples_precomputed_scores['indices'],
                                     p=self.samples_precomputed_scores['probs'])
        else:
            if self.epoch_len > 0:
                index = random.randrange(0, len(self.dataset_samples))

        sample = self.get_sample(index)

        self.sample_object(sample)
        image = sample['image']
        mask = self.selected_mask
        points = np.array(self.sample_points(), dtype=np.float32)

        output = {
            'images': torch.tensor(image),
            'points': np.array(points, dtype=np.float32),
            'instances': mask
        }
        self._visualize_data(image, mask, 99)

        if self.with_image_info:
            output['image_info'] = sample['id']

        return output

    def __len__(self):
        return self.get_samples_number()

    def load_samples(self):
        file_paths = self.file_paths
        self.dataset_samples = []

        for file_path in tqdm(file_paths, desc='Loading files'):
            tensor_data = np.load(file_path, allow_pickle=True).item()
            image = np.array(tensor_data['image'])
            image = self.normalize_to_rgb(image)
            image = np.stack((image, image, image), axis=0)

            if self.split == 'train':
                label = np.array(tensor_data['label'])
                label = self.normalize_to_rgb(label)
                self.dataset_samples.append({"image": image, "label": label})
                self._visualize_data(image, label, 99)
            else:
                self.dataset_samples.append({"image": image})

    def get_sample(self, index):
        dataset_sample = self.dataset_samples[index]

        image = dataset_sample['image']
        if self.split == 'train':
            label = dataset_sample['label']

        instance_map = label if self.split == 'train' else np.zeros(
            image.shape[1:], dtype=np.int32)

        return {'image': image, 'label': instance_map, 'id': index}

    def get_samples_number(self):
        return len(self.dataset_samples)

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
        pos_points = self._multi_mask_sample_points(self._selected_masks,
                                                    is_negative=[False] * len(self._selected_masks),
                                                    with_first_click=False)

        neg_strategy = [(self._neg_masks[k], prob)
                        for k, prob in zip(self.neg_strategies, self.neg_strategies_prob)]
        neg_masks = self._neg_masks['required'] + [mask for mask, _ in neg_strategy]
        neg_points = self._multi_mask_sample_points(neg_masks,
                                                    is_negative=[True] * (len(self._neg_masks['required']) + len(neg_strategy)))

        points = pos_points + neg_points

        return points

    def _multi_mask_sample_points(self, masks, is_negative, with_first_click=False):
        points = []

        for i, mask in enumerate(masks):
            points.extend(self._sample_points(
                i, mask, is_negative=is_negative[i], with_first_click=with_first_click and i == 0))

        if len(points) > self.max_num_points:
            selected_indices = np.random.choice(len(points), self.max_num_points, replace=False)
            points = [points[i] for i in selected_indices]

        return points

    def _sample_points(self, index, mask, is_negative=False, with_first_click=False):
        num_points = np.random.choice(np.arange(1, self.max_num_points + 1))
        sampled_points = []
        if is_negative:
            for i, _mask in enumerate(mask):
                points = self.extract_points_from_mask(_mask, num_points, is_negative, with_first_click, i)
                sampled_points.extend(points)
            return sampled_points
        else:
            points = self.extract_points_from_mask(mask, num_points, is_negative, with_first_click, index)
            sampled_points.extend(points)
            return sampled_points

    def extract_points_from_mask(self, mask, num_points, is_negative, with_first_click, click_index_base):
        points = []
        indices = np.argwhere(mask)

        for j in range(num_points):
            first_click = with_first_click and j == 0

            if first_click:
                point_indices = self.get_point_candidates(mask, k=self.sfc_inner_k, full_prob=self.sfc_full_inner_prob)
            else:
                point_indices = indices

            num_indices = len(point_indices)
            if num_indices > 0:
                selected_index = np.random.randint(0, num_indices)
                click = point_indices[selected_index].tolist() + [click_index_base + j]  # 항상 인덱스 추가
                points.append(click)

        return points

    def _get_border_mask(self, mask):
        expand_r = int(np.ceil(self.expand_ratio * np.sqrt(mask.sum())))
        kernel = np.ones((3, 3), np.uint8)
        expanded_mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=expand_r)
        expanded_mask[mask.astype(bool)] = 0
        return expanded_mask

    def get_point_candidates(obj_mask, k=1.7, full_prob=0.0):
        if full_prob > 0 and random.random() < full_prob:
            return obj_mask

        padded_mask = np.pad(obj_mask, ((1, 1), (1, 1)), 'constant')

        dt = cv2.distanceTransform(padded_mask.astype(np.uint8), cv2.DIST_L2, 0)[1:-1, 1:-1]
        if k > 0:
            inner_mask = dt > dt.max() / k
            return np.argwhere(inner_mask)
        else:
            prob_map = dt.flatten()
            prob_map /= max(prob_map.sum(), 1e-6)
            click_indx = np.random.choice(len(prob_map), p=prob_map)
            click_coords = np.unravel_index(click_indx, dt.shape)
            return np.array([click_coords])

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

    def _visualize_data(self, image, label, idx):
        image = image.transpose(1, 2, 0)
        image_path = self.vis_dir / f'image_{idx}.png'
        cv2.imwrite(str(image_path), image)
        label_path = self.vis_dir / f'label_{idx}.png'
        cv2.imwrite(str(label_path), label)

    def normalize_to_rgb(self, image):
        min_val, max_val = np.min(image), np.max(image)
        image = (image - min_val) / (max_val - min_val) * 255
        image = image.astype(np.uint8)
        return image
