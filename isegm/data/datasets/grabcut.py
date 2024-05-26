import cv2
import numpy as np
import torch
import random
from pathlib import Path
from tqdm import tqdm
from ..points_sampler import MultiPointSampler


class GrabCutDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, images_dir_name='data_GT', masks_dir_name='boundary_GT',
                 augmentator=None, temp=False, stuff_prob=0.0, points_sampler=MultiPointSampler(max_num_points=12),
                 min_object_area=0, keep_background_prob=0.0, with_image_info=False,
                 samples_scores_path=None, samples_first_click_prob=0.0, epoch_len=-1, **kwargs):
        super(GrabCutDataset, self).__init__(**kwargs)
        self.dataset_path = Path(dataset_path)
        self._images_path = self.dataset_path / images_dir_name
        self._insts_path = self.dataset_path / masks_dir_name

        self.dataset_samples = [x.name for x in sorted(self._images_path.glob('*.*'))]
        self._masks_paths = {x.stem: x for x in self._insts_path.glob('*.*')}

        self.vis_dir = Path("./experiments/saved_images")
        self.vis_dir.mkdir(parents=True, exist_ok=True)
        self.epoch_len = epoch_len
        self.min_object_area = min_object_area
        self.keep_background_prob = keep_background_prob
        self.points_sampler = points_sampler
        self.with_image_info = with_image_info
        self.samples_precomputed_scores = self._load_samples_scores(samples_scores_path)

        self.dataset_samples = self._load_samples()

    def __getitem__(self, index):
        if self.samples_precomputed_scores is not None:
            index = np.random.choice(self.samples_precomputed_scores['indices'],
                                     p=self.samples_precomputed_scores['probs'])
        else:
            if self.epoch_len > 0:
                index = random.randrange(0, len(self.dataset_samples))

        sample = self.dataset_samples[index]
        image, mask = sample['image'], sample['mask']
        points = np.array(self.points_sampler.sample_points(mask), dtype=np.float32)

        output = {
            'images': torch.tensor(image.transpose(2, 0, 1)),
            'points': points,
            'instances': torch.tensor(mask)
        }
        self._visualize_data(image, mask, index)

        if self.with_image_info:
            output['image_info'] = sample['id']

        return output

    def __len__(self):
        return len(self.dataset_samples)

    def _load_samples(self):
        samples = []
        for img_filename in tqdm(self.dataset_samples, desc='Loading images'):
            img_path = self._images_path / img_filename
            mask_path = self._insts_path / (img_path.stem + '.png')
            image = cv2.imread(str(img_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            samples.append({'image': image, 'mask': mask, 'id': img_path.stem})
        return samples

    def get_sample(self, index):
        dataset_sample = self.dataset_samples[index]

        image = dataset_sample['image']
        label = dataset_sample['mask']

        instance_map = np.zeros(image.shape[1:], dtype=np.int32)

        return {'image': image, 'label': label, 'id': index}

    def _visualize_data(self, image, mask, idx):
        image_path = self.vis_dir / f'image_{idx}.png'
        cv2.imwrite(str(image_path), image)
        mask_path = self.vis_dir / f'mask_{idx}.png'
        cv2.imwrite(str(mask_path), mask)

    @staticmethod
    def _load_samples_scores(samples_scores_path):
        if samples_scores_path is None:
            return None
        with open(samples_scores_path, 'rb') as f:
            images_scores = pickle.load(f)
        probs = np.array([score for _, _, score in images_scores])
        probs /= probs.sum()
        return {'indices': list(range(len(images_scores))), 'probs': probs}
