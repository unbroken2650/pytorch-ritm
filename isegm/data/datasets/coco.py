import cv2
import json
import numpy as np
import random
from pathlib import Path
from isegm.data.base import ISDataset
from isegm.data.sample import DSample


class CocoDataset(ISDataset):
    def __init__(self, dataset_path, split='train', stuff_prob=0.0, **kwargs):
        super(CocoDataset, self).__init__(**kwargs)
        self.split = split
        self.dataset_path = Path(dataset_path)
        self.stuff_prob = stuff_prob

        self.load_samples()

    def load_samples(self):
        annotation_json_path = self.dataset_path / 'annotations' / f'panoptic_{self.split}2017.json'
        self.images_path = self.dataset_path / f'{self.split}2017'

        with open(annotation_json_path, 'r') as f:
            self.annotations = json.load(f)

        self.dataset_samples = self.annotations['annotations']
        self._categories = self.annotations['categories']
        self._stuff_labels = [x['id'] for x in self._categories if x['isthing'] == 0]
        self._things_labels = [x['id'] for x in self._categories if x['isthing'] == 1]
        self._things_labels_set = set(self._things_labels)
        self._stuff_labels_set = set(self._stuff_labels)

    def get_sample(self, index) -> DSample:
        dataset_sample = self.dataset_samples[index]
        image_filename = dataset_sample['file_name'].replace('.png', '.jpg')
        image_path = self.images_path / image_filename

        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Cannot load image at {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        instance_map = np.zeros(image.shape[:2], dtype=np.int32)

        things_ids = []
        stuff_ids = []

        for segment_info in dataset_sample['segments_info']:
            obj_id = segment_info['id']
            mask = obj_id == obj_id
            if segment_info['category_id'] in self._things_labels_set:
                if segment_info['iscrowd'] == 1:
                    continue
                things_ids.append(obj_id)
            else:
                stuff_ids.append(obj_id)
            instance_map[mask] = obj_id

        instances_ids = things_ids if random.random() >= self.stuff_prob else things_ids + stuff_ids

        return DSample(image, instance_map, objects_ids=instances_ids)
