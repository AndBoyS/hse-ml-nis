from pathlib import Path
from typing import Union, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
import torchvision.transforms as T
from transformers import AutoFeatureExtractor

from .transforms import CutOut
from .utils import feature_extractor_to_numpy


class MeGlassDataset(torch.utils.data.Dataset):

    # {Путь к фото: таргет (1 если есть очки)}
    label_dict: Dict[Path, int]
    # [(Путь к фото, таргет), ...]
    # для оптимальности в __getitem__
    fps_and_labels: List[Tuple[Path, int]]

    def __init__(self,
                 dataset_fp: Union[Path, str],
                 labels_fp: Union[Path, str],
                 transform: Optional[T.Compose] = None,
                 idx: Optional[List[int]] = None,
                 hf_model_name: str = 'microsoft/resnet-18',
                 ):
        """
        Датасет MeGlass (https://github.com/cleardusk/MeGlass)

        :param dataset_fp:
            Папка с фото
        :param labels_fp:
            Путь к meta.txt (лейблы)
        :param transform:
            Трансформы для фото
        :param idx:
            Список айди фоток, которые нужно использовать (при None используются все фото)
        :param hf_model_name:
            Название предобученной модели на huggingface (например 'microsoft/resnet-50')
        """
        self.dataset_fp = Path(dataset_fp)
        self.idx = idx
        self._create_label_dict(labels_fp)
        self.transform = transform
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(hf_model_name)

        if self.transform is None:
            self.transform = self._create_default_transform()

    def _create_label_dict(self, label_fp: Union[Path, str]):
        self.label_dict = {}
        self.fps_and_labels = []
        meta_lines = Path(label_fp).read_text().split('\n')
        for line in meta_lines:
            if not line:
                continue
            filename, label = line.split()
            label = int(label)
            fp = self.dataset_fp / filename

            self.label_dict[fp] = label
            self.fps_and_labels.append((fp, label))

        if self.idx is not None:
            self.fps_and_labels = [self.fps_and_labels[i] for i in self.idx]

    def _create_default_transform(self):

        return T.Compose([
            T.ToTensor(),
            T.RandomHorizontalFlip(),
            CutOut(),
        ])

    def __getitem__(self, idx):
        fp, label = self.fps_and_labels[idx]
        im = Image.open(fp).convert('RGB')
        im = feature_extractor_to_numpy(self.feature_extractor, im)
        im = self.transform(im)

        item = {
            'image': im,
            'label': label,
        }
        return item

    def __len__(self):
        return len(self.fps_and_labels)
