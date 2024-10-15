"""
Copyright 2020 - present, Facebook, Inc
Copyright 2022 - present, Arthur Douillard

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""
Code adapted from the DyTox repository: https://github.com/arthurdouillard/dytox
@inproceedings{douillard2021dytox,
  title     = {DyTox: Transformers for Continual Learning with DYnamic TOken eXpansion},
  author    = {Douillard, Arthur and Ram\'e, Alexandre and Couairon, Guillaume and Cord, Matthieu},
  booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2022}
}
"""

"""
Modifications Copyright 2024, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
SPDX-License-Identifier: MIT
"""
# Code adapted from the CDDB repository: https://github.com/Coral79/CDDB
import os

from continuum import ClassIncremental
from continuum.datasets import _ContinuumDataset
from continuum.tasks import TaskType
from torchvision import transforms
from torchvision.transforms import functional as Fv

try:
    interpolation = Fv.InterpolationMode.BICUBIC
except:
    interpolation = 3

from typing import List, Tuple, Union

import numpy as np


class GANFakeDataset(_ContinuumDataset):
    """Continuum dataset for ganfake detection datasets.
    Continual Deepfake Detection Benchmark (CDDB) dataset preprocessing code pulled from CDDB project repo with minor modifications 
    https://github.com/Coral79/CDDB/tree/main
    @inproceedings{li2022continual,
        title={A Continual Deepfake Detection Benchmark: Dataset, Methods, and Essentials},
        author={Li, Chuqiao and Huang, Zhiwu and Paudel, Danda Pani and Wang, Yabin and Shahbazi, Mohamad and Hong, Xiaopeng and Van Gool, Luc},
        booktitle={Winter Conference on Applications of Computer Vision (WACV)},
        year={2023}
     }
    """

    def __init__(
            self,
            data_path: str,
            task_name: list,
            multiclass: list,
            train: bool = True,
            download: bool = True,
            data_type: TaskType = TaskType.IMAGE_PATH,
            one_real: bool = False
    ):
        self.data_path = data_path
        self._data_type = data_type
        self.one_real = one_real
        super().__init__(data_path=data_path, train=train, download=download)
        self.task_name = task_name
        self.multiclass = multiclass

        allowed_data_types = (TaskType.IMAGE_PATH, TaskType.SEGMENTATION)
        if data_type not in allowed_data_types:
            raise ValueError(f"Invalid data_type={data_type}, allowed={allowed_data_types}.")

    @property
    def data_type(self) -> TaskType:
        return self._data_type

    def get_data(self) -> Tuple[np.ndarray, np.ndarray, Union[None, np.ndarray]]:
        # codebase built with the assumption that real classes are listed inteh even indices! 
        dataset = []
        if self.train:
            for id, name in enumerate(self.task_name):
                root_ = os.path.join(self.data_path, name, 'train')
                print(root_)
                sub_classes = os.listdir(root_) if self.multiclass[id] else ['']
                for cls in sub_classes:
                    print(f' {os.path.join(root_, cls, "0_real")} {0 + 2 * id}')
                    print(f' {os.path.join(root_, cls, "1_fake")} {1 + 2 * id}')
                    for imgname in os.listdir(os.path.join(root_, cls, '0_real')):
                        
                        dataset.append((os.path.join(root_, cls, '0_real', imgname), 0 + 2 * id))
                    for imgname in os.listdir(os.path.join(root_, cls, '1_fake')):
                        
                        dataset.append((os.path.join(root_, cls, '1_fake', imgname), 1 + 2 * id))
        else:
            for id, name in enumerate(self.task_name):
                root_ = os.path.join(self.data_path, name, 'val')
                sub_classes = os.listdir(root_) if self.multiclass[id] else ['']
                for cls in sub_classes:
                    for imgname in os.listdir(os.path.join(root_, cls, '0_real')):
                        dataset.append((os.path.join(root_, cls, '0_real', imgname), 0 + 2 * id))
                    for imgname in os.listdir(os.path.join(root_, cls, '1_fake')):
                        dataset.append((os.path.join(root_, cls, '1_fake', imgname), 1 + 2 * id))

        self.dataset = dataset
        x, y, t = self._format(self.dataset)
        self.list_classes = np.unique(y)
        return x, y, t
        

    @staticmethod
    def _format(raw_data: List[Tuple[str, int]]) -> Tuple[np.ndarray, np.ndarray, None]:
        x = np.empty(len(raw_data), dtype="S255")
        y = np.empty(len(raw_data), dtype=np.int16)
        y_remap = np.empty(len(raw_data), dtype=np.int16)

        for i, (path, target) in enumerate(raw_data):
            x[i] = path
            y[i] = target
            is_odd = target%2
            target = (target // 2) + 1.0
            y_remap[i] = int(target*is_odd)

        return x, y, None#y_remap


def   build_dataset(is_train, args):
    transform = build_transform(is_train, args)
    
    if args.data_set.lower() == 'ganfake':
        dataset = GANFakeDataset(args.data_path, args.task_name, args.multiclass, train=is_train)
    else:
        raise ValueError(f'Unknown dataset {args.data_set}.')
    
    scenario = ClassIncremental(
        dataset,
        initial_increment=args.initial_increment,
        increment=args.increment,
        transformations=transform.transforms,
        class_order=args.class_order
    )
    nb_classes = scenario.nb_classes

    return scenario, nb_classes


def build_transform(is_train, args):

    if is_train:
        t= [
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
        ]
        return transforms.Compose(t)
    else:
        t= [
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
        ]
        return transforms.Compose(t)
