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

import torch
from utils import samplers

def get_backbone(args):
    print(f"Creating model: {args.model}")
    if args.model == 'convit_pretrain':
        from timm.models.convit import convit_base
        model = convit_base(pretrained=True)
    elif args.model == 'convit_pretrain_small':
        from timm.models.convit  import convit_base, convit_small
        model = convit_small(pretrained=True)
    elif args.model == 'convit_pretrain_tiny':
        from timm.models.convit import convit_tiny
        model = convit_tiny(pretrained=True)
    else:
        print(f'Unknown backbone {args.model}')
        raise NotImplementedError(f'Unknown backbone {args.model}')

    return model

def get_loaders(dataset_train, dataset_val, args, finetuning=False):
    sampler_train, sampler_val = samplers.get_sampler(dataset_train, dataset_val, args)

    loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler= sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=len(sampler_train) > args.batch_size,
    )

    loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=int(1.5 * args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    return loader_train, loader_val


def get_train_loaders(dataset_train, args, batch_size=None, drop_last=True):
    batch_size = batch_size or args.batch_size

    sampler_train = samplers.get_train_sampler(dataset_train, args)

    loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=drop_last,
    )

    return loader_train

class InfiniteLoader:
    def __init__(self, loader):
        self.loader = loader
        self.reset()

    def reset(self):
        self.it = iter(self.loader)

    def get(self):
        try:
            return next(self.it)
        except StopIteration:
            self.reset()
            return self.get()
