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
DyTox repository: https://github.com/arthurdouillard/dytox
@inproceedings{douillard2021dytox,
  title     = {DyTox: Transformers for Continual Learning with DYnamic TOken eXpansion},
  author    = {Douillard, Arthur and Ram\'e, Alexandre and Couairon, Guillaume and Cord, Matthieu},
  booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2022}
}
""" 


import torch
from timm.utils import dispatch_clip_grad


class ContinualScaler:
    state_dict_key = "amp_scaler"

    def __init__(self, disable_amp):
        self._scaler = torch.cuda.amp.GradScaler(enabled=not disable_amp)

    def __call__(
        self, loss, optimizer, model_without_ddp, clip_grad=None, clip_mode='norm',
        parameters=None, create_graph=False,
        hook=True
    ):
        self.pre_step(loss, optimizer, parameters, create_graph, clip_grad, clip_mode)
        self.post_step(optimizer, model_without_ddp, hook)

    def pre_step(self, loss, optimizer, parameters=None, create_graph=False, clip_grad=None, clip_mode='norm'):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
        if clip_grad is not None:
            assert parameters is not None
            dispatch_clip_grad(parameters, clip_grad, mode=clip_mode)

    def post_step(self, optimizer, model_without_ddp, hook=True):
        if hook and hasattr(model_without_ddp, 'hook_before_update'):
            model_without_ddp.hook_before_update()

        self._scaler.step(optimizer)

        if hook and hasattr(model_without_ddp, 'hook_after_update'):
            model_without_ddp.hook_after_update()

        self.update()

    def update(self):
        self._scaler.update()

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)
