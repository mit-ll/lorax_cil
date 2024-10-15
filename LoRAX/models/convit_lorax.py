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

import copy

import peft
import torch
import torch.nn as nn
import torch.nn.functional as F


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class SimpleLinear(nn.Module):
    '''
    Reference:
    https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/linear.py
    '''
    def __init__(self, in_features, out_features, bias=True):
        super(SimpleLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, nonlinearity='linear')
        nn.init.constant_(self.bias, 0)

    def forward(self, input):
        return {'logits': F.linear(input, self.weight, self.bias)}

device = 'cuda'
class LoRAX(nn.Module):
    def __init__(self, base_model, peft_config, out_dim):
        super(LoRAX, self).__init__()
        self.model = base_model
        self.model.head = Identity()
        self.peft_config = peft_config

        self.adapter_names = [] 
        self.add_adapter()
        
        self.embed_dim = base_model.embed_dim#768
        self.out_dim = out_dim
        
        self.head =  self.generate_fc(self.embed_dim, out_dim)
        self.div_head=None
        

    def feature_dim(self):
        return self.embed_dim*len(self.adapter_names)
    
    def generate_fc(self, in_dim, out_dim):
        fc = SimpleLinear(in_dim, out_dim)
        return fc
    
    def freeze(self):
        npar = sum(p.numel() for p in self.head.parameters() if p.requires_grad)
        pytorch_total_params = sum(p.numel() for p in self.model.parameters())
        pytorch_grad_params = sum(p.numel() for p in  self.model.parameters() if p.requires_grad)

    def end_finetuning(self):
        """Start FT mode, usually with backbone freezed and balanced classes."""
        self.in_finetuning = False

    def begin_finetuning(self):
        """End FT mode, usually with backbone freezed and balanced classes."""
        self.in_finetuning = True

    def reset_classifier(self,nb_classes):
        
        new_head = self.generate_fc(self.feature_dim(),nb_classes) 
        
        if self.head is not None:
            nb_output = self.head.out_features
            weight = copy.deepcopy(self.head.weight.data)
            bias = copy.deepcopy(self.head.bias.data)
            new_head.weight.data[:weight.shape[0],:weight.shape[1]] = weight
            new_head.bias.data[:nb_output] = bias

        self.head = new_head
        
    def add_adapter(self):
        i = len(self.adapter_names)
        adapter_name=f'task{i}'

        if i == 0:
            peft_model = peft.get_peft_model(self.model, self.peft_config, adapter_name=adapter_name).to(device)
            self.model = peft_model
        else:
            self.model.add_adapter(adapter_name,self.peft_config)

        self.adapter_names.append(adapter_name)
        
        print(f'added adapter {adapter_name}')
        print(f'all adapters {self.adapter_names}')
    
    def set_active_adapter(self):
        self.model.set_adapter(self.adapter_names[-1])
        print(f'active adapter {self.adapter_names[-1]}')

    def forward_features(self, x):
        features = [self.forward_adapter(adapter_name,x) for adapter_name in self.adapter_names]
        features = torch.cat(features, 1)
        self.model.set_adapter(self.adapter_names[-1])
        return features, None, None
    
    def forward_adapter(self, adapter_name, x):
        self.model.set_adapter(adapter_name)
        assert self.model.active_adapters[0] == adapter_name
        assert len(self.model.active_adapters) == 1 
        #feats=self.model.forward_features(x)[0]
        feats=self.model.forward(x)
        return feats
    
    def forward(self, x):
        features = [self.forward_adapter(adapter_name,x) for adapter_name in self.adapter_names]
        features = torch.cat(features, 1)
        out = self.head(features)['logits']

        if self.div_head is not None:

            f_dim = self.embed_dim
            aux_logits=self.div_head(features[:,-f_dim:])
            return { 'logits': out,'div': aux_logits}


        return out
                
    def get_internal_losses(self, clf_loss):
        """If you want to compute some internal loss, like a EWC loss for example.

        :param clf_loss: The main classification loss (if you wanted to use its gradient for example).
        :return: a dictionnary of losses, all values will be summed in the final loss.
        """
        int_losses = {}
        return int_losses  
    
    def epoch_log(self):
        log = {}
        return log
       
    
    def reset_div_head(self, one_real=False):
        #del div head
        if hasattr(self,'div_head'):
            del self.div_head

        n_features = self.embed_dim 
        # TODO: need to handle single real scenario 
        if one_real:
            new_head = nn.Linear(n_features, 2)
        else:   
            new_head = nn.Linear(n_features, 3)

        self.div_head = new_head
        print('added diversity head')
