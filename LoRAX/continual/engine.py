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


"""
Train and eval functions used in main.py
"""
import json
import os
import math
from typing import Iterable, Optional

import torch
from timm.data import Mixup

import utils.utils as utils


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, task_id: int, loss_scaler, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True, debug=False, args=None,
                    teacher_model: torch.nn.Module = None,
                    model_without_ddp: torch.nn.Module = None,
                    sam: torch.optim.Optimizer = None,
                    loader_memory=None):
    
    model.train(set_training_mode)
         
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Task: [{}] Epoch: [{}]'.format(task_id, epoch)
    print_freq = 20
    if len(data_loader)==0:
        import pdb;pdb.set_trace()

    all_targets = []
    for batch_index, (samples, targets, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        all_targets += targets.tolist()
        
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad()


        lam = None

        with torch.cuda.amp.autocast(enabled=not args.no_amp):
            #loss is a tuple because we have diff components - classification, divergence, distill... 
            loss_tuple = forward(samples, targets, model, teacher_model, criterion, lam, args)

        loss = sum(filter(lambda x: x is not None, loss_tuple))
        internal_losses = model_without_ddp.get_internal_losses(loss)
        for internal_loss_value in internal_losses.values():
            loss += internal_loss_value

        check_loss(loss)

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order

        loss_scaler(loss, optimizer, model_without_ddp, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()

        metric_logger.update_dict(internal_losses)
        metric_logger.update(loss=loss_tuple[0])
        metric_logger.update(div=loss_tuple[2])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if debug:
            print('Debug, only doing one epoch!')
            break

    if hasattr(model_without_ddp, 'hook_after_epoch'):
        model_without_ddp.hook_after_epoch()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def check_loss(loss):
    if not math.isfinite(loss.item()):
        raise Exception('Loss is {}, stopping training'.format(loss.item()))


#this function assumes targets ARE NOT REMAPPED 
def forward(samples, targets, model, teacher_model, criterion, lam, args):
    main_output, div_output = None, None
    outputs = model(samples)

    if isinstance(outputs, dict):
        main_output = outputs['logits']
        div_output = outputs['div']
    else:
        # simple classification loss 
        main_output = outputs

    loss = criterion(main_output, targets)

    div_loss = None
    if div_output is not None:
        nb_classes = main_output.shape[1]
        # hardcoded to expect 2 new classes per task
        nb_new_classes = 2
        nb_old_classes = nb_classes - nb_new_classes

        div_targets = torch.clone(targets)
        mask_old_cls = div_targets < nb_old_classes
        mask_new_cls = ~mask_old_cls

        div_targets[mask_old_cls] = 0
        div_targets[mask_new_cls] -= nb_old_classes - 1

        div_loss = args.head_div * criterion(div_output, div_targets)

    #LoRAX does not use these loss types 
    loss_binary = None
    kd_loss = None

    return loss, kd_loss, div_loss, loss_binary


@torch.no_grad()
def evaluate(data_loader, model, device, logger,ONE_REAL):

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    all_targets = []
    for images, target, task_ids in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target_orig = target

        all_targets += target_orig.tolist()
        target = target.to(device, non_blocking=True)
        is_odd_tar = torch.remainder(target, 2)
        
        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            if isinstance(output, dict):
                output = output['logits']
            loss = torch.tensor(0)

        """Computes the binary accuracy."""
        _, predicted = (output).max(1)
        is_odd_pred = torch.remainder(predicted, 2)


        bacc = ((predicted%2).eq(target%2)).sum().item()/images.shape[0]
        group_acc = ((predicted*is_odd_pred).eq(target*is_odd_tar)).sum().item()/images.shape[0]
        
        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())

        metric_logger.meters['bacc'].update(bacc, n=batch_size)
        metric_logger.meters['acc1'].update(group_acc, n=batch_size)

        logger.add([output.cpu().argmax(dim=1), target.cpu(), task_ids], subset='test')

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} BAcc {bacc.global_avg: .3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, bacc=metric_logger.bacc, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def eval_and_log(args, output_dir, model, model_without_ddp, optimizer, lr_scheduler,
                 epoch, task_id, loss_scaler, max_accuracy, accuracy_list,
                 n_parameters, device, data_loader_val, train_stats, log_store, log_path, logger,
                 model_log, skipped_task=False):
    if args.output_dir:
        if os.path.isdir(args.resume):
            checkpoint_paths = [os.path.join(args.resume, f'checkpoint_{task_id}.pth')]
        else:
            checkpoint_paths = [output_dir / f'checkpoint_{task_id}.pth']
        for checkpoint_path in checkpoint_paths:
            if skipped_task:
                print('skipped task?')
                continue
            #print(f'save on master {checkpoint_path}')
            utils.save_on_master({
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'task_id': task_id,
                'scaler': loss_scaler.state_dict(),
                'args': args,
            }, checkpoint_path)
            print(f'done: save on master {checkpoint_path}')
    ONE_REAL = False
    test_stats = evaluate(data_loader_val, model, device, logger,ONE_REAL)
    print(f"Accuracy of the network on the {len(data_loader_val.dataset)} test images: {test_stats['acc1']:.1f}%")

    accuracy_list.append(test_stats['acc1'])

    log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                **{f'test_{k}': v for k, v in test_stats.items()},
                'epoch': epoch}

    if log_store is not None:
        log_store['results'][task_id] = log_stats

    if log_path is not None and utils.is_main_process():
        with open(log_path, 'a+') as f:

            f.write(json.dumps({
                'task': task_id,
                'epoch': epoch,
                'acc_per_task_remapped': [round(100 * acc_t, 2) for acc_t in accuracy_per_task(logger)],
                'bacc_per_task': [round(100 * acc_t, 2) for acc_t in accuracy_bacc_per_task(logger)],
                'train_lr': log_stats.get('train_lr', 0.),
                'train_loss': round(log_stats.get('train_loss', 0.), 5),
                'test_loss': round(log_stats['test_loss'], 5),
                **model_log
            }) + '\n')
    #if args.output_dir and utils.is_main_process():
    #    with (output_dir / "log.txt").open("a") as f:
    #        f.write(json.dumps(log_stats) + "\n")
    json_data = {
                'task': task_id,
                'epoch': epoch,
                'acc_per_task_remapped': [round(100 * acc_t, 2) for acc_t in accuracy_per_task(logger)],
                'bacc_per_task': [round(100 * acc_t, 2) for acc_t in accuracy_bacc_per_task(logger)],
                'train_lr': log_stats.get('train_lr', 0.),
                'train_loss': round(log_stats.get('train_loss', 0.), 5),
                'test_loss': round(log_stats['test_loss'], 5),
                **model_log
            }
    return max_accuracy, json_data

import numpy as np
def accuracy_bacc_per_task(logger):
    #binary accuracy
    all_preds, all_targets, all_tasks = logger._get_best_epochs(subset="test")
    last_preds, last_targets, last_tasks = all_preds[-1], all_targets[-1], all_tasks[-1]
    correct_pred = last_preds%2 == last_targets%2

    acc_per_task = []
    for task_id in np.unique(last_tasks):
        indexes = last_tasks == task_id
        acc_per_task.append(correct_pred[indexes].mean())

    return acc_per_task

def accuracy_per_task(logger):
    #class accuracy
    """ without val data, logger assumes best epoch for each task is the last one"""
    all_preds, all_targets, all_tasks = logger._get_best_epochs(subset="test")
    last_preds, last_targets, last_tasks = all_preds[-1], all_targets[-1], all_tasks[-1]
    correct_pred = last_preds*(last_preds%2) == last_targets*(last_targets%2) 
    acc_per_task = []
    for task_id in np.unique(last_tasks):
        indexes = last_tasks == task_id
        acc_per_task.append(correct_pred[indexes].mean())

    return acc_per_task

def indexes_task_outputs(logits, targets, increment_per_task):
    if increment_per_task[0] != increment_per_task[1]:
        raise NotImplementedError(f'Not supported yet for non equal task size')

    inc = increment_per_task[0]
    indexes = torch.zeros(len(logits), inc).long()

    for r in range(indexes.shape[0]):
        for c in range(indexes.shape[1]):
            indexes[r, c] = (targets[r] // inc) * inc + r * logits.shape[1] + c

    indexed_logits = logits.view(-1)[indexes.view(-1)].view(len(logits), inc)
    indexed_targets = targets % inc

    return indexed_logits, indexed_targets
