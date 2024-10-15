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

# Code adapted from the DyTox repository: https://github.com/arthurdouillard/dytox

import argparse
import copy
import datetime
import json
import os
import statistics
import time
import warnings
from pathlib import Path

import numpy as np
import peft
import torch
import torch.backends.cudnn as cudnn
import utils.utils as utils
import yaml
from continual.engine import eval_and_log, train_one_epoch

from continuum.metrics import Logger
from continuum.tasks import split_train_val
from models.convit_lorax import LoRAX

from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
from utils import factory, scaler
from utils.datasets import build_dataset
from utils.incremental_utils.rehearsal import Memory
from utils.plot_metrics import process_metrics

warnings.filterwarnings("ignore")

from configs.arg_parser import get_args_parser


def main(args):

    print(f'world size {utils.get_world_size()}')

    #store task metrics  
    task_json_data = []

    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"Number of GPUs available: {num_gpus}")
        for i in range(num_gpus):
            gpu_name = torch.cuda.get_device_name(i)
            print(f"GPU {i}: {gpu_name}")
    else:
        print("No GPUs are available.")

    ONE_REAL = False # one real class not implemented 
    print(args)
    logger = Logger(list_subsets=['train', 'test'])

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    scenario_train, args.nb_classes = build_dataset(is_train=True, args=args)
    scenario_val, _ = build_dataset(is_train=False, args=args)

    mixup_fn = None

    # LoRA Configuration 
    base_model = factory.get_backbone(args) 
    r=args.r_param
    lora_alpha=int(r*args.r_alpha_ratio)
    print(f'r {r}, alpha {lora_alpha}')
    print(f'reg {args.reg}')
    peft_config = peft.LoraConfig(target_modules=args.reg,r=r,lora_alpha=lora_alpha,lora_dropout=0.1,bias="none")#, modules_to_save=["head"]
    print(peft_config )

    # Create LoRAX model
    model = LoRAX(base_model, peft_config, 2)

    model.to(device)
    model_without_ddp = model

    n_parameters = sum(p.numel() for p in model_without_ddp.parameters() if p.requires_grad)
    
        
    # Start the logging process on disk ----------------------------------------
    if args.name:
        args.output_dir = os.path.join(args.output_basedir, f"{datetime.datetime.now().strftime('%y-%m-%d')}_{args.name}_{args.r_param}_{args.reg}_{args.trial_id}")
        print(f'Results output directory: {args.output_dir}')
        log_path = os.path.join(args.output_dir, f"logs_{args.trial_id}.json")
        long_log_path = os.path.join(args.output_dir, f"long_logs_{args.trial_id}.json")

        if utils.is_main_process():
            os.system("echo '\ek{}\e\\'".format(args.name))
            os.makedirs(args.output_dir, exist_ok=True)
            with open(os.path.join(args.output_dir, f"config_{args.trial_id}.json"), 'w+') as f:
                config = vars(args)
                config["nb_parameters"] = n_parameters
                config["peft_config"] = str(peft_config)
                json.dump(config, f, indent=2)
            with open(log_path, 'w+') as f:
                pass  # touch
            with open(long_log_path, 'w+') as f:
                pass  # touch
        log_store = {'results': {}}   
    else:
        log_store = None
        log_path = long_log_path = None

    if args.output_dir and utils.is_main_process():
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if args.distributed:
        torch.distributed.barrier()

    output_dir = Path(args.output_dir)

    loss_scaler = scaler.ContinualScaler(args.no_amp)
    criterion = torch.nn.CrossEntropyLoss()#get_criterion(args)

    # ----------- exemplars / rehersal memory
    if args.memory_size > 0:
        print('using rehersal memory')
        memory = Memory(
            args.memory_size, scenario_train.nb_classes, args.rehearsal, args.fixed_memory
        )
    else:
        print('no rehersal memory ')
        memory = None

    # ----------- class increment per task
    nb_classes = args.initial_increment
    base_lr = args.lr
    accuracy_list = []
    start_time = time.time()

    if args.debug:
        args.base_epochs = 1
        args.epochs = 1

    args.increment_per_task = [args.initial_increment] + [args.increment for _ in range(len(scenario_train) - 1)]
    print(f' Increments per task {args.increment_per_task}')
    
    
    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------    
    # --------------------------------------------------------------------------
    #
    # Begin of the task loop
    #
    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------

    dataset_true_val = None

    for task_id, dataset_train in enumerate(scenario_train):
        if args.max_task == task_id:
            print(f"Stop training because of max task")
            break
        print(f"Starting task id {task_id}/{len(scenario_train) - 1}")

        # ----------------------------------------------------------------------
        # Data
        dataset_val = scenario_val[:task_id + 1]
        if args.validation > 0.:  # use validation split instead of test
            print('validation split instead of test')
            if task_id == 0:
                dataset_train, dataset_val = split_train_val(dataset_train, args.validation)
                dataset_true_val = dataset_val
            else:
                dataset_train, dataset_val = split_train_val(dataset_train, args.validation)
                dataset_true_val.concat(dataset_val)
            dataset_val = dataset_true_val

        for i in range(3):  # Quick check to ensure same preprocessing between train/test
            assert abs(dataset_train.trsf.transforms[-1].mean[i] - dataset_val.trsf.transforms[-1].mean[i]) < 0.0001
            assert abs(dataset_train.trsf.transforms[-1].std[i] - dataset_val.trsf.transforms[-1].std[i]) < 0.0001

        loader_memory = None
        if task_id > 0 and memory is not None:
            dataset_memory = memory.get_dataset(dataset_train)
            loader_memory = factory.InfiniteLoader(factory.get_train_loaders(
                dataset_memory, args,
                args.replay_memory if args.replay_memory > 0 else args.batch_size
            ))
            if not args.sep_memory:
                for _ in range(args.oversample_memory):
                    dataset_train.add_samples(*memory.get())

        torch.cuda.empty_cache()

        #add specialized
        if task_id >0: # dont need to add for first tasks, already have 
            model_without_ddp.add_adapter()
            n_class = (task_id+1)* 2 #if  not ONE_REAL else 2 + task_id
            model_without_ddp.reset_classifier(n_class )
            n_parameters = sum(p.numel() for p in model_without_ddp.parameters() if p.requires_grad)
            if args.head_div:
                model_without_ddp.reset_div_head(one_real=False)
        else:
            # hardcoded for 2 tasks for inital task
            model_without_ddp.reset_classifier(2)

        model_without_ddp.to(device)

        if task_id > 0:
            model_without_ddp.freeze()
        # ----------------------------------------------------------------------

        # ----------------------------------------------------------------------
        # Data
        loader_train, loader_val = factory.get_loaders(dataset_train, dataset_val, args)


        # ----------------------------------------------------------------------
        # Learning rate and optimizer
        if task_id > 0 and args.incremental_batch_size:
            args.batch_size = args.incremental_batch_size

        if args.incremental_lr is not None and task_id > 0:
            linear_scaled_lr = args.incremental_lr * args.batch_size * utils.get_world_size() / 512.0
        else:
            linear_scaled_lr = base_lr * args.batch_size * utils.get_world_size() / 512.0

        args.lr = linear_scaled_lr

        model_without_ddp.set_active_adapter()
        optimizer = create_optimizer(args, model_without_ddp)
        lr_scheduler, _ = create_scheduler(args, optimizer)

        # ----------------------------------------------------------------------

        epoch = 0
        if task_id == 0:
            epochs = args.base_epochs
        else:
            epochs = args.epochs

        if args.distributed:
            del model
            model = torch.nn.parallel.DistributedDataParallel(model_without_ddp, device_ids=[args.gpu], find_unused_parameters=True)
            torch.distributed.barrier()
        else:
            model = model_without_ddp

        model_without_ddp.nb_epochs = epochs
        model_without_ddp.nb_batch_per_epoch = len(loader_train)

        # sam support removed
        sam = None

        print(f"Start training for {epochs} epochs")
        max_accuracy = 0.0
        for epoch in range(0, epochs):
            if args.distributed:
                loader_train.sampler.set_epoch(epoch)

            assert model.module.model.active_adapters[0] == model.module.adapter_names[-1]
            assert len(model.module.model.active_adapters) == 1 
             
            train_stats = train_one_epoch(
                model, criterion, loader_train,
                optimizer, device, epoch, task_id, loss_scaler,
                args.clip_grad, mixup_fn,
                debug=args.debug,
                args=args,
                teacher_model=None,
                model_without_ddp=model_without_ddp,
                sam=sam,
                loader_memory=loader_memory
            )

            lr_scheduler.step(epoch)

            if args.save_every_epoch is not None and epoch % args.save_every_epoch == 0:
                if os.path.isdir(args.resume):
                    with open(os.path.join(args.resume, 'save_log.txt'), 'w+') as f:
                        f.write(f'task={task_id}, epoch={epoch}\n')

                    checkpoint_paths = [os.path.join(args.resume, f'checkpoint_{task_id}.pth')]
                    for checkpoint_path in checkpoint_paths:
                        if (task_id < args.start_task and args.start_task > 0) and os.path.isdir(args.resume) and os.path.exists(checkpoint_path):
                            continue

                        utils.save_on_master({
                            'model': model_without_ddp.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'lr_scheduler': lr_scheduler.state_dict(),
                            'epoch': epoch,
                            'task_id': task_id,
                            'scaler': loss_scaler.state_dict(),
                            'args': args,
                        }, checkpoint_path)
                

            if args.eval_every and (epoch % args.eval_every  == 0 ):
                eval_and_log(
                    args, output_dir, model, model_without_ddp, optimizer, lr_scheduler,
                    epoch, task_id, loss_scaler, max_accuracy,
                    [], n_parameters, device, loader_val, train_stats, None, long_log_path,
                    logger, model_without_ddp.epoch_log()
                )
                logger.end_epoch()




        if memory is not None and args.distributed_memory:
            task_memory_path = os.path.join(args.resume, f'dist_memory_{task_id}-{utils.get_rank()}.npz')
            if os.path.isdir(args.resume) and os.path.exists(task_memory_path):
                # Resuming this task step, thus reloading saved memory samples
                # without needing to re-compute them
                memory.load(task_memory_path)
            else:
                task_set_to_rehearse = scenario_train[task_id]
                if args.rehearsal_test_trsf:
                    task_set_to_rehearse.trsf = scenario_val[task_id].trsf

                memory.add(task_set_to_rehearse, model, args.initial_increment if task_id == 0 else args.increment)
                #memory.add(scenario_train[task_id], model, args.initial_increment if task_id == 0 else args.increment)

                if args.resume != '':
                    memory.save(task_memory_path)
                else:
                    memory.save(os.path.join(args.output_dir, f'dist_memory_{task_id}-{utils.get_rank()}.npz'))

        if memory is not None and not args.distributed_memory:
            task_memory_path = os.path.join(args.resume, f'memory_{task_id}.npz')
            if utils.is_main_process():
                if os.path.isdir(args.resume) and os.path.exists(task_memory_path):
                    # Resuming this task step, thus reloading saved memory samples
                    # without needing to re-compute them
                    memory.load(task_memory_path)
                else:
                    task_set_to_rehearse = scenario_train[task_id]
                    if args.rehearsal_test_trsf:
                        task_set_to_rehearse.trsf = scenario_val[task_id].trsf

                    memory.add(task_set_to_rehearse, model, args.initial_increment if task_id == 0 else args.increment)

                    if args.resume != '':
                        memory.save(task_memory_path)
                    else:
                        memory.save(os.path.join(args.output_dir, f'memory_{task_id}-{utils.get_rank()}.npz'))

            assert len(memory) <= args.memory_size, (len(memory), args.memory_size)
            torch.distributed.barrier()

            if not utils.is_main_process():
                if args.resume != '':
                    memory.load(task_memory_path)
                else:
                    memory.load(os.path.join(args.output_dir, f'memory_{task_id}-0.npz'))
                    memory.save(os.path.join(args.output_dir, f'memory_{task_id}-{utils.get_rank()}.npz'))

            torch.distributed.barrier()

        # ----------------------------------------------------------------------       
        # eval and log 
        # ----------------------------------------------------------------------         
        print(f'done task {task_id} eval and log')
        _, json_data = eval_and_log(
            args, output_dir, model, model_without_ddp, optimizer, lr_scheduler,
            epoch, task_id, loss_scaler, max_accuracy,
            accuracy_list, n_parameters, device, loader_val, train_stats, log_store, log_path,
            logger, model_without_ddp.epoch_log(), False
        )
        task_json_data.append(json_data)
        print('TASK JSON DATA ')
        print(task_json_data)
        logger.end_task()

        nb_classes += args.increment

    print('-DONE TRAINING----------------------------------')
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    #print(f'Setting {args.data_set} with {args.initial_increment}-{args.increment}')
    #print(f"All accuracies: {accuracy_list}")
    #print(f"Average Incremental Accuracy: {statistics.mean(accuracy_list)}")


    print(f"+---------------------Final Accuracy-------------------------+")
    from continual.engine import evaluate
    for task_id, dataset_train in enumerate(scenario_train):
        dataset_val = scenario_val[task_id]
        loader_train, loader_val = factory.get_loaders(dataset_train, dataset_val, args)
        print(str(task_id))
        print("num images: " + str(len(loader_val.dataset)))
        test_stats = evaluate(loader_val, model, device, logger, ONE_REAL)
    print(f"+------------------------------------------------------------+")


    print("-EXPORTING ACCURACY MATRIX TABLE----------------------------------")
    config_path = os.path.join(args.output_dir,f"config_{args.trial_id}.json")
    export_path = os.path.join(args.output_dir, f"metrics_{args.trial_id}.txt")
    plot_export_path = os.path.join(args.output_dir, f"metrics_plot_{args.trial_id}.png")
    print("log path: {}".format(log_path))
    print("config path: {}".format(config_path))
    print("metrics export path: {}".format(export_path))
    print("plot export path: {}".format(plot_export_path))
    print('rpcess metrics ')
    print(args.model, args.name)
    print(task_json_data)
    process_metrics(task_json_data, config_path, args.model, args.name, export_path, plot_export_path)



def load_options(args, options):
    varargs = vars(args)

    name = []
    for o in options:
        with open(o) as f:
            new_opts = yaml.safe_load(f)
        for k, v in new_opts.items():
            if k not in varargs:
                raise ValueError(f'Option {k}={v} doesnt exist!')
        varargs.update(new_opts)
        name.append(o.split("/")[-1].replace('.yaml', ''))

    return '_'.join(name) 


if __name__ == '__main__':
    parser = argparse.ArgumentParser('LoRAX training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    args.debug = False
    utils.init_distributed_mode(args)

    if args.options:
        name = load_options(args, args.options)
        if not args.name:
            args.name = name

    args.output_basedir = os.path.join(
        args.output_basedir, args.data_set.lower(), args.model,str(args.epochs),str(args.memory_size)
    )


    # for testing multiple class orders 
    if isinstance(args.class_order, list) and isinstance(args.class_order[0], list):
        print(f'Running {len(args.class_order)} different class orders.')
        class_orders = copy.deepcopy(args.class_order)

        for i, order in enumerate(class_orders, start=1):
            print(f'Running class ordering {i}/{len(class_orders)}.')
            args.trial_id = i
            args.class_order = order
            main(args)
    else:
        args.trial_id = 1
        main(args)
