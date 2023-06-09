#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 15:39:18 2023

@author: minyoungpark
"""

import argparse
import builtins
import os
import random
import warnings

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
from timm.scheduler.cosine_lr import CosineLRScheduler

from config import get_config
from models.DiffuSE import DiffuSE
from models.tsc_diffusion import TSCNet
from datasets.voicebank_dataset import VoicebankDataset, Collator
from core.function import train, validate, train_tsc_diffusion, validate_tsc_diffusion
from core.criterion import build_criterion
from core.optimizer import build_optimizer
from utils.utils import create_logger, save_checkpoint

model_names = ['diffuse', 'tsc-diffuse'
               ]

def parse_option():
    parser = argparse.ArgumentParser(description='Speech enhancement training script')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='diffuse',
                        choices=model_names,
                        help='model architecture: ' +
                            ' | '.join(model_names) +
                            ' (default: diffuse)')

    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
    parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                        help='number of data loading workers (default: 32)')
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=64, type=int,
                        metavar='N',
                        help='mini-batch size (default: 4096), this is the total '
                             'batch size of all GPUs on all nodes when '
                             'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                        metavar='LR', help='initial (base) learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=0.01, type=float,
                        metavar='W', help='weight decay (default: 1e-6)',
                        dest='weight_decay')
    parser.add_argument('--max-norm', default=0.0, type=float, metavar='M',
                        help='maximum normalization threshold of the gradients')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')
    parser.add_argument('--debug', action='store_true',
                        help='Use torch.autograd.set_detect_anomaly(True) for debudding')
    
    parser.add_argument('--optimizer', default='sgd', type=str,
                        choices=['sgd', 'adamw', 'lars', 'lamb'],
                        help='optimizer used (default: sgd)')
    parser.add_argument('--criterion', default='l1', type=str,
                        choices=['mae', 'l1', 'mse', 'l2', 'quantile'],
                        help='criterion used (default: l1)')
    
    parser.add_argument('--crop-len', default=1, type=int,
                        help='Length to crop audio signals in second (default: 1 sec)')
    parser.add_argument('--comp-type', default='pow', type=str,
                        choices=['norm', 'log', 'pow', 'none'],
                        help='Type of final spectrogram compression method (default: pow)')

    args, unparsed = parser.parse_known_args()
    config = get_config(args)

    return args, config


def main():
    args, config = parse_option()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args, config))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args, config)


def main_worker(gpu, ngpus_per_node, args, config):
    args.gpu = gpu

    # suppress printing if not first GPU on each node
    if args.multiprocessing_distributed and (args.gpu != 0 or args.rank != 0):
        def print_pass(*args):
            pass
        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
        torch.distributed.barrier()
    # create model
    print("=> creating model '{}'".format(args.arch))
    print(len(config.NOISE_SCHEDULE))
    if args.arch.startswith('diffuse'):
        model = DiffuSE(
            config.DILATION_CYCLE_LENGTH,
            config.HOP_SAMPLES,
            config.N_SPECS,
            config.NOISE_SCHEDULE,
            config.RESIDUAL_CHANNELS,
            config.RESIDUAL_LAYERS,
            )
    elif args.arch.startswith('tsc'):
        model = TSCNet(num_channel=64, num_features=config.N_FFT// 2 + 1,
                       noise_schedule=config.NOISE_SCHEDULE)
    
    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # apply SyncBN
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / args.world_size)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu],)
            # find_unused_parameters=True)
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        # comment out the following line for debugging
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather/rank implementation in this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    print(model) # print model after SyncBatchNorm

    optimizer = build_optimizer(args, model)

    scaler = torch.cuda.amp.GradScaler()

    best_loss = 1e8
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scaler.load_state_dict(checkpoint['scaler'])
            best_loss = checkpoint['best_loss']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    logger = create_logger(output_dir=config.OUTPUT, dist_rank=args.rank, name=f"{args.arch}")

    # Data loading code
    train_dataset = VoicebankDataset(config.DATA.TRAIN_CLEAN_DIR,
                                     config.DATA.TRAIN_NOISY_DIR,
                                     samples_per_frame=config.HOP_SAMPLES,
                                     crop_frames=config.CROP_FRAMES,
                                     random_crop=False)
    valid_dataset = VoicebankDataset(config.DATA.TEST_CLEAN_DIR,
                                     config.DATA.TEST_NOISY_DIR,
                                     samples_per_frame=config.HOP_SAMPLES,
                                     crop_frames=config.CROP_FRAMES,
                                     random_crop=False)

    print('Audio signals cropped to {} seconds long'.format(config.CROP_LEN))
    
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset)
    else:
        train_sampler = None
        valid_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=False,
        collate_fn=Collator(samples_per_frame=config.HOP_SAMPLES, 
                            crop_frames=config.CROP_FRAMES,
                            crop_len=config.CROP_LEN).collate,)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=valid_sampler, drop_last=False,
        collate_fn=Collator(samples_per_frame=config.HOP_SAMPLES, 
                            crop_frames=config.CROP_FRAMES,
                            crop_len=config.CROP_LEN).collate,)

    criterion = build_criterion(args.criterion).cuda(args.gpu)

    # lr_scheduler = CosineLRScheduler(
    #             optimizer,
    #             t_initial=len(train_loader)*args.epochs//4,
    #             cycle_decay=0.5,
    #             lr_min=args.lr*1e-2,
    #             warmup_lr_init=args.lr*1e-3,
    #             warmup_t=10,
    #             cycle_limit=4,
    #             t_in_epochs=True,
    #         )

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
            valid_sampler.set_epoch(epoch)
        
        if args.arch.startswith('diffuse'):
        # train & validate for one epoch
            train_loss = train(train_loader, model, criterion, optimizer, 
                               # lr_scheduler,
                               scaler, logger, epoch, args, config)
            valid_loss = validate(valid_loader, model, criterion, scaler, logger, 
                                  epoch, args, config)
        elif args.arch.startswith('tsc'):
            train_loss = train_tsc_diffusion(train_loader, model, criterion,
                                             optimizer, scaler, logger, epoch, 
                                             args, config)
            valid_loss = validate_tsc_diffusion(valid_loader, model, criterion, 
                                                scaler, logger, epoch, args, 
                                                config)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed \
                and args.rank == 0): # only the first GPU saves checkpoint

            if valid_loss <= best_loss:
                best_loss = valid_loss
                is_best = True
            else:
                is_best = False

            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'scaler': scaler.state_dict(),
                'best_loss': best_loss,
            }, config.OUTPUT, is_best=is_best, filename='checkpoint_{:04d}.pth.tar'.format(epoch))
            logger.info('=> saving checkpoint: checkpoint_{:04d}.pth.tar'.format(epoch))


        msg = 'Train Loss: {:.3f}\t Validation Loss: {:.3f}'.format(
                    train_loss, valid_loss)
        logger.info(msg)

if __name__ == '__main__':
    main()
