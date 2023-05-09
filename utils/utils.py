#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 17:32:41 2023

@author: minyoungpark
"""
import os
import sys
import math
import torch
import shutil
import logging
import functools
from pathlib import Path
from termcolor import colored

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


@functools.lru_cache()
def create_logger(output_dir, dist_rank=0, name=''):
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)

    # create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # create formatter
    fmt = '[%(asctime)s %(name)s] (%(filename)s %(lineno)d): %(levelname)s %(message)s'
    color_fmt = colored('[%(asctime)s %(name)s]', 'green') + \
                colored('(%(filename)s %(lineno)d)', 'yellow') + ': %(levelname)s %(message)s'

    # create console handlers for master process
    if dist_rank == 0:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(
            logging.Formatter(fmt=color_fmt, datefmt='%Y-%m-%d %H:%M:%S'))
        logger.addHandler(console_handler)

    # create file handlers
    file_handler = logging.FileHandler(os.path.join(output_dir, f'log_rank{dist_rank}.txt'), mode='a')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(file_handler)

    return logger


def save_checkpoint(state, path, is_best, filename='checkpoint.pth.tar'):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    torch.save(state, os.path.join(path, filename))
    if is_best:
        shutil.copyfile(os.path.join(path, filename),
                        os.path.join(path, 'model_best.pth.tar'))


def adjust_learning_rate(optimizers, epoch, config):
    """Decays the learning rate with half-cycle cosine after warmup"""
    cycle_length = config.TRAIN.SCHEDULER.EPOCHS//(config.TRAIN.SCHEDULER.CYCLE_LIMIT)
    q, r = divmod(epoch, cycle_length)
    if r < config.TRAIN.SCHEDULER.WARMUP_EPOCHS:
        lr = 0.5**(q) * config.TRAIN.SCHEDULER.LR * r / config.TRAIN.SCHEDULER.WARMUP_EPOCHS
    else:
        lr = config.TRAIN.SCHEDULER.LR * (0.5**(q+1)) * (1. + math.cos(math.pi * (r - config.TRAIN.SCHEDULER.WARMUP_EPOCHS) / \
                                            (cycle_length - config.TRAIN.SCHEDULER.WARMUP_EPOCHS)))
    for optimizer in optimizers:
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    return lr+config.TRAIN.SCHEDULER.MIN_LR
