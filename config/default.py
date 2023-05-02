# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 22:39:20 2023

@author: robin
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import yaml
import numpy as np
from yacs.config import CfgNode as CN

_C = CN()

_C.SAMPLE_RATE = 16000
_C.N_SPECS = 201
_C.N_FFT = 400
_C.HOP_SAMPLES = 100
_C.CROP_FRAMES = 160
_C.RESIDUAL_LAYERS = 30
_C.RESIDUAL_CHANNELS = 64
_C.DILATION_CYCLE_LENGTH = 10
_C.NOISE_SCHEDULE = np.linspace(1e-4, 0.035, 50).tolist()
_C.INFERENCE_NOISE_SCHEDULE = [0.0001, 0.001, 0.01, 0.05, 0.2, 0.35]

# Dataset settings
# -----------------------------------------------------------------------------
_C.DATA = CN()
_C.DATA.TRAIN_CLEAN_DIR =  'D:/data/DS_10283_2791/clean_trainset_56spk_wav'
_C.DATA.TRAIN_NOISY_DIR = 'D:/data/DS_10283_2791/noisy_trainset_56spk_wav'
_C.DATA.TEST_CLEAN_DIR =  'D:/data/DS_10283_2791/clean_testset_wav'
_C.DATA.TEST_NOISY_DIR = 'D:/data/DS_10283_2791/noisy_testset_wav'
_C.DATA.NPY_DIR = 'D:/data/npy_data'
_C.DATA.BATCH_SIZE = 32

_C.TRAIN = CN()

_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.NAME = 'sgd'

_C.TRAIN.CRITERION = CN()
_C.TRAIN.CRITERION.NAME = 'l1'

_C.MODEL = CN()
_C.MODEL.NAME = 'diffuse'
_C.MODEL.RESUME = ''

# Path to output folder, overwritten by command line argument
_C.OUTPUT = ''
# Tag of experiment, overwritten by command line argument
_C.TAG = 'default'


def _update_config_from_file(config, cfg_file):
    config.defrost()
    with open(cfg_file, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    for cfg in yaml_cfg.setdefault('BASE', ['']):
        if cfg:
            _update_config_from_file(
                config, os.path.join(os.path.dirname(cfg_file), cfg)
            )
    print('=> merge config from {}'.format(cfg_file))
    config.merge_from_file(cfg_file)
    config.freeze()


def update_config(config, args):
    _update_config_from_file(config, args.cfg)

    config.defrost()
    if args.opts:
        config.merge_from_list(args.opts)

    def _check_args(name):
        if hasattr(args, name) and eval(f'args.{name}'):
            return True
        return False

    # merge from specific arguments
    if _check_args('batch_size'):
        config.DATA.BATCH_SIZE = args.batch_size
    if _check_args('arch'):
        config.MODEL.NAME = args.arch
    if _check_args('resume'):
        config.MODEL.RESUME = args.resume
    if _check_args('output'):
        config.OUTPUT = args.output

    ## Overwrite optimizer if not None, currently we use it for [fused_adam, fused_lamb]
    if _check_args('optim'):
        config.TRAIN.OPTIMIZER.NAME = args.optim

    # set local rank for distributed training
    config.RANK = args.rank

    # output folder
    config.OUTPUT = os.path.join(config.OUTPUT, config.MODEL.NAME, config.TAG)

    config.freeze()


def get_config(args):
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = _C.clone()
    update_config(config, args)

    return config
