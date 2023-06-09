#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 16:35:30 2023

@author: minyoungpark
"""

import torch.nn as nn

def build_criterion(criterion_type):
    if criterion_type.lower() == "mae" or criterion_type.lower() == "l1":
        print('Criterion: MAE Loss')
        criterion  = nn.L1Loss()
    elif criterion_type.lower() == "mse" or criterion_type.lower() == "l2":
        print('Criterion: MSE Loss')
        criterion  = nn.MSELoss()
    else:
        print("Invalid criterion!")

    return criterion
