#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 15:10:08 2023

@author: minyoungpark
"""

from tqdm import tqdm
import numpy as np
from torch.utils.data import Dataset


class CustomTimeSeriesDataset(Dataset):
    def __init__(self, df, col_names, seq_len, pred_len, target_type='processed'):
        self.df = df
        self.target_type = target_type
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.data, self.target = self._transform(df, col_names)
        
    def _transform(self, df, col_names):
        data = []
        target = []
        tic_wise_col = col_names['tic_wise']
        date_wise_col = col_names['date_wise']
        print("Building dataset...")
        for i in tqdm(df.time_idx.drop_duplicates()[:-(self.seq_len+self.pred_len)+1]):
            seq_idx = np.arange(i, i+self.seq_len)
            pred_idx = np.arange(i+self.seq_len, i+self.seq_len+self.pred_len)
            processed_data = df[df.time_idx.isin(seq_idx)][self.target_type].to_numpy().reshape(self.seq_len, -1)
            # close_data = df[df.time_idx.isin(seq_idx)]['close'].to_numpy().reshape(self.seq_len, -1)
            frame_data = df[df.time_idx.isin(seq_idx)][tic_wise_col].to_numpy().reshape(self.seq_len, -1)
            index_data = df[df.time_idx.isin(seq_idx)][date_wise_col].drop_duplicates().to_numpy().astype(np.float32).reshape(self.seq_len, -1)
            data.append(np.concatenate([processed_data, frame_data, index_data], axis=1))
            target.append(df[df.time_idx.isin(pred_idx)][self.target_type].to_numpy().reshape(self.pred_len, -1))
            
        return data, target

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        input = self.data[idx]
        target = self.target[idx]
        
        input = np.stack(input, axis=0)
        target = np.stack(target, axis=0)
        
        return input, target