# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 15:58:37 2023

@author: robin
"""

import numpy as np
import os
import random
import torch
import torchaudio
import librosa
from pathlib import PureWindowsPath, PurePosixPath, Path
import posixpath
from tqdm import tqdm
from glob import glob
from torch.utils.data.distributed import DistributedSampler
from models.discriminator import pesq_loss

class VoicebankDataset(torch.utils.data.Dataset):
    def __init__(self, clean_path, noisy_path, samples_per_frame=100, 
                 crop_frames=160, random_crop=False,):
        super().__init__()
        self.clean_path = clean_path
        self.noisy_path = noisy_path
        self.random_crop = random_crop
        self.data_paths = sorted(glob(f'{noisy_path}/*.wav', recursive=True))
        self.samples_per_frame = samples_per_frame
        self.crop_frames = crop_frames
            
    def __len__(self):
        return len(self.data_paths)
    
    def _get_data(self, idx):
        noisy_file_path = self.data_paths[idx]
        if isinstance(Path(noisy_file_path), PureWindowsPath):
            noisy_file_path = noisy_file_path.replace(os.sep, posixpath.sep)
        clean_file_path = noisy_file_path.replace(self.noisy_path, self.clean_path)
        signal, _ = librosa.load(clean_file_path, sr=16000)
        noisy_signal, _ = librosa.load(noisy_file_path, sr=16000)  
        
        return signal, noisy_signal
    
    def random_cropping(self, signal, noisy_signal):
        L = self.crop_frames*self.samples_per_frame
        start = random.randint(0, len(signal) - L)
        end = start + L
        signal = signal[start:end]
        noisy_signal = noisy_signal[start:end]
        return signal, noisy_signal
            
    def __getitem__(self, idx):
        signal, noisy_signal = self._get_data(idx)
        if self.random_crop:
            signal, noisy_signal = self.random_cropping(signal, noisy_signal)
        return {
                'audio': signal,
                'noisy': noisy_signal,
                }

class Collator:
    def __init__(self, samples_per_frame, crop_frames, crop_len=1):
        self.samples_per_frame = samples_per_frame
        self.crop_frames = crop_frames
        self.L = self.crop_frames*self.samples_per_frame
        self.crop_len = self.L*crop_len
    
    def recrop(self, record, chances):
        clean = record['audio']
        noisy = record['noisy']
        length = len(record['audio'])
        if length < self.crop_len:
            units = self.crop_len // length
            clean_final = []
            noisy_final = []
            for i in range(units):
                clean_final.append(clean)
                noisy_final.append(noisy)
            clean_final.append(clean[: self.crop_len%length])
            noisy_final.append(noisy[: self.crop_len%length])
            clean = np.concatenate(clean_final, axis=-1)
            noisy = np.concatenate(noisy_final, axis=-1)
        else:
            start = random.randint(0, length - self.crop_len)
            end = start + self.crop_len
            
            clean, noisy = clean[start:end], noisy[start:end]
        if pesq_loss(clean, noisy) == -1:
            chances -= 1
            succeeded = 0
        else:
            succeeded = 1
        return chances, succeeded, clean, noisy
        
    def collate(self, minibatch):
        for record in minibatch:
            chances, succeeded = 10, 0
            
            # Ten more chances to avoid getting a silent signal
            while chances > 0 and not succeeded:
                chances, succeeded, clean, noisy = self.recrop(record, chances)
            
            if succeeded:
                record['audio'], record['noisy'] = clean, noisy
            else:
                del record['audio']
                del record['noisy']
                continue
            # print(record['audio'].shape, record['noisy'].shape)
        audio = np.stack([record['audio'] for record in minibatch if 'audio' in record])
        noisy = np.stack([record['noisy'] for record in minibatch if 'noisy' in record])
        
        return {
            'audio': torch.from_numpy(audio),
            'noisy': torch.from_numpy(noisy),
            }
            
        
