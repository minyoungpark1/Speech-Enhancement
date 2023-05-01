# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 15:58:37 2023

@author: robin
"""

import numpy as np
import os
import random
import torch
import librosa
from pathlib import PureWindowsPath, PurePosixPath, Path

from tqdm import tqdm
from glob import glob
from torch.utils.data.distributed import DistributedSampler


class VoicebankDataset(torch.utils.data.Dataset):
    def __init__(self, wav_path, noisy_path, npy_dir, se, voicebank=False,
                 samples_per_frame=256, crop_mel_frames=62
                 ):
        super().__init__()
        self.wav_path = wav_path
        self.noisy_path = noisy_path
        self.specnames = []
        self.se = se
        self.voicebank = voicebank
        npy_paths = [os.path.join(npy_dir, os.path.basename(wav_path)),
                     os.path.join(npy_dir, os.path.basename(noisy_path))]
        for path in npy_paths:
            self.specnames += glob(f'{path}/*.wav.spec.npy', recursive=True)
        self.samples_per_frame = samples_per_frame
        self.crop_mel_frames = crop_mel_frames
            
    def __len__(self):
        return len(self.specnames)
    
    def _get_data(self, idx):
        spec_filename = self.specnames[idx]
        if isinstance(Path(spec_filename), PureWindowsPath):
            spec_filename = str(PurePosixPath(spec_filename))
        if self.voicebank:
            spec_path = "/".join(spec_filename.split("/")[:-1])
            audio_filename = spec_filename.replace(spec_path, self.wav_path).replace(".spec.npy", "")
            noisy_filename = spec_filename.replace(spec_path, self.noisy_path).replace(".spec.npy", "")
        else:
            spec_path = "/".join(spec_filename.split("/")[:-2])+"/"
            if self.se:
                audio_filename = spec_filename.replace(spec_path, self.wav_path).replace(".wav.spec.npy", ".Clean.wav")
            else:
                audio_filename = spec_filename.replace(spec_path, self.wav_path).replace(".spec.npy", "")
            
        signal, _ = librosa.load(audio_filename, sr=16000)
        noisy_signal, _ = librosa.load(noisy_filename, sr=16000)  

        spectrogram = np.load(spec_filename)
        return signal, noisy_signal, spectrogram
        # return {
        #     'audio': signal,
        #     'noisy': noisy_signal,
        #     'spectrogram': spectrogram.T
        #     }
    
    def random_cropping(self, signal, noisy_signal, spectrogram):
        start = random.randint(0, spectrogram.shape[1] - self.crop_mel_frames)
        end = start + self.crop_mel_frames
        spectrogram = spectrogram[:, start:end]
      
        start *= self.samples_per_frame
        end *= self.samples_per_frame
        signal = signal[start:end]
        signal = np.pad(signal, (0, (end-start) - len(signal)), 
                                 mode='constant')
        noisy_signal = noisy_signal[start:end]
        noisy_signal = np.pad(noisy_signal, (0, (end-start) - len(noisy_signal)), 
                              mode='constant')
        return signal, noisy_signal, spectrogram
        
    def __getitem__(self, idx):
        signal, noisy_signal, spectrogram = self._get_data(idx)
        signal, noisy_signal, spectrogram = self.random_cropping(signal, 
                                                                noisy_signal, 
                                                                spectrogram)
        
        return torch.tensor(signal),\
            torch.tensor(noisy_signal),\
            torch.tensor(spectrogram)
    

# class Collator:
#   def __init__(self, params):
#     self.params = params

#   def collate(self, minibatch):
#     samples_per_frame = self.params.hop_samples
#     for record in tqdm(minibatch):
#       # Filter out records that aren't long enough.
#       if len(record['spectrogram']) < self.params.crop_mel_frames:
#         del record['spectrogram']
#         del record['audio']
#         del record['noisy']
#         continue

#       start = random.randint(0, record['spectrogram'].shape[0] - self.params.crop_mel_frames)
#       end = start + self.params.crop_mel_frames
#       record['spectrogram'] = record['spectrogram'][start:end].T

#       start *= samples_per_frame
#       end *= samples_per_frame
#       record['audio'] = record['audio'][start:end]
#       record['audio'] = np.pad(record['audio'], (0, (end-start) - len(record['audio'])), mode='constant')
#       record['noisy'] = record['noisy'][start:end]
#       record['noisy'] = np.pad(record['noisy'], (0, (end-start) - len(record['noisy'])), mode='constant')

#     audio = np.stack([record['audio'] for record in minibatch if 'audio' in record])
#     noisy = np.stack([record['noisy'] for record in minibatch if 'noisy' in record])
#     spectrogram = np.stack([record['spectrogram'] for record in minibatch if 'spectrogram' in record])
#     return {
#         'audio': torch.from_numpy(audio),
#         'noisy': torch.from_numpy(noisy),
#         'spectrogram': torch.from_numpy(spectrogram),
#     }    