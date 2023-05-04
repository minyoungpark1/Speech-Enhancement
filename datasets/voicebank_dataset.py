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
    def __init__(self, wav_path, noisy_path, npy_dir, se, voicebank=False,
                 samples_per_frame=100, crop_frames=160, get_spec=True, 
                 random_crop=False,
                 ):
        super().__init__()
        self.wav_path = wav_path
        self.noisy_path = noisy_path
        self.specnames = []
        self.se = se
        self.voicebank = voicebank
        self.get_spec = get_spec
        self.random_crop = random_crop
        npy_paths = [
            # os.path.join(npy_dir, os.path.basename(wav_path)),
            os.path.join(npy_dir, os.path.basename(noisy_path))
            ]
        for path in npy_paths:
            self.specnames += glob(f'{path}/*.wav.spec.npy', recursive=True)
        self.samples_per_frame = samples_per_frame
        self.crop_frames = crop_frames
            
    def __len__(self):
        return len(self.specnames)
    
    def _get_data(self, idx):
        spec_filename = self.specnames[idx]
        if isinstance(Path(spec_filename), PureWindowsPath):
            # spec_filename = str(PurePosixPath(spec_filename))
            spec_filename = spec_filename.replace(os.sep, posixpath.sep)
            # print(spec_filename)
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
        
        # print(audio_filename)
        signal, _ = librosa.load(audio_filename, sr=16000)
        noisy_signal, _ = librosa.load(noisy_filename, sr=16000)  
        
        # signal, _ = torchaudio.load(audio_filename)
        # noisy_signal, _ = torchaudio.load(noisy_filename)
        # signal = signal.squeeze()
        # noisy_signal = noisy_signal.squeeze()
        
        if self.get_spec:
            spectrogram = np.load(spec_filename)
            return signal, noisy_signal, spectrogram
        else:
            return signal, noisy_signal, None
    
    def random_cropping(self, signal, noisy_signal, spectrogram=None):
        if spectrogram is not None:
            start = random.randint(0, spectrogram.shape[1] - self.crop_frames)
            end = start + self.crop_frames
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
        else:
            L = self.crop_frames*self.samples_per_frame
            start = random.randint(0, len(signal) - L)
            end = start + L
            signal = signal[start:end]
            # signal = np.pad(signal, (0, (end-start) - len(signal)), 
            #                          mode='constant')
            noisy_signal = noisy_signal[start:end]
            # noisy_signal = np.pad(noisy_signal, (0, (end-start) - len(noisy_signal)), 
            #                       mode='constant')
            return signal, noisy_signal, None
            
        
    def __getitem__(self, idx):
        signal, noisy_signal, spectrogram = self._get_data(idx)
        if self.random_crop:
            signal, noisy_signal, spectrogram = self.random_cropping(signal, 
                                                                    noisy_signal, 
                                                                    spectrogram)
        
        if self.get_spec:
            # return signal, noisy_signal, torch.tensor(spectrogram)
            return {
                    'audio': signal,
                    'noisy': noisy_signal,
                    'spectrogram': spectrogram
                }
        else:
            # return signal, noisy_signal
            return {
                    'audio': signal,
                    'noisy': noisy_signal,
                    }

class Collator:
    def __init__(self, samples_per_frame, crop_frames, get_spec=True):
        self.samples_per_frame = samples_per_frame
        self.crop_frames = crop_frames
        self.get_spec = get_spec
        self.L = self.crop_frames*self.samples_per_frame
    
    def recrop(self, record, chances):
        start = random.randint(0, len(record['audio']) - self.L)
        end = start + self.L
        
        clean, noisy = record['audio'][start:end], record['noisy'][start:end]
        if pesq_loss(clean, noisy) == -1:
            chances -= 1
            succeeded = 0
        else:
            succeeded = 1
        return chances, succeeded, start, end
        
    def collate(self, minibatch):
        for record in minibatch:
            if len(record['audio']) < self.L:
                del record['audio']
                del record['noisy']
                if self.get_spec:
                    del record['spectrogram']
            
            chances, succeeded = 5, 0
            
            # Five more chance to avoid getting a silent signal
            while chances > 0 and not succeeded:
                chances, succeeded, start, end = self.recrop(record, chances)
            
            if succeeded:
                record['audio'], record['noisy'] = record['audio'][start:end], record['noisy'][start:end]
                start = start//self.samples_per_frame
                end = end//self.samples_per_frame
                if self.get_spec:
                    record['spectrogram'] = record['spectrogram'][:,start:end]
            else:
                del record['audio']
                del record['noisy']
                if self.get_spec:
                    del record['spectrogram']
                continue
        
        if len(minibatch) == 1:
            audio = minibatch[0]['audio'].unsqueeze(0)
            noisy = minibatch[0]['noisy'].unsqueeze(0)
            if self.get_spec:
                spectrogram = minibatch[0]['spectrogram'].unsqueeze(0)
        else:
            audio = np.stack([record['audio'] for record in minibatch if 'audio' in record])
            noisy = np.stack([record['noisy'] for record in minibatch if 'noisy' in record])
            if self.get_spec:
                spectrogram = np.stack([record['spectrogram'] for record \
                                        in minibatch if 'spectrogram' in record])
        
        if self.get_spec:
            return {
                'audio': torch.from_numpy(audio),
                'noisy': torch.from_numpy(noisy),
                'spectrogram': torch.from_numpy(spectrogram),
                }
        else:
            return {
                'audio': torch.from_numpy(audio),
                'noisy': torch.from_numpy(noisy),
                }
            
        
