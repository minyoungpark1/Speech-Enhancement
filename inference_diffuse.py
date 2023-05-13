# -*- coding: utf-8 -*-
"""
Created on Sat May  6 00:15:37 2023

@author: robin
"""

# Copyright 2020 LMNT, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import os
import torch
import random
import librosa
import posixpath
import torchaudio
import numpy as np
from glob import glob
from tqdm import tqdm
from pathlib import PureWindowsPath, Path
from collections import OrderedDict

from argparse import ArgumentParser

random.seed(23)

from models.DiffuSE import DiffuSE
from config import get_config
from utils.compute_metrics import compute_metrics


def parse_option():
    parser = ArgumentParser(description='runs inference on a spectrogram file generated by diffwave.preprocess')
    parser.add_argument('--output', '-o', type=str, required=True, 
                        help='output directory name')
    parser.add_argument('--model_path', '-m', type=str, required=True, metavar="FILE", 
                        help='path to trained model')
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", 
                        help='path to config file', )
    parser.add_argument('--save', action='store_true',
                        help='Choose whether you will save the results or not')
    parser.add_argument('--validate-epochs', action='store_true',
                        help='Validate all epoch checkpoints')
    
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
    parser.add_argument('--fast', dest='fast', action='store_true',
    help='fast sampling procedure')
    args, unparsed = parser.parse_known_args()
    
    config = get_config(args)

    return args, config
    

def load_model(model_path, config, device=torch.device('cuda')):
    model = DiffuSE(
        config.DILATION_CYCLE_LENGTH,
        config.HOP_SAMPLES,
        config.N_SPECS,
        config.NOISE_SCHEDULE,
        config.RESIDUAL_CHANNELS,
        config.RESIDUAL_LAYERS,
        ).to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint['state_dict']
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove 'module.'
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)    
    model.eval()
    
    return model


def inference_schedule(model, config, fast_sampling=False):
    training_noise_schedule = np.array(config.NOISE_SCHEDULE)
    inference_noise_schedule = np.array(model.INFERENCE_NOISE_SCHEDULE) \
        if fast_sampling else training_noise_schedule

    talpha = 1 - training_noise_schedule
    talpha_cum = np.cumprod(talpha)
    beta = inference_noise_schedule
    alpha = 1 - beta
    alpha_cum = np.cumprod(alpha)
    sigmas = [0 for i in alpha]
    for n in range(len(alpha) - 1, -1, -1):
        sigmas[n] = ((1.0 - alpha_cum[n-1]) / (1.0 - alpha_cum[n]) * beta[n])

    T = []
    for s in range(len(inference_noise_schedule)):
        for t in range(len(training_noise_schedule) - 1):
            if talpha_cum[t+1] <= alpha_cum[s] <= talpha_cum[t]:
                twiddle = (talpha_cum[t]**0.5 - alpha_cum[s]**0.5) / \
                    (talpha_cum[t]**0.5 - talpha_cum[t+1]**0.5)
                T.append(t + twiddle)
                break
    T = np.array(T, dtype=np.float32)

    m = [0 for i in alpha]
    gamma = [0 for i in alpha]
    delta = [0 for i in alpha]
    d_x = [0 for i in alpha]
    d_y = [0 for i in alpha]
    delta_cond = [0 for i in alpha]
    delta_bar = [0 for i in alpha]
    c1 = [0 for i in alpha]
    c2 = [0 for i in alpha]
    c3 = [0 for i in alpha]
    oc1 = [0 for i in alpha]
    oc3 = [0 for i in alpha]

    for n in range(len(alpha)):
        m[n] = min(((1- alpha_cum[n])/(alpha_cum[n]**0.5)),1)**0.5
        
    m[-1] = 1

    for n in range(len(alpha)):
        delta[n] = max(1-(1+m[n]**2)*alpha_cum[n],0)
        gamma[n] = sigmas[n]

    for n in range(len(alpha)):
        if n >0:
            d_x[n] = (1-m[n])/(1-m[n-1]) * (alpha[n]**0.5)
            d_y[n] = (m[n]-(1-m[n])/(1-m[n-1])*m[n-1])*(alpha_cum[n]**0.5)
            delta_cond[n] = delta[n] - (((1-m[n])/(1-m[n-1])))**2 * alpha[n] * \
                delta[n-1]
            delta_bar[n] = (delta_cond[n])* delta[n-1]/ delta[n]
        else:
            d_x[n] = (1-m[n])* (alpha[n]**0.5)
            d_y[n]= (m[n])*(alpha_cum[n]**0.5)
            delta_cond[n] = 0
            delta_bar[n] = 0

    for n in range(len(alpha)):
        oc1[n] = 1 / alpha[n]**0.5
        oc3[n] = oc1[n] * beta[n] / (1 - alpha_cum[n])**0.5
        if n >0:
            c1[n] = (1-m[n])/(1-m[n-1])*(delta[n-1]/delta[n])*alpha[n]**0.5 + \
                (1-m[n-1])*(delta_cond[n]/delta[n])/alpha[n]**0.5
            c2[n] = (m[n-1] * delta[n] - (m[n] *(1-m[n]))/(1-m[n-1])*alpha[n]*delta[n-1])*\
                (alpha_cum[n-1]**0.5/delta[n])
            c3[n] = (1-m[n-1])*(delta_cond[n]/delta[n])*(1-alpha_cum[n])**0.5/(alpha[n])**0.5
        else:
            c1[n] = 1 / alpha[n]**0.5
            c3[n] = c1[n] * beta[n] / (1 - alpha_cum[n])**0.5
            
    return alpha, beta, alpha_cum,sigmas, T, c1, c2, c3, delta, delta_bar


@torch.no_grad()
def predict(model, config, noisy_signal, alpha, beta, alpha_cum, 
            sigmas, T, c1, c2, c3, delta, delta_bar, device=torch.device('cuda')):
    # Expand rank 2 tensors by adding a batch dimension.
    hamming_window = torch.hamming_window(config.N_FFT).cuda()
    noisy_signal = torch.from_numpy(noisy_signal).to(device)
    spectrogram = torch.stft(noisy_signal, config.N_FFT, config.HOP_SAMPLES, 
                             window=hamming_window, onesided=True, return_complex=True)
    if len(spectrogram.shape) == 2:
        spectrogram = spectrogram.unsqueeze(0)
        
    spectrogram = spectrogram.to(device)
    audio = torch.randn(spectrogram.shape[0], 
                        config.HOP_SAMPLES * spectrogram.shape[-1], device=device)
    noise_scale = torch.from_numpy(alpha_cum**0.5).float().unsqueeze(1).to(device)
    noisy_audio = torch.zeros(spectrogram.shape[0], 
                              config.HOP_SAMPLES * spectrogram.shape[-1], device=device)
    noisy_audio[:,:noisy_signal.shape[0]] = noisy_signal
    audio = noisy_audio
    gamma = [0.2]
    for n in range(len(alpha) - 1, -1, -1):
        if n > 0:
            predicted_noise =  model(audio, spectrogram, 
                                     torch.tensor([T[n]], device=audio.device)).squeeze(1)
            audio = c1[n] * audio + c2[n] * noisy_audio - c3[n] * predicted_noise
            noise = torch.randn_like(audio)
            newsigma = delta_bar[n]**0.5
            audio += newsigma * noise
        else:
            predicted_noise =  model(audio, spectrogram, 
                                     torch.tensor([T[n]], device=audio.device)).squeeze(1)
            audio = c1[n] * audio - c3[n] * predicted_noise
            audio = (1-gamma[n])*audio+gamma[n]*noisy_audio
            audio = torch.clamp(audio, -1.0, 1.0)
            
    return audio

def inference(args, config, model_path, data_paths):
    model = load_model(model_path, config)
    alpha, beta, alpha_cum, sigmas, T, c1, c2, c3, \
        delta, delta_bar = inference_schedule(model, config, fast_sampling=args.fast)
    
    metrics_total = np.zeros(6)
    
    for i, noisy_file_path in tqdm(enumerate(data_paths)):
        if isinstance(Path(noisy_file_path), PureWindowsPath):
            noisy_file_path = noisy_file_path.replace(os.sep, posixpath.sep)
        if i == 0:
            output_path = Path(os.path.join(args.output, noisy_file_path.split("/")[-2]))
            output_path.mkdir(parents=True, exist_ok=True)
        
        clean_file_path = noisy_file_path.replace(config.DATA.TEST_NOISY_DIR, 
                                                  config.DATA.TEST_CLEAN_DIR)
        noisy_signal, _ = librosa.load(noisy_file_path, sr=16000)
        clean_signal, _ = librosa.load(clean_file_path, sr=16000)
        
        wlen = noisy_signal.shape[0]
        audio = predict(model, config, noisy_signal, alpha, beta, 
                            alpha_cum, sigmas, T,c1, c2, c3, delta, delta_bar)
        audio = audio[:,:wlen]
        metrics = compute_metrics(clean_signal, torch.flatten(audio).cpu().numpy(), 16000, 0)
        metrics = np.array(metrics)
        metrics_total += metrics

        # audio = snr_process(audio,noisy_signal)
        if args.save:
            output_name = os.path.join(output_path, noisy_file_path.split("/")[-1])
            torchaudio.save(output_name, audio.cpu(), sample_rate=16000)
        return metrics_total
    
def main():
    args, config = parse_option()
    
    data_paths = glob(f'{config.DATA.TEST_NOISY_DIR}/*.wav', recursive=True)
    num = len(data_paths)
    
    if args.validate_epochs:
        model_paths = sorted(glob(f'{args.model_path}/checkpoint*', recursive=True))
        best_pesq = 0
        best_epoch = 0
        for model_path in model_paths:
            epoch = int(os.path.basename(model_path).split('.')[-3].split('_')[-1])
            metrics_total = inference(args, config, model_path, data_paths)

            metrics_avg = metrics_total / num
            print(f'pesq: {metrics_avg[0]:.3f}\t '\
                  f'csig: {metrics_avg[1]:.3f}\t '\
                  f'cbak: {metrics_avg[2]:.3f}\t '\
                  f'covl: {metrics_avg[3]:.3f}\t '\
                  f'ssnr: {metrics_avg[4]:.3f}\t '\
                  f'stoi: {metrics_avg[5]:.3f}')
                
            if metrics_avg[0] > best_pesq:
                best_pesq = metrics_avg[0]
                best_epoch = epoch
        print(f'Best epoch: {best_epoch}\t best PESQ: {best_pesq}')
        
    else:
        metrics_total = inference(args, config, args.model_path, data_paths)
        metrics_avg = metrics_total / num
        print(f'pesq: {metrics_avg[0]:.3f}\t '\
              f'csig: {metrics_avg[1]:.3f}\t '\
              f'cbak: {metrics_avg[2]:.3f}\t '\
              f'covl: {metrics_avg[3]:.3f}\t '\
              f'ssnr: {metrics_avg[4]:.3f}\t '\
              f'stoi: {metrics_avg[5]:.3f}')
        

if __name__ == '__main__':
    main()
