# -*- coding: utf-8 -*-
"""
Created on Sat May  6 16:47:22 2023

@author: robin
"""

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

from config import get_config
from models.generator import TSCNet
from core.function import compressed_stft, uncompressed_istft, power_compress
from utils.compute_metrics import compute_metrics

model_names = ['diffuse', 'tsc-diffuse'
               ]

def parse_option():
    parser = ArgumentParser(description='runs inference on a spectrogram file generated by diffwave.preprocess')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='diffuse',
                        choices=model_names,
                        help='model architecture: ' +
                            ' | '.join(model_names) +
                            ' (default: diffuse)')
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
    parser.add_argument('--start', default=None, type=int,
                        help='Start epoch to validate')
    parser.add_argument('--end', default=None, type=int,
                        help='End epoch to validate')
    parser.add_argument('--gpu', default=0, type=int,
                        help='GPU id to use.')
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
    args, unparsed = parser.parse_known_args()
    
    config = get_config(args)

    return args, config
    

def load_model(model_path, config, device=torch.device('cuda')):
    model = TSCNet(num_channel=64, num_features=config.N_FFT// 2 + 1).to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint['gen_state_dict']
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove 'module.'
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)    
    model.eval()
    
    return model


@torch.no_grad()
def predict(model, config, noisy_signal, device=torch.device('cuda')):
    noisy = torch.tensor(noisy_signal).unsqueeze(0).to(device)
    hamming_window = torch.hamming_window(config.N_FFT).to(device)
    c = torch.sqrt(noisy.size(-1) / torch.sum((noisy ** 2.0), dim=-1))
    noisy = torch.transpose(noisy, 0, 1)
    noisy = torch.transpose(noisy * c, 0, 1)

    length = noisy.size(-1)
    frame_num = int(np.ceil(length / 100))
    padded_len = frame_num * 100
    padding_len = padded_len - length
    noisy = torch.cat([noisy, noisy[:, :padding_len]], dim=-1)
    
    noisy_spec = compressed_stft(noisy, config.N_FFT, config.HOP_SAMPLES, hamming_window)
    est_real, est_imag = model(noisy_spec)
    est_real, est_imag = est_real.permute(0, 1, 3, 2), est_imag.permute(0, 1, 3, 2)
    est_complex = torch.complex(est_real, est_imag).squeeze(1)
    est_audio = uncompressed_istft(est_complex, config.N_FFT, config.HOP_SAMPLES, 
                                   hamming_window)
    est_audio = est_audio / c
    est_audio = torch.flatten(est_audio)[:length].cpu().numpy()
    
    assert len(est_audio) == length, "Estimated audio and the origin audio must have the same length"
    
    return est_audio

def inference(args, config, model_path, data_paths):
    device = 'cuda:{}'.format(args.gpu)
    model = load_model(model_path, config, device)
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
        
        audio = predict(model, config, noisy_signal, device)
        metrics = compute_metrics(clean_signal, audio, 16000, 0)
        metrics = np.array(metrics)
        metrics_total += metrics
        if args.save:
            output_name = os.path.join(output_path, noisy_file_path.split("/")[-1])
            torchaudio.save(output_name, torch.tensor(audio).unsqueeze(0), sample_rate=16000)
            
    return metrics_total
    
def main():
    args, config = parse_option()
    data_paths = sorted(glob(f'{config.DATA.TEST_NOISY_DIR}/*.wav', recursive=True))
    num = len(data_paths)
    
    if args.validate_epochs:
        best_pesq = 0
        best_epoch = 0
        for epoch in range(args.start, args.end):
            model_path = os.path.join(args.model_path, 'checkpoint_{:04d}.pth.tar'.format(epoch))
            metrics_total = inference(args, config, model_path, data_paths)
            metrics_avg = metrics_total / num
            print('Epoch: {}'.format(epoch))
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

# CDifuSE
# pesq: 2.205      csig: 3.634     cbak: 2.891     covl: 2.953     ssnr: 4.250     stoi: 0.901

# CMGAN
# pesq: 3.231      csig: 4.522     cbak: 3.804     covl: 3.958     ssnr: 10.34     stoi: 0.953
# pesq: 3.387	 csig: 4.623	 cbak: 3.914	 covl: 4.103	 ssnr: 10.879	 stoi: 0.957

# SCP-GAN
# pesq: 3.163	 csig: 4.371	 cbak: 3.760	 covl: 3.838	 ssnr: 10.163	 stoi: 0.954
# pesq: 3.115	 csig: 4.516	 cbak: 3.756	 covl: 3.887	 ssnr: 10.461	 stoi: 0.956
# pesq: 3.121	 csig: 4.521	 cbak: 3.757	 covl: 3.892	 ssnr: 10.440	 stoi: 0.956

# SCP-GAN PyTorch 2.0 last checkpoint
# pesq: 2.673	 csig: 4.033	 cbak: 3.537	 covl: 3.401	 ssnr: 10.420	 stoi: 0.943
# pesq: 2.533	 csig: 3.827	 cbak: 3.459	 covl: 3.229	 ssnr: 10.185	 stoi: 0.948
# pesq: 2.535	 csig: 3.836	 cbak: 3.458	 covl: 3.235	 ssnr: 10.160	 stoi: 0.948
# 