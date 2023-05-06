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

from models.generator import TSCNet
from core.function import power_compress, power_uncompress
from config import get_config
from utils.compute_metrics import compute_metrics


def parse_option():
    parser = ArgumentParser(description='runs inference on a spectrogram file generated by diffwave.preprocess')
    parser.add_argument('--output', '-o', type=str, required=True, 
                        help='output path name')
    parser.add_argument('--model_path', '-m', type=str, required=True, metavar="FILE", 
                        help='path to trained model')
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", 
                        help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
    args, unparsed = parser.parse_known_args()
    
    config = get_config(args)

    return args, config
    

def load_model(args, config, device=torch.device('cuda')):
    model = TSCNet(num_channel=64, num_features=config.N_FFT// 2 + 1).to(device)
    
    checkpoint = torch.load(args.model_path, map_location=device)
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
    
    c = torch.sqrt(noisy.size(-1) / torch.sum((noisy ** 2.0), dim=-1))
    noisy = torch.transpose(noisy, 0, 1)
    noisy = torch.transpose(noisy * c, 0, 1)

    length = noisy.size(-1)
    frame_num = int(np.ceil(length / 100))
    padded_len = frame_num * 100
    padding_len = padded_len - length
    noisy = torch.cat([noisy, noisy[:, :padding_len]], dim=-1)

    noisy_spec = torch.view_as_real(torch.stft(noisy, config.N_FFT, config.HOP_SAMPLES,
                                                window=torch.hamming_window(config.N_FFT).cuda(), 
                                                onesided=True, return_complex=True))
    noisy_spec = power_compress(noisy_spec).permute(0, 1, 3, 2)
    est_real, est_imag = model(noisy_spec)
    est_real, est_imag = est_real.permute(0, 1, 3, 2), est_imag.permute(0, 1, 3, 2)

    est_spec_uncompress = power_uncompress(est_real, est_imag).squeeze(1)
    est_audio = torch.istft(est_spec_uncompress, config.N_FFT, config.HOP_SAMPLES,
                            window=torch.hamming_window(config.N_FFT).cuda(),
                            onesided=True)
    est_audio = est_audio / c
    est_audio = torch.flatten(est_audio)[:length].cpu().numpy()
    
    assert len(est_audio) == length, "Estimated audio and the origin audio must have the same length"
    
    return est_audio


def main():
    args, config = parse_option()
    
    specnames = []
    npy_paths = [
        os.path.join(config.DATA.NPY_DIR, os.path.basename(config.DATA.TEST_NOISY_DIR))
        ]
    for path in npy_paths:
        specnames += glob(f'{path}/*.wav.spec.npy', recursive=True)

    model = load_model(args, config)
    num = len(specnames)
    metrics_total = np.zeros(6)

    for i, spec in tqdm(enumerate(specnames)):
        if isinstance(Path(spec), PureWindowsPath):
            spec = spec.replace(os.sep, posixpath.sep)
        if i == 0:
            output_path = Path(os.path.join(args.output, spec.split("/")[-2]))
            output_path.mkdir(parents=True, exist_ok=True)
            
        noisy_signal, _ = librosa.load(os.path.join(config.DATA.TEST_NOISY_DIR,
                                                    spec.split("/")[-1].replace(".spec.npy","")),
                                       sr=16000)
        clean_signal, _ = librosa.load(os.path.join(config.DATA.TEST_CLEAN_DIR,
                                                    spec.split("/")[-1].replace(".spec.npy","")),
                                       sr=16000)
        
        audio = predict(model, config, noisy_signal)
        metrics = compute_metrics(clean_signal, audio, 16000, 0)
        metrics = np.array(metrics)
        metrics_total += metrics

        output_name = os.path.join(output_path, spec.split("/")[-1].replace(".spec.npy", ""))
        torchaudio.save(output_name, torch.tensor(audio).unsqueeze(0), sample_rate=16000)
        
    metrics_avg = metrics_total / num
    print('pesq: ', metrics_avg[0], 'csig: ', metrics_avg[1], 'cbak: ', metrics_avg[2], 'covl: ',
          metrics_avg[3], 'ssnr: ', metrics_avg[4], 'stoi: ', metrics_avg[5])


if __name__ == '__main__':
    main()

# Baseline
# pesq: 2.205      csig: 3.634     cbak: 2.891     covl: 2.953     ssnr: 4.250     stoi: 0.901
