# -*- coding: utf-8 -*-
"""
Created on Sun May 14 16:55:27 2023

@author: robin
"""
from models.conformer import ConformerBlock
import torch.nn as nn
import torch
import numpy as np
from models.DiffuSE import DiffusionEmbedding
from models.generator import DenseEncoder, TSCB, MaskDecoder, ComplexDecoder


class MergeBlock(nn.Module):
    def __init__(self, num_channel, noise_schedule):
        super().__init__()
        self.diffusion_embedding = DiffusionEmbedding(len(noise_schedule))
        self.diffusion_projection = nn.Linear(512, num_channel)
        
        self.merge_diffusion = nn.Conv2d(num_channel, num_channel*2, 1)
        self.conditioner_projection = nn.Conv2d(num_channel, num_channel*2, 1)
        
        self.output_residual = nn.Conv1d(num_channel, num_channel, 1)
    
    def forward(self, x, conditioner, diffusion_step):
        diffusion_step = self.diffusion_embedding(diffusion_step)
        diffusion_step = self.diffusion_projection(diffusion_step)
        diffusion_step = diffusion_step[:,:,None,None]
        
        conditioner = self.conditioner_projection(conditioner)
        y = x + diffusion_step
        y = self.merge_diffusion(y) + conditioner
    
        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)
    
        residual = self.output_residual(y)
    
        return (x + residual) / np.sqrt(2.0)
    
    
class TSCNet(nn.Module):
    def __init__(self, num_channel=64, num_features=201, noise_schedule=None):
        super(TSCNet, self).__init__()
        self.dense_encoder = DenseEncoder(in_channel=3, channels=num_channel)
        self.dense_encoder_noisy = DenseEncoder(in_channel=3, channels=num_channel)
        
        self.merge_block = MergeBlock(num_channel, noise_schedule)
        self.TSCB_1 = TSCB(num_channel=num_channel)
        self.TSCB_2 = TSCB(num_channel=num_channel)
        self.TSCB_3 = TSCB(num_channel=num_channel)
        self.TSCB_4 = TSCB(num_channel=num_channel)

        self.mask_decoder = MaskDecoder(num_features, num_channel=num_channel, out_channel=1)
        self.complex_decoder = ComplexDecoder(num_channel=num_channel)
        
            
    def forward(self, x, noisy_spec, diffusion_step=None):
        x_mag = x.abs().unsqueeze(1).permute(0, 1, 3, 2)
        x_angle = x.angle().unsqueeze(1).permute(0, 1, 3, 2)
        x_in = torch.cat([x_mag, 
                          x.real.unsqueeze(1).permute(0, 1, 3, 2), 
                          x.imag.unsqueeze(1).permute(0, 1, 3, 2)], 
                         dim=1)
        
        noisy_mag = noisy_spec.abs().unsqueeze(1).permute(0, 1, 3, 2)
        noisy_in = torch.cat([noisy_mag, 
                          noisy_spec.real.unsqueeze(1).permute(0, 1, 3, 2), 
                          noisy_spec.imag.unsqueeze(1).permute(0, 1, 3, 2)], 
                         dim=1)
        
        
        out = self.dense_encoder(x_in)
        out_noisy = self.dense_encoder_noisy(noisy_in)
        
        out = self.TSCB_1(self.merge_block(out, out_noisy, diffusion_step))
        out = self.TSCB_2(self.merge_block(out, out_noisy, diffusion_step))
        out = self.TSCB_3(self.merge_block(out, out_noisy, diffusion_step))
        out = self.TSCB_4(self.merge_block(out, out_noisy, diffusion_step))

        mask = self.mask_decoder(out)
        out_mag = mask * x_mag

        complex_out = self.complex_decoder(out)
        mag_real = out_mag * torch.cos(x_angle)
        mag_imag = out_mag * torch.sin(x_angle)
        final_real = mag_real + complex_out[:, 0, :, :].unsqueeze(1)
        final_imag = mag_imag + complex_out[:, 1, :, :].unsqueeze(1)

        return final_real, final_imag