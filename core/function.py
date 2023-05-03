#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 15:58:51 2023

@author: minyoungpark
"""

import time
import torch
import numpy as np
import datetime
from timm.utils import AverageMeter
import torch.nn.functional as F

from models.discriminator import batch_pesq
from utils.utils import ProgressMeter


def get_accuracy(pred, label):
    output_class = torch.argmax(pred.detach(), dim=1)
    acc = torch.sum(output_class == label.detach()) / len(output_class)

    return acc

def add_noise(audio, noisy, noise_schedule):
    N, T = audio.shape
    device = audio.device

    beta = np.array(noise_schedule)
    noise_level = np.cumprod(1 - beta)
    noise_level = torch.tensor(noise_level.astype(np.float32)).to(device)
    t = torch.randint(0, len(noise_schedule), [N], device=audio.device)
    noise_scale = noise_level[t].unsqueeze(1)
    noise_scale_sqrt = noise_scale**0.5
    m = (((1-noise_level[t])/noise_level[t]**0.5)**0.5).unsqueeze(1)

    noise = torch.randn_like(audio)
    noisy_audio = (1-m) * noise_scale_sqrt  * audio + \
        m * noise_scale_sqrt * noisy  + \
        (1.0 - (1+m**2) *noise_scale)**0.5 * noise
    combine_noise = (m * noise_scale_sqrt * (noisy-audio) + \
                     (1.0 - (1+m**2) *noise_scale)**0.5 * noise) / (1-noise_scale)**0.5
    return noisy_audio, combine_noise, t

def train(train_loader, model, criterion, optimizer, lr_scheduler, scaler, 
          logger, epoch, args, config):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    learning_rates = AverageMeter()
    losses = AverageMeter()

    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses],
        prefix="Epoch: [{}]".format(epoch))

    model.train()

    start = time.time()
    end = time.time()
    iters_per_epoch = len(train_loader)

    for idx, (signal, noisy_signal, spectrogram) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        lr_scheduler.step(epoch*iters_per_epoch+idx)
        learning_rates.update(optimizer.param_groups[0]['lr'])

        if args.gpu is not None:
            signal = signal.to(args.gpu, non_blocking=True).float()
            noisy_signal = noisy_signal.to(args.gpu, non_blocking=True).float()
            spectrogram = spectrogram.to(args.gpu, non_blocking=True).float()

        # compute output
        with torch.cuda.amp.autocast(True):
            noisy_audio, combine_noise, t = add_noise(signal, noisy_signal,
                                                      config.NOISE_SCHEDULE)
            predicted = model(noisy_audio, spectrogram, t)
            loss = criterion(predicted.squeeze(1), combine_noise)

        losses.update(loss.item(), signal.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        scaler.scale(loss).backward()

        if args.max_norm != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)

        scaler.step(optimizer)
        scaler.update()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % args.print_freq == 0:
            lr = optimizer.param_groups[0]['lr']
            wd = optimizer.param_groups[0]['weight_decay']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (iters_per_epoch - idx)
            logger.info(
                f'Train: [{epoch}/{args.epochs}][{idx}/{iters_per_epoch}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t wd {wd:.4f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'loss {losses.val:.4f} ({losses.avg:.4f})\t'
                f'mem {memory_used:.0f}MB')

            progress.display(idx)

    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")

    return losses.avg

@torch.no_grad()
def validate(valid_loader, model, criterion, scaler, logger, epoch, args, config):
    batch_time = AverageMeter()
    losses = AverageMeter()
    # accuracies = AverageMeter()

    progress = ProgressMeter(
            len(valid_loader),
            [batch_time, losses,],
            # [batch_time, losses, accuracies],
            prefix='Test: ')

    model.eval()

    end = time.time()

    # with torch.no_grad():
    for idx, (signal, noisy_signal, spectrogram) in enumerate(valid_loader):
        if args.gpu is not None:
            signal = signal.to(args.gpu, non_blocking=True).float()
            noisy_signal = noisy_signal.to(args.gpu, non_blocking=True).float()
            spectrogram = spectrogram.to(args.gpu, non_blocking=True).float()

        # compute output
        with torch.cuda.amp.autocast(True):
            noisy_audio, combine_noise, t = add_noise(signal, noisy_signal,
                                                      config.NOISE_SCHEDULE)
            predicted = model(noisy_audio, spectrogram, t)
            loss = criterion(predicted.squeeze(1), combine_noise)

        # acc = get_accuracy(output, target)

        losses.update(loss.item(), signal.size(0))
        # accuracies.update(acc.item(), target.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % args.print_freq == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logger.info(
                f'Test: [{idx}/{len(valid_loader)}]\t'
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                # f'Acc {accuracies.val:.3f} ({accuracies.avg:.3f})\t'
                f'Mem {memory_used:.0f}MB')

            progress.display(idx)

    return losses.avg
    # return losses.avg, accuracies.avg

# TODO Merge train/valid functions of  GAN models in the future
# def train_scpgan(train_loader, model, discriminator, optimizer, optimizer_disc, 
#               lr_scheduler, scaler, logger, epoch, args, config):

#     batch_time = AverageMeter()
#     data_time = AverageMeter()
#     learning_rates = AverageMeter()
#     losses = AverageMeter()
#     losses = AverageMeter()

#     progress = ProgressMeter(
#         len(train_loader),
#         [batch_time, data_time, losses],
#         prefix="Epoch: [{}]".format(epoch))

#     model.train()

#     start = time.time()
#     end = time.time()
#     iters_per_epoch = len(train_loader)

#     for idx, batch in enumerate(train_loader):
#         # measure data loading time
#         data_time.update(time.time() - end)
#         lr_scheduler.step(epoch*iters_per_epoch+idx)
#         learning_rates.update(optimizer.param_groups[0]['lr'])
#         optimizer.zero_grad()
        
#         clean, clean_spec, noisy_spec, \
#         clean_real, clean_imag, one_labels, hamming_window= batch_stft(batch, args, config)
            
#         # compute output
#         with torch.cuda.amp.autocast(True):
#             est_real, est_imag = model(noisy_spec)
#             est_real, est_imag = est_real.permute(0, 1, 3, 2), est_imag.permute(0, 1, 3, 2)
#             est_mag = torch.sqrt(est_real**2 + est_imag**2)
#             clean_mag = torch.sqrt(clean_real**2 + clean_imag**2)
            
#             predict_fake_metric = discriminator(clean_mag, est_mag)
#             gen_loss_GAN = F.mse_loss(predict_fake_metric.flatten(), one_labels.float())
            
#             loss_ri = F.mse_loss(est_real, clean_real) + F.mse_loss(est_imag, clean_imag)
            
#             est_spec_uncompress = power_uncompress(est_real, est_imag).squeeze(1)
            
            
#             ################### Consistency Presering Network #################
#             # Enhanced audio pipeline
#             est_audio = torch.istft(est_spec_uncompress, config.N_FFT, 
#                                     config.HOP_SAMPLES, window=hamming_window, onesided=True)
#             est_spec_prime = torch.stft(est_audio, config.n_fft, config.hop, 
#                                     window=hamming_window,
#                                     onesided=True, return_complex=True)
#             est_spec_prime_real = est_spec_prime[:, 0, :, :].unsqueeze(1)
#             est_spec_prime_imag = est_spec_prime[:, 1, :, :].unsqueeze(1)
#             est_mag_prime = torch.sqrt(est_spec_prime_real**2 + est_spec_prime_imag**2)
            
#             # Clean* audio pipeline
#             clean_spec_uncompress = power_uncompress(clean_real, clean_imag).squeeze(1)
#             clean_audio_prime = torch.istft(clean_spec_uncompress, config.N_FFT, 
#                                     config.HOP_SAMPLES, window=hamming_window, onesided=True)
#             clean_audio_prime_spec = torch.stft(clean_audio_prime, config.n_fft, config.hop, 
#                                     window=hamming_window,
#                                     onesided=True, return_complex=True)
#             clean_audio_prime_real = clean_audio_prime_spec[:, 0, :, :].unsqueeze(1)
#             clean_audio_prime_imag = clean_audio_prime_spec[:, 1, :, :].unsqueeze(1)
#             clean_audio_prime_mag = torch.sqrt(clean_audio_prime_real**2 +\
#                                                clean_audio_prime_imag**2)
            
#             # loss_mag = F.mse_loss(est_mag, clean_mag)
#             # time_loss = torch.mean(torch.abs(est_audio - clean))
#             loss_mag = F.mse_loss(est_mag_prime, clean_audio_prime_mag)
#             time_loss = torch.mean(torch.abs(est_audio - clean_audio_prime))
#             length = est_audio.size(-1)
#             loss = config.LOSS_WEIGHTS[0] * loss_ri + \
#                 config.LOSS_WEIGHTS[1] * loss_mag + \
#                     config.LOSS_WEIGHTS[2] * time_loss + \
#                         config.LOSS_WEIGHTS[3] * gen_loss_GAN
#             loss.backward()
#             optimizer.step()
            
#             est_audio_list = list(est_audio.detach().cpu().numpy())
#             clean_audio_list = list(clean.cpu().numpy()[:, :length])
#             pesq_score = batch_pesq(clean_audio_list, est_audio_list)
            
#             # The calculation of PESQ can be None due to silent part
#             if pesq_score is not None:
#                 optimizer_disc.zero_grad()
#                 predict_enhance_metric = discriminator(clean_mag, est_mag.detach())
#                 predict_max_metric = discriminator(clean_mag, clean_mag)
#                 L_E = F.mse_loss(predict_enhance_metric.flatten(), pesq_score)
#                 discrim_loss_metric = F.mse_loss(predict_max_metric.flatten(), one_labels) + L_E
                                      
#                 discrim_loss_metric.backward()
#                 optimizer_disc.step()
#             else:
#                 discrim_loss_metric = torch.tensor([0.])
        
#         # loss.item(), discrim_loss_metric.item()
        
#         losses.update(loss.item(), signal.size(0))

#         # compute gradient and do SGD step
#         scaler.scale(loss).backward()

#         if args.max_norm != 0.0:
#             scaler.unscale_(optimizer)
#             torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)

#         scaler.step(optimizer)
#         scaler.update()

#         # measure elapsed time
#         batch_time.update(time.time() - end)
#         end = time.time()

#         if idx % args.print_freq == 0:
#             lr = optimizer.param_groups[0]['lr']
#             wd = optimizer.param_groups[0]['weight_decay']
#             memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
#             etas = batch_time.avg * (iters_per_epoch - idx)
#             logger.info(
#                 f'Train: [{epoch}/{args.epochs}][{idx}/{iters_per_epoch}]\t'
#                 f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t wd {wd:.4f}\t'
#                 f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
#                 f'loss {losses.val:.4f} ({losses.avg:.4f})\t'
#                 f'mem {memory_used:.0f}MB')

#             progress.display(idx)

#     epoch_time = time.time() - start
#     logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")

#     return losses.avg


def train_cmgan(train_loader, model, discriminator, optimizer, optimizer_disc, 
              lr_scheduler_G, lr_scheduler_D, scaler, logger, epoch, args, config):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    learning_rates = AverageMeter()
    gen_losses = AverageMeter()
    disc_losses = AverageMeter()

    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, gen_losses, disc_losses],
        prefix="Epoch: [{}]".format(epoch))

    model.train()

    start = time.time()
    end = time.time()
    iters_per_epoch = len(train_loader)

    for idx, batch in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        lr_scheduler_G.step(epoch*iters_per_epoch+idx)
        lr_scheduler_D.step(epoch*iters_per_epoch+idx)
        learning_rates.update(optimizer.param_groups[0]['lr'])
        
        with torch.cuda.amp.autocast(True):
            clean, noisy, clean_spec, noisy_spec,  clean_real, clean_imag, \
                one_labels, hamming_window= batch_stft(batch, args, config)
                
            est_real, est_imag = model(noisy_spec)
            est_real, est_imag = est_real.permute(0, 1, 3, 2), est_imag.permute(0, 1, 3, 2)
            est_mag = torch.sqrt(est_real**2 + est_imag**2)
            clean_mag = torch.sqrt(clean_real**2 + clean_imag**2)
            
            predict_fake_metric = discriminator(clean_mag, est_mag)
            gen_loss_GAN = F.mse_loss(predict_fake_metric.flatten(), one_labels.float())
            
            loss_ri = F.mse_loss(est_real, clean_real) + F.mse_loss(est_imag, clean_imag)
            
            est_spec_uncompress = power_uncompress(est_real, est_imag).squeeze(1)
            est_audio = torch.istft(est_spec_uncompress, config.N_FFT, 
                                    config.HOP_SAMPLES, window=hamming_window, onesided=True)
            
            loss_mag = F.mse_loss(est_mag, clean_mag)
            time_loss = torch.mean(torch.abs(est_audio - clean))
            length = est_audio.size(-1)
            loss = config.LOSS_WEIGHTS[0] * loss_ri + \
                config.LOSS_WEIGHTS[1] * loss_mag + \
                    config.LOSS_WEIGHTS[2] * time_loss + \
                        config.LOSS_WEIGHTS[3] * gen_loss_GAN
                        
        optimizer.zero_grad()
        scaler.scale(loss).backward()

        if args.max_norm != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), args.max_norm)

        scaler.step(optimizer)
        scaler.update()
        
        est_audio_list = list(est_audio.detach().cpu().numpy())
        clean_audio_list = list(clean.cpu().numpy()[:, :length])
        pesq_score = batch_pesq(clean_audio_list, est_audio_list)
        
        with torch.cuda.amp.autocast(True):
            # The calculation of PESQ can be None due to silent part
            if pesq_score is not None:
                optimizer_disc.zero_grad()
                predict_enhance_metric = discriminator(clean_mag, est_mag.detach())
                predict_max_metric = discriminator(clean_mag, clean_mag)
                L_E = F.mse_loss(predict_enhance_metric.flatten(), pesq_score)
                discrim_loss_metric = F.mse_loss(predict_max_metric.flatten(), one_labels) + L_E
            else:
                discrim_loss_metric = torch.tensor([0.])
        
        if pesq_score is not None:
            optimizer_disc.zero_grad()
            scaler.scale(discrim_loss_metric).backward()
    
            if args.max_norm != 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), args.max_norm)
    
            scaler.step(optimizer_disc)
            scaler.update()
            
        gen_losses.update(loss.item(), clean.size(0))
        disc_losses.update(discrim_loss_metric.item(), clean.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % args.print_freq == 0:
            lr = optimizer.param_groups[0]['lr']
            wd = optimizer.param_groups[0]['weight_decay']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (iters_per_epoch - idx)
            logger.info(
                f'Train: [{epoch}/{args.epochs}][{idx}/{iters_per_epoch}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t wd {wd:.4f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'generator loss {gen_losses.val:.4f} ({gen_losses.avg:.4f})\t'
                f'discriminator loss {disc_losses.val:.4f} ({disc_losses.avg:.4f})\t'
                f'mem {memory_used:.0f}MB')

            progress.display(idx)

    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")

    return gen_losses.avg, disc_losses.avg

@torch.no_grad()
def validate_cmgan(valid_loader, model, discriminator, scaler, logger,
                   epoch, args, config):
    batch_time = AverageMeter()
    gen_losses = AverageMeter()
    disc_losses = AverageMeter()
    # accuracies = AverageMeter()
    model.eval()
    discriminator.eval()
    end = time.time()

    progress = ProgressMeter(
            len(valid_loader),
            [batch_time, gen_losses, disc_losses],
            # [batch_time, losses, accuracies],
            prefix='Test: ')

    for idx, batch in enumerate(valid_loader):
        with torch.cuda.amp.autocast(True):
            clean, noisy, clean_spec, noisy_spec, clean_real, clean_imag, \
                one_labels, hamming_window = batch_stft(batch, args, config)
                
            c = torch.sqrt(noisy.size(-1) / torch.sum((noisy ** 2.0), dim=-1))
            noisy, clean = torch.transpose(noisy, 0, 1), torch.transpose(clean, 0, 1)
            noisy, clean = torch.transpose(noisy * c, 0, 1), torch.transpose(clean * c, 0, 1)
        
            noisy_spec = torch.view_as_real(torch.stft(noisy, config.N_FFT, config.HOP_SAMPLES, 
                                    window=hamming_window, onesided=True, return_complex=True))
            clean_spec = torch.view_as_real(torch.stft(clean, config.N_FFT, config.HOP_SAMPLES, 
                                    window=hamming_window, onesided=True, return_complex=True))
            noisy_spec = power_compress(noisy_spec).permute(0, 1, 3, 2)
            clean_spec = power_compress(clean_spec)
            clean_real = clean_spec[:, 0, :, :].unsqueeze(1)
            clean_imag = clean_spec[:, 1, :, :].unsqueeze(1)
        
            est_real, est_imag = model(noisy_spec)
            est_real, est_imag = est_real.permute(0, 1, 3, 2), est_imag.permute(0, 1, 3, 2)
            est_mag = torch.sqrt(est_real ** 2 + est_imag ** 2)
            clean_mag = torch.sqrt(clean_real ** 2 + clean_imag ** 2)
        
            predict_fake_metric = discriminator(clean_mag, est_mag)
            gen_loss_GAN = F.mse_loss(predict_fake_metric.flatten(), one_labels.float())
        
            loss_mag = F.mse_loss(est_mag, clean_mag)
            loss_ri = F.mse_loss(est_real, clean_real) + F.mse_loss(est_imag, clean_imag)
        
            est_spec_uncompress = power_uncompress(est_real, est_imag).squeeze(1)
            est_audio = torch.istft(est_spec_uncompress, config.N_FFT, config.HOP_SAMPLES, 
                                    window=hamming_window, onesided=True)
        
            time_loss = torch.mean(torch.abs(est_audio - clean))
            length = est_audio.size(-1)
            
            loss = config.LOSS_WEIGHTS[0] * loss_ri + \
                config.LOSS_WEIGHTS[1] * loss_mag + \
                    config.LOSS_WEIGHTS[2] * time_loss + \
                        config.LOSS_WEIGHTS[3] * gen_loss_GAN
        
            est_audio_list = list(est_audio.detach().cpu().numpy())
            clean_audio_list = list(clean.cpu().numpy()[:, :length])
            pesq_score = batch_pesq(clean_audio_list, est_audio_list)
            if pesq_score is not None:
                predict_enhance_metric = discriminator(clean_mag, est_mag.detach())
                predict_max_metric = discriminator(clean_mag, clean_mag)
                discrim_loss_metric = F.mse_loss(predict_max_metric.flatten(), one_labels) + \
                                      F.mse_loss(predict_enhance_metric.flatten(), pesq_score)
            else:
                discrim_loss_metric = torch.tensor([0.])
    
        gen_losses.update(loss.item(), clean.size(0))
        disc_losses.update(discrim_loss_metric.item(), clean.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % args.print_freq == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logger.info(
                f'Test: [{idx}/{len(valid_loader)}]\t'
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'generator loss {gen_losses.val:.4f} ({gen_losses.avg:.4f})\t'
                f'discriminator loss {disc_losses.val:.4f} ({disc_losses.avg:.4f})\t'
                f'Mem {memory_used:.0f}MB')

            progress.display(idx)
            
    return gen_losses.avg, disc_losses.avg


def batch_stft(batch, args, config):
    clean = batch[0]
    noisy = batch[1]
    one_labels = torch.ones(args.batch_size)
    hamming_window = torch.hamming_window(config.N_FFT)
    
    if args.gpu is not None:
        clean = clean.to(args.gpu, non_blocking=True)
        noisy = noisy.to(args.gpu, non_blocking=True)
        one_labels = one_labels.to(args.gpu, non_blocking=True)
        hamming_window = hamming_window.to(args.gpu, non_blocking=True)
        
    # Normalization
    c = torch.sqrt(noisy.size(-1) / torch.sum((noisy ** 2.0), dim=-1))
    noisy, clean = torch.transpose(noisy, 0, 1), torch.transpose(clean, 0, 1)
    noisy, clean = torch.transpose(noisy * c, 0, 1), torch.transpose(clean * c, 0, 1)

    noisy_spec = torch.view_as_real(torch.stft(noisy, config.N_FFT, config.HOP_SAMPLES, 
                                               window=hamming_window, onesided=True, 
                                               return_complex=True))
    clean_spec = torch.view_as_real(torch.stft(clean, config.N_FFT, config.HOP_SAMPLES, 
                                               window=hamming_window, onesided=True,
                                               return_complex=True))
    
    noisy_spec = power_compress(noisy_spec).permute(0, 1, 3, 2)
    clean_spec = power_compress(clean_spec)
    clean_real = clean_spec[:, 0, :, :].unsqueeze(1)
    clean_imag = clean_spec[:, 1, :, :].unsqueeze(1)
    
    return clean, noisy, clean_spec, noisy_spec, clean_real, clean_imag, \
        one_labels, hamming_window

def power_compress(x):
    real = x[..., 0]
    imag = x[..., 1]
    spec = torch.complex(real, imag)
    mag = torch.abs(spec)
    phase = torch.angle(spec)
    mag = mag**0.3
    real_compress = mag * torch.cos(phase)
    imag_compress = mag * torch.sin(phase)
    
    return torch.stack([real_compress, imag_compress], 1)

def power_uncompress(real, imag):
    spec = torch.complex(real, imag)
    mag = torch.abs(spec)
    phase = torch.angle(spec)
    mag = mag**(1./0.3)
    real_compress = mag * torch.cos(phase)
    imag_compress = mag * torch.sin(phase)
    
    return torch.complex(real_compress, imag_compress)
    # return torch.stack([real_compress, imag_compress], -1)

def compute_angle(x1, x2):
    cos = F.cosine_similarity(x1.flatten(), x2.flatten(), dim=0)
    angle = torch.abs(torch.acos(cos))%torch.pi
    
    return angle


# def compute_self_correcting_weights():
    