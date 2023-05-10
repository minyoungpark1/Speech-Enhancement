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

from models.discriminator import batch_pesq
from utils.utils import ProgressMeter, adjust_learning_rate


def get_accuracy(pred, label):
    output_class = torch.argmax(pred.detach(), dim=1)
    acc = torch.sum(output_class == label.detach()) / len(output_class)

    return acc

def add_noise(audio, noisy, noise_schedule):
    N, T = audio.shape
    device = audio.device

    beta = np.array(noise_schedule)
    noise_level = np.cumprod(1 - beta)
    noise_level = torch.tensor(noise_level.astype(np.float32)).cuda(device)
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

def train(train_loader, model, criterion, optimizer, scaler, logger, epoch, 
          args, config):
# def train(train_loader, model, criterion, optimizer, lr_scheduler, scaler, 
#           logger, epoch, args, config):

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

    for idx, batch in enumerate(train_loader):
        signal = batch['audio']
        noisy_signal = batch['noisy']
        spectrogram = batch['spectrogram']
        
        # measure data loading time
        data_time.update(time.time() - end)
        lr = adjust_learning_rate([optimizer], epoch + idx / iters_per_epoch, config)
        learning_rates.update(lr)
        # lr_scheduler.step(epoch*iters_per_epoch+idx)
        # learning_rates.update(optimizer.param_groups[0]['lr'])

        if args.gpu is not None:
            signal = signal.cuda(args.gpu, non_blocking=True).float()
            noisy_signal = noisy_signal.cuda(args.gpu, non_blocking=True).float()
            spectrogram = spectrogram.cuda(args.gpu, non_blocking=True).float()

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
    for idx, batch in enumerate(valid_loader):
        signal = batch['audio']
        noisy_signal = batch['noisy']
        spectrogram = batch['spectrogram']
        
        if args.gpu is not None:
            signal = signal.cuda(args.gpu, non_blocking=True).float()
            noisy_signal = noisy_signal.cuda(args.gpu, non_blocking=True).float()
            spectrogram = spectrogram.cuda(args.gpu, non_blocking=True).float()

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
# def train_gan(train_loader, model, discriminator, criterion, optimizer, optimizer_disc, 
#               logger, epoch, args, config):
def train_gan(train_loader, model, discriminator, criterion, optimizer, optimizer_disc, 
              lr_scheduler_G, lr_scheduler_D, logger, epoch, args, config):
    torch.autograd.set_detect_anomaly(True)
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
    discriminator.train()
    
    start = time.time()
    end = time.time()
    iters_per_epoch = len(train_loader)

    for idx, batch in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        
        lr_scheduler_G.step(epoch*iters_per_epoch+idx)
        lr_scheduler_D.step(epoch*iters_per_epoch+idx)
        learning_rates.update(optimizer.param_groups[0]['lr'])
        # lr = adjust_learning_rate([optimizer, optimizer_disc], 
        #                           epoch + idx / iters_per_epoch, config)
        # learning_rates.update(lr)
        optimizer.zero_grad()
        
        clean, noisy, clean_spec, noisy_spec, clean_real, clean_imag, \
            one_labels, hamming_window= batch_stft(batch, args, config)
            
        est_real, est_imag = model(noisy_spec)
        est_real, est_imag = est_real.permute(0, 1, 3, 2), est_imag.permute(0, 1, 3, 2)
        
        est_complex = torch.complex(est_real, est_imag).squeeze(1)
        est_mag = est_complex.abs().unsqueeze(1)
        clean_mag = clean_spec.abs().unsqueeze(1)
        
        est_audio = torch.istft(est_complex, config.N_FFT, 
                                config.HOP_SAMPLES, window=hamming_window, 
                                onesided=True, normalized=True)
            
        if args.arch == 'scp-gan':
            ################### Consistency Preserving Network #################
            # Enhanced audio pipeline
            est_prime_mag, \
                est_prime_real,\
                    est_prime_imag= compute_mag(est_audio, config.N_FFT, 
                                                config.HOP_SAMPLES, hamming_window,)
            
            # Clean* audio pipeline
            clean_audio_prime = torch.istft(clean_spec, config.N_FFT, 
                                    config.HOP_SAMPLES, window=hamming_window, 
                                    onesided=True, normalized=True)
            clean_audio_prime_mag, \
                clean_audio_prime_real, \
                    clean_audio_prime_imag = compute_mag(clean_audio_prime, config.N_FFT, 
                                                         config.HOP_SAMPLES, hamming_window,)
                                                         # log_exp_mag_compress=args.log_exp_mag,
                                                         # last_compress=args.last_compress)
            
            loss_mag = criterion(est_prime_mag, clean_audio_prime_mag)
            time_loss = torch.mean(torch.abs(est_audio - clean_audio_prime))
            loss_ri = criterion(est_prime_real, clean_audio_prime_real) + \
                criterion(est_prime_imag, clean_audio_prime_imag)
        else:
            loss_mag = criterion(est_mag, clean_mag)
            time_loss = torch.mean(torch.abs(est_audio - clean))
            loss_ri = criterion(est_real, clean_real) + criterion(est_imag, clean_imag)
        
        if epoch >= int(args.epochs*0.3):
            predict_fake_metric = discriminator(clean_mag, est_mag)
            gen_loss_GAN = criterion(predict_fake_metric.flatten(), one_labels.float())
            
            loss = config.LOSS_WEIGHTS[0] * loss_ri + \
                config.LOSS_WEIGHTS[1] * loss_mag + \
                    config.LOSS_WEIGHTS[2] * time_loss + \
                        config.LOSS_WEIGHTS[3] * gen_loss_GAN
        else:
            gen_loss_GAN = torch.tensor([0.])
            loss = config.LOSS_WEIGHTS[0] * loss_ri + \
                config.LOSS_WEIGHTS[1] * loss_mag + \
                    config.LOSS_WEIGHTS[2] * time_loss
        logger.info(f'{loss_ri.item():.4f}\t {loss_mag.item():.4f}\t {time_loss.item():.4f}\t {gen_loss_GAN.item():.4f}')
        loss.backward()
        if args.max_norm != 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
        optimizer.step()
        
        optimizer_disc.zero_grad()
        
        if epoch >= int(args.epochs*0.3):
            length = est_audio.size(-1)
            est_audio_list = list(est_audio.detach().cpu().numpy())
            clean_audio_list = list(clean.cpu().numpy()[:, :length])
            
            D_Gx_y = discriminator(est_mag.detach(), clean_mag)
            Q_Gx_y = batch_pesq(clean_audio_list, est_audio_list)
            D_y_y = discriminator(clean_mag, clean_mag)
            
            L_E = criterion(D_Gx_y.flatten(), Q_Gx_y)
            
            if args.arch == 'scp-gan':
                Q_y_y = batch_pesq(clean_audio_list, clean_audio_list)
                L_C = criterion(D_y_y.flatten(), Q_y_y)
                
                noisy_real = noisy_spec[:,0,...].unsqueeze(1).permute(0, 1, 3, 2)
                noisy_imag = noisy_spec[:,1,...].unsqueeze(1).permute(0, 1, 3, 2)
                noisy_mag = torch.sqrt(noisy_real**2 + noisy_imag**2)
                D_x_y = discriminator(noisy_mag, clean_mag)
                
                noisy_audio_list = list(noisy.cpu().numpy()[:, :length])
                Q_x_y = batch_pesq(clean_audio_list, noisy_audio_list)
                L_N = criterion(D_x_y.flatten(), Q_x_y)
                
                discrim_loss_metric = compute_self_correcting_loss_weights(discriminator,
                                                                optimizer_disc,
                                                                L_C, L_E, L_N)
            else:
                L_C = criterion(D_y_y.flatten(), one_labels)
                discrim_loss_metric = L_C + L_E
                                  
            discrim_loss_metric.backward()
            if args.max_norm != 0.0:
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), args.max_norm)
            optimizer_disc.step()
        else:
            discrim_loss_metric = torch.tensor([0.])
            
        torch.cuda.synchronize()
        
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
def validate_gan(valid_loader, model, discriminator, criterion, logger, 
                   epoch, args, config):
    batch_time = AverageMeter()
    gen_losses = AverageMeter()
    disc_losses = AverageMeter()
    model.eval()
    discriminator.eval()
    end = time.time()

    progress = ProgressMeter(
            len(valid_loader),
            [batch_time, gen_losses, disc_losses],
            prefix='Test: ')

    for idx, batch in enumerate(valid_loader):
        clean, noisy, clean_spec, noisy_spec, clean_real, clean_imag, \
            one_labels, hamming_window = batch_stft(batch, args, config)
    
        est_real, est_imag = model(noisy_spec)
        est_real, est_imag = est_real.permute(0, 1, 3, 2), est_imag.permute(0, 1, 3, 2)
        est_complex = torch.complex(est_real, est_imag).squeeze(1)
        est_mag = est_complex.abs().unsqueeze(1)
        clean_mag = clean_spec.abs().unsqueeze(1)
        
        est_audio = torch.istft(est_complex, config.N_FFT, config.HOP_SAMPLES, 
                                window=hamming_window, onesided=True, normalized=True)
    
        if args.arch == 'scp-gan':
            ################### Consistency Presering Network #################
            # Enhanced audio pipeline
            est_prime_mag, \
                est_prime_real,\
                    est_prime_imag= compute_mag(est_audio, config.N_FFT, 
                                                config.HOP_SAMPLES, hamming_window,)
            # Clean* audio pipeline
            clean_audio_prime = torch.istft(clean_spec, config.N_FFT, 
                                    config.HOP_SAMPLES, window=hamming_window, 
                                    onesided=True, normalized=True)
            clean_audio_prime_mag, \
                clean_audio_prime_real, \
                    clean_audio_prime_imag = compute_mag(clean_audio_prime, config.N_FFT, 
                                                         config.HOP_SAMPLES, hamming_window,)
            
            loss_mag = criterion(est_prime_mag, clean_audio_prime_mag)
            time_loss = torch.mean(torch.abs(est_audio - clean_audio_prime))
            loss_ri = criterion(est_prime_real, clean_audio_prime_real) + \
                criterion(est_prime_imag, clean_audio_prime_imag)
        else:
            loss_mag = criterion(est_mag, clean_mag)
            time_loss = torch.mean(torch.abs(est_audio - clean))
            loss_ri = criterion(est_real, clean_real) + criterion(est_imag, clean_imag)
        
        predict_fake_metric = discriminator(clean_mag, est_mag)
        gen_loss_GAN = criterion(predict_fake_metric.flatten(), one_labels.float())
            
        length = est_audio.size(-1)
        loss = config.LOSS_WEIGHTS[0] * loss_ri + \
            config.LOSS_WEIGHTS[1] * loss_mag + \
                config.LOSS_WEIGHTS[2] * time_loss + \
                    config.LOSS_WEIGHTS[3] * gen_loss_GAN
    
        est_audio_list = list(est_audio.detach().cpu().numpy())
        clean_audio_list = list(clean.cpu().numpy()[:, :length])
        
        D_Gx_y = discriminator(est_mag.detach(), clean_mag)
        Q_Gx_y = batch_pesq(est_audio_list, clean_audio_list)
        D_y_y = discriminator(clean_mag, clean_mag)
        
        L_C = criterion(D_y_y.flatten(), one_labels)
        L_E = criterion(D_Gx_y.flatten(), Q_Gx_y)
        
        # Unable to compute validation self-correcting loss
        discrim_loss_metric = L_C + L_E
    
        torch.cuda.synchronize()
        
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
    clean = batch['audio']
    noisy = batch['noisy']
    one_labels = torch.ones(len(clean))
    hamming_window = torch.hamming_window(config.N_FFT)
    
    if args.gpu is not None:
        clean = clean.cuda(args.gpu, non_blocking=True)
        noisy = noisy.cuda(args.gpu, non_blocking=True)
        one_labels = one_labels.cuda(args.gpu, non_blocking=True)
        hamming_window = hamming_window.cuda(args.gpu, non_blocking=True)
        
    # Normalization
    c = torch.sqrt(noisy.size(-1) / torch.sum((noisy ** 2.0), dim=-1))
    noisy, clean = torch.transpose(noisy, 0, 1), torch.transpose(clean, 0, 1)
    noisy, clean = torch.transpose(noisy * c, 0, 1), torch.transpose(clean * c, 0, 1)

    noisy_spec = torch.stft(noisy, config.N_FFT, config.HOP_SAMPLES, 
                                               window=hamming_window, onesided=True, 
                                               return_complex=True, normalized=True)
    clean_spec = torch.stft(clean, config.N_FFT, config.HOP_SAMPLES, 
                                               window=hamming_window, onesided=True,
                                               return_complex=True, normalized=True)
    
    clean_real = clean_spec.real.unsqueeze(1)
    clean_imag = clean_spec.imag.unsqueeze(1)
    
    return clean, noisy, clean_spec, noisy_spec, clean_real, clean_imag, \
        one_labels, hamming_window

def compute_mag(signal, n_fft, hop_length, window):
    spec = torch.stft(signal, n_fft, hop_length, window=window, onesided=True, 
                    return_complex=True, normalized=True)   
    return spec.abs(), spec.real, spec.imag

def compute_self_correcting_loss_weights(discriminator, optimizer_disc, L_C, L_E, L_N):
    # resetting gradient back to zero
    optimizer_disc.zero_grad()

    L_C.backward(retain_graph=True)
    
    # tensor with clean gradients
    grad_C_tensor = [param.grad.clone() for _, param in discriminator.named_parameters()]
    grad_C_list = torch.cat([grad.reshape(-1) for grad in grad_C_tensor], dim=0)
    
    # resetting gradient back to zero
    optimizer_disc.zero_grad()

    L_E.backward(retain_graph=True)
    # tensor with enhanced gradients
    grad_E_tensor = [param.grad.clone() for _, param in discriminator.named_parameters()]
    grad_E_list = torch.cat([grad.reshape(-1) for grad in grad_E_tensor], dim=0)
    EdotE = torch.dot(grad_E_list, grad_E_list).item() + 1e-6
    
    # resetting gradient back to zero
    optimizer_disc.zero_grad()
    L_N.backward(retain_graph=True)
    # tensor with noisy gradients
    grad_N_tensor = [param.grad.clone() for _, param in discriminator.named_parameters()]
    grad_N_list = torch.cat([grad.reshape(-1) for grad in grad_N_tensor], dim=0)
    NdotN = torch.dot(grad_N_list, grad_N_list).item() + 1e-6
    
    # dot product between gradients
    CdotE = torch.dot(grad_C_list, grad_E_list).item()
    CdotN = torch.dot(grad_C_list, grad_N_list).item()
    EdotN = torch.dot(grad_E_list, grad_N_list).item()

    if CdotE > 0:
        w_C, w_E = 1, 1
        if torch.dot(w_C*grad_C_list + w_E*grad_E_list, grad_N_list).item() > 0:
            w_N = 1
        else:
            w_N = -(CdotN)/(NdotN)- (EdotN)/(NdotN)
    else:
        w_C = 1
        w_E = -(CdotE)/(EdotE)
        if torch.dot(w_C*grad_C_list + w_E*grad_E_list, grad_N_list).item() > 0:
            w_N = 1
        else:
            w_N = -(CdotN)/(NdotN) + (CdotE * EdotN)/(EdotE * NdotN)

    optimizer_disc.zero_grad()
    # calculating self correcting loss
    self_correcting_loss = w_C*L_C + w_E*L_E + w_N*L_N

    # # updating gradient, i.e. getting self correcting gradient
    for index, (_, param) in enumerate(discriminator.named_parameters()):
        param.grad = w_C * grad_C_tensor[index] + \
            w_E * grad_E_tensor[index] + \
                w_N * grad_N_tensor[index]

    return self_correcting_loss