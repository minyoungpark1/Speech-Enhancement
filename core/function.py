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
from utils.utils import ProgressMeter
from timm.utils import AverageMeter
from torch.autograd import Variable


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

    with torch.no_grad():
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
