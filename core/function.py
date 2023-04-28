#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 15:58:51 2023

@author: minyoungpark
"""

import time
import torch
import datetime
from utils.utils import ProgressMeter
from timm.utils import AverageMeter
from torch.autograd import Variable


def get_accuracy(pred, label):
    output_class = torch.argmax(pred.detach(), dim=1)
    acc = torch.sum(output_class == label.detach()) / len(output_class)

    return acc

def train(train_loader, model, criterion, optimizer, lr_scheduler, scaler, 
          logger, epoch, args):
    
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
    for idx, (image, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        lr_scheduler.step(epoch*iters_per_epoch+idx)
        learning_rates.update(optimizer.param_groups[0]['lr'])
        
        if args.gpu is not None:
            image = image.to(args.gpu, non_blocking=True).float()
            target = target.to(args.gpu, non_blocking=True).float()
            
        # compute output
        with torch.cuda.amp.autocast(True):
            output = model(image)
            
        loss = criterion(output, target)
        losses.update(loss.item(), target.size(0))

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


def validate(valid_loader, model, criterion, scaler, logger, epoch, args):
    
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
        for idx, (image, target) in enumerate(valid_loader):
            if args.gpu is not None:
                image = image.to(args.gpu, non_blocking=True).float()
                target = target.long().to(args.gpu, non_blocking=True).float()
            
            with torch.cuda.amp.autocast(True):
                output = model(image)
                
            loss = criterion(output, target)
            
            # acc = get_accuracy(output, target)
            
            losses.update(loss.item(), target.size(0))
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
