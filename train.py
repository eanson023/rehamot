#! python3
# -*- encoding: utf-8 -*-
"""
Created on Fri Nov  18 20:36:29 2022

@author: eanson
"""
import logging
import os
import shutil
import time
from pathlib import Path

from tensorboardX import SummaryWriter
import torch
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import numpy as np
import random

import model.launch.prepare  # NOQA
from evaluation import AverageMeter, LogCollector, encode_data, m2t, t2m
from model.data.utils import get_loaders

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="configs", config_name="train")
def _train(cfg: DictConfig):
    return main(cfg)


def main(cfg: DictConfig) -> None:
    global tb_writer
    working_dir = cfg.path.working_dir
    code_dir = cfg.path.code_dir
    logger.info("Training script. The outputs will be stored in:")
    logger.info(f"{working_dir}")

    tb_writer = SummaryWriter(working_dir, flush_secs=5)

    seed_everything(cfg.seed)
    logger.info(f"Seed everything: {cfg.seed}")

    opt = cfg.machine
    device = opt.device

    logger.info("Loading data module")
    train_loader, val_loader = get_loaders(
        dataname=opt.dataname, batch_size=opt.batch_size, workers=opt.num_workers, cfg=cfg.data)
    logger.info(f"Data module '{cfg.data.dataname}' loaded")

    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

    # Construct the model
    logger.info("Loading model")
    model = instantiate(cfg.model,
                        device=device,
                        learning_rate=opt.learning_rate,
                        grad_clip=opt.grad_clip,
                        nfeats=train_loader.dataset.nfeats,
                        # Avoid recursive early loading of encoders
                        _recursive_=False)
    logger.info(model.motionencoder)
    logger.info(model.textencoder)
    logger.info(f"Model '{cfg.model.modelname}' loaded")

    ##############################################################################################
    start_epoch = 0
    # optionally resume from a checkpoint
    if cfg.resume:
        ckpt_path = code_dir / cfg.resume
        if os.path.isfile(ckpt_path):
            logger.info("=> loading checkpoint '{}'".format(cfg.resume))
            checkpoint = torch.load(ckpt_path)
            start_epoch = checkpoint['epoch']
            best_rsum = checkpoint['best_rsum']
            model.load_state_dict(checkpoint['model'])
            # Eiters is used to show logs as the continuation of another
            # training
            model.Eiters = checkpoint['Eiters']
            logger.info("=> loaded checkpoint '{}' (epoch {}, best_rsum {})"
                        .format(cfg.resume, start_epoch + 1, best_rsum))
        else:
            logger.info("=> no checkpoint found at '{}'".format(cfg.resume))

    # evaluate on validation set
    validate(opt, val_loader, model)

    # Train the Model
    best_rsum = 0
    use_hard_negative_mining = False
    lr_updated = False

    for epoch in range(start_epoch, opt.num_epochs):
        if epoch >= opt.warm_up and not use_hard_negative_mining:
            model.hard_negative_mining()
            use_hard_negative_mining = True
            logger.info('use hard negative mining')

        if epoch >= opt.lr_update and not lr_updated:
            adjust_learning_rate(opt, model.optimizer, epoch)
            lr_updated = True

        # train for one epoch
        train(opt, train_loader, model, epoch, val_loader)

        # evaluate on validation set
        rsum = validate(opt, val_loader, model)

        # remember best R@ sum and save checkpoint
        is_best = rsum > best_rsum
        best_rsum = max(rsum, best_rsum)
        save_checkpoint({
            'epoch': epoch + 1,
            'model': model.state_dict(),
            'best_rsum': best_rsum,
            'Eiters': model.Eiters,
        }, is_best, prefix=working_dir)

    tb_writer.close()
    logger.info("all done!")


def train(opt, train_loader, model, epoch, val_loader):
    # average meters to record the training statistics
    batch_time = AverageMeter()
    data_time = AverageMeter()
    train_logger = LogCollector(tb_writer)

    start_new_epoch = True

    # switch to train mode
    model.train_start()

    end = time.time()
    for i, train_data in enumerate(train_loader):
        # Always reset to train mode, this is not the default behavior
        model.train_start()

        # measure data loading time
        data_time.update(time.time() - end)

        # make sure train logger is used
        model.logger = train_logger

        # Update the model
        model.train_emb(**train_data, val_step=opt.val_step,
                        init=start_new_epoch)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # Print log info
        if model.Eiters % opt.log_step == 0:
            logging.info(
                'Epoch: [{0}][{1}/{2}]\t'
                '{e_log}\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                .format(
                    epoch + 1, i + 1, len(train_loader), batch_time=batch_time,
                    data_time=data_time, e_log=str(model.logger)))

        # Record logs in tensorboard
        tb_writer.add_scalar('epoch', epoch, global_step=model.Eiters)
        tb_writer.add_scalar('step', i, global_step=model.Eiters)
        tb_writer.add_scalar('batch_time', batch_time.val,
                             global_step=model.Eiters)
        tb_writer.add_scalar('data_time', data_time.val,
                             global_step=model.Eiters)
        model.logger.tb_log(global_step=model.Eiters)

        # validate at every val_step
        if model.Eiters % opt.val_step == 0:
            validate(opt, val_loader, model)

        start_new_epoch = False


def validate(opt, val_loader, model):
    # compute the encoding for all the validation images and captions
    motion_embs, cap_embs, ids = encode_data(
        model, val_loader, tb_writer, opt.log_step, logging.info)

    # motion retrieval
    (r1, r5, r10, medr, meanr) = m2t(
        motion_embs, cap_embs, ids, val_loader.dataset.texts_data)
    # image retrieval
    (r1i, r5i, r10i, medri, meanri) = t2m(
        motion_embs, cap_embs, ids, val_loader.dataset.texts_data)

    logging.info("Motion to text: %.1f, %.1f, %.1f, %.1f, %.1f" %
                 (r1, r5, r10, medr, meanr))
    logging.info("Text to motion: %.1f, %.1f, %.1f, %.1f, %.1f" %
                 (r1i, r5i, r10i, medri, meanri))
    # sum of recalls to be used for early stopping
    currscore = r1 + r5 + r10 + r1i + r5i + r10i
    logging.info("Currscore: %.1f" % currscore)

    # record metrics in tensorboard
    tb_writer.add_scalar('r1', r1, global_step=model.Eiters)
    tb_writer.add_scalar('r5', r5, global_step=model.Eiters)
    tb_writer.add_scalar('r10', r10, global_step=model.Eiters)
    tb_writer.add_scalar('medr', medr, global_step=model.Eiters)
    tb_writer.add_scalar('meanr', meanr, global_step=model.Eiters)
    tb_writer.add_scalar('r1i', r1i, global_step=model.Eiters)
    tb_writer.add_scalar('r5i', r5i, global_step=model.Eiters)
    tb_writer.add_scalar('r10i', r10i, global_step=model.Eiters)
    tb_writer.add_scalar('medri', medri, global_step=model.Eiters)
    tb_writer.add_scalar('meanri', meanri, global_step=model.Eiters)
    tb_writer.add_scalar('rsum', currscore, global_step=model.Eiters)

    return currscore


def save_checkpoint(state, is_best, prefix: Path, filename='latest.ckpt'):
    path = prefix / 'checkpoints'
    path.mkdir(exist_ok=True, parents=True)
    torch.save(state, path / filename)
    if is_best:
        shutil.copyfile(path / filename, path / 'model_best.ckpt')


def adjust_learning_rate(opt, optimizer, epoch):
    """Sets the learning rate to the initial LR
       decayed by 10 at [lr_update] epoch"""
    lr_multiplier = (0.1 ** (1 if (epoch // opt.lr_update) > 0 else 0))
    for param_group in optimizer.param_groups:
        param_group['lr'] *= lr_multiplier


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    _train()
