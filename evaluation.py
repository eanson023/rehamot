from __future__ import print_function

import logging
import os
import time
from collections import OrderedDict
from pathlib import Path

import hydra
import numpy
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from model.data.utils import get_loader_single

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import model.launch.prepare  # NOQA

logger = logging.getLogger(__name__)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=0):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / (.0001 + self.count)

    def __str__(self):
        """String representation for logging
        """
        # for values that should be recorded exactly e.g. iteration number
        if self.count == 0:
            return str(self.val)
        # for stats
        return '%.4f (%.4f)' % (self.val, self.avg)


class LogCollector(object):
    """A collection of logging objects that can change from train to val"""

    def __init__(self, tb_writer):
        self.tb_writer = tb_writer
        # to keep the order of logged variables deterministic
        self.meters = OrderedDict()

    def update(self, k, v, n=0):
        # create a new meter if previously not recorded
        if k not in self.meters:
            self.meters[k] = AverageMeter()
        self.meters[k].update(v, n)

    def __str__(self):
        """Concatenate the meters in one log line
        """
        s = ''
        for i, (k, v) in enumerate(self.meters.items()):
            if i > 0:
                s += '  '
            s += k + ' ' + str(v)
        return s

    def tb_log(self, prefix='', global_step=None):
        """Log using tensorboard
        """
        for k, v in self.meters.items():
            self.tb_writer.add_scalar(
                prefix + k, v.val, global_step=global_step)

    def tb_scalar(self, k, v, global_step=None):
        self.tb_writer.add_scalar(k, v, global_step=global_step)

    def tb_figure(self, k, v, global_step=None):
        self.tb_writer.add_figure(k, v, global_step=global_step)


def encode_data(model, data_loader, tb_writer=None, log_step=10, logging=logger.info):
    """Encode all motions and texts loadable by `data_loader`
    """
    batch_time = AverageMeter()
    val_logger = LogCollector(tb_writer)

    # switch to evaluate mode
    model.val_start()

    end = time.time()

    # numpy array to keep all the embeddings
    motion_embs = None
    text_embs = None
    with torch.no_grad():
        batch_size = data_loader.batch_size
        for i, batch in enumerate(data_loader):
            keyids, cap_indice, indice = batch['keyid'], batch['text_index'], batch['index']
            # make sure val logger is used
            model.logger = val_logger

            # compute the embeddings
            motion_emb, text_emb, motion_emb_m, text_emb_m = model.forward_emb(
                **batch)

            # initialize the numpy arrays given the size of the embeddings
            if motion_embs is None:
                motion_embs = numpy.zeros(
                    (len(data_loader.dataset), motion_emb.size(1)))
                text_embs = numpy.zeros(
                    (len(data_loader.dataset), text_emb.size(1)))
                ids = []

            # preserve the embeddings by copying from gpu and converting to numpy
            for j, id in enumerate(keyids):
                index = i * batch_size + j
                motion_embs[index] = motion_emb.data.cpu().numpy().copy()[j]
                text_embs[index] = text_emb.data.cpu().numpy().copy()[j]
                ids.append({'keyid': id, 'text_index': cap_indice[j]})

            # measure accuracy and record loss
            model.forward_loss(motion_emb, text_emb,
                               motion_emb_m, text_emb_m, indice, is_train=False)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % log_step == 0:
                logging('Test: [{0}/{1}]\t'
                        '{e_log}\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        .format(
                            i + 1, len(data_loader), batch_time=batch_time,
                            e_log=str(model.logger)))

    return motion_embs, text_embs, ids


@hydra.main(version_base=None, config_path="configs", config_name="evaluation")
def evalrank(newcfg: DictConfig) -> None:
    """
    Evaluate a trained model on either dev or test.
    """
    # load model and options
    # Load last config
    output_dir = Path(hydra.utils.to_absolute_path(newcfg.folder))
    ckpt_path = newcfg.last_ckpt_path

    # Load previous config
    prevcfg = OmegaConf.load(output_dir / ".hydra/config.yaml")
    # Overload it
    cfg = OmegaConf.merge(prevcfg, newcfg)

    opt = cfg.machine
    device = opt.device
    # device = 'cpu'

    checkpoint = torch.load(
        ckpt_path, map_location=device)

    logger.info('Loading dataset')
    data_loader = get_loader_single(
        dataname=opt.dataname,
        split=cfg.split,
        batch_size=opt.batch_size,
        workers=opt.num_workers,
        shuffle=False,
        drop_last=False,
        data_cfg=cfg.data)
    logger.info(f"dataset '{cfg.data.dataname}' loaded")

    # Construct the model
    logger.info("Loading model")
    model = instantiate(cfg.model,
                        device=device,
                        learning_rate=opt.learning_rate,
                        grad_clip=opt.grad_clip,
                        nfeats=data_loader.dataset.nfeats,
                        # 避免递归提前加载encoder
                        _recursive_=False)

    # load model state
    model.load_state_dict(checkpoint['model'])

    logger.info(f"Model '{cfg.model.modelname}' loaded")
    logger.info('checkpoint info: epoch:{} best_rsum:{}'.format(
        checkpoint['epoch'], checkpoint['best_rsum']))
    logger.info('Computing results...')
    mot_embs, cap_embs, ids = encode_data(model, data_loader)

    logger.info('Motions: %d, Captions: %d' %
                (data_loader.dataset._num_motions, data_loader.dataset._num_texts))

    r, rt = m2t(mot_embs, cap_embs, ids,
                data_loader.dataset.texts_data, return_ranks=True)
    ri, rti = t2m(mot_embs, cap_embs, ids,
                  data_loader.dataset.texts_data, return_ranks=True)
    record_log(rt[2], rti[2], output_dir)

    ar = (r[0] + r[1] + r[2]) / 3
    ari = (ri[0] + ri[1] + ri[2]) / 3
    rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
    logger.info("rsum: %.1f" % rsum)
    logger.info("Average m2t Recall: %.1f" % ar)
    logger.info("Motion to text: %.1f %.1f %.1f %.1f %.1f" % r)
    logger.info("Average t2m Recall: %.1f" % ari)
    logger.info("Text to motion: %.1f %.1f %.1f %.1f %.1f" % ri)
    torch.save({'rt': rt, 'rti': rti}, output_dir / 'ranks.pth.tar')


def m2t(motions, captions, ids, texts_data, return_ranks=False):
    """
    Motions->Text (Motion Annotation)

    Args:
        motions: motion embeddings
        captions: caption embeddings
        ids: correspondence between keyid and text_index
        texts_data: text description of all ids
    """
    npts = len(texts_data)
    index_list = []

    ranks = numpy.zeros(npts)
    top1 = numpy.zeros(npts)

    ranks_info = []
    index = 0
    eit = 0
    while index < len(ids):
        keyid = ids[index]['keyid']
        texts = texts_data[keyid]
        step = len(texts)

        # Get query motion
        m = motions[index].reshape(1, motions.shape[1])

        # Compute scores
        d = numpy.dot(m, captions.T).flatten()
        # inds: The index of the stored score from high to low (i.e. the index of the value of d in descending order)
        inds = numpy.argsort(d)[::-1]
        index_list.append(inds[0])

        # Score
        rank = 1e20
        rank_index = 0
        # Return the highest index (ranking) in the n sentences (ground-truth) corresponding to the picture after sorting
        for i in range(index, index + step, 1):
            tmp = numpy.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
                rank_index = i
        # The minimum ranking of the most relevant positive samples of the eit graph
        ranks[eit] = rank
        # The most relevant sentence index for the image eit
        top1[eit] = inds[0]
        # ###############################################################################
        if return_ranks:
            cap_idx = ids[rank_index]['text_index']
            rank_info = {'keyid': keyid, 'rank': rank,
                         'text_index': cap_idx, 'text': texts[cap_idx], 'result': []}
            for idx in inds[:rank + 1]:
                keyid = ids[idx]['keyid']
                cap_idx = ids[idx]['text_index']
                sim = numpy.dot(motions[rank_index], captions[idx].T)
                sim_m = numpy.dot(motions[rank_index], motions[idx].T)
                sim_t = numpy.dot(captions[rank_index], captions[idx].T)
                rank_info['result'].append({'keyid': keyid, 'text_index': cap_idx,
                                            'text': texts_data[keyid][cap_idx], 'sim': sim, 'sim_m': sim_m, 'sim_t': sim_t})
            ranks_info.append(rank_info)
        # ###############################################################################

        index += step
        eit += 1

    # Compute metrics
    r1 = 100.0 * len(numpy.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(numpy.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(numpy.where(ranks < 10)[0]) / len(ranks)
    medr = numpy.floor(numpy.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1, ranks_info)
    else:
        return (r1, r5, r10, medr, meanr)


def t2m(motions, captions, ids, texts_data, return_ranks=False):
    """
    Text->Motions (Motion Search)

    Args:
        motions: motion embeddings
        captions: caption embeddings
        ids: correspondence between keyid and text_index
        texts_data: text description of all ids
    """
    motions_indice = []
    i = numpy.size(captions, 0) - 1
    while i >= 0:
        motions_indice.append(i)
        i = i - (ids[i]['text_index'] + 1)
    motions_indice.reverse()
    ims = motions[motions_indice]

    ranks = numpy.zeros(numpy.size(captions, 0))
    top1 = numpy.zeros(numpy.size(captions, 0))

    ranks_info = []
    index = 0
    # traverse the index of the key
    eit = 0
    while index < len(ids):
        keyid = ids[index]['keyid']
        texts = texts_data[keyid]
        step = len(texts)

        # Get query captions
        queries = captions[index: index + step]

        # Compute scores
        d = numpy.dot(queries, ims.T)
        inds = numpy.zeros(d.shape)
        for i in range(len(inds)):
            # inds: The index of the score of the i-th sentence under index from high to low (i.e. the index of the value of d in descending order)
            inds[i] = numpy.argsort(d[i])[::-1]
            # Find the ranking of ground-truth and record it in the overall ranks
            ranks[index + i] = numpy.where(inds[i] == eit)[0][0]
            rank = ranks[index + i].astype(int)

            # #########################################################################
            if return_ranks:
                cap_idx = ids[index + i]['text_index']
                rank_info = {'keyid': keyid, 'rank': rank,
                             'text_index': cap_idx, 'text': texts[cap_idx], 'result': []}
                for idx in inds[i, :rank + 1]:
                    key_index = motions_indice[int(idx)]
                    key_keyid = ids[key_index]['keyid']
                    key_cap_idx = ids[key_index]['text_index']
                    sim = numpy.dot(captions[index + i], motions[key_index].T)
                    sim_m = numpy.dot(motions[index + i], motions[key_index].T)
                    sim_t = numpy.dot(
                        captions[index + i], captions[key_index].T)
                    rank_info['result'].append({'keyid': key_keyid, 'text_index': key_cap_idx,
                                                'text': texts_data[key_keyid][key_cap_idx], 'sim': sim, 'sim_m': sim_m, 'sim_t': sim_t})
                ranks_info.append(rank_info)
            # ############################################################################

            top1[index + i] = inds[i][0]

        index += step
        eit += 1

    # Compute metrics
    r1 = 100.0 * len(numpy.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(numpy.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(numpy.where(ranks < 10)[0]) / len(ranks)
    medr = numpy.floor(numpy.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1, ranks_info)
    else:
        return (r1, r5, r10, medr, meanr)


def record_log(r_m2t, r_t2m, path):
    interval = [-1, 1, 5, 10, 50, 100, 100000]
    m2t_files = dict()
    t2m_files = dict()
    path = Path(path)
    for i, x in enumerate(interval[:-1]):
        for item in r_m2t:
            rank = item['rank']
            if x <= rank < interval[i + 1]:
                filepath = path / f"eval_m2t_r{interval[i + 1]}.txt"
                mode = 'a'
                if filepath not in m2t_files.keys():
                    m2t_files[filepath] = 0
                    # 覆盖以前的文件
                    mode = 'w'
                m2t_files[filepath] += 1
                with open(filepath, mode=mode, encoding='utf-8') as file_obj:
                    string = "keyid:%6s==>rank:%d\ttext_id:%d\ttext:%s\n" % (
                        item['keyid'], rank + 1, item['text_index'] + 1, item['text'])
                    res_string = ["%d. keyid:%6s\ttext_id:%d\tsim:%.2f\tsim_m:%.2f\tsim_t:%.2f\ttext:%s\n" % (
                        no + 1, y['keyid'], y['text_index'] + 1, y['sim'], y['sim_m'], y['sim_t'], y['text']) for no, y in enumerate(item['result'])]
                    file_obj.write(string)
                    file_obj.writelines(res_string)
                    file_obj.write('\n')
        for item in r_t2m:
            rank = item['rank']
            if x <= rank < interval[i + 1]:
                filepath = path / f"eval_t2m_r{interval[i + 1]}.txt"
                mode = 'a'
                if filepath not in t2m_files.keys():
                    t2m_files[filepath] = 0
                    mode = 'w'
                t2m_files[filepath] += 1
                with open(filepath, mode=mode, encoding='utf-8') as file_obj:
                    string = "keyid:%6s==>rank:%d\ttext_id:%d\ttext:%s\n" % (
                        item['keyid'], rank + 1, item['text_index'] + 1, item['text'])
                    res_string = ["%d. keyid:%6s\ttext_id:%d\tsim:%.2f\tsim_m:%.2f\tsim_t:%.2f\ttext:%s\n" % (
                        no + 1, y['keyid'], y['text_index'] + 1, y['sim'], y['sim_m'], y['sim_t'], y['text']) for no, y in enumerate(item['result'])]
                    file_obj.write(string)
                    file_obj.writelines(res_string)
                    file_obj.write('\n')

    m2t_files.update(t2m_files)
    for filepath, value in m2t_files.items():
        with open(filepath, mode='r+', encoding='utf-8') as file_obj:
            old = file_obj.read()
            file_obj.seek(0)
            file_obj.write(f"Count: {value}\n")
            file_obj.write(old)

    logger.info("evaluation results saved at {}".format(path))


if __name__ == '__main__':
    evalrank()
