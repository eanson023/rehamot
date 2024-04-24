#! python3
# -*- encoding: utf-8 -*-
"""
Created on Tue Feb  07 21:52:10 2023

loss合集

@author: eanson
"""
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from einops import reduce


def cross_perceptual_salience_mapping(motion, query, c_mask, q_mask):
    # (batch_size, m_seq_len, q_seq_len)
    score = torch.einsum('amd,bqd->abmq', motion, query)
    # query-wise softmax (m_batch_size, q_batch_size, m_seq_len, q_seq_len)
    score_m = nn.Softmax(dim=3)(mask_logits(score, q_mask.unsqueeze(0).unsqueeze(2)))
    # motion-wise softmax (m_batch_size, q_batch_size, m_seq_len, q_seq_len)
    score_t = nn.Softmax(dim=2)(mask_logits(score, c_mask.unsqueeze(1).unsqueeze(-1)))
    # (m_batch_size, q_batch_size, q_seq_len, m_seq_len)
    score_t = score_t.transpose(2, 3) 
    # m2t perceptual similarity
    score_m = reduce(score * score_m, 'a b m q -> a b m', 'sum')
    # m2t salience similarity
    score_m = reduce(score_m, 'a b m -> a b', 'max')
    # t2m perceptual similarity
    score_t = reduce(score.transpose(2, 3) * score_t, 'a b q m -> a b q', 'sum')
    # t2m salience similarity (q_batch_size, m_batch_size)
    score_t = reduce(score_t, 'a b q -> a b', 'max')
    score = 1/2 * score_m + 1/2 * score_t
    return score


def mask_logits(inputs, mask, mask_value=-1e30):
    mask = mask.type(torch.float32)
    return inputs + (1.0 - mask) * mask_value


def cosine_sim(mm, s, *args):
    """Cosine similarity between all the motion and sentence pairs
    """
    return mm.mm(s.t())


def order_sim(mm, s):
    """Order embeddings similarity measure $max(0, s-mm)$
    """
    YmX = (s.unsqueeze(1).expand(s.size(0), mm.size(0), s.size(1))
           - mm.unsqueeze(0).expand(s.size(0), mm.size(0), s.size(1)))
    score = -YmX.clamp(min=0).pow(2).sum(2).sqrt().t()
    return score


class InfoNCELoss(nn.Module):
    """
    Compute symmetric cross entropy loss 
    """

    def __init__(self, sim, temp: float = 0.07, threshold_hetero=1.0, threshold_homo=1.0, **kwargs):
        super(InfoNCELoss, self).__init__()
        self.sim = sim
        self.max_violation = False
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.threshold_hetero = threshold_hetero
        self.threshold_homo = threshold_homo
    
    def _forward_once(self, scores):
        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_emb1 = logit_scale * scores

        sim_targets = torch.arange(len(scores), device=scores.device)

        loss = self.cross_entropy_loss(logits_per_emb1, sim_targets)
        return loss

    def forward(self, emb1, emb2, emb1_mask, emb2_mask):
        scores = self.sim(emb1, emb2, emb1_mask, emb2_mask)

        scores_emb1 = self.sim(emb1, emb1, emb1_mask, emb1_mask)
        scores_emb2 = self.sim(emb2, emb2, emb2_mask, emb2_mask)

        # Intra-Modal
        cost_intra = self._forward_once(scores_emb1) + self._forward_once(scores_emb2)
        # cost_intra = 0
        # clear false negative samples
        if self.max_violation:
            scores_emb1 = scores_emb1.detach() 
            scores_emb2 = scores_emb2.detach() 
            mask = torch.eye(scores.size(0)) > .5
            I = Variable(mask).to(scores.device)
            scores_emb1 = scores_emb1 * ~I
            scores_emb2 = scores_emb2 * ~I
            mask_emb1 = scores_emb1 > self.threshold_hetero
            mask_emb2 = scores_emb2 > self.threshold_homo
            mask = (mask_emb1 | mask_emb2)
            I = Variable(mask).to(scores.device)
            scores = scores * ~I
        
        # Inter-Modal
        loss = self._forward_once(scores) + self._forward_once(scores.T) + cost_intra

        return loss

class ContrastiveLoss(nn.Module):
    """
    Compute triplet loss
    """

    def __init__(self, sim, margin=0, max_violation=False, **kwargs):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.sim = sim

        self.max_violation = max_violation
    
    def _forward_once(self, scores):
        diagonal = scores.diag().view(scores.size(0), 1)

        d1 = diagonal.expand_as(scores)

        # compare every diagonal score to scores in its column
        # text retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        I = Variable(mask).to(scores.device)

        cost_s = cost_s.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
        
        return cost_s.sum()

    def forward(self, mm, s, m_mask, s_mask):

        # compute inter-modal triplet loss
        scores = self.sim(mm, s, m_mask, s_mask)
        cost_inter = self._forward_once(scores) + self._forward_once(scores.T)

        return cost_inter, 0

class SoftContrastiveLoss(nn.Module):
    """
    Compute triplet loss
    """

    def __init__(self, sim, margin=0, max_violation=False, threshold_hetero=1.0, threshold_homo=1.0, **kwargs):
        super(SoftContrastiveLoss, self).__init__()
        self.margin = margin
        self.sim = sim

        self.max_violation = max_violation
        self.threshold_hetero = threshold_hetero
        self.threshold_homo = threshold_homo

   
    def forward(self, mm, s, m_mask, s_mask):
        # compute inter-modal triplet loss
        scores = self.sim(mm, s, m_mask, s_mask)

        scores_emb1 = self.sim(mm, mm, m_mask, m_mask)
        scores_emb2 = self.sim(s, s, s_mask, s_mask)

        # cost_intra = self._forward_once(scores_emb1) + self._forward_once(scores_emb2)
        cost_intra = 0
        # clear false negative samples
        drop_num = 0
        if self.max_violation:
            scores_emb1 = scores_emb1.detach() 
            scores_emb2 = scores_emb2.detach() 
            mask = torch.eye(scores.size(0)) > .5
            I = Variable(mask).to(scores.device)
            scores_emb1 = scores_emb1 * ~I
            scores_emb2 = scores_emb2 * ~I
            mask_emb1 = scores_emb1 > self.threshold_hetero
            mask_emb2 = scores_emb2 > self.threshold_homo
            mask = (mask_emb1 | mask_emb2)
            I = Variable(mask).to(scores.device)
            scores = scores * ~I
            drop_num = I.sum()

        cost = self._forward_once(scores) + self._forward_once(scores.T)+ cost_intra

        return cost, drop_num
    
    def _forward_once(self, scores):
        diagonal = scores.diag().view(scores.size(0), 1)

        d1 = diagonal.expand_as(scores)

        # compare every diagonal score to scores in its column
        # text retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        I = Variable(mask).to(scores.device)

        cost_s = cost_s.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
        
        return cost_s.sum()


class ContrastiveLossMoco(nn.Module):
    """
    Compute triplet loss
    """

    def __init__(self, margin=0, max_violation=False, queue_size: int = 65536, **kwargs):
        super(ContrastiveLossMoco, self).__init__()
        self.margin = margin
        self.sim = cosine_sim

        self.max_violation = max_violation

        self.queue_size = queue_size
        self.register_buffer("idx_queue", torch.full(
            (1, self.queue_size), -100))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    def forward(self, emb1, emb2, idx):
        # compute motion-sentence score matrix
        scores = self.sim(emb1, emb2)

        idx = torch.tensor(idx, device=emb1.device).view(-1, 1)
        idx_all = torch.cat([idx.t(), self.idx_queue.clone().detach()], dim=1)
        pos_mask = torch.eq(idx, idx_all)

        # positive-pair score
        diagonal = scores.diag().view(-1, 1)

        # 向右扩展
        d = diagonal.expand_as(scores)
        # 给定 emb1 检索 emb2中的条数
        cost_emb1 = (self.margin + scores - d).clamp(min=0)

        # clear positive pairs
        I = Variable(pos_mask).to(emb1.device)
        cost_emb1 = cost_emb1.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            # always hardest negative
            # cost_emb1 = cost_emb1.max(1)[0]

            # A negative emb2 is sampled for each emb1 based on the similarity score
            embeds_neg = []
            for b in range(scores.size(0)):
                neg_idx = torch.multinomial(cost_emb1[b] + 1e-8, 1).item()
                embeds_neg.append(cost_emb1[b, neg_idx])
            cost_emb1 = torch.stack(embeds_neg, dim=0)

        # dequeue and enqueue
        self._dequeue_and_enqueue(idx)

        return cost_emb1.sum()

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        # array out of bounds check
        if ptr + batch_size >= self.queue_size:
            split = self.queue_size - ptr
            prefix = batch_size - split
            self.idx_queue[:, ptr:] = keys[:split].T
            self.idx_queue[:, :prefix] = keys[split:].T
        else:
            # replace the keys at ptr (dequeue and enqueue)
            self.idx_queue[:, ptr: ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.queue_size  # move pointer

        self.queue_ptr[0] = ptr
