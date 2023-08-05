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
from torch.autograd import Variable


def cosine_sim(mm, s):
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

    def __init__(self, temp: float = 0.07, queue_size: int = 65536, **kwargs):
        super(InfoNCELoss, self).__init__()
        self.queue_size = queue_size
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.cross_entropy_loss = nn.CrossEntropyLoss()

        self.register_buffer("idx_queue", torch.full(
            (1, self.queue_size), -100))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    def forward(self, emb1, emb2, idx=None):
        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_emb1 = logit_scale * emb1 @ emb2.T

        # symmetric loss function
        # compute label targets
        if idx is None:
            sim_targets = torch.arange(len(emb1), device=emb1.device)
        else:
            idx = torch.tensor(idx, device=emb1.device).view(-1, 1)
            idx_all = torch.cat(
                [idx.t(), self.idx_queue.clone().detach()], dim=1)
            pos_idx = torch.eq(idx, idx_all).float()
            sim_targets = pos_idx / pos_idx.sum(1, keepdim=True)

            # dequeue and enqueue
            self._dequeue_and_enqueue(idx)

        loss = self.cross_entropy_loss(logits_per_emb1, sim_targets)

        return loss

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


class ContrastiveLoss(nn.Module):
    """
    Compute triplet loss
    """

    def __init__(self, margin=0, max_violation=False, **kwargs):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.sim = cosine_sim

        self.max_violation = max_violation

    def forward(self, mm, s):
        # compute motion-sentence score matrix
        scores = self.sim(mm, s)

        diagonal = scores.diag().view(mm.size(0), 1)

        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # text retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # motion retrieval
        cost_m = (self.margin + scores - d2).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        I = Variable(mask).to(mm.device)

        cost_s = cost_s.masked_fill_(I, 0)
        cost_m = cost_m.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_m = cost_m.max(0)[0]

        return cost_s.sum() + cost_m.sum()


class SoftContrastiveLoss(nn.Module):
    """
    Compute triplet loss
    """

    def __init__(self, margin=0, max_violation=False, threshold_hetero=1.0, threshold_homo=1.0, **kwargs):
        super(SoftContrastiveLoss, self).__init__()
        self.margin = margin
        self.sim = cosine_sim

        self.max_violation = max_violation
        self.threshold_hetero = threshold_hetero
        self.threshold_homo = threshold_homo

    def forward(self, motion_emb, text_emb):
        return self.compute(motion_emb, text_emb) + self.compute(text_emb, motion_emb)

    def compute(self, emb1, emb2):
        # compute motion-sentence score matrix
        scores = self.sim(emb1, emb2)

        # Soft hard negative mining
        # DropTrip loss function implementation
        if self.max_violation:
            scores_emb1 = self.sim(emb1, emb1)
            scores_emb2 = self.sim(emb2, emb2)
            mask_emb1 = (scores_emb1 > self.threshold_hetero) & (
                scores_emb1 < 1 - 1e-6)
            mask_emb2 = (scores_emb2 > self.threshold_homo) & (
                scores_emb2 < 1 - 1e-6)
            scores[mask_emb1 | mask_emb2] = 0

        # positive-pair score
        diagonal = scores.diag().view(-1, 1)

        # Expand to the right
        d = diagonal.expand_as(scores)
        # Given emb1 retrieves the number of entries in emb2
        cost_emb1 = (self.margin + scores - d).clamp(min=0)

        # clear positive pairs
        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        I = Variable(mask).to(emb1.device)
        cost_emb1 = cost_emb1.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            # always hardest negative
            cost_emb1 = cost_emb1.max(1)[0]

        return cost_emb1.sum()


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
