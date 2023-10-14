# -*- coding: utf-8 -*-

from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional, Tuple

import torch


class Metric(object):

    def __init__(self, reverse: Optional[bool] = None, eps: float = 1e-12) -> Metric:
        super().__init__()

        self.n = 0.0
        self.count = 0.0
        self.total_loss = 0.0
        self.reverse = reverse
        self.eps = eps

    def __repr__(self):
        return f"loss: {self.loss:.4f} - " + ' '.join([f"{key}: {val:6.2%}" for key, val in self.values.items()])

    def __lt__(self, other: Metric) -> bool:
        if not hasattr(self, 'score'):
            return True
        if not hasattr(other, 'score'):
            return False
        return (self.score < other.score) if not self.reverse else (self.score > other.score)

    def __le__(self, other: Metric) -> bool:
        if not hasattr(self, 'score'):
            return True
        if not hasattr(other, 'score'):
            return False
        return (self.score <= other.score) if not self.reverse else (self.score >= other.score)

    def __gt__(self, other: Metric) -> bool:
        if not hasattr(self, 'score'):
            return False
        if not hasattr(other, 'score'):
            return True
        return (self.score > other.score) if not self.reverse else (self.score < other.score)

    def __ge__(self, other: Metric) -> bool:
        if not hasattr(self, 'score'):
            return False
        if not hasattr(other, 'score'):
            return True
        return (self.score >= other.score) if not self.reverse else (self.score <= other.score)

    def __add__(self, other: Metric) -> Metric:
        return other

    @property
    def score(self):
        raise AttributeError

    @property
    def loss(self):
        return self.total_loss / (self.count + self.eps)

    @property
    def values(self):
        raise AttributeError


class AttachmentMetric(Metric):

    def __init__(
        self,
        loss: Optional[float] = None,
        preds: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        golds: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        mask: Optional[torch.BoolTensor] = None,
        reverse: bool = False,
        eps: float = 1e-12
    ) -> AttachmentMetric:
        super().__init__(reverse=reverse, eps=eps)

        self.n_ucm = 0.0
        self.n_lcm = 0.0
        self.total = 0.0
        self.correct_arcs = 0.0
        self.correct_rels = 0.0

        if loss is not None:
            self(loss, preds, golds, mask)

    def __call__(
        self,
        loss: float,
        preds: Tuple[torch.Tensor, torch.Tensor],
        golds: Tuple[torch.Tensor, torch.Tensor],
        mask: torch.BoolTensor
    ) -> AttachmentMetric:
        lens = mask.sum(1)
        arc_preds, rel_preds, arc_golds, rel_golds = *preds, *golds
        arc_mask = arc_preds.eq(arc_golds) & mask
        rel_mask = rel_preds.eq(rel_golds) & arc_mask
        arc_mask_seq, rel_mask_seq = arc_mask[mask], rel_mask[mask]

        self.n += len(mask)
        self.count += 1
        self.total_loss += float(loss)
        self.n_ucm += arc_mask.sum(1).eq(lens).sum().item()
        self.n_lcm += rel_mask.sum(1).eq(lens).sum().item()

        self.total += len(arc_mask_seq)
        self.correct_arcs += arc_mask_seq.sum().item()
        self.correct_rels += rel_mask_seq.sum().item()
        return self

    def __add__(self, other: AttachmentMetric) -> AttachmentMetric:
        metric = AttachmentMetric(eps=self.eps)
        metric.n = self.n + other.n
        metric.count = self.count + other.count
        metric.total_loss = self.total_loss + other.total_loss
        metric.n_ucm = self.n_ucm + other.n_ucm
        metric.n_lcm = self.n_lcm + other.n_lcm
        metric.total = self.total + other.total
        metric.correct_arcs = self.correct_arcs + other.correct_arcs
        metric.correct_rels = self.correct_rels + other.correct_rels
        metric.reverse = self.reverse or other.reverse
        return metric

    @property
    def score(self):
        return self.las

    @property
    def ucm(self):
        return self.n_ucm / (self.n + self.eps)

    @property
    def lcm(self):
        return self.n_lcm / (self.n + self.eps)

    @property
    def uas(self):
        return self.correct_arcs / (self.total + self.eps)

    @property
    def las(self):
        return self.correct_rels / (self.total + self.eps)

    @property
    def pre_arc(self):
        return self.correct_arcs / (self.total + self.eps)
    @property
    def rec_arc(self):
        return self.correct_arcs / (self.total + self.eps)
    @property
    def f1_arc(self):
        return 2*self.pre_arc*self.rec_arc / (self.pre_arc + self.rec_arc)

    @property
    def pre_rel(self):
        return self.correct_rels / (self.total + self.eps)

    @property
    def rec_rel(self):
        return self.correct_rels / (self.total + self.eps)

    @property
    def f1_rel(self):
        return 2 * self.pre_rel * self.rec_rel / (self.pre_rel + self.rec_rel)

    @property
    def values(self) -> Dict:
        return {'f1_arc': self.f1_arc,
                'f1_rel': self.f1_rel,
                'UAS': self.uas,
                'LAS': self.las}


class AttachmentMetricMulti(Metric):

    def __init__(
        self,
        loss: Optional[float] = None,
        preds: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        golds: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        mask: Optional[torch.BoolTensor] = None,
        reverse: bool = False,
        eps: float = 1e-12
    ) -> AttachmentMetricMulti:
        super().__init__(reverse=reverse, eps=eps)

        self.correct_arcs = 0.0
        self.correct_rels = 0.0
        self.pred = 0.0
        self.golden = 0.0

        if loss is not None:
            self(loss, preds, golds, mask)

    def __call__(
        self,
        loss: float,
        preds: Tuple[torch.Tensor, torch.Tensor],
        golds: Tuple[torch.Tensor, torch.Tensor],
        mask: torch.BoolTensor
    ) -> AttachmentMetric:
        # lens = mask.sum(1)
        arc_preds, rel_preds, arc_golds, rel_golds = *preds, *golds
        # mask_multi = torch.tensor([1 for i in range(arc_golds.size(1))]).bool()
        # mask_multi[0] = False
        mask_arc = arc_golds.gt(0)
        mask_rel = rel_golds.gt(0)
        sum_no_par = arc_golds.sum(dim=1).eq(0).sum().item()
        sum_arcs = arc_golds.sum().item()
        arc_mask = arc_preds.eq(arc_golds) & mask_arc
        rel_mask = rel_preds.eq(rel_golds) & mask_rel

        self.n += len(mask)
        self.count += 1
        self.total_loss += float(loss)
        self.golden += sum_no_par + sum_arcs
        self.pred += arc_golds.size(1) - sum_no_par + 1
        self.correct_arcs += arc_mask.sum().item() + 1
        self.correct_rels += rel_mask.sum().item() + 1
        return self

    def __add__(self, other: AttachmentMetricMulti) -> AttachmentMetricMulti:
        metric = AttachmentMetricMulti(eps=self.eps)
        metric.n = self.n + other.n
        metric.count = self.count + other.count
        metric.total_loss = self.total_loss + other.total_loss
        metric.golden = self.golden + other.golden
        metric.pred = self.pred + other.pred
        metric.correct_arcs = self.correct_arcs + other.correct_arcs
        metric.correct_rels = self.correct_rels + other.correct_rels
        metric.reverse = self.reverse or other.reverse
        return metric

    @property
    def pre_arc(self):
        return self.correct_arcs / (self.pred + self.eps)
    @property
    def rec_arc(self):
        return self.correct_arcs / (self.golden + self.eps)
    @property
    def f1_arc(self):
        return 2*self.pre_arc*self.rec_arc / (self.pre_arc + self.rec_arc)

    @property
    def pre_rel(self):
        return self.correct_rels / (self.pred + self.eps)

    @property
    def rec_rel(self):
        return self.correct_rels / (self.golden + self.eps)

    @property
    def f1_rel(self):
        return 2 * self.pre_rel * self.rec_rel / (self.pre_rel + self.rec_rel)

    @property
    def values(self) -> Dict:
        return {'f1_arc': self.f1_arc,
                'f1_rel': self.f1_rel}