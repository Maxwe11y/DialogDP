# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from models.base import Model
from modules import MLP, Biaffine, max_min_scale
from utils.utils_loss import multilabel_soft_margin_loss
import torch.nn.functional as F


class TopDownBottomUp(nn.Module):
    def __init__(self, configs=None, shared_mlp=False):
        super(TopDownBottomUp, self).__init__()
        self.args = configs
        self.arc_mlp_d = MLP(n_in=self.args.n_encoder_hidden, n_out=self.args.n_arc_mlp, dropout=self.args.mlp_dropout)
        self.arc_mlp_h = MLP(n_in=self.args.n_encoder_hidden, n_out=self.args.n_arc_mlp, dropout=self.args.mlp_dropout)
        if shared_mlp:
            self.rel_mlp_d = self.arc_mlp_d
            self.rel_mlp_h = self.arc_mlp_h
        else:
            self.rel_mlp_d = MLP(n_in=self.args.n_encoder_hidden, n_out=self.args.n_rel_mlp, dropout=self.args.mlp_dropout)
            self.rel_mlp_h = MLP(n_in=self.args.n_encoder_hidden, n_out=self.args.n_rel_mlp, dropout=self.args.mlp_dropout)

        self.arc_attn = Biaffine(n_in=self.args.n_arc_mlp, scale=self.args.scale, bias_x=True, bias_y=False)
        self.rel_attn = Biaffine(n_in=self.args.n_rel_mlp, n_out=self.args.n_rels, bias_x=True, bias_y=True)

    def forward(self, inputs):
        arc_d = self.arc_mlp_d(inputs)
        arc_h = self.arc_mlp_h(inputs)
        rel_d = self.rel_mlp_d(inputs)
        rel_h = self.rel_mlp_h(inputs)

        # [batch_size, seq_len, seq_len]
        s_arc = self.arc_attn(arc_d, arc_h)

        # [batch_size, seq_len, seq_len, n_rels]
        s_rel = self.rel_attn(rel_d, rel_h).permute(0, 2, 3, 1)

        return s_arc, s_rel


class BiaffineDependencyModel(Model):

    def __init__(self, configs=None, **kwargs):
        super(BiaffineDependencyModel, self).__init__(configs)
        self.bottomup = TopDownBottomUp(configs)
        self.topdown = TopDownBottomUp(configs, shared_mlp=configs.shared_mlp)
        self.criterion_arc = nn.CrossEntropyLoss()
        self.criterion_rel = nn.CrossEntropyLoss()

        self.criterion_multilabel_arc = multilabel_soft_margin_loss
        self.criterion_multilabel_rel = nn.MultiLabelSoftMarginLoss()
        self.klloss = nn.KLDivLoss(reduction='batchmean', log_target=True)
        self.celoss = nn.CrossEntropyLoss()
        if configs.supervision_loss == 'mse':
            self.sup_loss = self.loss_supervision
        elif configs.supervision_loss == 'kl':
            self.sup_loss = self.loss_supervision_kl
        elif configs.supervision_loss == 'ce':
            self.sup_loss = self.loss_supervision_ce
        else:
            print('Invalid supervision loss!')
        # self.criterion_multilabel_rel = multilabel_soft_margin_loss
        # self.distribution =torch.log(torch.Tensor([6137, 2389, 1195, 582, 343, 156, 102, 57, 44, 24, 22, 14, 11, 12, 3, 3,
        #                                   7, 1, 3, 3, 2, 1, 1, 1, 1, 1]))   # STAC
        self.distribution = torch.log(torch.Tensor([45747, 15136, 5197, 2137, 1033, 638, 291, 156, 70, 35, 3, 1]))  # Molweni

        self.class_weight = 1.0 /torch.Tensor([0.6474, 1.4443, 0.3767, 1.6739, 0.2563, 3.2614, 1.4235, 0.5023, 7.9849, 1.6944,
                                          0.7312, 9.9241, 7.3903, 1.1875, 3.5809, 5.8872])  # STAC
        # self.class_weight_mol = torch.Tensor([0.9416, 4.1496, 0.1956, 0.2598, 0.3107, 4.8065, 5.0031, 1.9302, 16.9337, 2.5202,
        #                                   2.7111, 6.2628, 21.5821, 2.0748, 26.5226, 25.0156])
        self.gen_dist_vec()

    def forward(self, inputs, inputs_additional, plm_tok=30522):
        r"""
        Args:
            words (~torch.LongTensor): ``[batch_size, seq_len]``.
                Word indices.

        Returns:
            ~torch.Tensor, ~torch.Tensor:
        """

        x = self.encode(inputs, inputs_additional, plm_tok).unsqueeze(0)

        s_arc_bu, s_rel_bu = self.bottomup(x)

        s_arc_td, s_rel_td = self.topdown(x)

        mask = self.soft_gradient_window_mask(s_arc_bu.size(1))
        s_arc_bu = self.soft_masking(s_arc_bu, mask)
        s_arc_td = self.soft_masking(s_arc_td, mask)

        return s_arc_bu, s_rel_bu, s_arc_td, s_rel_td

    def loss(self, s_arc, s_arc_td, s_rel, s_rel_td, arcs, arcs_td, rels, rels_td, mask):
        loss_bu_arc, loss_bu_rel, rel_bu = self.loss_bu(s_arc, s_rel, arcs, rels, mask)
        loss_td_arc, loss_td_rel, rel_td = self.loss_td(s_arc_td, s_rel_td, arcs_td, rels_td)
        loss_super_arc = self.sup_loss(s_arc, s_arc_td)
        loss_super_rel = self.loss_supervision_rel_kl(rel_bu, rel_td)

        return float(self.args.alpha[0])*loss_bu_arc + float(self.args.alpha[1])*loss_bu_rel \
            + float(self.args.alpha[2])*loss_td_arc + float(self.args.alpha[3])*loss_td_rel \
            + float(self.args.alpha[4])*loss_super_arc \
            + float(self.args.alpha[5])*loss_super_rel, \
            loss_bu_arc.data.item(), loss_bu_rel.data.item(),\
            loss_td_arc.data.item(), loss_td_rel.data.item(), \
            loss_super_arc.data.item(), loss_super_rel.data.item()

    def loss_bu(self, s_arc, s_rel, arcs, rels, mask, multi_par=True):
        r"""
        Args:
            s_arc (~torch.Tensor): ``[batch_size, seq_len, seq_len]``.
                Scores of all possible arcs.
            s_rel (~torch.Tensor): ``[batch_size, seq_len, seq_len, n_labels]``.
                Scores of all possible labels on each arc.
            arcs (~torch.LongTensor): ``[batch_size, seq_len]``.
                The tensor of gold-standard arcs.
            rels (~torch.LongTensor): ``[batch_size, seq_len]``.
                The tensor of gold-standard labels.
            mask (~torch.BoolTensor): ``[batch_size, seq_len]``.
                The mask for covering the unpadded tokens.
            partial (bool):
                ``True`` denotes the trees are partially annotated. Default: ``False``.

        Returns:
            ~torch.Tensor:
                The training loss.
        """

        tril_mask_ini = torch.triu(torch.ones(s_arc.size(0), s_arc.size(-1), s_arc.size(-1)), diagonal=0).to(s_arc.device)
        s_arc_ = s_arc.masked_fill(tril_mask_ini.bool(), float('-inf'))
        mask_bu = torch.sum(arcs, dim=-1).ne(0)
        s_arc_, arcs = s_arc_[mask_bu], arcs[mask_bu]
        s_rel, rels = s_rel[mask_bu], rels[mask_bu]
        tril_mask_ini = tril_mask_ini[mask_bu]
        arc_loss = self.criterion_multilabel_arc(s_arc_, arcs, mask=tril_mask_ini.bool())
        rel_loss_mask = rels.contiguous().view(-1, 16).sum(dim=1).ne(0)
        # rel_loss = self.criterion_multilabel_rel(s_rel.contiguous().view(-1, 16)[rel_loss_mask],
        #                                             rels.contiguous().view(-1, 16)[rel_loss_mask])
        s_rel_masked, rels_masked = s_rel.contiguous().view(-1, 16)[rel_loss_mask], (rels.contiguous().view(-1, 16)[rel_loss_mask]).argmax(dim=-1)
        rel_loss = F.cross_entropy(s_rel_masked, rels_masked) # add class_weight or not  weight=self.class_weight.to(s_rel_masked.device)

        return arc_loss, rel_loss, s_rel_masked


    def loss_td(self, s_arc, s_rel, arcs, rels):
        r"""
        Args:
            s_arc (~torch.Tensor): ``[batch_size, seq_len, seq_len]``.
                Scores of all possible arcs.
            s_rel (~torch.Tensor): ``[batch_size, seq_len, seq_len, n_labels]``.
                Scores of all possible labels on each arc.
            arcs (~torch.LongTensor): ``[batch_size, seq_len]``.
                The tensor of gold-standard arcs.
            rels (~torch.LongTensor): ``[batch_size, seq_len]``.
                The tensor of gold-standard labels.
            mask (~torch.BoolTensor): ``[batch_size, seq_len]``.
                The mask for covering the unpadded tokens.
            partial (bool):
                ``True`` denotes the trees are partially annotated. Default: ``False``.

        Returns:
            ~torch.Tensor:
                The training loss.
        """

        tril_mask_ini = torch.tril(torch.ones(s_arc.size(0), s_arc.size(-1), s_arc.size(-1)), diagonal=0).to(
            s_arc.device)
        s_arc_ = s_arc.masked_fill(tril_mask_ini.bool(), float('-inf'))
        mask_td = torch.sum(arcs, dim=-1).ne(0)
        s_arc_, arcs = s_arc_[mask_td], arcs[mask_td]
        s_rel, rels = s_rel[mask_td], rels[mask_td]
        tril_mask_ini = tril_mask_ini[mask_td]

        arc_loss_td = self.criterion_multilabel_arc(s_arc_, arcs, mask=tril_mask_ini.bool())
        rel_loss_mask = rels.contiguous().view(-1, 16).sum(dim=1).ne(0)
        rel_loss_mask_supervise = rels.transpose(0, 1).contiguous().view(-1, 16).sum(dim=1).ne(0)
        # s_rel_masked, rels_masked = s_rel.contiguous().view(-1, 16)[rel_loss_mask], rels.contiguous().view(-1, 16)[rel_loss_mask]
        # rel_loss_td = self.criterion_multilabel_rel(s_rel_masked, rels_masked)
        s_rel_supervise = s_rel.transpose(0, 1).contiguous().view(-1, 16)[rel_loss_mask_supervise]
        s_rel_masked, rels_masked = s_rel.contiguous().view(-1, 16)[rel_loss_mask], rels.contiguous().view(-1, 16)[
            rel_loss_mask].argmax(dim=-1)
        rel_loss_td = F.cross_entropy(s_rel_masked, rels_masked)

        return arc_loss_td, rel_loss_td, s_rel_supervise

    def loss_supervision(self, s_arc_bu, s_arc_td):
        s_arc_bu = s_arc_bu.squeeze(0)
        s_arc_td = s_arc_td.transpose(1, 2).squeeze(0)
        mask = torch.triu(torch.ones(s_arc_bu.size(0), s_arc_bu.size(-1)), diagonal=0).to(
            s_arc_bu.device).bool()

        return nn.functional.mse_loss(s_arc_bu.masked_fill_(mask, 0), s_arc_td.masked_fill_(mask, 0)) \
            + nn.functional.mse_loss(s_arc_td.masked_fill_(mask, 0), s_arc_bu.masked_fill_(mask, 0))

    def loss_supervision_rel_kl(self, s_rel_bu, s_rel_td):
        # s_rel_td = s_rel_td.transpose(0, 1).contiguous().view(-1, 16)
        s_rel_td = s_rel_td.contiguous().view(-1, 16)

        s_rel_bu = F.log_softmax(s_rel_bu, dim=1)
        s_rel_td = F.log_softmax(s_rel_td, dim=1)

        return self.klloss(s_rel_bu, s_rel_td) + self.klloss(s_rel_td, s_rel_bu)

    def loss_supervision_kl(self, s_arc_bu, s_arc_td):
        s_arc_bu = s_arc_bu.squeeze(0)
        s_arc_td = s_arc_td.transpose(1, 2).squeeze(0)
        mask = torch.triu(torch.ones(s_arc_bu.size(0), s_arc_bu.size(-1)), diagonal=0).to(
            s_arc_bu.device).bool()
        s_arc_bu = F.log_softmax(s_arc_bu, dim=1).masked_fill(mask, 0)
        s_arc_td = F.log_softmax(s_arc_td, dim=1).masked_fill(mask, 0)

        return self.klloss(s_arc_bu, s_arc_td) + self.klloss(s_arc_td, s_arc_bu)

    def loss_supervision_ce(self, s_arc_bu, s_arc_td):
        s_arc_bu = s_arc_bu.squeeze(0)
        s_arc_td = s_arc_td.transpose(1, 2).squeeze(0)
        mask = torch.triu(torch.ones(s_arc_bu.size(0), s_arc_bu.size(-1)), diagonal=0).to(
            s_arc_bu.device).bool()
        s_arc_td_ = s_arc_td.softmax(dim=1).masked_fill(mask, 0)
        loss_1 = self.celoss(s_arc_bu, s_arc_td_)
        s_arc_bu_ = s_arc_bu.softmax(dim=1).masked_fill(mask, 0)
        loss_2 = self.celoss(s_arc_td, s_arc_bu_)

        return loss_1 + loss_2

    def soft_gradient_window_mask_(self, seq_len):
        # distribution is a log_2 distribution
        mask = torch.ones(seq_len, seq_len)
        # mid1 = int(0.2*seq_len) + 1 if int(0.2*seq_len) + 1< seq_len else int(0.2*seq_len)
        # mid2 = int(0.6*seq_len) + 1 if int(0.6*seq_len) + 1< seq_len else int(0.6*seq_len)
        mid1 = min(4, seq_len-1)  # 3 + 1
        mid2 = min(11, seq_len-1) # 8 + 1
        dist_1 = self.distribution[0:mid1]
        dist_1 = max_min_scale(dist_1, new_min=1.0, new_max=1.0)
        # mask[:, 1:mid1 + 1] = mask[:, 1:mid1 + 1] - torch.sign(mask[:, 1:mid1 + 1])*dist_1_delta.unsqueeze(0)
        mask[:,1:mid1+1] = mask[:,1:mid1+1]*dist_1.unsqueeze(0)
        mask[1:mid1 + 1] = mask[1:mid1 + 1] * dist_1.unsqueeze(1)
        if seq_len > mid1+1:

            dist_2 = self.distribution[mid1:mid2]
            dist_2 = max_min_scale(dist_2, new_min=0.95, new_max=1.0)
            mask[:,mid1+1:mid2+1] = mask[:,mid1+1:mid2+1]*dist_2.unsqueeze(0)
            mask[mid1 + 1:mid2 + 1] = mask[mid1 + 1:mid2 + 1] * dist_2.unsqueeze(1)
            if seq_len > mid2+1:
                mid3 = min(seq_len-1, self.distribution.size(0))
                dist_3 = self.distribution[mid2: mid3]
                dist_3 = max_min_scale(dist_3, new_min=0.8, new_max=0.95)
                mask[:,mid2+1:mid3+1] = mask[:,mid2+1:mid3+1]*dist_3.unsqueeze(0)
                mask[mid2 + 1:mid3 + 1] = mask[mid2 + 1:mid3 + 1] * dist_3.unsqueeze(1)
                if seq_len > mid3 + 1:
                    mask[:,mid3 + 1:] = mask[:,mid3 + 1:]*0.7
                    mask[mid3 + 1:] = mask[mid3 + 1:] * 0.7

        return mask

    def soft_gradient_window_mask(self, seq_len,  max_len=14): # max_len=14 for Molweni max_len=89 for STAC
        indices = torch.LongTensor([[i for i in range(i_start, seq_len + i_start)] for i_start in range(0, -seq_len, -1)]) + max_len
        mask = self.dist_vec[indices]
        return mask

    def gen_dist_vec(self, mid_1=8, mid_2=10, max_len=14): #mid_1=8, mid_2=10 max_len=14 for Molweni    mid_1=10, mid_2=20, max_len=89 for STAC
        dist_1 = self.distribution[0:mid_1]
        dist_1 = max_min_scale(dist_1, new_min=1.0, new_max=1.0)
        dist_2 = self.distribution[mid_1:mid_2]
        dist_2 = max_min_scale(dist_2, new_min=0.95, new_max=1.0)
        dist_3 = self.distribution[mid_2:]
        dist_3 = max_min_scale(dist_3, new_min=0.8, new_max=0.95)
        mid_3 = self.distribution.size(0)
        self.dist_vec = torch.ones(max_len*2+1)*0.7
        self.dist_vec[max_len - mid_3: max_len - mid_2] = dist_3.flip(dims=(0,))
        self.dist_vec[max_len + mid_2 + 1 : max_len + mid_3 + 1] = dist_3
        self.dist_vec[max_len - mid_2: max_len - mid_1] = dist_2.flip(dims=(0,))
        self.dist_vec[max_len + mid_1 + 1: max_len + mid_2 + 1] = dist_2
        self.dist_vec[max_len - mid_1: max_len] = dist_1.flip(dims=(0,))
        self.dist_vec[max_len + 1: max_len + mid_1 + 1] = dist_1

        self.dist_vec[max_len] = 1.0

    def soft_masking(self, s_arc, mask):
        mask = mask.to(s_arc.device)
        s_arc = s_arc - torch.abs(s_arc)*(1.0 - mask)

        return s_arc

    def decode(self, s_arc, s_rel, mask):
        r"""
        Args:
            s_arc (~torch.Tensor): ``[batch_size, seq_len, seq_len]``.
                Scores of all possible arcs.
            s_rel (~torch.Tensor): ``[batch_size, seq_len, seq_len, n_labels]``.
                Scores of all possible labels on each arc.
            mask (~torch.BoolTensor): ``[batch_size, seq_len]``.
                The mask for covering the unpadded tokens.
            tree (bool):
                If ``True``, ensures to output well-formed trees. Default: ``False``.
            proj (bool):
                If ``True``, ensures to output projective trees. Default: ``False``.

        Returns:
            ~torch.LongTensor, ~torch.LongTensor:
                Predicted arcs and labels of shape ``[batch_size, seq_len]``.
        """

        # lens = mask.sum(1)

        tril_mask_ini = torch.triu(torch.ones(s_arc.size(0), s_arc.size(-1), s_arc.size(-1)), diagonal=0).to(s_arc.device)
        s_arc = s_arc.masked_fill_(tril_mask_ini.bool(), float('-inf'))

        # triu_mask = torch.tril(torch.ones(s_arc.size(0), s_arc.size(-1), s_arc.size(-1)), diagonal=-1).to(s_arc.device)
        # s_arc = s_arc*triu_mask
        arc_preds = s_arc.argmax(-1)
        arc_preds = nn.functional.one_hot(arc_preds, num_classes=arc_preds.size(1))

        # rel_preds = s_rel.argmax(-1).gather(-1, arc_preds.unsqueeze(-1)).squeeze(-1)
        rel_preds = nn.functional.one_hot(s_rel.argmax(-1).squeeze(0), num_classes=16).unsqueeze(0)*arc_preds.unsqueeze(-1)
        return arc_preds, rel_preds
