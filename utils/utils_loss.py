'''
Author: Li Wei
Email: wei008@e.ntu.edu.sg
'''
import torch.nn as nn
import torch

def multilabel_soft_margin_loss(
    pred,
    target,
    mask,
    weight=None,
    reduction: str = "mean",
):
    r"""multilabel_soft_margin_loss(input, target, weight=None, size_average=None, reduce=None, reduction='mean') -> Tensor

    See :class:`~torch.nn.MultiLabelSoftMarginLoss` for details.
    """
    logsigmoid = nn.LogSigmoid()
    loss = -(target * logsigmoid(pred).masked_fill_(mask, 0) + (1 - target) * logsigmoid(-pred).masked_fill_(mask, 0))

    if weight is not None:
        loss = loss * weight

    class_dim = pred.dim() - 1
    # C = pred.size(class_dim)
    # loss = loss.sum(dim=class_dim) / C  # only return N loss values
    loss = torch.div(loss.sum(dim=class_dim), mask.sum(dim=class_dim)+1e-12)  # small loss
    # loss = torch.div(loss.sum(dim=class_dim), target.sum(dim=class_dim) + 1e-12) # large loss

    if reduction == "none":
        ret = loss
    elif reduction == "mean":
        ret = loss.mean()
    elif reduction == "sum":
        ret = loss.sum()
    else:
        ret = pred
        raise ValueError(reduction + " is not valid")
    return ret