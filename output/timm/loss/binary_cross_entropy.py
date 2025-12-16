import mindspore as ms
import mindspore.nn as msnn
import mindspore.ops as msops
import mindspore.mint as mint
from mindspore.mint import nn, ops
""" Binary Cross Entropy w/ a few extras

Hacked together by / Copyright 2021 Ross Wightman
"""
from typing import Optional, Union

# import torch
# import torch.nn as nn


class BinaryCrossEntropy(msnn.Cell):
    """ BCE with optional one-hot from dense targets, label smoothing, thresholding
    NOTE for experiments comparing CE to BCE /w label smoothing, may remove
    """
    def __init__(
            self,
            smoothing=0.1,
            target_threshold: Optional[float] = None,
            weight: Optional[torch.Tensor] = None,
            reduction: str = 'mean',
            sum_classes: bool = False,
            pos_weight: Optional[Union[torch.Tensor, float]] = None,
    ):
        super(BinaryCrossEntropy, self).__init__()
        assert 0. <= smoothing < 1.0
        if pos_weight is not None:
            if not isinstance(pos_weight, torch.Tensor):
                pos_weight = ms.Tensor(pos_weight)  # 'torch.tensor':默认参数名不一致(position 0): PyTorch=data, MindSpore=input_data;
        self.smoothing = smoothing
        self.target_threshold = target_threshold
        self.reduction = 'none' if sum_classes else reduction
        self.sum_classes = sum_classes
        self.register_buffer('weight', weight)
        self.register_buffer('pos_weight', pos_weight)

    def construct(self, x: ms.Tensor, target: ms.Tensor) -> ms.Tensor:
        batch_size = x.shape[0]
        assert batch_size == target.shape[0]

        if target.shape != x.shape:
            # NOTE currently assume smoothing or other label softening is applied upstream if targets are already sparse
            num_classes = x.shape[-1]
            # FIXME should off/on be different for smoothing w/ BCE? Other impl out there differ
            off_value = self.smoothing / num_classes
            on_value = 1. - self.smoothing + off_value
            target = target.long().view(-1, 1)
            target = mint.full(
                size = ((batch_size, num_classes), off_value), dtype = x.dtype).scatter_(1, target, on_value)  # 'torch.full':没有对应的mindspore参数 'device' (position 5);

        if self.target_threshold is not None:
            # Make target 0, or 1 if threshold set
            target = target.gt(self.target_threshold).to(dtype=target.dtype)

        loss = nn.functional.binary_cross_entropy_with_logits(
            x, target, self.weight, reduction = self.reduction, pos_weight = self.pos_weight)
        if self.sum_classes:
            loss = loss.sum(-1).mean()
        return loss
