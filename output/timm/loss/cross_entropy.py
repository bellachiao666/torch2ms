import mindspore as ms
import mindspore.nn as msnn
import mindspore.ops as msops
import mindspore.mint as mint
from mindspore.mint import nn, ops
""" Cross Entropy w/ smoothing or soft targets

Hacked together by / Copyright 2021 Ross Wightman
"""


class LabelSmoothingCrossEntropy(msnn.Cell):
    """ NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

    def construct(self, x: ms.Tensor, target: ms.Tensor) -> ms.Tensor:
        logprobs = mint.special.log_softmax(x, dim = -1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class SoftTargetCrossEntropy(msnn.Cell):

    def __init__(self):
        super(SoftTargetCrossEntropy, self).__init__()

    def construct(self, x: ms.Tensor, target: ms.Tensor) -> ms.Tensor:
        loss = mint.sum(-target * mint.special.log_softmax(x, dim = -1))
        return loss.mean()
