import mindspore as ms
import mindspore.nn as msnn
import mindspore.ops as msops
import mindspore.mint as mint
from mindspore.mint import nn, ops
""" Linear layer (alternate definition)
"""
# import torch
# from torch import nn as nn


class Linear(nn.Linear):
    r"""Applies a linear transformation to the incoming data: :math:`y = xA^T + b`

    Wraps torch.nn.Linear to support AMP + torchscript usage by manually casting
    weight & bias to input.dtype to work around an issue w/ torch.addmm in this use case.
    """
    def forward(self, input: ms.Tensor) -> ms.Tensor:
        if torch.jit.is_scripting():
            bias = self.bias.to(dtype=input.dtype) if self.bias is not None else None
            return nn.functional.linear(input, self.weight.to(dtype=input.dtype), bias = bias)
        else:
            return nn.functional.linear(input, self.weight, self.bias)
