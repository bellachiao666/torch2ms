import mindspore as ms
import mindspore.nn as msnn
import mindspore.ops as msops
import mindspore.mint as mint
from mindspore.mint import nn, ops
""" Activations

A collection of activations fn and modules with a common interface so that they can
easily be swapped. All have an `inplace` arg even if not used.

Hacked together by / Copyright 2020 Ross Wightman
"""
# from torch import nn as nn


def swish(x, inplace: bool = False):
    """Swish - Described in: https://arxiv.org/abs/1710.05941
    """
    return x.mul_(x.sigmoid()) if inplace else x.mul(x.sigmoid())


class Swish(msnn.Cell):
    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace

    def construct(self, x):
        return swish(x, self.inplace)


def mish(x, inplace: bool = False):
    """Mish: A Self Regularized Non-Monotonic Neural Activation Function - https://arxiv.org/abs/1908.08681
    NOTE: I don't have a working inplace variant
    """
    return x.mul(ms.Tensor.tanh())


class Mish(msnn.Cell):
    """Mish: A Self Regularized Non-Monotonic Neural Activation Function - https://arxiv.org/abs/1908.08681
    """
    def __init__(self, inplace: bool = False):
        super().__init__()

    def construct(self, x):
        return mish(x)


def sigmoid(x, inplace: bool = False):
    return x.sigmoid_() if inplace else x.sigmoid()


# PyTorch has this, but not with a consistent inplace argument interface
class Sigmoid(msnn.Cell):
    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace

    def construct(self, x):
        return x.sigmoid_() if self.inplace else x.sigmoid()


def tanh(x, inplace: bool = False):
    return x.tanh_() if inplace else x.tanh()


# PyTorch has this, but not with a consistent inplace argument interface
class Tanh(msnn.Cell):
    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace

    def construct(self, x):
        return x.tanh_() if self.inplace else x.tanh()


def hard_swish(x, inplace: bool = False):
    inner = nn.functional.relu6(x + 3.).div_(6.)  # 'torch.nn.functional.relu6.div_' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
    return x.mul_(inner) if inplace else x.mul(inner)


class HardSwish(msnn.Cell):
    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace

    def construct(self, x):
        return hard_swish(x, self.inplace)


def hard_sigmoid(x, inplace: bool = False):
    if inplace:
        return x.add_(3.).clamp_(0., 6.).div_(6.)
    else:
        return nn.functional.relu6(x + 3.) / 6.


class HardSigmoid(msnn.Cell):
    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace

    def construct(self, x):
        return hard_sigmoid(x, self.inplace)


def hard_mish(x, inplace: bool = False):
    """ Hard Mish
    Experimental, based on notes by Mish author Diganta Misra at
      https://github.com/digantamisra98/H-Mish/blob/0da20d4bc58e696b6803f2523c58d3c8a82782d0/README.md
    """
    if inplace:
        return x.mul_(0.5 * (x + 2).clamp(min=0, max=2))
    else:
        return 0.5 * x * (x + 2).clamp(min=0, max=2)


class HardMish(msnn.Cell):
    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace

    def construct(self, x):
        return hard_mish(x, self.inplace)


class PReLU(nn.PReLU):
    """Applies PReLU (w/ dummy inplace arg)
    """
    def __init__(self, num_parameters: int = 1, init: float = 0.25, inplace: bool = False) -> None:
        super().__init__(num_parameters=num_parameters, init=init)

    def forward(self, input: ms.Tensor) -> ms.Tensor:
        return nn.functional.prelu(input, self.weight)


def gelu(x: ms.Tensor, inplace: bool = False) -> ms.Tensor:
    return nn.functional.gelu(x)


class GELU(msnn.Cell):
    """Applies the Gaussian Error Linear Units function (w/ dummy inplace arg)
    """
    def __init__(self, inplace: bool = False):
        super().__init__()

    def construct(self, input: ms.Tensor) -> ms.Tensor:
        return nn.functional.gelu(input)


def gelu_tanh(x: ms.Tensor, inplace: bool = False) -> ms.Tensor:
    return nn.functional.gelu(x)


class GELUTanh(msnn.Cell):
    """Applies the Gaussian Error Linear Units function (w/ dummy inplace arg)
    """
    def __init__(self, inplace: bool = False):
        super().__init__()

    def construct(self, input: ms.Tensor) -> ms.Tensor:
        return nn.functional.gelu(input)


def quick_gelu(x: ms.Tensor, inplace: bool = False) -> ms.Tensor:
    return x * mint.sigmoid(1.702 * x)


class QuickGELU(msnn.Cell):
    """Applies the Gaussian Error Linear Units function (w/ dummy inplace arg)
    """
    def __init__(self, inplace: bool = False):
        super().__init__()

    def construct(self, input: ms.Tensor) -> ms.Tensor:
        return quick_gelu(input)
