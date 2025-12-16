import mindspore as ms
import mindspore.nn as msnn
import mindspore.ops as msops
import mindspore.mint as mint
from mindspore.mint import nn, ops
""" Activations (memory-efficient w/ custom autograd)

A collection of activations fn and modules with a common interface so that they can
easily be swapped. All have an `inplace` arg even if not used.

These activations are not compatible with jit scripting or ONNX export of the model, please use
basic versions of the activations.

Hacked together by / Copyright 2020 Ross Wightman
"""

# import torch
# from torch import nn as nn


def swish_fwd(x):
    return x.mul(mint.sigmoid(x))


def swish_bwd(x, grad_output):
    x_sigmoid = mint.sigmoid(x)
    return grad_output * (x_sigmoid * (1 + x * (1 - x_sigmoid)))


# 'torch.autograd.Function' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
class SwishAutoFn(torch.autograd.Function):
    """ optimised Swish w/ memory-efficient checkpoint
    Inspired by conversation btw Jeremy Howard & Adam Pazske
    https://twitter.com/jeremyphoward/status/1188251041835315200
    """
    @staticmethod
    def symbolic(g, x):
        return g.op("Mul", x, g.op("Sigmoid", x))

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return swish_fwd(x)

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]
        return swish_bwd(x, grad_output)


def swish_me(x, inplace=False):
    return SwishAutoFn.apply(x)


class SwishMe(msnn.Cell):
    def __init__(self, inplace: bool = False):
        super().__init__()

    def construct(self, x):
        return SwishAutoFn.apply(x)


def mish_fwd(x):
    return x.mul(mint.tanh(nn.functional.softplus(x)))


def mish_bwd(x, grad_output):
    x_sigmoid = mint.sigmoid(x)
    x_tanh_sp = ms.Tensor.tanh()
    return grad_output.mul(x_tanh_sp + x * x_sigmoid * (1 - x_tanh_sp * x_tanh_sp))


# 'torch.autograd.Function' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
class MishAutoFn(torch.autograd.Function):
    """ Mish: A Self Regularized Non-Monotonic Neural Activation Function - https://arxiv.org/abs/1908.08681
    A memory efficient variant of Mish
    """
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return mish_fwd(x)

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]
        return mish_bwd(x, grad_output)


def mish_me(x, inplace=False):
    return MishAutoFn.apply(x)


class MishMe(msnn.Cell):
    def __init__(self, inplace: bool = False):
        super().__init__()

    def construct(self, x):
        return MishAutoFn.apply(x)


def hard_sigmoid_fwd(x, inplace: bool = False):
    return (x + 3).clamp(min=0, max=6).div(6.)


def hard_sigmoid_bwd(x, grad_output):
    m = mint.ones_like(x) * ((x >= -3.) & (x <= 3.)) / 6.
    return grad_output * m


# 'torch.autograd.Function' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
class HardSigmoidAutoFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return hard_sigmoid_fwd(x)

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]
        return hard_sigmoid_bwd(x, grad_output)


def hard_sigmoid_me(x, inplace: bool = False):
    return HardSigmoidAutoFn.apply(x)


class HardSigmoidMe(msnn.Cell):
    def __init__(self, inplace: bool = False):
        super().__init__()

    def construct(self, x):
        return HardSigmoidAutoFn.apply(x)


def hard_swish_fwd(x):
    return x * (x + 3).clamp(min=0, max=6).div(6.)


def hard_swish_bwd(x, grad_output):
    m = mint.ones_like(x) * (x >= 3.)
    m = mint.where((x >= -3.) & (x <= 3.), x / 3. + .5, m)
    return grad_output * m


# 'torch.autograd.Function' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
class HardSwishAutoFn(torch.autograd.Function):
    """A memory efficient HardSwish activation"""
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return hard_swish_fwd(x)

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]
        return hard_swish_bwd(x, grad_output)

    @staticmethod
    def symbolic(g, self):
        input = g.op("Add", self, g.op('Constant', value_t=ms.Tensor(3, dtype = ms.float)))  # 'torch.tensor':默认参数名不一致(position 0): PyTorch=data, MindSpore=input_data;
        hardtanh_ = g.op("Clip", input, g.op('Constant', value_t=ms.Tensor(0, dtype = ms.float)), g.op('Constant', value_t=ms.Tensor(6, dtype = ms.float)))  # 'torch.tensor':默认参数名不一致(position 0): PyTorch=data, MindSpore=input_data;
        hardtanh_ = g.op("Div", hardtanh_, g.op('Constant', value_t=ms.Tensor(6, dtype = ms.float)))  # 'torch.tensor':默认参数名不一致(position 0): PyTorch=data, MindSpore=input_data;
        return g.op("Mul", self, hardtanh_)


def hard_swish_me(x, inplace=False):
    return HardSwishAutoFn.apply(x)


class HardSwishMe(msnn.Cell):
    def __init__(self, inplace: bool = False):
        super().__init__()

    def construct(self, x):
        return HardSwishAutoFn.apply(x)


def hard_mish_fwd(x):
    return 0.5 * x * (x + 2).clamp(min=0, max=2)


def hard_mish_bwd(x, grad_output):
    m = mint.ones_like(x) * (x >= -2.)
    m = mint.where((x >= -2.) & (x <= 0.), x + 1., m)
    return grad_output * m


# 'torch.autograd.Function' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
class HardMishAutoFn(torch.autograd.Function):
    """ A memory efficient variant of Hard Mish
    Experimental, based on notes by Mish author Diganta Misra at
      https://github.com/digantamisra98/H-Mish/blob/0da20d4bc58e696b6803f2523c58d3c8a82782d0/README.md
    """
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return hard_mish_fwd(x)

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]
        return hard_mish_bwd(x, grad_output)


def hard_mish_me(x, inplace: bool = False):
    return HardMishAutoFn.apply(x)


class HardMishMe(msnn.Cell):
    def __init__(self, inplace: bool = False):
        super().__init__()

    def construct(self, x):
        return HardMishAutoFn.apply(x)



