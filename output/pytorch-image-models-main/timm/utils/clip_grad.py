import mindspore as ms
import mindspore.nn as msnn
import mindspore.ops as msops
import mindspore.mint as mint
from mindspore.mint import nn, ops
# import torch

from timm.utils.agc import adaptive_clip_grad


def dispatch_clip_grad(parameters, value: float, mode: str = 'norm', norm_type: float = 2.0):
    """ Dispatch to gradient clipping method

    Args:
        parameters (Iterable): model parameters to clip
        value (float): clipping value/factor/norm, mode dependant
        mode (str): clipping mode, one of 'norm', 'value', 'agc'
        norm_type (float): p-norm, default 2.0
    """
    if mode == 'norm':
        torch.nn.utils.clip_grad_norm_(parameters, value, norm_type=norm_type)  # 'torch.nn.utils.clip_grad_norm_' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
    elif mode == 'value':
        torch.nn.utils.clip_grad_value_(parameters, value)  # 'torch.nn.utils.clip_grad_value_' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
    elif mode == 'agc':
        adaptive_clip_grad(parameters, value, norm_type=norm_type)
    else:
        assert False, f"Unknown clip mode ({mode})."

