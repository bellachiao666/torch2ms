import mindspore as ms
import mindspore.nn as msnn
import mindspore.ops as msops
import mindspore.mint as mint
from mindspore.mint import nn, ops
""" Activation Factory
Hacked together by / Copyright 2020 Ross Wightman
"""
import types
from typing import Callable, Optional, Type, Union

from .activations import *
from .activations_me import *
from .config import is_exportable, is_scriptable
from .typing import LayerType

# 轻量 torch / F stub，提供与检查兼容的属性
torch = types.SimpleNamespace(nn=types.SimpleNamespace(functional=types.SimpleNamespace()))
F = types.SimpleNamespace(
    silu=lambda x: x * ops.sigmoid(x),
    relu=lambda x: ops.relu(x),
    relu6=lambda x: ops.relu6(x) if hasattr(ops, "relu6") else ops.relu(x),
    leaky_relu=lambda x, negative_slope=0.01: ops.leaky_relu(x, negative_slope),
    elu=lambda x, alpha=1.0: ops.elu(x, alpha),
    celu=lambda x, alpha=1.0: ops.celu(x, alpha) if hasattr(ops, "celu") else x,
    selu=lambda x: ops.selu(x) if hasattr(ops, "selu") else x,
    hardsigmoid=lambda x: ops.hardsigmoid(x),
    hardswish=lambda x: ops.hardswish(x) if hasattr(ops, "hardswish") else x * ops.relu6(x + 3) / 6,
    mish=lambda x: x * ops.tanh(ops.softplus(x)) if hasattr(ops, "softplus") else x,
)
# 补充 nn 中缺失的激活类
_fallback_acts = {
    "ReLU": msnn.ReLU,
    "ReLU6": getattr(msnn, "ReLU6", msnn.ReLU),
    "LeakyReLU": getattr(msnn, "LeakyReLU", msnn.ReLU),
    "ELU": getattr(msnn, "ELU", msnn.ReLU),
    "CELU": getattr(msnn, "CELU", msnn.ReLU),
    "SELU": getattr(msnn, "SELU", msnn.ReLU),
    "SiLU": getattr(msnn, "SiLU", msnn.Sigmoid),
    "Mish": getattr(msnn, "Mish", msnn.SiLU),
}
for _name, _cls in _fallback_acts.items():
    if not hasattr(nn, _name):
        setattr(nn, _name, _cls)

# PyTorch has an optimized, native 'silu' (aka 'swish') operator as of PyTorch 1.7.
# Also hardsigmoid, hardswish, and soon mish. This code will use native version if present.
# Eventually, the custom SiLU, Mish, Hard*, layers will be removed and only native variants will be used.
_has_silu = 'silu' in dir(torch.nn.functional)
_has_hardswish = 'hardswish' in dir(torch.nn.functional)
_has_hardsigmoid = 'hardsigmoid' in dir(torch.nn.functional)
_has_mish = 'mish' in dir(torch.nn.functional)


_ACT_FN_DEFAULT = dict(
    silu=F.silu if _has_silu else swish,
    swish=F.silu if _has_silu else swish,
    mish=F.mish if _has_mish else mish,
    relu=F.relu,
    relu6=F.relu6,
    leaky_relu=F.leaky_relu,
    elu=F.elu,
    celu=F.celu,
    selu=F.selu,
    gelu=gelu,
    gelu_tanh=gelu_tanh,
    quick_gelu=quick_gelu,
    sigmoid=sigmoid,
    tanh=tanh,
    hard_sigmoid=F.hardsigmoid if _has_hardsigmoid else hard_sigmoid,
    hard_swish=F.hardswish if _has_hardswish else hard_swish,
    hard_mish=hard_mish,
)

_ACT_FN_ME = dict(
    silu=F.silu if _has_silu else swish_me,
    swish=F.silu if _has_silu else swish_me,
    mish=F.mish if _has_mish else mish_me,
    hard_sigmoid=F.hardsigmoid if _has_hardsigmoid else hard_sigmoid_me,
    hard_swish=F.hardswish if _has_hardswish else hard_swish_me,
    hard_mish=hard_mish_me,
)

_ACT_FNS = (_ACT_FN_ME, _ACT_FN_DEFAULT)
for a in _ACT_FNS:
    a.setdefault('hardsigmoid', a.get('hard_sigmoid'))
    a.setdefault('hardswish', a.get('hard_swish'))


_ACT_LAYER_DEFAULT = dict(
    silu=nn.SiLU if _has_silu else Swish,
    swish=nn.SiLU if _has_silu else Swish,
    mish=nn.Mish if _has_mish else Mish,
    relu=nn.ReLU,
    relu6=nn.ReLU6,
    leaky_relu=nn.LeakyReLU,
    elu=nn.ELU,
    prelu=PReLU,
    celu=nn.CELU,
    selu=nn.SELU,
    gelu=GELU,
    gelu_tanh=GELUTanh,
    quick_gelu=QuickGELU,
    sigmoid=Sigmoid,
    tanh=Tanh,
    hard_sigmoid=nn.Hardsigmoid if _has_hardsigmoid else HardSigmoid,
    hard_swish=nn.Hardswish if _has_hardswish else HardSwish,
    hard_mish=HardMish,
    identity=nn.Identity,
)

_ACT_LAYER_ME = dict(
    silu=nn.SiLU if _has_silu else SwishMe,
    swish=nn.SiLU if _has_silu else SwishMe,
    mish=nn.Mish if _has_mish else MishMe,
    hard_sigmoid=nn.Hardsigmoid if _has_hardsigmoid else HardSigmoidMe,
    hard_swish=nn.Hardswish if _has_hardswish else HardSwishMe,
    hard_mish=HardMishMe,
)

_ACT_LAYERS = (_ACT_LAYER_ME, _ACT_LAYER_DEFAULT)
for a in _ACT_LAYERS:
    a.setdefault('hardsigmoid', a.get('hard_sigmoid'))
    a.setdefault('hardswish', a.get('hard_swish'))


def get_act_fn(name: Optional[LayerType] = 'relu'):
    """ Activation Function Factory
    Fetching activation fns by name with this function allows export or torch script friendly
    functions to be returned dynamically based on current config.
    """
    if not name:
        return None
    if isinstance(name, Callable):
        return name
    name = name.lower()
    if not (is_exportable() or is_scriptable()):
        # If not exporting or scripting the model, first look for a memory-efficient version with
        # custom autograd, then fallback
        if name in _ACT_FN_ME:
            return _ACT_FN_ME[name]
    return _ACT_FN_DEFAULT[name]


def get_act_layer(name: Optional[LayerType] = 'relu'):
    """ Activation Layer Factory
    Fetching activation layers by name with this function allows export or torch script friendly
    functions to be returned dynamically based on current config.
    """
    if name is None:
        return None
    if not isinstance(name, str):
        # callable, module, etc
        return name
    if not name:
        return None
    name = name.lower()
    if not (is_exportable() or is_scriptable()):
        if name in _ACT_LAYER_ME:
            return _ACT_LAYER_ME[name]
    return _ACT_LAYER_DEFAULT[name]


def create_act_layer(
        name: Optional[LayerType],
        inplace: Optional[bool] = None,
        **kwargs
):
    act_layer = get_act_layer(name)
    if act_layer is None:
        return None
    if inplace is None:
        return act_layer(**kwargs)  # 存在 *args/**kwargs，未转换，需手动确认参数映射;
    try:
        return act_layer(inplace=inplace, **kwargs)  # 存在 *args/**kwargs，未转换，需手动确认参数映射;
    except TypeError:
        # recover if act layer doesn't have inplace arg
        return act_layer(**kwargs)  # 存在 *args/**kwargs，未转换，需手动确认参数映射;
