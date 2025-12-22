import mindspore as ms
import mindspore.nn as msnn
import mindspore.ops as msops
import mindspore.mint as mint
from mindspore.mint import nn, ops
import pytest
# import torch.nn as nn

from timm.layers import create_act_layer, set_layer_config, get_act_layer, get_act_fn, Attention2d, MultiQueryAttentionV2

import importlib
import os

torch_backend = os.environ.get('TORCH_BACKEND')
if torch_backend is not None:
    importlib.import_module(torch_backend)
torch_device = os.environ.get('TORCH_DEVICE', 'cpu')

class MLP(msnn.Cell):
    def __init__(self, act_layer="relu", inplace=True):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(1000, 100)
        self.act = create_act_layer(act_layer, inplace=inplace)
        self.fc2 = nn.Linear(100, 10)

    def construct(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


def _run_act_layer_grad(act_type, inplace=True):
    x = mint.rand(10, 1000) * 10
    m = MLP(act_layer=act_type, inplace=inplace)

    def _run(x, act_layer=''):
        if act_layer:
            # replace act layer if set
            m.act = create_act_layer(act_layer, inplace=inplace)
        out = m(x)
        l = (out - 0).pow(2).sum()
        return l

    x = x.to(device=torch_device)
    m.to(device=torch_device)

    out_me = _run(x)

    with set_layer_config(scriptable=True):
        out_jit = _run(x, act_type)

    assert mint.isclose(out_jit, out_me)

    with set_layer_config(no_jit=True):
        out_basic = _run(x, act_type)

    assert mint.isclose(out_basic, out_jit)


def test_swish_grad():
    for _ in range(100):
        _run_act_layer_grad('swish')


def test_mish_grad():
    for _ in range(100):
        _run_act_layer_grad('mish')


def test_hard_sigmoid_grad():
    for _ in range(100):
        _run_act_layer_grad('hard_sigmoid', inplace=None)


def test_hard_swish_grad():
    for _ in range(100):
        _run_act_layer_grad('hard_swish')


def test_hard_mish_grad():
    for _ in range(100):
        _run_act_layer_grad('hard_mish')

def test_get_act_layer_empty_string():
    # Empty string should return None
    assert get_act_layer('') is None


def test_create_act_layer_inplace_error():
    class NoInplaceAct(msnn.Cell):
        def __init__(self):
            super().__init__()
        def construct(self, x):
            return x
    
    # Should recover when inplace arg causes TypeError
    layer = create_act_layer(NoInplaceAct, inplace=True)
    assert isinstance(layer, NoInplaceAct)


def test_create_act_layer_edge_cases():
    # Test None input
    assert create_act_layer(None) is None
    
    # Test TypeError handling for inplace
    class CustomAct(msnn.Cell):
        def __init__(self, **kwargs):
            super().__init__()
        def construct(self, x):
            return x
            
    result = create_act_layer(CustomAct, inplace=True)
    assert isinstance(result, CustomAct)


def test_get_act_fn_callable():
    def custom_act(x): 
        return x
    assert get_act_fn(custom_act) is custom_act


def test_get_act_fn_none():
    assert get_act_fn(None) is None
    assert get_act_fn('') is None


@pytest.mark.parametrize("dim", [128])
@pytest.mark.parametrize("dim_out", [128, 256])
@pytest.mark.parametrize("use_m", [True, False])
def test_mqa_v2(dim, dim_out, use_m):
    mqa = MultiQueryAttentionV2(dim, dim_out)
    
    x = mint.randn(1, dim, 32, 48)
    if use_m:
        m = mint.randn(1, dim, 16, 24)
    else:
        m = None
        
    y = mqa(x, m=m)
    
    assert (y.shape) == (1, dim_out, 32, 48)


@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("expand_first", [True, False])
@pytest.mark.parametrize("head_first", [True, False])
@pytest.mark.parametrize("attn_mask", [True, False])
def test_attn2d(bias, expand_first, head_first, attn_mask):
    x = mint.randn(1, 128, 32, 48)
    attn = Attention2d(
        128, 128, num_heads=4, bias=bias, expand_first=expand_first, head_first=head_first
    )
    
    if attn_mask:
        mask = mint.randint(0, 1, size=(32 * 48, 32 * 48), dtype=ms.float32)
    else:
        mask = None
    
    o1 = attn(x, mask)
    attn.fused_attn = False
    o2 = attn(x, mask)
    
    assert mint.allclose(o1, o2, atol=1e-5), f"{mint.abs(o1 - o2).max()}"
