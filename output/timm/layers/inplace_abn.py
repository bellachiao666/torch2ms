import mindspore as ms
import mindspore.nn as msnn
import mindspore.ops as msops
import mindspore.mint as mint
from mindspore.mint import nn, ops
# from torch import nn as nn

try:
    from inplace_abn.functions import inplace_abn, inplace_abn_sync
    has_iabn = True
except ImportError:
    has_iabn = False

    def inplace_abn(x, weight, bias, running_mean, running_var,
                    training=True, momentum=0.1, eps=1e-05, activation="leaky_relu", activation_param=0.01):
        raise ImportError(
            "Please install InplaceABN:'pip install git+https://github.com/mapillary/inplace_abn.git@v1.0.12'")

    def inplace_abn_sync(**kwargs):
        inplace_abn(**kwargs)

from ._fx import register_notrace_module


@register_notrace_module
class InplaceAbn(msnn.Cell):
    """Activated Batch Normalization

    This gathers a BatchNorm and an activation function in a single module

    Parameters
    ----------
    num_features : int
        Number of feature channels in the input and output.
    eps : float
        Small constant to prevent numerical issues.
    momentum : float
        Momentum factor applied to compute running statistics.
    affine : bool
        If `True` apply learned scale and shift transformation after normalization.
    act_layer : str or nn.Module type
        Name or type of the activation functions, one of: `leaky_relu`, `elu`
    act_param : float
        Negative slope for the `leaky_relu` activation.
    """

    def __init__(
            self,
            num_features,
            eps=1e-5,
            momentum=0.1,
            affine=True,
            apply_act=True,
            act_layer="leaky_relu",
            act_param=0.01,
            drop_layer=None,
    ):
        super().__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps
        self.momentum = momentum
        if apply_act:
            if isinstance(act_layer, str):
                assert act_layer in ('leaky_relu', 'elu', 'identity', '')
                self.act_name = act_layer if act_layer else 'identity'
            else:
                # convert act layer passed as type to string
                if act_layer == nn.ELU:
                    self.act_name = 'elu'
                elif act_layer == nn.LeakyReLU:
                    self.act_name = 'leaky_relu'
                elif act_layer is None or act_layer == msnn.Identity:
                    self.act_name = 'identity'
                else:
                    assert False, f'Invalid act layer {act_layer.__name__} for IABN'
        else:
            self.act_name = 'identity'
        self.act_param = act_param
        if self.affine:
            self.weight = ms.Parameter(mint.ones(num_features))
            self.bias = ms.Parameter(mint.zeros(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.register_buffer('running_mean', mint.zeros(num_features))
        self.register_buffer('running_var', mint.ones(num_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.running_mean, 0)  # 'torch.nn.init.constant_' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
        nn.init.constant_(self.running_var, 1)  # 'torch.nn.init.constant_' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
        if self.affine:
            nn.init.constant_(self.weight, 1)  # 'torch.nn.init.constant_' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
            nn.init.constant_(self.bias, 0)  # 'torch.nn.init.constant_' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;

    def construct(self, x):
        output = inplace_abn(
            x, self.weight, self.bias, self.running_mean, self.running_var,
            self.training, self.momentum, self.eps, self.act_name, self.act_param)
        if isinstance(output, tuple):
            output = output[0]
        return output
