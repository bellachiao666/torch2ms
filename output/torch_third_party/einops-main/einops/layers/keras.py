import mindspore as ms
import mindspore.nn as msnn
import mindspore.ops as msops
import mindspore.mint as mint
from mindspore.mint import nn, ops
__author__ = "Alex Rogozhnikov"

from einops.layers.tensorflow import EinMix, Rearrange, Reduce

keras_custom_objects = {
    Rearrange.__name__: Rearrange,
    Reduce.__name__: Reduce,
    EinMix.__name__: EinMix,
}
