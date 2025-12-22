import mindspore as ms
import mindspore.nn as msnn
import mindspore.ops as msops
import mindspore.mint as mint
from mindspore.mint import nn, ops
dependencies = ['torch']
import timm
globals().update(timm.models._registry._model_entrypoints)
