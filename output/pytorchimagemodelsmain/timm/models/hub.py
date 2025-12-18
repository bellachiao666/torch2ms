import mindspore as ms
import mindspore.nn as msnn
import mindspore.ops as msops
import mindspore.mint as mint
from mindspore.mint import nn, ops
from ._hub import *

import warnings
warnings.warn(f"Importing from {__name__} is deprecated, please import via timm.models", FutureWarning)
