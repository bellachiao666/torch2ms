import mindspore as ms
import mindspore.nn as msnn
import mindspore.ops as msops
import mindspore.mint as mint
from mindspore.mint import nn, ops
try:
    # from torch import _assert
except ImportError:
    def _assert(condition: bool, message: str):
        assert condition, message


def _float_to_int(x: float) -> int:
    """
    Symbolic tracing helper to substitute for inbuilt `int`.
    Hint: Inbuilt `int` can't accept an argument of type `Proxy`
    """
    return int(x)
