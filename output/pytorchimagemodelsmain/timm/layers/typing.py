import mindspore as ms
import mindspore.nn as msnn
import mindspore.ops as msops
import mindspore.mint as mint
from mindspore.mint import nn, ops
from contextlib import nullcontext
from functools import wraps
from typing import Callable, Optional, Tuple, Type, TypeVar, Union, overload, ContextManager

# 轻量 torch stub
class _TorchStub:
    def __init__(self):
        self.compiler = type("C", (), {"disable": None})()


torch = _TorchStub()

__all__ = ["LayerType", "PadType", "nullwrap", "disable_compiler"]


LayerType = Union[str, Callable, Type[msnn.Cell]]
PadType = Union[str, int, Tuple[int, int]]

F = TypeVar("F", bound=Callable[..., object])


@overload
def nullwrap(fn: F) -> F: ...  # decorator form

@overload
def nullwrap(fn: None = ...) -> ContextManager: ...  # context‑manager form

def nullwrap(fn: Optional[F] = None):
    # as a context manager
    if fn is None:
        return nullcontext()  # `with nullwrap():`

    # as a decorator
    @wraps(fn)
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)  # 存在 *args/**kwargs，未转换，需手动确认参数映射;
    return wrapper  # `@nullwrap`


disable_compiler = getattr(getattr(torch, "compiler", None), "disable", None) or nullwrap
