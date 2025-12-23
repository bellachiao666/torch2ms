import os
import random
import pickle
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, Iterable, List, Sequence, Tuple

import numpy as np
import mindspore as ms
import mindspore.nn as msnn
import mindspore.ops as ops
import mindspore.mint as mint
from mindspore.common.initializer import Constant, XavierNormal, initializer


def _as_tensor(value: Any, dtype=None) -> ms.Tensor:
    if isinstance(value, ms.Tensor):
        return value if dtype is None else value.astype(dtype)
    return ms.Tensor(value, dtype=dtype)


def _assign_tensor_data(tensor: ms.Tensor, value: ms.Tensor) -> ms.Tensor:
    if hasattr(tensor, "set_data"):
        tensor.set_data(value)
    elif hasattr(tensor, "assign_value"):
        tensor.assign_value(value)
    return tensor


def _tensor_cuda(self: ms.Tensor):
    setattr(self, "_is_cuda", True)
    return self


def _tensor_cpu(self: ms.Tensor):
    setattr(self, "_is_cuda", False)
    return self


def _tensor_is_cuda(self: ms.Tensor):
    return bool(getattr(self, "_is_cuda", False))


def _tensor_numpy(self: ms.Tensor):
    return self.asnumpy()


def _tensor_long(self: ms.Tensor):
    return self.astype(ms.int64)


def _tensor_float(self: ms.Tensor):
    return self.astype(ms.float32)


def _tensor_contiguous(self: ms.Tensor):
    return self


def _tensor_view(self: ms.Tensor, *shape):
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = tuple(shape[0])
    return ops.reshape(self, shape)


def _tensor_repeat(self: ms.Tensor, *sizes):
    if len(sizes) == 1 and isinstance(sizes[0], (tuple, list)):
        sizes = tuple(sizes[0])
    else:
        sizes = tuple(sizes)
    if len(sizes) > len(self.shape):
        pad = (1,) * (len(sizes) - len(self.shape))
        data = ops.reshape(self, pad + tuple(self.shape))
    else:
        data = self
    return ops.tile(data, sizes)


def _tensor_size(self: ms.Tensor, dim=None):
    return self.shape if dim is None else self.shape[dim]


def _tensor_eq(self: ms.Tensor, other):
    other_tensor = other.data if hasattr(other, "data") else other
    return ops.equal(self, other_tensor)


def _tensor_max(self: ms.Tensor, dim=None, keepdim: bool = False):
    if dim is None:
        return ops.max(self)
    values = ops.max(self, axis=dim, keep_dims=keepdim)
    indices = ops.argmax(self, axis=dim)
    return values, indices


ms.Tensor.cuda = _tensor_cuda  # type: ignore[attr-defined]
ms.Tensor.cpu = _tensor_cpu  # type: ignore[attr-defined]
ms.Tensor.numpy = _tensor_numpy  # type: ignore[attr-defined]
ms.Tensor.long = _tensor_long  # type: ignore[attr-defined]
ms.Tensor.float = _tensor_float  # type: ignore[attr-defined]
ms.Tensor.contiguous = _tensor_contiguous  # type: ignore[attr-defined]
ms.Tensor.view = _tensor_view  # type: ignore[attr-defined]
ms.Tensor.repeat = _tensor_repeat  # type: ignore[attr-defined]
ms.Tensor.size = _tensor_size  # type: ignore[attr-defined]
ms.Tensor.eq = _tensor_eq  # type: ignore[attr-defined]
ms.Tensor.max = _tensor_max  # type: ignore[attr-defined]
ms.Tensor.data = property(lambda self: self)  # type: ignore[attr-defined]
ms.Tensor.is_cuda = property(_tensor_is_cuda)  # type: ignore[attr-defined]


class Dataset:
    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


def _collate(items: List[Any]):
    first = items[0]
    if isinstance(first, tuple):
        transposed = list(zip(*items))
        return tuple(_collate(list(component)) for component in transposed)
    if isinstance(first, list):
        transposed = list(zip(*items))
        return [_collate(list(component)) for component in transposed]
    if isinstance(first, ms.Tensor):
        stacked = np.stack([x.asnumpy() for x in items], axis=0)
        return ms.Tensor(stacked)
    if isinstance(first, np.ndarray):
        return ms.Tensor(np.stack(items, axis=0))
    if np.isscalar(first):
        return ms.Tensor(np.array(items))
    try:
        return ms.Tensor(np.stack(items, axis=0))
    except Exception:
        return items


class DataLoader:
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 1,
        shuffle: bool = False,
        num_workers: int = 0,
        drop_last: bool = False,
        pin_memory: bool = False,
        worker_init_fn=None,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.drop_last = drop_last
        self.pin_memory = pin_memory
        self.worker_init_fn = worker_init_fn

    def __len__(self):
        length = len(self.dataset)
        if self.drop_last:
            return length // self.batch_size
        return (length + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            random.shuffle(indices)
        if self.worker_init_fn is not None:
            try:
                self.worker_init_fn(0)
            except Exception:
                pass
        batch: List[Any] = []
        for idx in indices:
            batch.append(self.dataset[idx])
            if len(batch) == self.batch_size:
                yield _collate(batch)
                batch = []
        if batch and not self.drop_last:
            yield _collate(batch)


def manual_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    ms.set_seed(seed)


class _Cuda:
    @staticmethod
    def manual_seed_all(seed: int):
        manual_seed(seed)


cuda = _Cuda()


def from_numpy(array: np.ndarray):
    return _as_tensor(np.asarray(array))


@contextmanager
def no_grad():
    yield


class _Init:
    @staticmethod
    def xavier_normal_(tensor: ms.Tensor):
        value = initializer(XavierNormal(), tensor.shape, tensor.dtype)
        return _assign_tensor_data(tensor, value)

    @staticmethod
    def constant_(tensor: ms.Tensor, val):
        value = initializer(Constant(val), tensor.shape, tensor.dtype)
        return _assign_tensor_data(tensor, value)


class _NN:
    init = _Init()


nn = _NN()


if hasattr(mint, "optim") and hasattr(mint.optim, "SGD"):
    _BaseSGD = mint.optim.SGD  # type: ignore[attr-defined]
else:
    _BaseSGD = msnn.SGD  # type: ignore[assignment]


class SGD(_BaseSGD):  # type: ignore[misc]
    def __init__(self, params: Iterable[ms.Tensor], lr: float = 0.01, momentum: float = 0.0, **kwargs):
        if _BaseSGD is msnn.SGD:
            super().__init__(params, learning_rate=lr, momentum=momentum, **kwargs)
        else:
            super().__init__(params, lr=lr, momentum=momentum, **kwargs)


class StepLR:
    def __init__(self, optimizer, step_size: int, gamma: float = 0.1):
        self.optimizer = optimizer
        self.step_size = step_size
        self.gamma = gamma
        self.last_epoch = 0

    def step(self):
        self.last_epoch += 1
        if self.last_epoch % self.step_size != 0:
            return
        if hasattr(self.optimizer, "param_groups"):
            for group in self.optimizer.param_groups:
                if "lr" in group:
                    group["lr"] *= self.gamma
        if hasattr(self.optimizer, "learning_rate"):
            try:
                self.optimizer.learning_rate = self.optimizer.learning_rate * self.gamma
            except Exception:
                pass


class _Optim:
    SGD = SGD
    lr_scheduler = SimpleNamespace(StepLR=StepLR)


optim = _Optim()


def save(obj: Any, path: str):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


utils = SimpleNamespace(data=SimpleNamespace(DataLoader=DataLoader, Dataset=Dataset))


# Legacy torch.autograd.Variable shim: simply returns a Tensor
def Variable(x):
    return _as_tensor(x)


__all__ = [
    "cuda",
    "from_numpy",
    "manual_seed",
    "no_grad",
    "optim",
    "nn",
    "save",
    "load",
    "utils",
    "Dataset",
    "DataLoader",
    "Variable",
]
