import mindspore as ms
import mindspore.nn as msnn
import mindspore.ops as msops
import mindspore.mint as mint
from mindspore.mint import nn, ops
from abc import abstractmethod


class Reader:
    def __init__(self):
        pass

    @abstractmethod
    def _filename(self, index, basename=False, absolute=False):
        pass

    def filename(self, index, basename=False, absolute=False):
        return self._filename(index, basename=basename, absolute=absolute)

    def filenames(self, basename=False, absolute=False):
        return [self._filename(index, basename=basename, absolute=absolute) for index in range(len(self))]

