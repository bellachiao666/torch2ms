import mindspore as ms
import mindspore.nn as msnn
import mindspore.ops as msops
import mindspore.mint as mint
from mindspore.mint import nn, ops
from multiprocessing import Value


class SharedCount:
    def __init__(self, epoch: int = 0):
        self.shared_epoch = Value('i', epoch)

    @property
    def value(self):
        return self.shared_epoch.value

    @value.setter
    def value(self, epoch):
        self.shared_epoch.value = epoch
