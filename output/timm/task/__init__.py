import mindspore as ms
import mindspore.nn as msnn
import mindspore.ops as msops
import mindspore.mint as mint
from mindspore.mint import nn, ops
"""Training task abstractions for timm.

This module provides task-based abstractions for training loops where each task
encapsulates both the forward pass and loss computation, returning a dictionary
with loss components and outputs for logging.
"""
from .task import TrainingTask
from .classification import ClassificationTask
from .distillation import DistillationTeacher, LogitDistillationTask, FeatureDistillationTask

__all__ = [
    'TrainingTask',
    'ClassificationTask',
    'DistillationTeacher',
    'LogitDistillationTask',
    'FeatureDistillationTask',
]
