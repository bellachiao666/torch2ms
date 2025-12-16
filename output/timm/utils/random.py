import mindspore as ms
import mindspore.nn as msnn
import mindspore.ops as msops
import mindspore.mint as mint
from mindspore.mint import nn, ops
import random
import numpy as np
# import torch


def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)  # 'torch.manual_seed' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
    np.random.seed(seed + rank)
    random.seed(seed + rank)
