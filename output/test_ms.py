import torch
import torch.nn as nn
import mindspore as ms  # 让转换器识别 ms.nn 前缀

layer = ms.nn.Sequential(
    ms.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, has_bias=True, pad_mode="zeros"), # 默认值不一致: has_bias (PyTorch=True, MindSpore=False); 默认值不一致: pad_mode (PyTorch=zeros, MindSpore=same)
    ms.nn.Sequential(
        ms.nn.ReLU(), # 没有对应的mindspore参数 'inplace'
        ms.nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            has_bias=ms.nn.Dense(3, 3)
        )
    )
)