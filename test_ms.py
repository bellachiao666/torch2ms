import torch
import torch.nn as nn
import mindspore as ms  # 让转换器识别 ms.nn 前缀

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = ms.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, has_bias=False, pad_mode="zeros")  # 默认值不一致: pad_mode (PyTorch=zeros, MindSpore=same)
        self.bn = ms.nn.BatchNorm2d(num_features=64, momentum=0.1, use_batch_statistics=True)  # 默认值不一致: momentum (PyTorch=0.1, MindSpore=0.9); 默认值不一致: use_batch_statistics (PyTorch=True, MindSpore=None)
        self.fc = ms.nn.Dense(in_channels=128, out_channels=10)
        self.relu = ms.nn.ReLU()  # 没有对应的mindspore参数 'inplace'

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x