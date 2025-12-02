import torch
import torch.nn as nn
import mindspore as ms  # 让转换器识别 ms.nn 前缀

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, bias=False)
        self.bn = nn.BatchNorm2d(num_features=64)
        self.fc = nn.Linear(in_features=128, out_features=10)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x