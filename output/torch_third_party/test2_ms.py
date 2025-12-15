import mindspore as ms
import mindspore.nn as msnn
import mindspore.ops as msops
import mindspore.mint as mint
from mindspore.mint import nn, ops
# import torch.nn as nn


# 模拟复杂命名空间，测试前缀匹配的健壮性
class Wrapper:
    class SubModule:
        Conv2d = nn.Conv2d

myconv = Wrapper.SubModule.Conv2d


class MintLikeModel(msnn.Cell):
    def __init__(self, in_channels, num_classes, **kwargs):
        super().__init__()

        # 基础卷积
        self.stem = msnn.SequentialCell(
            nn.Conv2d(in_channels, 32, kernel_size = 3, padding = 1, bias = False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )  # 'torch.nn.ReLU':没有对应的mindspore参数 'inplace' (position 0);

        # 测试 mint.nn 前缀，你的转换器需要把这些都转换掉
        # 包含关键字缺省、参数顺序、字段不一致等情况
        self.block = msnn.SequentialCell(
            nn.Conv2d(
                32, 64, kernel_size = 3, stride = 1, padding = 1, bias = nn.Linear(1, 1)),

            msnn.SequentialCell(
                nn.ReLU(),
                nn.Conv2d(
                    in_channels = 64, out_channels = 64, kernel_size = 3, padding = 1, bias = False),
            ),

            # 使用 wrapper 前缀调用
            myconv(
                64, 128, kernel_size = 3, padding = 1, bias = False),
        )  # 'torch.nn.ReLU':没有对应的mindspore参数 'inplace' (position 0);

        # 全连接部分
        self.classifier = msnn.SequentialCell(
            nn.Linear(128 * 7 * 7, num_classes),
            nn.ReLU(),
            nn.Linear(num_classes, num_classes)
        )  # 'torch.nn.ReLU':没有对应的mindspore参数 'inplace' (position 0);


    def forward(self, x):

        # 嵌套调用 + 多个连续的 API 使用
        x = self.stem(x)
        x = self.block(x)

        # 测试多行函数调用
        x = nn.functional.adaptive_avg_pool2d(
            x, (7, 7))

        # 测试 view/reshape 不应被转换器破坏
        x = x.view(x.size(0), -1)

        # 测试 classifier
        x = self.classifier(x)

        # 测试 F.xxx 不应误伤
        x = nn.functional.softmax(x, dim = 1)

        return x
