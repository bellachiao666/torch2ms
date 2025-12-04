import torch
import torch.nn as nn
import torch.nn.functional as F


# 模拟复杂命名空间，测试前缀匹配的健壮性
class Wrapper:
    class SubModule:
        Conv2d = nn.Conv2d

myconv = Wrapper.SubModule.Conv2d


class MintLikeModel(nn.Module):
    def __init__(self, in_channels, num_classes, **kwargs):
        super().__init__()

        # 基础卷积
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # 测试 mint.nn 前缀，你的转换器需要把这些都转换掉
        # 包含关键字缺省、参数顺序、字段不一致等情况
        self.block = nn.Sequential(
            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=nn.Linear(1, 1)  # 参数中嵌套 API
            ),

            nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    in_channels=64,
                    out_channels=64,
                    kernel_size=3,
                    padding=1,
                    bias=False
                ),
            ),

            # 使用 wrapper 前缀调用
            myconv(
                64,
                128,
                kernel_size=3,
                padding=1,
                bias=False
            ),
        )

        # 全连接部分
        self.classifier = nn.Sequential(
            nn.Linear(128 * 7 * 7, num_classes),
            nn.ReLU(inplace=True),
            nn.Linear(num_classes, num_classes)
        )


    def forward(self, x):

        # 嵌套调用 + 多个连续的 API 使用
        x = self.stem(x)
        x = self.block(x)

        # 测试多行函数调用
        x = F.adaptive_avg_pool2d(
            x,
            (7, 7)
        )

        # 测试 view/reshape 不应被转换器破坏
        x = x.view(x.size(0), -1)

        # 测试 classifier
        x = self.classifier(x)

        # 测试 F.xxx 不应误伤
        x = F.softmax(x, dim=1)

        return x
