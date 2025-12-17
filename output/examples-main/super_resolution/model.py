import mindspore as ms
import mindspore.nn as msnn
import mindspore.ops as msops
import mindspore.mint as mint
from mindspore.mint import nn, ops
# import torch.nn as nn
# import torch.nn.init as init


class Net(msnn.Cell):
    def __init__(self, upscale_factor):
        super(Net, self).__init__()

        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(1, 64, (5, 5), (1, 1), (2, 2))
        self.conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
        self.conv4 = nn.Conv2d(32, upscale_factor ** 2, (3, 3), (1, 1), (1, 1))
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)  # 'torch.nn.PixelShuffle' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;

        self._initialize_weights()

    def construct(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.pixel_shuffle(self.conv4(x))
        return x

    def _initialize_weights(self):
        init.orthogonal_(self.conv1.weight, init.calculate_gain('relu'))  # 'torch.nn.init.calculate_gain' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;; 'torch.nn.init.orthogonal_' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
        init.orthogonal_(self.conv2.weight, init.calculate_gain('relu'))  # 'torch.nn.init.calculate_gain' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;; 'torch.nn.init.orthogonal_' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
        init.orthogonal_(self.conv3.weight, init.calculate_gain('relu'))  # 'torch.nn.init.calculate_gain' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;; 'torch.nn.init.orthogonal_' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
        init.orthogonal_(self.conv4.weight)  # 'torch.nn.init.orthogonal_' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
