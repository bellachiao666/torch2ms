import mindspore as ms
import mindspore.nn as msnn
import mindspore.ops as msops
import mindspore.mint as mint
from mindspore.mint import nn, ops
# import torch
# import torch.nn as nn
# from torch.autograd import Variable
import numpy as np


class STN3d(msnn.Cell):
    def __init__(self, channel):
        super(STN3d, self).__init__()
        self.conv1 = nn.Conv1d(channel, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def construct(self, x):
        batchsize = x.size()[0]
        x = nn.functional.relu(self.bn1(self.conv1(x)))
        x = nn.functional.relu(self.bn2(self.conv2(x)))
        x = nn.functional.relu(self.bn3(self.conv3(x)))
        x = mint.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = nn.functional.relu(self.bn4(self.fc1(x)))
        x = nn.functional.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32))).view(1, 9).repeat(
            batchsize, 1)  # 'torch.from_numpy' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;; 'torch.autograd.Variable' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;; 'torch.autograd.Variable.view' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;; 'torch.autograd.Variable.view.repeat' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class STNkd(msnn.Cell):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def construct(self, x):
        batchsize = x.size()[0]
        x = nn.functional.relu(self.bn1(self.conv1(x)))
        x = nn.functional.relu(self.bn2(self.conv2(x)))
        x = nn.functional.relu(self.bn3(self.conv3(x)))
        x = mint.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = nn.functional.relu(self.bn4(self.fc1(x)))
        x = nn.functional.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1, self.k * self.k).repeat(
            batchsize, 1)  # 'torch.from_numpy' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;; 'torch.autograd.Variable' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;; 'torch.autograd.Variable.view' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;; 'torch.autograd.Variable.view.repeat' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


class PointNetEncoder(msnn.Cell):
    def __init__(self, global_feat=True, feature_transform=False, channel=3):
        super(PointNetEncoder, self).__init__()
        self.stn = STN3d(channel)
        self.conv1 = nn.Conv1d(channel, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def construct(self, x):
        B, D, N = x.size()
        trans = self.stn(x)
        x = x.transpose(2, 1)
        if D > 3:
            feature = x[:, :, 3:]
            x = x[:, :, :3]
        x = mint.bmm(x, trans)
        if D > 3:
            x = mint.cat([x, feature], dim=2)
        x = x.transpose(2, 1)
        x = nn.functional.relu(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = mint.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        pointfeat = x
        x = nn.functional.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = mint.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, N)
            return mint.cat([x, pointfeat], 1), trans, trans_feat


def feature_transform_reguliarzer(trans):
    d = trans.size()[1]
    I = mint.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = mint.mean(mint.norm(mint.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))
    return loss
