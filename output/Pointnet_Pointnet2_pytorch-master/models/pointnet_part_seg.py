import mindspore as ms
import mindspore.nn as msnn
import mindspore.ops as msops
import mindspore.mint as mint
from mindspore.mint import nn, ops
# import torch.nn as nn
from pointnet_utils import STN3d, STNkd, feature_transform_reguliarzer


class get_model(msnn.Cell):
    def __init__(self, part_num=50, normal_channel=True):
        super(get_model, self).__init__()
        if normal_channel:
            channel = 6
        else:
            channel = 3
        self.part_num = part_num
        self.stn = STN3d(channel)
        self.conv1 = nn.Conv1d(channel, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 128, 1)
        self.conv4 = nn.Conv1d(128, 512, 1)
        self.conv5 = nn.Conv1d(512, 2048, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(2048)
        self.fstn = STNkd(k=128)
        self.convs1 = nn.Conv1d(4944, 256, 1)
        self.convs2 = nn.Conv1d(256, 256, 1)
        self.convs3 = nn.Conv1d(256, 128, 1)
        self.convs4 = nn.Conv1d(128, part_num, 1)
        self.bns1 = nn.BatchNorm1d(256)
        self.bns2 = nn.BatchNorm1d(256)
        self.bns3 = nn.BatchNorm1d(128)

    def construct(self, point_cloud, label):
        B, D, N = point_cloud.size()
        trans = self.stn(point_cloud)
        point_cloud = point_cloud.transpose(2, 1)
        if D > 3:
            point_cloud, feature = point_cloud.split(3, dim=2)
        point_cloud = mint.bmm(point_cloud, trans)
        if D > 3:
            point_cloud = mint.cat([point_cloud, feature], dim=2)

        point_cloud = point_cloud.transpose(2, 1)

        out1 = nn.functional.relu(self.bn1(self.conv1(point_cloud)))
        out2 = nn.functional.relu(self.bn2(self.conv2(out1)))
        out3 = nn.functional.relu(self.bn3(self.conv3(out2)))

        trans_feat = self.fstn(out3)
        x = out3.transpose(2, 1)
        net_transformed = mint.bmm(x, trans_feat)
        net_transformed = net_transformed.transpose(2, 1)

        out4 = nn.functional.relu(self.bn4(self.conv4(net_transformed)))
        out5 = self.bn5(self.conv5(out4))
        out_max = mint.max(out5, 2, keepdim=True)[0]
        out_max = out_max.view(-1, 2048)

        out_max = mint.cat([out_max,label.squeeze(1)],1)
        expand = out_max.view(-1, 2048+16, 1).repeat(1, 1, N)
        concat = mint.cat([expand, out1, out2, out3, out4, out5], 1)
        net = nn.functional.relu(self.bns1(self.convs1(concat)))
        net = nn.functional.relu(self.bns2(self.convs2(net)))
        net = nn.functional.relu(self.bns3(self.convs3(net)))
        net = self.convs4(net)
        net = net.transpose(2, 1).contiguous()
        net = mint.special.log_softmax(net.view(-1, self.part_num), dim=-1)
        net = net.view(B, N, self.part_num) # [B, N, 50]

        return net, trans_feat


class get_loss(msnn.Cell):
    def __init__(self, mat_diff_loss_scale=0.001):
        super(get_loss, self).__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale

    def construct(self, pred, target, trans_feat):
        loss = nn.functional.nll_loss(pred, target)
        mat_diff_loss = feature_transform_reguliarzer(trans_feat)
        total_loss = loss + mat_diff_loss * self.mat_diff_loss_scale
        return total_loss