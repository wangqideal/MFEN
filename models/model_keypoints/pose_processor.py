import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from .config import cfg as pose_config
from .gaussian_blur import GaussianBlur


class HeatmapProcessor(nn.Module):
    """post process of the heatmap, group and normalize"""

    def __init__(self, normalize_heatmap=False, group_mode="sum", gaussion_smooth=None, norm_scale=1.0):
        super(HeatmapProcessor, self).__init__()
        self.num_joints = pose_config.MODEL.NUM_JOINTS
        self.groups = pose_config.MODEL.JOINTS_GROUPS
        self.gaussion_smooth = gaussion_smooth
        self.norm_scale = norm_scale
        print('^'*30)
        print(self.gaussion_smooth)
        assert group_mode in ['sum', 'max'], "only support sum or max"
        self.group_mode = group_mode
        print("groupmod", self.group_mode)
        self.normalize_heatmap = normalize_heatmap
        if self.normalize_heatmap:
            print("normalize scoremap")
        else:
            print("no normalize scoremap")
        if self.gaussion_smooth:
            kernel, sigma = self.gaussion_smooth
            self.gaussion_blur = GaussianBlur(kernel, sigma)
            print("gaussian blur:", kernel, sigma)
        else:
            self.gaussion_blur = None
            print("no gaussian blur")

    def forward(self, x):
        n, c, h, w = x.shape
        x = F.interpolate(x, [16, 8], mode='bilinear', align_corners=False)
        n, c, h, w = x.shape

        if not self.training:
            # if in eval phase, we calculate the max value and its position of each channel of heatmap
            n, c, h, w = x.shape

            x_reshaped = x.reshape((n, c, -1))
            idx = torch.argmax(x_reshaped, 2)
            max_response, _ = torch.max(x_reshaped, 2)

            idx = idx.reshape((n, c, 1))
            max_response = max_response.reshape((n, c))
            max_index = torch.empty((n, c, 2))
            max_index[:, :, 0] = idx[:, :, 0] % w  # column
            max_index[:, :, 1] = idx[:, :, 0] // w  # row
            '''
             #max_index :  每个通道的最大值坐标索引 ，那3通道[bs,ch,h,w],ch==3,做例子.
             比如：[2,1] 代表bs == 1中 ，通道0，的最大值在第index == 1行,第index==2 列
            tensor([[[2., 0.],
                    [1., 0.],
                    [0., 1.]],

                    [[2., 1.],
                    [3., 0.],
                    [1., 0.]]])
            '''

        if self.gaussion_blur:
            x = self.gaussion_blur(x)

        if self.group_mode == 'sum':
            heatmap = torch.sum(x[:, self.groups[0]], dim=1, keepdim=True)
            max_response_2 = torch.mean(max_response[:, self.groups[0]], dim=1, keepdim=True)

            for i in range(1, len(self.groups)):
                heatmapi = torch.sum(x[:, self.groups[i]], dim=1, keepdim=True)
                heatmap = torch.cat((heatmap, heatmapi), dim=1)

                max_response_i = torch.mean(max_response[:, self.groups[i]], dim=1, keepdim=True)
                max_response_2 = torch.cat((max_response_2, max_response_i), dim=1)


        elif self.group_mode == 'max':
            heatmap, _ = torch.max(x[:, self.groups[0]], dim=1, keepdim=True)
            max_response_2, _ = torch.max(max_response[:, self.groups[0]], dim=1, keepdim=True)

            for i in range(1, len(self.groups)):
                heatmapi, _ = torch.max(x[:, self.groups[i]], dim=1, keepdim=True)
                heatmap = torch.cat((heatmap, heatmapi), dim=1)

                max_response_i, _ = torch.max(max_response[:, self.groups[i]], dim=1, keepdim=True)
                max_response_2 = torch.cat((max_response_2, max_response_i), dim=1)

        if self.normalize_heatmap:
            heatmap = self.normalize(heatmap, norm_scale=self.norm_scale)

        if self.training:
            return heatmap
        else:
            return heatmap, max_response_2, max_index

    def normalize(self, in_tensor, norm_scale):
        n, c, h, w = in_tensor.shape
        in_tensor_reshape = in_tensor.reshape((n, c, -1))

        normalized_tensor = F.softmax(norm_scale * in_tensor_reshape, dim=2)
        normalized_tensor = normalized_tensor.reshape((n, c, h, w))

        return normalized_tensor


class HeatmapProcessor2(nn.Module):

    def __init__(self, normalize_heatmap=True, group_mode="sum", norm_scale=1.0):
        super(HeatmapProcessor2, self).__init__()
        self.num_joints = pose_config.MODEL.NUM_JOINTS
        self.groups = pose_config.MODEL.JOINTS_GROUPS

        self.group_mode = group_mode
        self.normalize_heatmap = normalize_heatmap
        self.norm_scale = norm_scale
        assert group_mode in ['sum', 'max'], "only support sum or max"

    def __call__(self, x):
        # 将 HRNet 输出的 热图  64, 32 or 64,48 or.... 他与encoder 输出的特征图时不一样的大小的
        # 这里进行resize 到 encoder 的 shape  [bs,ch,16, 8]
        x = F.interpolate(x, [16, 8], mode='bilinear', align_corners=False)
        n, c, h, w = x.shape

        x_reshaped = x.reshape((n, c, -1)) # 将所有的 h,w 展开，一个通道展开到一起，方便找到每个通道的最大值
        idx = torch.argmax(x_reshaped, 2)  # 返回每个channel 最大的index
        max_response, _ = torch.max(x_reshaped, 2) # 返回每个channel 最大的 概率值与 index

        idx = idx.reshape((n, c, 1))
        max_response = max_response.reshape((n, c))
        max_index = torch.empty((n, c, 2))
        max_index[:, :, 0] = idx[:, :, 0] % w  # column
        max_index[:, :, 1] = idx[:, :, 0] // w  # row

        if self.group_mode == 'sum':
            heatmap = torch.sum(x[:, self.groups[0]], dim=1, keepdim=True)
            max_response_2 = torch.mean(max_response[:, self.groups[0]], dim=1, keepdim=True)

            for i in range(1, len(self.groups)):
                heatmapi = torch.sum(x[:, self.groups[i]], dim=1, keepdim=True)
                heatmap = torch.cat((heatmap, heatmapi), dim=1)

                max_response_i = torch.mean(max_response[:, self.groups[i]], dim=1, keepdim=True)
                max_response_2 = torch.cat((max_response_2, max_response_i), dim=1)

        elif self.group_mode == 'max':
            heatmap, _ = torch.max(x[:, self.groups[0]], dim=1, keepdim=True)
            max_response_2, _ = torch.max(max_response[:, self.groups[0]], dim=1, keepdim=True)

            for i in range(1, len(self.groups)):
                heatmapi, _ = torch.max(x[:, self.groups[i]], dim=1, keepdim=True)
                heatmap = torch.cat((heatmap, heatmapi), dim=1)

                max_response_i, _ = torch.max(max_response[:, self.groups[i]], dim=1, keepdim=True)
                max_response_2 = torch.cat((max_response_2, max_response_i), dim=1)

        if self.normalize_heatmap:
            heatmap = self.normalize(heatmap, self.norm_scale)

        return heatmap, max_response_2, max_index

    def normalize(self, in_tensor, norm_scale):
        n, c, h, w = in_tensor.shape
        in_tensor_reshape = in_tensor.reshape((n, c, -1))

        normalized_tensor = F.softmax(norm_scale * in_tensor_reshape, dim=2)
        normalized_tensor = normalized_tensor.reshape((n, c, h, w))

        return normalized_tensor
