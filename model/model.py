# This doc is used for construct few-shot sar model.
# The model is using metric-learning to classify the class of
# samples which is a pix in sar iamge(more like semantic segmentation).
# The model is working like:
# first, we have same labeled samples and their position.
# and then we can use these labeled samples' feature to metric
# other samples class.

import torch.nn as nn
import torch
import numpy as np
import math
# from torch.utils import data
# import numpy as np
# from PIL import Image
from .complex_module import Complex, \
    C_MaxPooling, C_conv2d, C_BatchNorm2d, \
    C_ReLU, complex_weight_init, C_Linear, C_BatchNorm, C_AvePooling, C_convtranspose2d

from collections import Counter


class ConvBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, padding=1, kernel_size=3, batch_norm=True,
                 is_complex=False):
        super().__init__()
        if is_complex:
            relu = C_ReLU
            conv2d = C_conv2d
            bn = C_BatchNorm
        else:
            relu = nn.ReLU
            conv2d = nn.Conv2d
            bn = nn.BatchNorm2d
        conv_relu = []
        conv_relu.append(conv2d(in_channels=in_channels, out_channels=middle_channels,
                                kernel_size=kernel_size, padding=padding, stride=1))
        if batch_norm:
            conv_relu.append(bn(middle_channels))
        conv_relu.append(relu())
        conv_relu.append(conv2d(in_channels=middle_channels, out_channels=out_channels,
                                kernel_size=3, padding=1, stride=1))
        if batch_norm:
            conv_relu.append(bn(out_channels))
        conv_relu.append(relu())
        self.conv_ReLU = nn.Sequential(*conv_relu)

    def forward(self, x):
        out = self.conv_ReLU(x)
        return out


class SarModel(torch.nn.Module):
    def __init__(self, in_channels, out_channels, is_complex, batch_norm=True):
        super(SarModel, self).__init__()
        # left model
        if is_complex:
            max_pooling = C_MaxPooling
            convTranspose2d = C_convtranspose2d
        else:
            max_pooling = nn.MaxPool2d
            convTranspose2d = nn.ConvTranspose2d
        self.left_conv_1 = ConvBlock(in_channels=in_channels, middle_channels=64, out_channels=64, padding=0,
                                     kernel_size=5,
                                     batch_norm=batch_norm, is_complex=is_complex)
        self.pool_1 = max_pooling(kernel_size=2, stride=2)

        self.left_conv_2 = ConvBlock(in_channels=64, middle_channels=128, out_channels=128, padding=1,
                                     batch_norm=batch_norm, is_complex=is_complex)
        self.pool_2 = max_pooling(kernel_size=2, stride=2)

        self.left_conv_3 = ConvBlock(in_channels=128, middle_channels=256, out_channels=256, padding=1,
                                     batch_norm=batch_norm, is_complex=is_complex)
        self.pool_3 = max_pooling(kernel_size=2, stride=2)

        self.left_conv_4 = ConvBlock(in_channels=256, middle_channels=512, out_channels=512, padding=1,
                                     batch_norm=batch_norm, is_complex=is_complex)
        self.pool_4 = max_pooling(kernel_size=2, stride=2)
        # padding = 1
        self.left_conv_5 = ConvBlock(in_channels=512, middle_channels=1024, out_channels=1024, padding=1,
                                     batch_norm=batch_norm, is_complex=is_complex)
        # 定义右半部分网络
        self.deconv_1 = convTranspose2d(in_channels=1024, out_channels=512, kernel_size=3, stride=2, padding=1,
                                        output_padding=1)
        self.right_conv_1 = ConvBlock(in_channels=1024, middle_channels=512, out_channels=512, batch_norm=batch_norm,
                                      is_complex=is_complex)

        self.deconv_2 = convTranspose2d(in_channels=512, out_channels=256, kernel_size=3, padding=1, stride=2,
                                        output_padding=1)
        self.right_conv_2 = ConvBlock(in_channels=512, middle_channels=256, out_channels=256, batch_norm=batch_norm,
                                      is_complex=is_complex)

        self.deconv_3 = convTranspose2d(in_channels=256, out_channels=128, kernel_size=3, padding=1, stride=2,
                                        output_padding=1)
        self.right_conv_3 = ConvBlock(in_channels=256, middle_channels=128, out_channels=128, batch_norm=batch_norm,
                                      is_complex=is_complex)

        self.deconv_4 = convTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, output_padding=1,
                                        padding=1)
        self.right_conv_4 = ConvBlock(in_channels=128, middle_channels=64, out_channels=64, batch_norm=batch_norm,
                                      is_complex=is_complex)
        # 最后是1x1的卷积，用于将通道数化为3
        self.right_conv_5 = nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # 1：进行编码过程
        feature_1 = self.left_conv_1(x)
        feature_1_pool = self.pool_1(feature_1)

        feature_2 = self.left_conv_2(feature_1_pool)
        feature_2_pool = self.pool_2(feature_2)

        feature_3 = self.left_conv_3(feature_2_pool)
        feature_3_pool = self.pool_3(feature_3)

        feature_4 = self.left_conv_4(feature_3_pool)
        feature_4_pool = self.pool_4(feature_4)

        feature_5 = self.left_conv_5(feature_4_pool)

        de_feature_1 = self.deconv_1(feature_5)  # N*512*H*W
        # 特征拼接
        temp = torch.cat((feature_4, de_feature_1), dim=1)
        de_feature_1_conv = self.right_conv_1(temp)  # N*512*H*W

        de_feature_2 = self.deconv_2(de_feature_1_conv)
        temp = torch.cat((feature_3, de_feature_2), dim=1)
        de_feature_2_conv = self.right_conv_2(temp)

        de_feature_3 = self.deconv_3(de_feature_2_conv)

        temp = torch.cat((feature_2, de_feature_3), dim=1)
        de_feature_3_conv = self.right_conv_3(temp)

        de_feature_4 = self.deconv_4(de_feature_3_conv)
        temp = torch.cat((feature_1, de_feature_4), dim=1)
        de_feature_4_conv = self.right_conv_4(temp)

        out = self.right_conv_5(de_feature_4_conv)  # 1*64*H*W
        if torch.isnan(out).sum():
            print(0)
        return out


class MetricModel(torch.nn.Module):
    def __init__(self, feature_dim, loss, is_complex=False):
        super(MetricModel, self).__init__()
        if is_complex:
            fc = C_Linear
            relu = C_ReLU
        else:
            fc = nn.Linear
            relu = nn.ReLU
        self.feature_dim = feature_dim
        self.fc1 = fc(feature_dim, 16)
        self.relu = relu
        self.loss = loss

    def forward(self, batch_x, batch_label, batch_mask):
        total_loss = 0
        total_acc_matrix = []
        for i_batch in range(batch_x.shape[0]):
            label = batch_label[i_batch:i_batch + 1]
            mask = batch_mask[i_batch:i_batch + 1]
            x = batch_x[i_batch:i_batch + 1]
            mask_label = mask * label
            # Only for masked_label
            classes_label = torch.unique(mask_label.reshape(-1))
            classes_label = [c for c in classes_label.int() if c != 17 and c != 0]
            base_features = torch.zeros([len(classes_label), self.feature_dim]).cuda(x.device)
            # label for sub-image to calculate loss
            local_label = torch.zeros(label.shape).long().cuda(0)
            for i, cls in enumerate(classes_label):
                # For get loss.
                local_label[label == cls] = i
                # Get base_features
                mask_class_i = label == cls
                pos = torch.nonzero(mask_class_i)
                # labeled sample pos in heat-map
                pos = pos[:, -2:]
                base_features[i] = x[0, :, pos[:, 0], pos[:, 1]].mean(-1)
                if torch.isnan(base_features).sum():
                    print(0)
            pd_0 = self.L1Metric(x, base_features, len(classes_label))
            # one_hot = torch.zeros(pd.shape).cuda(x.device)
            # one_hot = one_hot.scatter_(0, local_label.reshape((1, *(local_label.shape[-2:]))), 1)
            # To allevoiate gradient explosion
            pd = pd_0 - torch.min(pd_0, 1, keepdim=True)[0]
            loss = self.loss(reduce=False)(pd, local_label)
            loss = (loss * mask.float().squeeze(1)).sum()
            loss = loss / mask.sum()
            # Get acc matrix.
            acc_matrix = {}
            pd_label = pd.argmax(1)
            mask = mask.reshape(pd_label.shape)
            pd_label[mask == 0] = -1
            for i, cls in enumerate(classes_label):
                indexs = torch.nonzero(pd_label == i)
                gt_i = Counter(local_label[indexs[:, 0], indexs[:, 1], indexs[:, 2]].reshape(-1).cpu().numpy())
                # Add zero sample class
                for j in range(len(classes_label)):
                    if j not in gt_i:
                        gt_i[j] = 0
                acc_matrix[cls] = gt_i.values()
            total_loss += loss
            total_acc_matrix.append(acc_matrix)
        return total_loss / x.shape[0], total_acc_matrix

    def L1Metric(self, x, base_features, num_classes):
        x = x.unsqueeze(1)
        print(x.shape)
        print(x)
        x = x.repeat((1, num_classes, 1, 1, 1))

        base_features = base_features.view(x.shape[0], -1, self.feature_dim, 1, 1)
        pd = torch.exp(-(x - base_features).abs().sum(2))
        return pd
