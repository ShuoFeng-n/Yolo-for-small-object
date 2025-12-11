###################### RFA  ####     START   by  AI&CV  ###############################

# https://arxiv.org/pdf/2304.03198.pdf
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import math

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class SEWeightModule(nn.Module):

    def __init__(self, channels, reduction=16):
        super(SEWeightModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.avg_pool(x)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        weight = self.sigmoid(out)

        return weight


class MixConv2d(nn.Module):
    # Mixed Depth-wise Conv https://arxiv.org/abs/1907.09595
    def __init__(self, c1, c2, k=(1, 3), s=1, equal_ch=True):  # ch_in, ch_out, kernel, stride, ch_strategy
        super().__init__()
        n = len(k)  # number of convolutions
        if equal_ch:  # equal c_ per group
            i = torch.linspace(0, n - 1E-6, c2).floor()  # c2 indices
            c_ = [(i == g).sum() for g in range(n)]  # intermediate channels
        else:  # equal weight.numel() per group
            b = [c2] + [0] * n
            a = np.eye(n + 1, n, k=-1)
            a -= np.roll(a, 1, axis=1)
            a *= np.array(k) ** 2
            a[0] = 1
            c_ = np.linalg.lstsq(a, b, rcond=None)[0].round()  # solve for equal weight indices, ax = b

        self.m = nn.ModuleList([
            nn.Conv2d(c1, int(c_), k, s, k // 2, groups=math.gcd(c1, int(c_)), bias=False) for k, c_ in zip(k, c_)])
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(torch.cat([m(x) for m in self.m], 1)))


class CAConv(nn.Module):
    def __init__(self, inp, oup, kernel_size, stride, reduction=32):
        super(CAConv, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        self.conv = nn.Sequential(nn.Conv2d(inp, oup, kernel_size, padding=kernel_size // 2, stride=stride),
                                  nn.BatchNorm2d(oup),
                                  nn.ReLU())

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return self.conv(out)


class CBAMConv(nn.Module):
    def __init__(self, channel, out_channel, kernel_size, stride, reduction=16, spatial_kernel=7):
        super().__init__()

        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )

        self.spatital = nn.Conv2d(2, 1, kernel_size=spatial_kernel,
                                  padding=spatial_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

        self.conv = nn.Sequential(nn.Conv2d(channel, out_channel, kernel_size, padding=kernel_size // 2, stride=stride),
                                  nn.BatchNorm2d(out_channel),
                                  nn.ReLU())
        kernel_sizes = [3, 5, 7]
        self.attention_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(2, 1, kernel_size=k, padding=k // 2),
                nn.Sigmoid()
            ) for k in kernel_sizes
        ])

    def forward(self, x):
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        x = channel_out * x

        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        spatial_out = self.sigmoid(self.spatital(torch.cat([max_out, avg_out], dim=1)))
        spatial_out = [att_layer(torch.cat([max_out, avg_out], dim=1)) for att_layer in self.attention_layers]
        spatial_out = self.sigmoid(sum(spatial_out))  # / len(spatial_out)
        x = spatial_out * x
        return self.conv(x)


class CAMConv(nn.Module):
    def __init__(self, channel, out_channel, kernel_size, stride, reduction=16, spatial_kernel=7):
        super().__init__()

        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        self.conv = nn.Sequential(nn.Conv2d(channel, out_channel, kernel_size, padding=kernel_size // 2, stride=stride),
                                  nn.BatchNorm2d(out_channel),
                                  nn.ReLU())

    def forward(self, x):
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        x = channel_out * x
        return self.conv(x)


class RFCAConv(nn.Module):
    def __init__(self, c1, c2, kernel_size, stride):
        super(RFCAConv, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.group_conv1 = Conv(c1, 9 * c1, k=3, g=c1)
        self.group_conv2 = Conv(c1, 9 * c1, k=3, g=c1)
        self.group_conv3 = Conv(c1, 9 * c1, k=3, g=c1)

        self.softmax = nn.Softmax(dim=1)

        self.group_conv = Conv(c1, 9 * c1, k=3, g=c1)
        self.convDown = Conv(3 * c1, c1, k=3, s=3, g=c1)
        self.CA = CAConv(c1, c2, kernel_size, stride)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x)

        group1 = self.softmax(self.group_conv1(y))
        group2 = self.softmax(self.group_conv2(y))
        group3 = self.softmax(self.group_conv3(y))
        # g1 =  torch.cat([group1, group2, group3], dim=1)

        g2 = self.group_conv(x)

        out1 = g2 * group1
        out2 = g2 * group2
        out3 = g2 * group3

        # out = sum([out1, out2, out3])
        out = torch.cat([out1, out2, out3], dim=1)
        # 获取输入特征图的形状
        batch_size, channels, height, width = out.shape

        # 计算输出特征图的通道数
        output_channels = channels // 9

        # 重塑并转置特征图以将通道数分成3x3个子通道并扩展高度和宽度
        # out = out.view(batch_size, output_channels, 3, 3, height, width).permute(0, 1, 4, 2, 5,3).\
        #                                         reshape(batch_size, output_channels, 3 * height, 3 * width)
        out = out.view(batch_size, output_channels, height * 3, width * 3)
        out = self.convDown(out)
        out = self.CA(out)
        return out


class RFAConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(RFAConv, self).__init__()

        # Adaptive Average Pooling
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # Non-shared Convolution layers
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size // 2,
                               groups=in_channels)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size // 2,
                               groups=in_channels)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size // 2,
                               groups=in_channels)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU
        self.softmax = nn.Softmax(dim=1)
        self.conv4 = nn.Conv2d(in_channels, out_channels, 1, 1)

    def forward(self, x):
        b, c, _, _ = x.size()

        y = self.avg_pool(x)
        y = y.view(b, c)

        w1 = self.softmax(y)
        w2 = self.softmax(y)
        w3 = self.softmax(y)

        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        if w1.shape[1] == x1.shape[1]:
            out = w1.unsqueeze(2).unsqueeze(3) * x1 + \
                  w2.unsqueeze(2).unsqueeze(3) * x2 + \
                  w3.unsqueeze(2).unsqueeze(3) * x3
        else:
            out = self.conv4(w1.unsqueeze(2).unsqueeze(3)) * x1 + \
                  self.conv4(w2.unsqueeze(2).unsqueeze(3)) * x2 + \
                  self.conv4(w3.unsqueeze(2).unsqueeze(3)) * x3

        return out


class PSAMixConv(nn.Module):

    def __init__(self, inplans, planes, kernel_size, stride, conv_kernels=[[1, 3], [3, 5], [5, 7], [7, 9]],
                 conv_groups=[1, 4, 8, 16]):
        super(PSAMixConv, self).__init__()
        self.conv_1 = MixConv2d(inplans, planes // 4, k=conv_kernels[0])
        self.conv_2 = MixConv2d(inplans, planes // 4, k=conv_kernels[1])
        self.conv_3 = MixConv2d(inplans, planes // 4, k=conv_kernels[2])
        self.conv_4 = MixConv2d(inplans, planes // 4, k=conv_kernels[3])
        self.se = SEWeightModule(planes // 4)
        self.split_channel = planes // 4
        self.softmax = nn.Softmax(dim=1)
        self.conv = nn.Sequential(nn.Conv2d(inplans, planes, kernel_size, padding=kernel_size // 2, stride=stride),
                                  nn.BatchNorm2d(planes),
                                  nn.ReLU())

    def forward(self, x):
        batch_size = x.shape[0]
        x1 = self.conv_1(x)
        x2 = self.conv_2(x)
        x3 = self.conv_3(x)
        x4 = self.conv_4(x)

        feats = torch.cat((x1, x2, x3, x4), dim=1)
        feats = feats.view(batch_size, 4, self.split_channel, feats.shape[2], feats.shape[3])

        x1_se = self.se(x1)
        x2_se = self.se(x2)
        x3_se = self.se(x3)
        x4_se = self.se(x4)

        x_se = torch.cat((x1_se, x2_se, x3_se, x4_se), dim=1)
        attention_vectors = x_se.view(batch_size, 4, self.split_channel, 1, 1)
        attention_vectors = self.softmax(attention_vectors)
        feats_weight = feats * attention_vectors
        for i in range(4):
            x_se_weight_fp = feats_weight[:, i, :, :]
            if i == 0:
                out = x_se_weight_fp
            else:
                out = torch.cat((x_se_weight_fp, out), 1)
        out = self.conv(out)
        return out


class Conv_L(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        # self.bn = nn.LayerNorm((c2, s, s))  # 修改此处，其中s表示height和width
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.conv(x))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class DyMCAConv(nn.Module):
    def __init__(self, inp, oup, kernel_size, stride, reduction=32):
        super(DyMCAConv, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        self.conv = nn.Sequential(nn.Conv2d(inp, oup, kernel_size, padding=kernel_size // 2, stride=stride),
                                  nn.BatchNorm2d(oup),
                                  nn.SiLU())

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.downsample = nn.MaxPool2d(2)

        self.dynamic_weight_fc = nn.Sequential(
            nn.Linear(inp * 3, 3),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        identity = x
        b, c, h, w = x.shape
        # Multi-scale context information
        x_down = self.downsample(x)
        x_up = self.upsample(x)

        x_down = F.interpolate(x_down, size=(h, w), mode='bilinear', align_corners=False)
        x_up = F.interpolate(x_up, size=(h, w), mode='bilinear', align_corners=False)

        x_scales = [x, x_down, x_up]

        # Compute dynamic weights
        x_concat = torch.cat(x_scales, dim=1)
        x_avg_pool = nn.AdaptiveAvgPool2d(1)(x_concat)
        x_avg_pool = x_avg_pool.view(x_concat.size(0), -1)
        dynamic_weights = self.dynamic_weight_fc(x_avg_pool)

        out = 0
        for i, x_scale in enumerate(x_scales):
            n, c, h, w = x_scale.size()
            x_h = self.pool_h(x_scale)
            x_w = self.pool_w(x_scale).permute(0, 1, 3, 2)

            y = torch.cat([x_h, x_w], dim=2)
            y = self.conv1(y)
            y = self.bn1(y)
            y = self.act(y)

            x_h, x_w = torch.split(y, [h, w], dim=2)
            x_w = x_w.permute(0, 1, 3, 2)

            a_h = self.conv_h(x_h).sigmoid()
            a_w = self.conv_w(x_w).sigmoid()

            out_scale = identity * a_w * a_h
            out += dynamic_weights[:, i].view(-1, 1, 1, 1) * out_scale

        return self.conv(out)


class DyCAConv(nn.Module):
    def __init__(self, inp, oup, kernel_size, stride, reduction=32):
        super(DyCAConv, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        self.conv = nn.Sequential(nn.Conv2d(inp, oup, kernel_size, padding=kernel_size // 2, stride=stride),
                                  nn.BatchNorm2d(oup),
                                  nn.SiLU())

        self.dynamic_weight_fc = nn.Sequential(
            nn.Linear(inp, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        # Compute dynamic weights
        x_avg_pool = nn.AdaptiveAvgPool2d(1)(x)
        x_avg_pool = x_avg_pool.view(x.size(0), -1)
        dynamic_weights = self.dynamic_weight_fc(x_avg_pool)

        out = identity * (dynamic_weights[:, 0].view(-1, 1, 1, 1) * a_w +
                          dynamic_weights[:, 1].view(-1, 1, 1, 1) * a_h)

        return self.conv(out)


class CAConv2(nn.Module):
    def __init__(self, inp, oup, kernel_size, stride, reduction=16):
        super(CAConv2, self).__init__()
        self.pool_h1 = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w1 = nn.AdaptiveAvgPool2d((1, None))

        self.pool_h2 = nn.AdaptiveAvgPool2d((None, 2))
        self.pool_w2 = nn.AdaptiveAvgPool2d((2, None))

        self.pool_h3 = nn.AdaptiveAvgPool2d((None, 4))
        self.pool_w3 = nn.AdaptiveAvgPool2d((4, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        self.conv = nn.Sequential(nn.Conv2d(inp, oup, kernel_size, padding=kernel_size // 2, stride=stride),
                                  nn.BatchNorm2d(oup),
                                  nn.SiLU())

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h1 = self.pool_h1(x)  # n,c,h,1
        x_w1 = self.pool_w1(x).permute(0, 1, 3, 2)  # n,c,h,1

        x_h2 = self.pool_h2(x)  # n,c,h,1
        x_w2 = self.pool_w2(x).permute(0, 1, 3, 2)  # n,c,h,1

        x_h3 = self.pool_h3(x)  # n,c,h,1
        x_w3 = self.pool_w3(x).permute(0, 1, 3, 2)  # n,c,h,1

        x_h2 = F.interpolate(x_h2, size=(h, 1), mode='nearest')
        x_w2 = F.interpolate(x_w2, size=(w, 1), mode='nearest')

        x_h3 = F.interpolate(x_h3, size=(h, 1), mode='nearest')
        x_w3 = F.interpolate(x_w3, size=(w, 1), mode='nearest')

        x_h = x_h1 + x_h2 + x_h3
        x_w = x_w1 + x_w2 + x_w3

        # # Combine the pooling results using concatenation
        # x_h = torch.cat([x_h1, F.interpolate(x_h2, size=(h, 1), mode='nearest'),
        #                  F.interpolate(x_h3, size=(h, 1), mode='nearest')], dim=1)
        # x_w = torch.cat([x_w1, F.interpolate(x_w2, size=(1, w), mode='nearest'),
        #                  F.interpolate(x_w3, size=(1, w), mode='nearest')], dim=1)
        y = torch.cat([x_h, x_w], dim=2)  # n,c,2h,1
        y = self.conv1(y)  # n,c/16,2h,1
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)  # n,c/16,h,1
        x_w = x_w.permute(0, 1, 3, 2)  # n,c/16,1,w

        a_h = self.conv_h(x_h).sigmoid()  # n,c,1,w
        a_w = self.conv_w(x_w).sigmoid()  # n,c,1,w

        out = identity * a_w * a_h

        return self.conv(out)


import collections


class SKConv(nn.Module):

    def __init__(self, channel=512, c2=512, kernel_size=3, stride=2, kernels=[1, 3, 5, 7], reduction=32, group=16,
                 L=32):
        super().__init__()
        inp = channel
        group = channel
        oup = c2
        self.d = max(L, channel // reduction)
        self.convs = nn.ModuleList([])
        for k in kernels:
            self.convs.append(
                nn.Sequential(collections.OrderedDict([
                    ('conv', nn.Conv2d(channel, channel, kernel_size=k, padding=k // 2, groups=group)),
                    ('bn', nn.BatchNorm2d(channel)),
                    ('relu', nn.ReLU()),
                ]))
            )
        self.fc = nn.Linear(channel * 21, self.d)
        self.fcs = nn.ModuleList([])
        for i in range(len(kernels)):
            self.fcs.append(nn.Linear(self.d, channel))
        self.softmax = nn.Softmax(dim=0)
        self.avg_pool1 = nn.AdaptiveAvgPool2d(1)
        self.avg_pool2 = nn.AdaptiveAvgPool2d(2)
        self.avg_pool4 = nn.AdaptiveAvgPool2d(4)
        self.conv = nn.Sequential(nn.Conv2d(inp, oup, kernel_size, padding=kernel_size // 2, stride=stride),
                                  nn.BatchNorm2d(oup),
                                  nn.ReLU())

    def forward(self, x):
        bs, c, _, _ = x.size()
        conv_outs = []
        ### split
        for conv in self.convs:
            conv_outs.append(conv(x))
        feats = torch.stack(conv_outs, 0)  # k,bs,channel,h,w

        ### fuse
        U = sum(conv_outs)  # bs,c,h,w

        y1 = self.avg_pool1(x).reshape((bs, -1))
        y2 = self.avg_pool2(x).reshape((bs, -1))
        y3 = self.avg_pool4(x).reshape((bs, -1))
        S = torch.cat((y1, y2, y3), 1)
        ### reduction channel
        # S=U.mean(-1).mean(-1) #bs,c
        Z = self.fc(S)  # bs,d

        ### calculate attention weight
        weights = []
        for fc in self.fcs:
            weight = fc(Z)
            weights.append(weight.view(bs, c, 1, 1))  # bs,channel
        attention_weughts = torch.stack(weights, 0)  # k,bs,channel,1,1
        attention_weughts = self.softmax(attention_weughts)  # k,bs,channel,1,1

        ### fuse
        V = (attention_weughts * feats).sum(0)
        V = self.conv(V)
        return V
###################### RFA  ####     END  by  AI&CV  ###############################