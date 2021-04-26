# -*- coding: utf-8 -*-
# @Time    : 2021/4/20 0020 23:05
# @Author  : Daniel Zhang
# @Email   : zhangdan_nuaa@163.com
# @File    : model.py
# @Software: PyCharm


import pdb
from utils import concat_elu, discretized_mix_logistic_loss
from layers import down_shifted_conv2d, gated_resnet, down_right_shifted_conv2d, \
    down_shifted_deconv2d, down_right_shifted_deconv2d, nin

import torch
import torch.nn as nn
import torch.nn.functional as F
torch.set_printoptions(sci_mode=False)


class PixelCNNLayer_up(nn.Module):
    '''
    一个完整的论文示意图流程
    u是vertical stack的流程。ul是horizontal stack的流程
    '''
    def __init__(self, nr_resnet, nr_filters, resnet_nonlinearity):
        super(PixelCNNLayer_up, self).__init__()
        self.nr_resnet = nr_resnet
        # stream from pixels above
        self.u_stream = nn.ModuleList([gated_resnet(nr_filters, down_shifted_conv2d,
                                                    resnet_nonlinearity, skip_connection=0)
                                       for _ in range(nr_resnet)])

        # stream from pixels above and to thes left
        self.ul_stream = nn.ModuleList([gated_resnet(nr_filters, down_right_shifted_conv2d,
                                                     resnet_nonlinearity, skip_connection=1)
                                        for _ in range(nr_resnet)])

    def forward(self, u, ul):
        u_list, ul_list = [], []

        for i in range(self.nr_resnet):
            u = self.u_stream[i](u)
            ul = self.ul_stream[i](ul, a=u)
            u_list += [u]
            ul_list += [ul]

        return u_list, ul_list


class PixelCNNLayer_down(nn.Module):
    def __init__(self, nr_resnet, nr_filters, resnet_nonlinearity):
        super(PixelCNNLayer_down, self).__init__()
        self.nr_resnet = nr_resnet
        # stream from pixels above
        self.u_stream = nn.ModuleList([gated_resnet(nr_filters, down_shifted_conv2d,
                                                    resnet_nonlinearity, skip_connection=1)
                                       for _ in range(nr_resnet)])

        # stream from pixels above and to thes left
        self.ul_stream = nn.ModuleList([gated_resnet(nr_filters, down_right_shifted_conv2d,
                                                     resnet_nonlinearity, skip_connection=2)
                                        for _ in range(nr_resnet)])

    def forward(self, u, ul, u_list, ul_list):
        for i in range(self.nr_resnet):
            u = self.u_stream[i](u, a=u_list.pop())
            ul = self.ul_stream[i](ul, a=torch.cat((u, ul_list.pop()), 1))

        return u, ul


class PixelCNN(nn.Module):
    '''
    u_init是长方形的卷积核，如论文所说提取当前点上方所有像素信息
    ul_init中组合起来的卷积核，实际就是[[1, 1, 1],
                                    [1, 0, 0],
                                    [0, 0, 0]
    u_init和ul_init是所谓的MaskA，即不包括本身位置的卷积
    up_layers和down_layers有类似unet的跳连接，分别代表上采样和下采样阶段的resnet模块
    downsize和upsize是利用stride=2的conv和deconv完成的上下采样，维度上的变化
    最终输出的通道数为num_mix * nr_logistic_mix
    '''
    def __init__(self, nr_resnet=5, nr_filters=80, nr_logistic_mix=10,
                    resnet_nonlinearity='concat_elu', input_channels=3):
        super(PixelCNN, self).__init__()
        if resnet_nonlinearity == 'concat_elu':
            self.resnet_nonlinearity = lambda x: concat_elu(x)
        else:
            raise Exception('right now only concat elu is supported as resnet nonlinearity.')

        self.nr_filters = nr_filters
        self.input_channels = input_channels
        self.nr_logistic_mix = nr_logistic_mix
        self.right_shift_pad = nn.ZeroPad2d((1, 0, 0, 0))
        self.down_shift_pad = nn.ZeroPad2d((0, 0, 1, 0))

        down_nr_resnet = [nr_resnet] + [nr_resnet + 1] * 2
        self.down_layers = nn.ModuleList([PixelCNNLayer_down(down_nr_resnet[i], nr_filters,
                                                             self.resnet_nonlinearity) for i in range(3)])

        self.up_layers = nn.ModuleList([PixelCNNLayer_up(nr_resnet, nr_filters,
                                                         self.resnet_nonlinearity) for _ in range(3)])

        self.downsize_u_stream = nn.ModuleList([down_shifted_conv2d(nr_filters, nr_filters,
                                                                    stride=(2, 2)) for _ in range(2)])

        self.downsize_ul_stream = nn.ModuleList([down_right_shifted_conv2d(nr_filters, nr_filters,
                                                                           stride=(2, 2)) for _ in range(2)])

        self.upsize_u_stream = nn.ModuleList([down_shifted_deconv2d(nr_filters, nr_filters,
                                                                    stride=(2, 2)) for _ in range(2)])

        self.upsize_ul_stream = nn.ModuleList([down_right_shifted_deconv2d(nr_filters, nr_filters,
                                                                           stride=(2, 2)) for _ in range(2)])

        self.u_init = down_shifted_conv2d(input_channels + 1, nr_filters,
                                          filter_size=(2, 3), shift_output_down=True)
        self.ul_init = nn.ModuleList([down_shifted_conv2d(input_channels + 1, nr_filters,
                                                          filter_size=(1, 3), shift_output_down=True),
                                      down_right_shifted_conv2d(input_channels + 1, nr_filters,
                                                                filter_size=(2, 1), shift_output_right=True)])

        num_mix = 3 if self.input_channels == 1 else 10
        self.nin_out = nin(nr_filters, num_mix * nr_logistic_mix)
        self.init_padding = None

    def forward(self, x, sample=False):
        # similar as done in the tf repo :
        if self.init_padding is None and not sample:
            xs = [int(y) for y in x.size()]
            padding = torch.ones(xs[0], 1, xs[2], xs[3]).requires_grad_(False)  # 为什么要在通道上加全1的padding呢？
            self.init_padding = padding.cuda() if x.is_cuda else padding

        if sample:
            xs = [int(y) for y in x.size()]
            padding = torch.ones(xs[0], 1, xs[2], xs[3]).requires_grad_(False)
            padding = padding.cuda() if x.is_cuda else padding
            x = torch.cat((x, padding), 1)

        ### UP PASS ###
        x = x if sample else torch.cat((x, self.init_padding), 1)
        u_list = [self.u_init(x)]
        ul_list = [self.ul_init[0](x) + self.ul_init[1](x)]

        for i in range(3):
            # resnet block
            u_out, ul_out = self.up_layers[i](u_list[-1], ul_list[-1])
            u_list += u_out
            ul_list += ul_out

            if i != 2:
                # downscale (only twice)
                u_list += [self.downsize_u_stream[i](u_list[-1])]
                ul_list += [self.downsize_ul_stream[i](ul_list[-1])]

        ### DOWN PASS ###
        u = u_list.pop()
        ul = ul_list.pop()

        for i in range(3):
            # resnet block
            u, ul = self.down_layers[i](u, ul, u_list, ul_list)

            # upscale (only twice)
            if i != 2:
                u = self.upsize_u_stream[i](u)
                ul = self.upsize_ul_stream[i](ul)

        x_out = self.nin_out(F.elu(ul))
        assert len(u_list) == len(ul_list) == 0, pdb.set_trace()

        return x_out


if __name__ == '__main__':
    img = torch.zeros(8, 3, 32, 32).float().uniform_(-1, 1).cuda()
    # img = torch.zeros(8, 3, 32, 32).float().cuda()
    model = PixelCNN(nr_resnet=3, nr_filters=100, input_channels=img.size(1)).cuda()
    out = model(img)

    # loss = discretized_mix_logistic_loss(img, out)
    # print('loss : %s' % loss.item())
