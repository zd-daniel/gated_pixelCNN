# -*- coding: utf-8 -*-
# @Time    : 2021/4/20 0020 21:41
# @Author  : Daniel Zhang
# @Email   : zhangdan_nuaa@163.com
# @File    : utils.py
# @Software: PyCharm


import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.autograd import Variable

from datasets import is_image_file, MyDataset, MyDataLoader


def load_data(mode='cifar10', batch_size=16):
    assert mode in ['cifar10', 'mnist', 'faces'], '未知数据集'

    if mode == 'faces':
        root_path = 'G:/Dataset/celebAHQ/'

        image_list = [x for x in os.listdir(root_path) if is_image_file(x)]
        train_list = image_list[:int(0.8 * len(image_list))]
        test_list = image_list[int(0.8 * len(image_list)):]
        assert len(train_list) > 0
        assert len(test_list) >= 0

        trainset = MyDataset(train_list, root_path, input_height=None, crop_height=None,
                             output_height=32, is_mirror=True)
        testset = MyDataset(test_list, root_path, input_height=None, crop_height=None,
                            output_height=32, is_mirror=False)
        trainloader = MyDataLoader(trainset, batch_size)
        testloader = MyDataLoader(testset, batch_size, shuffle=False)
        classes = None

        return trainset, trainloader, testset, testloader, classes

    elif mode == 'cifar10':
        root_path = 'G:/Dataset/cifar10/'
        classes = ('plane', 'car', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'trunk')
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                        ])
        trainset = torchvision.datasets.CIFAR10(root=root_path, train=True,
                                                download=False, transform=transform)
        testset = torchvision.datasets.CIFAR10(root=root_path, train=False,
                                               download=False, transform=transform)

    elif mode == 'mnist':
        root_path = 'G:/Dataset/mnist/'
        classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5, ), (0.5, ))
                                        ])
        trainset = torchvision.datasets.MNIST(root=root_path, train=True,
                                              download=False, transform=transform)
        testset = torchvision.datasets.MNIST(root=root_path, train=False,
                                             download=False, transform=transform)

    trainloader = DataLoader(trainset, batch_size=batch_size,
                             shuffle=True, pin_memory=True,
                             drop_last=False, num_workers=2)
    testloader = DataLoader(trainset, batch_size=batch_size,
                            shuffle=True, pin_memory=True,
                            drop_last=False, num_workers=2)

    return trainset, trainloader, testset, testloader, classes


def load_part_of_model(model, path):
    params = torch.load(path)
    added = 0
    for name, param in params.items():
        if name in model.state_dict().keys():
            try:
                model.state_dict()[name].copy_(param)
                added += 1
            except Exception as e:
                print(e)
                pass
    print('added %s of params:' % (added / float(len(model.state_dict().keys()))))


def concat_elu(x):
    """ like concatenated ReLU (http://arxiv.org/abs/1603.05201), but then with ELU """
    # TODO trick？这样做的好处是什么
    axis = len(x.size()) - 3
    return F.elu(torch.cat([x, -x], dim=axis))


def down_shift(x, pad=None):
    # Pytorch ordering
    xs = [int(y) for y in x.size()]
    # when downshifting, the last row is removed
    x = x[:, :, :xs[2] - 1, :]
    # padding left, padding right, padding top, padding bottom
    pad = nn.ZeroPad2d((0, 0, 1, 0)) if pad is None else pad
    return pad(x)


def right_shift(x, pad=None):
    # Pytorch ordering
    xs = [int(y) for y in x.size()]
    # when righshifting, the last column is removed
    x = x[:, :, :, :xs[3] - 1]
    # padding left, padding right, padding top, padding bottom
    pad = nn.ZeroPad2d((1, 0, 0, 0)) if pad is None else pad
    return pad(x)


def log_sum_exp(x):
    """ numerically stable log_sum_exp implementation that prevents overflow """
    # TF ordering
    axis = len(x.size()) - 1
    m, _ = torch.max(x, dim=axis)
    m2, _ = torch.max(x, dim=axis, keepdim=True)
    return m + torch.log(torch.sum(torch.exp(x - m2), dim=axis))


def log_prob_from_logits(x):
    """
    返回log_softmax, 用了logsumexp防止下溢的技巧
    log[softmax(X)] = x - logsumexp(x) = x - (m + logsumexp(x - m)), m取最大值
    """
    # TF ordering
    axis = len(x.size()) - 1
    m, _ = torch.max(x, dim=axis, keepdim=True)
    return x - m - torch.log(torch.sum(torch.exp(x - m), dim=axis, keepdim=True))


def discretized_mix_logistic_loss(x, l, nr_mix):
    """ log-likelihood for mixture of discretized logistics, assumes the data has been rescaled to [-1,1] interval """
    '''
    l最后一维的100中，10个模型10个π，30个u，30个s，30个coeff
    x[:, :, :, 0:3, :]即RGB
    coeffs[:, :, :, 0, :] coeffs[:, :, :, 1, :] coeffs[:, :, :, 2, :]分别是公式3中的α β γ
    means即代表根据先前像素点信息预测的R值，根据先前像素+R预测的G值，根据先前像素+RG预测的B值
    nr_mix是公式中预设的K，混合logistic分布的数量，文中说5就足够
    logit_probs是各个模型的占比，即公式中的π，所以要经过softmax归一化
    所以模型输出维度包括，nr_mix个π，3*nr_mix个均值，3*nr_mix个coef，3*nr_mix个scale
    '''
    x = x.permute(0, 2, 3, 1)
    l = l.permute(0, 2, 3, 1)
    xs = [int(y) for y in x.size()]
    ls = [int(y) for y in l.size()]

    # here and below: unpacking the params of the mixture of logistics
    logit_probs = l[:, :, :, :nr_mix]
    l = l[:, :, :, nr_mix:].reshape(xs + [nr_mix * 3])  # 3 for mean, scale, coef

    means = l[:, :, :, :, :nr_mix]
    log_scales = torch.clamp(l[:, :, :, :, nr_mix:2 * nr_mix], min=-7.)
    coeffs = torch.tanh(l[:, :, :, :, 2 * nr_mix:3 * nr_mix])

    x = x.contiguous()
    x = x.unsqueeze(-1) + torch.zeros(xs + [nr_mix]).cuda().requires_grad_(False)

    m2 = (means[:, :, :, 1, :] + coeffs[:, :, :, 0, :] * x[:, :, :, 0, :]).view(xs[0], xs[1], xs[2], 1, nr_mix)
    m3 = (means[:, :, :, 2, :] +
          coeffs[:, :, :, 1, :] * x[:, :, :, 0, :] +
          coeffs[:, :, :, 2, :] * x[:, :, :, 1, :]).view(xs[0], xs[1], xs[2], 1, nr_mix)

    means = torch.cat((means[:, :, :, 0, :].unsqueeze(3), m2, m3), dim=3)

    # 根据文章公式2，为什么是 ± 1/255而不是0.5，pixelCNN将[0,255]约束到[-1,1]，所以这里要在0.5基础上除以 255/2
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1. / 255.)
    cdf_plus = torch.sigmoid(plus_in)
    min_in = inv_stdv * (centered_x - 1. / 255.)
    cdf_min = torch.sigmoid(min_in)
    cdf_delta = cdf_plus - cdf_min  # probability for all other cases  # 公式2的P  32*32*3*10每个像素点3个通道的生成概率

    # log_cdf_plus，log_one_minus_cdf_min这两步按照文中所说，是edge case取±无穷的情况。以下两个操作已经取了log
    # log probability for edge case of 0 (before scaling)
    log_cdf_plus = plus_in - F.softplus(plus_in)
    # log probability for edge case of 255 (before scaling)
    log_one_minus_cdf_min = -F.softplus(min_in)

    # TODO 这个是什么不理解
    mid_in = inv_stdv * centered_x
    # log probability in the center of the bin, to be used in extreme cases
    # (not actually used in our code)
    log_pdf_mid = mid_in - log_scales - 2. * F.softplus(mid_in)

    inner_inner_cond = (cdf_delta > 1e-5).float()
    inner_inner_out = inner_inner_cond * torch.log(torch.clamp(cdf_delta, min=1e-12)) + (1. - inner_inner_cond) * (
                log_pdf_mid - np.log(127.5))
    inner_cond = (x > 0.999).float()  # 这里即超过255的情况，用log_one_minus_cdf_min代替
    inner_out = inner_cond * log_one_minus_cdf_min + (1. - inner_cond) * inner_inner_out
    cond = (x < -0.999).float()  # 这里即小于0的情况，用log_cdf_plus代替
    log_probs = cond * log_cdf_plus + (1. - cond) * inner_out

    # 这一步是相乘=取log相加，即公式2的相乘
    # log_prob_from_logits(logit_probs)是π取log，log_probs是公式2中括号取log
    log_probs = torch.sum(log_probs, dim=3) + log_prob_from_logits(logit_probs)

    # 多分类的log likelihood，防止下溢用log_sum_exp
    return -torch.sum(log_sum_exp(log_probs))


def discretized_mix_logistic_loss_1d(x, l, nr_mix):
    """ log-likelihood for mixture of discretized logistics, assumes the data has been rescaled to [-1,1] interval """
    # Pytorch ordering
    x = x.permute(0, 2, 3, 1)
    l = l.permute(0, 2, 3, 1)
    xs = [int(y) for y in x.size()]
    ls = [int(y) for y in l.size()]

    # here and below: unpacking the params of the mixture of logistics
    logit_probs = l[:, :, :, :nr_mix]
    l = l[:, :, :, nr_mix:].contiguous().view(xs + [nr_mix * 2])  # 2 for mean, scale

    means = l[:, :, :, :, :nr_mix]
    log_scales = torch.clamp(l[:, :, :, :, nr_mix:2 * nr_mix], min=-7.)
    # here and below: getting the means and adjusting them based on preceding
    # sub-pixels
    x = x.contiguous()
    x = x.unsqueeze(-1) + torch.zeros(xs + [nr_mix]).cuda().requires_grad_(False)

    # means = torch.cat((means[:, :, :, 0, :].unsqueeze(3), m2, m3), dim=3)
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1. / 255.)
    cdf_plus = torch.sigmoid(plus_in)
    min_in = inv_stdv * (centered_x - 1. / 255.)
    cdf_min = torch.sigmoid(min_in)
    cdf_delta = cdf_plus - cdf_min  # probability for all other cases

    # log probability for edge case of 0 (before scaling)
    log_cdf_plus = plus_in - F.softplus(plus_in)
    # log probability for edge case of 255 (before scaling)
    log_one_minus_cdf_min = -F.softplus(min_in)

    mid_in = inv_stdv * centered_x
    # log probability in the center of the bin, to be used in extreme cases
    # (not actually used in our code)
    log_pdf_mid = mid_in - log_scales - 2. * F.softplus(mid_in)

    inner_inner_cond = (cdf_delta > 1e-5).float()
    inner_inner_out = inner_inner_cond * torch.log(torch.clamp(cdf_delta, min=1e-12)) + (1. - inner_inner_cond) * (
                log_pdf_mid - np.log(127.5))
    inner_cond = (x > 0.999).float()
    inner_out = inner_cond * log_one_minus_cdf_min + (1. - inner_cond) * inner_inner_out
    cond = (x < -0.999).float()
    log_probs = cond * log_cdf_plus + (1. - cond) * inner_out
    log_probs = torch.sum(log_probs, dim=3) + log_prob_from_logits(logit_probs)

    return -torch.sum(log_sum_exp(log_probs))


def to_one_hot(tensor, n, fill_with=1.):
    # we perform one hot encore with respect to the last axis
    one_hot = torch.FloatTensor(tensor.size() + (n,)).zero_()
    if tensor.is_cuda:
        one_hot = one_hot.cuda()
    one_hot.scatter_(len(tensor.size()), tensor.unsqueeze(-1), fill_with)
    return one_hot


def sample_from_discretized_mix_logistic(l, nr_mix):
    # Pytorch ordering
    l = l.permute(0, 2, 3, 1)
    ls = [int(y) for y in l.size()]
    xs = ls[:-1] + [3]

    # unpack parameters
    logit_probs = l[:, :, :, :nr_mix]
    l = l[:, :, :, nr_mix:].reshape(xs + [nr_mix * 3])
    # sample mixture indicator from softmax
    temp = torch.FloatTensor(logit_probs.size())
    if l.is_cuda:
        temp = temp.cuda()
    temp.uniform_(1e-5, 1. - 1e-5)
    temp = logit_probs.data - torch.log(- torch.log(temp))
    _, argmax = temp.max(dim=3)

    one_hot = to_one_hot(argmax, nr_mix)
    sel = one_hot.view(xs[:-1] + [1, nr_mix])  # 根据logit_probs，nr_mix个模型中选出当前像素属于哪个模型
    # select logistic parameters
    means = torch.sum(l[:, :, :, :, :nr_mix] * sel, dim=4)
    log_scales = torch.clamp(torch.sum(l[:, :, :, :, nr_mix:2 * nr_mix] * sel, dim=4), min=-7.)
    coeffs = torch.sum(torch.tanh(l[:, :, :, :, 2 * nr_mix:3 * nr_mix]) * sel, dim=4)
    # sample from logistic & clip to interval
    # we don't actually round to the nearest 8bit value when sampling
    u = torch.FloatTensor(means.size())  # TODO 这里的u是为了增加随机性吗？
    if l.is_cuda:
        u = u.cuda()
    u.uniform_(1e-5, 1. - 1e-5)
    u = Variable(u)
    x = means + torch.exp(log_scales) * (torch.log(u) - torch.log(1. - u))
    x0 = torch.clamp(torch.clamp(x[:, :, :, 0], min=-1.), max=1.)
    x1 = torch.clamp(torch.clamp(x[:, :, :, 1] + coeffs[:, :, :, 0] * x0, min=-1.), max=1.)
    x2 = torch.clamp(torch.clamp(x[:, :, :, 2] + coeffs[:, :, :, 1] * x0 + coeffs[:, :, :, 2] * x1, min=-1.), max=1.)

    out = torch.cat([x0.view(xs[:-1] + [1]), x1.view(xs[:-1] + [1]), x2.view(xs[:-1] + [1])], dim=3)
    # put back in Pytorch ordering
    out = out.permute(0, 3, 1, 2)
    return out


def sample_from_discretized_mix_logistic_1d(l, nr_mix):
    # Pytorch ordering
    l = l.permute(0, 2, 3, 1)
    ls = [int(y) for y in l.size()]
    xs = ls[:-1] + [1]  # [3]

    # unpack parameters
    logit_probs = l[:, :, :, :nr_mix]
    l = l[:, :, :, nr_mix:].contiguous().view(xs + [nr_mix * 2])  # for mean, scale

    # sample mixture indicator from softmax
    temp = torch.FloatTensor(logit_probs.size())
    if l.is_cuda:
        temp = temp.cuda()
    temp.uniform_(1e-5, 1. - 1e-5)
    temp = logit_probs.data - torch.log(- torch.log(temp))
    _, argmax = temp.max(dim=3)

    one_hot = to_one_hot(argmax, nr_mix)
    sel = one_hot.view(xs[:-1] + [1, nr_mix])
    # select logistic parameters
    means = torch.sum(l[:, :, :, :, :nr_mix] * sel, dim=4)
    log_scales = torch.clamp(torch.sum(l[:, :, :, :, nr_mix:2 * nr_mix] * sel, dim=4), min=-7.)
    u = torch.FloatTensor(means.size())
    if l.is_cuda:
        u = u.cuda()
    u.uniform_(1e-5, 1. - 1e-5)
    u = Variable(u)
    x = means + torch.exp(log_scales) * (torch.log(u) - torch.log(1. - u))
    x0 = torch.clamp(torch.clamp(x[:, :, :, 0], min=-1.), max=1.)
    out = x0.unsqueeze(1)
    return out


if __name__ == '__main__':
    load_data()
