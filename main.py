# -*- coding: utf-8 -*-
# @Time    : 2021/4/20 0020 21:41
# @Author  : Daniel Zhang
# @Email   : zhangdan_nuaa@163.com
# @File    : main.py
# @Software: PyCharm


from utils import load_data, discretized_mix_logistic_loss, sample_from_discretized_mix_logistic, \
    load_part_of_model, discretized_mix_logistic_loss_1d, sample_from_discretized_mix_logistic_1d
from model import PixelCNN

import argparse
import numpy as np
import time
import os
from tqdm import tqdm
import torch
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler
from torchvision.utils import save_image

torch.set_printoptions(sci_mode=False)
np.set_printoptions(suppress=True)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def parse_arg():
    parser = argparse.ArgumentParser('参数管理')

    parser.add_argument('--nr_resnet', type=int, default=5,
                        help='Number of residual blocks per stage of the model')
    parser.add_argument('--nr_filters', type=int, default=160,
                        help='Number of filters to use across the model. Higher = larger model.')
    parser.add_argument('--nr_logistic_mix', type=int, default=10,
                        help='Number of logistic components in the mixture. Higher = more flexible model')

    parser.add_argument('--lr', type=float, default=0.0002, help='Base learning rate')
    parser.add_argument('--lr_decay', type=float, default=0.999995,
                        help='Learning rate decay, applied every step of the optimization')
    parser.add_argument('--max_epochs', type=int, default=5000,
                        help='How many epochs to run in total?')
    parser.add_argument('--batch_size', type=int, default=25, help='批大小')

    parser.add_argument('--load_params', type=str, default='',
                        help='Restore training from previous model checkpoint?')
    parser.add_argument('--save_interval', type=int, default=1,
                        help='Every how many epochs to write checkpoint/samples?')

    return parser.parse_known_args()[0]


def train(config, mode='cifar10'):
    model_name = 'pcnn_lr:{:.5f}_nr-resnet{}_nr-filters{}'.format(config.lr, config.nr_resnet, config.nr_filters)
    try:
        os.makedirs('models')
        os.makedirs('images')
        # print('mkdir:', config.outfile)
    except OSError:
        pass

    seed = np.random.randint(0, 10000)
    print("Random Seed: ", seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = True

    trainset, train_loader, testset, test_loader, classes = load_data(mode=mode, batch_size=config.batch_size)
    if mode == 'cifar10' or mode == 'faces':
        obs = (3, 32, 32)
        loss_op = lambda real, fake: discretized_mix_logistic_loss(real, fake, config.nr_logistic_mix)
        sample_op = lambda x: sample_from_discretized_mix_logistic(x, config.nr_logistic_mix)
    elif mode == 'mnist':
        obs = (1, 28, 28)
        loss_op = lambda real, fake: discretized_mix_logistic_loss_1d(real, fake, config.nr_logistic_mix)
        sample_op = lambda x: sample_from_discretized_mix_logistic_1d(x, config.nr_logistic_mix)
    sample_batch_size = 25
    rescaling_inv = lambda x: .5 * x + .5

    model = PixelCNN(nr_resnet=config.nr_resnet, nr_filters=config.nr_filters,
                     input_channels=obs[0], nr_logistic_mix=config.nr_logistic_mix).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=config.lr_decay)

    if config.load_params:
        load_part_of_model(model, config.load_params)
        print('model parameters loaded')

    def sample(model):
        model.train(False)
        data = torch.zeros(sample_batch_size, obs[0], obs[1], obs[2])
        data = data.cuda()
        with tqdm(total=obs[1] * obs[2]) as pbar:
            for i in range(obs[1]):
                for j in range(obs[2]):
                    with torch.no_grad():
                        data_v = data
                        out = model(data_v, sample=True)
                        out_sample = sample_op(out)
                        data[:, :, i, j] = out_sample.data[:, :, i, j]
                    pbar.update(1)
        return data

    print('starting training')
    for epoch in range(config.max_epochs):
        model.train()
        torch.cuda.synchronize()
        train_loss = 0.
        time_ = time.time()
        with tqdm(total=len(train_loader)) as pbar:
            for batch_idx, (data, label) in enumerate(train_loader):
                data = data.requires_grad_(True).cuda()

                output = model(data)
                loss = loss_op(data, output)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                pbar.update(1)

        deno = batch_idx * config.batch_size * np.prod(obs)
        print('train loss : %s' % (train_loss / deno), end='\t')

        # decrease learning rate
        scheduler.step()

        model.eval()
        test_loss = 0.
        with tqdm(total=len(test_loader)) as pbar:
            for batch_idx, (data, _) in enumerate(test_loader):
                data = data.requires_grad_(False).cuda()

                output = model(data)
                loss = loss_op(data, output)
                test_loss += loss.item()
                del loss, output
                pbar.update(1)
        deno = batch_idx * config.batch_size * np.prod(obs)
        print('test loss : {:.4f}, time : {:.4f}'.format((test_loss / deno), (time.time() - time_)))

        torch.cuda.synchronize()

        if (epoch + 1) % config.save_interval == 0:
            torch.save(model.state_dict(), 'models/{}_{}.pth'.format(model_name, epoch))
            print('sampling...')
            sample_t = sample(model)
            sample_t = rescaling_inv(sample_t)
            save_image(sample_t, 'images/{}_{}.png'.format(model_name, epoch), nrow=5, padding=0)


def pred(config, mode='cifar10'):
    if mode == 'cifar10':
        obs = (3, 32, 32)
    sample_batch_size = 25
    model = PixelCNN(nr_resnet=config.nr_resnet, nr_filters=config.nr_filters,
                     input_channels=obs[0], nr_logistic_mix=config.nr_logistic_mix).cuda()

    if config.load_params:
        load_part_of_model(model, config.load_params)
        print('model parameters loaded')
    sample_op = lambda x: sample_from_discretized_mix_logistic(x, config.nr_logistic_mix)
    rescaling_inv = lambda x: .5 * x + .5

    def sample(model):
        model.train(False)
        data = torch.zeros(sample_batch_size, obs[0], obs[1], obs[2])
        data = data.cuda()
        for i in range(obs[1]):
            for j in range(obs[2]):
                with torch.no_grad():
                    data_v = data
                    out = model(data_v, sample=True)
                    out_sample = sample_op(out)
                    data[:, :, i, j] = out_sample.data[:, :, i, j]
        return data

    print('sampling...')
    sample_t = sample(model)
    sample_t = rescaling_inv(sample_t)
    save_image(sample_t, 'images/sample.png', nrow=5, padding=0)


if __name__ == '__main__':
    config = parse_arg()
    disp_str = ''
    for attr in sorted(dir(config), key=lambda x: len(x)):
        if not attr.startswith('_'):
            disp_str += ' {} : {}\n'.format(attr, getattr(config, attr))
    # print(disp_str)

    train(config, mode='faces')
    # pred(config)
