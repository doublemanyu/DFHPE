# -*- coding: utf-8 -*-
# @Time    : 2022/6/10 14:18
# @Author  : Jiaan Chen, Hao Shi

from __future__ import print_function
import os

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from dataset import load_test
from model.simpleResnet import PoseNet
from model.Resnet import ResNet
import numpy as np
from torch.utils.data import DataLoader
from tools.utils import init_dir, IOStream, decode_batch_sa_simdr, accuracy, KLDiscretLoss
from tools.geometry_function import cal_2D_mpjpe
# from tqdm import tqdm
from os.path import join
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def calculate_mpjpe(pred, target):
    mpjpe_loss = []
    for b in range(pred.shape[0]):
        for i in range(17):
            pred_xy = torch.argmax(pred[b, i, :, :])
            target_xy = torch.argmax(target[b, i, :, :])
            # print(pred_xy.shape, pred_xy, pred[b, i, :, :].shape)
            loss = 4 * torch.sqrt((pred_xy / 320 - target_xy / 320) ** 2 + (pred_xy % 320 - target_xy % 320) ** 2)
            mpjpe_loss.append(loss.detach().cpu())
    return np.mean(mpjpe_loss)


def mse2D(y_true, y_pred):
    # print(y_true.shape, y_pred.shape)
    mean_over_ch = torch.mean(torch.square(y_pred - y_true), axis=-1)
    mean_over_w = torch.mean(mean_over_ch, axis=-1)
    mean_over_h = torch.mean(mean_over_w, axis=-1)
    mean_over_all = torch.mean(mean_over_h, axis=-1)
    return mean_over_all


def train(args):
    root_train_data_dir = r'/mnt/DHP19_our/event_stream/test/'
    root_valid_data_dir = r'/mnt/DHP19_our/event_stream/test/'
    root_dict = r"/mnt/mt/dict/"

    train_dataset = load_test.HPELoad(
        args,
        root_data_dir=root_train_data_dir + 'data//',
        root_label_dir=root_train_data_dir + 'label_17//',
        root_dict_dir=root_dict + 'Dict_train.npy',
    )
    valid_dataset = load_test.HPELoad(
        args,
        root_data_dir=root_valid_data_dir + 'data//',
        root_label_dir=root_valid_data_dir + 'label_17//',
        root_dict_dir=root_dict + 'Dict_test.npy',
    )

    train_loader = DataLoader(train_dataset,
                              num_workers=8,
                              batch_size=args.train_batch_size,
                              shuffle=False,
                              drop_last=True)
    valid_loader = DataLoader(valid_dataset,
                              num_workers=8,
                              batch_size=args.valid_batch_size,
                              shuffle=False,
                              drop_last=False)

    device = torch.device("cuda:{:d}".format(args.cuda_num) if args.cuda_num else "cpu")

    model = ResNet(args).to(device)

    opt = optim.Adam(model.parameters(), lr=0.0001)

    scheduler = MultiStepLR(opt, [15, 20], 0.1)

    criterion = mse2D

    global_train_steps = 0

    best_valid_MPJPE2D = 1e10
    model.train()

    outstr = ''

    for epoch in range(args.epochs):

        scheduler.step()

        # ********Train********
        train_loss_list = []

        for i, (data, label) in enumerate(train_loader):
            
            data, label = data.to(device), label.to(device)

            output = model(data.float())

            # KL loss
            loss = criterion(label, output)

            opt.zero_grad()
            loss.backward()
            opt.step()

            train_loss_list.append(loss.item())
            
            if i % 20 == 0 and i > 0:
                outstr = 'Train Step %d | %d epoch, Loss: %.6f' % (global_train_steps, epoch + 1,
                                                                    np.mean(train_loss_list))
                print(outstr)

                train_loss_list = []

            global_train_steps += 1
        print(outstr)

        # ********Valid********
        # if (epoch + 1) % 2 == 0 and (epoch >= args.epochs // 2) or (epoch == args.epochs - 1):
        if (epoch + 1) % 2 == 0 and (epoch >= 0) or (epoch == args.epochs - 1):
            valid_mpjpe2D_all = []
            model.eval()
            with torch.no_grad():
                for i, (data, label) in enumerate(valid_loader):
                    data = data.to(device)

                    batch_size = data.size()[0]

                    data, label = data.to(device), label.to(device)

                    output = model(data)

                    Loss2D = calculate_mpjpe(output, label)

                    valid_mpjpe2D_all.append(Loss2D)
            mpjpe_2d = np.mean(valid_mpjpe2D_all)
            outstr = 'Valid %d epoch, MPJPE2D: %.6f' % (epoch + 1, mpjpe_2d)
            print(outstr)
            if mpjpe_2d < best_valid_MPJPE2D:
                torch.save({'model_state_dict': model.state_dict()}, args.save_model + '/resnet_17_412.pth')
                best_valid_MPJPE2D = mpjpe_2d

            model.train()

    print('Best model is saved!')
    print('MPJPE2D: {:.2f}'.format(best_valid_MPJPE2D))


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='test HPE')
    parser.add_argument('--train_batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of train batch)')
    parser.add_argument('--valid_batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of valid batch)')
    parser.add_argument('--epochs', type=int, default=30, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--sensor_sizeH', type=int, default=720,
                        help='sensor_sizeH')
    parser.add_argument('--sensor_sizeW', type=int, default=1280,
                        help='sensor_sizeW')
    parser.add_argument('--num_joints', type=int, default=17,
                        help='number of joints')
    parser.add_argument('--name', type=str, default='Experiment1',
                        help='Name your exp')
    parser.add_argument('--cuda_num', type=int, default=0, metavar='N',
                        help='cuda device number')
    parser.add_argument('--in_channel', type=int, default=1,
                        help='in_channel')
    parser.add_argument('--save_model', type=str, default="/mnt/mt/savel_model/",
                        help='savel_model')

    args = parser.parse_args()

    train(args)

    print('******** Finish resnet-test ********')

    # python main.py --train_batch_size=8 --valid_batch_size=8 --cuda_num=1
    # python main.py --train_batch_size=32 --valid_batch_size=32 --cuda_num=1
    # python test.py --train_batch_size=32 --valid_batch_size=32 --cuda_num=2
