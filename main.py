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
from dataset import data_load
from model.simpleResnet import PoseNet
import numpy as np
from torch.utils.data import DataLoader
from tools.utils import init_dir, IOStream, decode_batch_sa_simdr, accuracy, KLDiscretLoss
from tools.geometry_function import cal_2D_mpjpe
from tqdm import tqdm
from os.path import join
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def calculate_mpjpe(pred, target):
    mpjpe_loss = torch.tensor(0)
    for i in range(13):
        pred_xy = torch.argmax(pred[i])
        target_xy = torch.argmax(target[i])
        loss = torch.sqrt((pred_xy / 320 - target_xy / 320) ** 2 + (pred_xy % 320 - target_xy % 320) ** 2)
        mpjpe_loss += loss
    return mpjpe_loss / 13


def train(args):
    # path of train data
    # /home/mt/EventPointPose_ourData/Test/test_num_dict/scy_01_Num_Dict.npy

    # root_train_data_dir = r'/mnt/data/ynn/DHP19_our/event_stream/test/'
    root_train_data_dir = r'/data/ynn/DHP19_our/event_stream/test/'
    # path of valid data
    # root_valid_data_dir = r'/mnt/data/ynn/DHP19_our/event_stream/test/'
    root_valid_data_dir = r'/data/ynn/DHP19_our/event_stream/test/'

    # root_dict = r"/mnt/data/ynn/mt/dict/"
    root_dict = r"/data/ynn/mt/dict/"

    train_dataset = data_load.DHP19EPC(
        args,
        root_data_dir=root_train_data_dir + 'data//',
        root_label_dir=root_train_data_dir + 'label//',
        root_dict_dir=root_dict + 'Dict_train.npy',
    )
    valid_dataset = data_load.DHP19EPC(
        args,
        root_data_dir=root_valid_data_dir + 'data//',
        root_label_dir=root_valid_data_dir + 'label//',
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

    criterion = KLDiscretLoss()

    global_train_steps = 0

    best_valid_MPJPE2D = 1e10
    model.train()

    outstr = ''

    for epoch in range(args.epochs):
        pbar = tqdm(total=len(train_loader))

        scheduler.step()

        # ********Train********
        train_loss_list = []
        train_acc_cnt_all = 0.0
        train_acc_final = 0.0

        for i, (data, x_label, y_label, weight) in enumerate(train_loader):

            data, x_label, y_label, weight = data.to(device), x_label.to(device), y_label.to(device), weight.to(device)

            output_x, output_y = model(data.float())

            # KL loss
            loss = criterion(output_x, output_y, x_label, y_label, weight)

            opt.zero_grad()
            loss.backward()
            opt.step()

            train_loss_list.append(loss.item())

            decode_batch_pred = decode_batch_sa_simdr(output_x, output_y)
            decode_batch_label = decode_batch_sa_simdr(x_label, y_label)

            acc, avg_acc, cnt, pred = accuracy(decode_batch_pred, decode_batch_label, hm_type='sa-simdr', thr=0.5)

            train_acc_cnt_all += cnt
            train_acc_final += avg_acc * cnt

            if i % 20 == 0 and i > 0:
                outstr = 'Train Step %d | %d epoch, Loss: %.6f, Acc: %.6f' % (global_train_steps, epoch + 1,
                                                                              np.mean(train_loss_list),
                                                                              train_acc_final / train_acc_cnt_all)
                print(outstr)

                train_loss_list = []
                train_acc_cnt_all = 0.0
                train_acc_final = 0.0

            global_train_steps += 1

            pbar.update(1)

        pbar.close()

        print(outstr)

        # ********Valid********
        # if (epoch + 1) % 2 == 0 and (epoch >= args.epochs // 2) or (epoch == args.epochs - 1):
        if (epoch + 1) % 2 == 0 and (epoch >= 0) or (epoch == args.epochs - 1):
            valid_mpjpe2D_all = []
            valid_acc_cnt_all = 0.0
            valid_acc_final = 0.0
            model.eval()
            with torch.no_grad():
                pbar = tqdm(total=len(valid_loader))
                for i, (data, x_label, y_label, weight) in enumerate(valid_loader):
                    data = data.to(device)

                    batch_size = data.size()[0]

                    data = data.float().to(device)

                    x_label, y_label, weight = x_label.to(device), y_label.to(device), weight.to(device)

                    output_x, output_y = model(data)

                    decode_batch_label = decode_batch_sa_simdr(x_label, y_label)

                    decode_batch_pred = decode_batch_sa_simdr(output_x, output_y)

                    pred2 = np.zeros((batch_size, 13, 2))
                    pred2[:, :, 1] = decode_batch_pred[:, :, 0]  # exchange x,y order
                    pred2[:, :, 0] = decode_batch_pred[:, :, 1]

                    Loss2D = cal_2D_mpjpe(decode_batch_label, weight.squeeze(dim=2).cpu(), decode_batch_pred)

                    valid_mpjpe2D_all.append(Loss2D)

                    acc, avg_acc, cnt, pred = accuracy(decode_batch_pred, decode_batch_label, hm_type='sa-simdr',
                                                       thr=0.5)

                    valid_acc_cnt_all += cnt
                    valid_acc_final += avg_acc * cnt

                    valid_acc_cnt_all += cnt
                    valid_acc_final += avg_acc * cnt

                    pbar.update(1)
                pbar.close()

            outstr = 'Valid %d epoch, Acc: %.6f, MPJPE2D: %.6f' % (epoch + 1,
                                                                   valid_acc_final / valid_acc_cnt_all,
                                                                   np.mean(valid_mpjpe2D_all))
            print(outstr)

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
    parser.add_argument('--num_joints', type=int, default=13,
                        help='number of joints')
    parser.add_argument('--name', type=str, default='Experiment1',
                        help='Name your exp')
    parser.add_argument('--cuda_num', type=int, default=0, metavar='N',
                        help='cuda device number')
    parser.add_argument('--in_channel', type=int, default=2,
                        help='in_channel')
    args = parser.parse_args()

    train(args)

    print('******** Finish resnet-test ********')

    # python main.py --train_batch_size=8 --valid_batch_size=8 --cuda_num=1
    # python main.py --train_batch_size=32 --valid_batch_size=32 --cuda_num=1
    # python test.py --train_batch_size=32 --valid_batch_size=32 --cuda_num=2
