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
import csv
from keras.models import load_model
from os.path import join
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["CUDA_VISIBLE_DEVICES"] = '2' #use GPU with ID=2, 3


def calculate_mpjpe(pred, target):
    res_pred = []
    res_target = []
    mpjpe_loss = []
    for b in range(1):
        for i in range(13):
            pred_xy = torch.argmax(pred[:, :, i])
            target_xy = torch.argmax(target[i, :, :])
            res_pred.append([pred_xy / 344, pred_xy % 344])
            res_target.append([target_xy / 320 + 40, target_xy % 320 + 12])
            # print(pred_xy.shape, pred_xy, pred[b, i, :, :].shape)
            loss = torch.sqrt((pred_xy / 320 - target_xy / 320 - 40) ** 2 + (pred_xy % 320 - target_xy % 320 - 12) ** 2)
            mpjpe_loss.append(loss.detach().cpu())
    return np.mean(mpjpe_loss), res_pred, res_target


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
    root_dir = r"/mnt/mt/"

    device = torch.device("cuda:{:d}".format(args.cuda_num) if args.cuda_num else "cpu")

    # model = ResNet(args).to(device)
    model=load_model(join('/mnt/mt/savel_model/DHP_CNN.model'), custom_objects={'mse2D': mse2D})

    # opt = optim.Adam(model.parameters(), lr=0.0001)

    # scheduler = MultiStepLR(opt, [15, 20], 0.1)

    criterion = mse2D

    global_train_steps = 0

    best_valid_MPJPE2D = 1e10
    # model.train()

    outstr = ''
    # model.load_state_dict(torch.load(args.save_model + '/resnet_412.pth')["model_state_dict"])

    num_dicts = ["jcl_06_dict.npy", "jcl_07_dict.npy", "jcl_10_dict.npy", "scy_01_dict.npy", "scy_07_dict.npy", "scy_09_dict.npy"]
    with open(os.path.join(root_dir, "dhp19_results_4.12.csv"), "a") as csv_file:
        result_writer = csv.writer(csv_file)
        result = []
        for num_dict in num_dicts:
            test_dataset = load_test.HPELoad(
                    args,
                    root_data_dir=root_train_data_dir + 'data//',
                    root_label_dir=root_train_data_dir + 'label//',
                    root_dict_dir=root_dir + '/dict/' + num_dict,
                    )

            valid_mpjpe2D_all = []
            # model.eval()
            best_MPJPE = 1e10
            bad_MPJPE = 0
            index = -1
            gt_label = []
            with torch.no_grad():
                for i in range(len(test_dataset)):

                    data, label = test_dataset[i]
                    # data, label = torch.tensor(data[0, :, :]), torch.tensor(label).unsqueeze(0)
                    # data = F.pad(data, (12, 12, 40, 40))
                    # data = data.unsqueeze(0).unsqueeze(-1)
                    # data, label = data.to(device), label.to(device)
                    data = np.pad(data[0, :, :], ((40, 40), (12, 12)))
                    data = np.expand_dims(data, axis=0)
                    # print(data.shape, type(data))

                    output = model.predict(np.expand_dims(data, axis=-1))

                    Loss2D, pred_label, target_label = calculate_mpjpe(torch.tensor(output), torch.tensor(label))

                    if Loss2D < best_MPJPE:
                        best_MPJPE = Loss2D
                        index = i
                        gt_label=[]
                        gt_label.append(pred_label)
                        gt_label.append(target_label)

                print("{} mpjpe loss : {}".format(num_dict[0:6], best_MPJPE))
                result.append(num_dict[0:6])
                result.append(best_MPJPE)
                result.append(index)
                result.append(gt_label)
                result_writer.writerow(result)
                result = []


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
