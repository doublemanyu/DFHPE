# -*- coding: utf-8 -*-
# @Time    : 2022/6/10 14:18
# @Author  : Jiaan Chen, Hao Shi

from __future__ import print_function
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from dataset.DHP19EPC import DHP19EPC
from model.simplePointTrans import PoseTrans
import numpy as np
from torch.utils.data import DataLoader
from tools.utils import init_dir, IOStream, decode_batch_sa_simdr, accuracy, KLDiscretLoss
from tools.geometry_function import get_pred_3d_batch, cal_2D_mpjpe, cal_3D_mpjpe
from tools.image_save import save_debug_images
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from os.path import join


def train(exp_name, args, io):
    # path of train data
    root_train_data_dir = r'/mnt/data/ynn/DHP19_our/event_stream/test/'
    # path of valid data
    root_valid_data_dir = r'/mnt/data/ynn/DHP19_our/event_stream/test/'
    root_dict = r"/mnt/data/ynn/mt/dict/"

    train_dataset = DHP19EPC(
        args,
        root_data_dir=root_train_data_dir + 'data//',
        root_label_dir=root_train_data_dir + 'label//',
        root_3Dlabel_dir=root_train_data_dir + '3Dlabel//',
        root_dict_dir=root_dict + 'Dict_train.npy',
        min_EventNum=1024, Test3D=False
    )
    valid_dataset = DHP19EPC(
        args,
        root_data_dir=root_valid_data_dir + 'data//',
        root_label_dir=root_valid_data_dir + 'label//',
        root_3Dlabel_dir=root_valid_data_dir + '3Dlabel//',
        root_dict_dir=root_dict + 'Dict_test.npy',
        min_EventNum=0, Test3D=True,
    )

    train_loader = DataLoader(train_dataset, num_workers=0,
                              batch_size=args.train_batch_size, shuffle=False, drop_last=True)
    valid_loader = DataLoader(valid_dataset, num_workers=0,
                              batch_size=args.valid_batch_size, shuffle=False, drop_last=False)

    device = torch.device("cuda:{}".format(args.cuda_num))

    # Try to load models
    model = PoseTrans(args).to(device)

    opt = optim.Adam(model.parameters(), lr=0.0001)

    scheduler = MultiStepLR(opt, [15, 20], 0.1)

    criterion = KLDiscretLoss()

    LogWriter = SummaryWriter(log_dir='logs/%s/' % exp_name)
    global_train_steps = 0

    # best_valid_MPJPE3D = 1e10
    best_valid_MPJPE2D = 1e10
    model.train()

    outstr = ''

    for epoch in range(args.epochs):

        scheduler.step()

        # ********Train********
        train_loss_list = []
        train_acc_cnt_all = 0.0
        train_acc_final = 0.0

        for i, (data, xlabel, ylabel, wlabel) in enumerate(train_loader):

            data, xlabel, ylabel, wlabel = data.to(device), xlabel.to(device), ylabel.to(device), wlabel.to(device)

            output_x, output_y = model(data.float())
            # print(output_x.shape, output_y.shape, xlabel.shape)
            # KL loss
            loss = criterion(output_x, output_y, xlabel, ylabel, wlabel)

            opt.zero_grad()
            loss.backward()
            opt.step()

            train_loss_list.append(loss.item())

            decode_batch_pred = decode_batch_sa_simdr(output_x, output_y)
            decode_batch_label = decode_batch_sa_simdr(xlabel, ylabel)

            acc, avg_acc, cnt, pred = accuracy(decode_batch_pred, decode_batch_label, hm_type='sa-simdr', thr=0.5)

            train_acc_cnt_all += cnt
            train_acc_final += avg_acc * cnt
            
            if i % 20 == 0 and i > 0:
                outstr = 'Train Step %d | %d epoch, Loss: %.6f, Acc: %.6f' % (global_train_steps, epoch + 1,
                                                                              np.mean(train_loss_list),
                                                                              train_acc_final / train_acc_cnt_all)
                io.cprint(outstr)

                train_loss_list = []
                train_acc_cnt_all = 0.0
                train_acc_final = 0.0

            global_train_steps += 1

        print(outstr)

        # ********Valid********
        # if (epoch + 1) % 2 == 0 and (epoch >= args.epochs // 2) or (epoch == args.epochs - 1):
        if (epoch + 1) % 2 == 0 and (epoch >= 0) or (epoch == args.epochs - 1):


            valid_mpjpe2D_all = []
            valid_acc_cnt_all = 0.0
            valid_acc_final = 0.0
            model.eval()
            with torch.no_grad():
                for i, (data, xlabel, ylabel, wlabel) in enumerate(valid_loader):
                    data = data.to(device)

                    batch_size = data.size()[0]

                    data = data.float().to(device)
                    xlabel = xlabel.to(device)
                    ylabel = ylabel.to(device)
                    wlabel = wlabel.to(device)

                    output_x, output_y = model(data)

                    decode_batch_label = decode_batch_sa_simdr(xlabel, ylabel)

                    decode_batch_pred = decode_batch_sa_simdr(output_x, output_y)
                    pred = np.zeros((batch_size, 13, 2))
                    pred[:, :, 1] = decode_batch_pred[:, :, 0]

                    Loss2D = cal_2D_mpjpe(decode_batch_label, wlabel.squeeze(dim=2).cpu(), decode_batch_pred)

                    valid_mpjpe2D_all.append(Loss2D)

                    acc, avg_acc, cnt, pred = accuracy(decode_batch_pred, decode_batch_label, hm_type='sa-simdr',
                                                       thr=0.5)

                    valid_acc_cnt_all += cnt
                    valid_acc_final += avg_acc * cnt

            outstr = 'Valid %d epoch, Acc: %.6f, MPJPE2D: %.6f' % (epoch + 1,
                                                             valid_acc_final / valid_acc_cnt_all,
                                                             np.mean(valid_mpjpe2D_all))
            io.cprint(outstr)

            if (np.mean(valid_mpjpe2D_all)) <= best_valid_MPJPE2D:
                best_valid_MPJPE2D = np.mean(valid_mpjpe2D_all)
                torch.save(model.state_dict(),
                           'checkpoints/{}/models/model.pth'.format(exp_name))

            model.train()

    print('Best model is saved!')
    print('MPJPE2D: {:.2f}'.format(best_valid_MPJPE2D))


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Event Point Cloud HPE')

    parser.add_argument('--model', type=str, default='PointNet', metavar='N',
                        choices=['PointNet', 'DGCNN', 'PointTrans'],
                        help='Model to use, [PointNet, DGCNN, PointTrans]')
    parser.add_argument('--train_batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of train batch)')
    parser.add_argument('--valid_batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of valid batch)')
    parser.add_argument('--epochs', type=int, default=30, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--num_points', type=int, default=7500,
                        help='Number of event points to use(after sample)')
    parser.add_argument('--sensor_sizeH', type=int, default=720,
                        help='sensor_sizeH')
    parser.add_argument('--sensor_sizeW', type=int, default=1280,
                        help='sensor_sizeW')
    parser.add_argument('--num_joints', type=int, default=13,
                        help='number of joints')
    parser.add_argument('--label', type=str, default='mean', metavar='N',
                        choices=['mean', 'last'],
                        help='label setting ablation, [MeanLabel, LastLabel]')
    parser.add_argument('--name', type=str, default='Experiment1',
                        help='Name your exp')
    parser.add_argument('--cuda_num', type=int, default=0, metavar='N',
                        help='cuda device number')
    parser.add_argument('--save_image', action='store_true',
                        help='save image for debug')
    args = parser.parse_args()

    exp_name = args.name

    init_dir(args)

    io = IOStream('checkpoints/' + exp_name + '/run.log')
    io.cprint(str(args))

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        io.cprint(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
        torch.cuda.manual_seed(args.seed)
    else:
        io.cprint('Using CPU')

    train(exp_name, args, io)

    print('******** Finish ' + exp_name + ' ********')
