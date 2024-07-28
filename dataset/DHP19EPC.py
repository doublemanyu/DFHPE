# -*- coding: utf-8 -*-
# @Time    : 2022/6/10 19:48
# @Author  : Jiaan Chen
# The SA-SimDR part is
# modified from repository of "https://github.com/leeyegy/SimDR"


import os
import sys
import glob
import h5py
import numpy as np
from torch.utils.data import Dataset
from os.path import join
import torch
import cv2
import copy
from time import time
from .sample import random_sample_point
from .rasterized import RasEventCloud


def adjust_target_weight(joint, target_weight, tmp_size, sx=1280, sy=720):
    mu_x = joint[0]
    mu_y = joint[1]
    # Check that any part of the gaussian is in-bounds
    ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
    br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
    if ul[0] >= sx or ul[1] >= sy or br[0] < 0 or br[1] < 0:
        # If not, just return the image as is
        target_weight = 0

    return target_weight


def generate_sa_simdr(joints, joints_vis, sigma=8, sx=1280, sy=720, num_joints=13):
    """
    joints:  [num_joints, 3]
    joints_vis: [num_joints, 3]

    return => target, target_weight(1: visible, 0: invisible)
    """

    target_weight = np.ones((num_joints, 1), dtype=np.float32)
    target_weight[:, 0] = joints_vis[:, 0]

    target_x = np.zeros((num_joints, int(sx)), dtype=np.float32)
    target_y = np.zeros((num_joints, int(sy)), dtype=np.float32)

    tmp_size = sigma * 3

    frame_size = np.array([sx, sy])
    frame_resize = np.array([sx, sy])
    feat_stride = frame_size / frame_resize

    for joint_id in range(num_joints):
        target_weight[joint_id] = \
            adjust_target_weight(joints[joint_id], target_weight[joint_id], tmp_size)
        if target_weight[joint_id] == 0:
            continue

        mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5)
        mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)

        x = np.arange(0, int(sx), 1, np.float32)
        y = np.arange(0, int(sy), 1, np.float32)

        v = target_weight[joint_id]
        if v > 0.5:
            target_x[joint_id] = (np.exp(- ((x - mu_x) ** 2) / (2 * sigma ** 2))) / (sigma * np.sqrt(np.pi * 2))
            target_y[joint_id] = (np.exp(- ((y - mu_y) ** 2) / (2 * sigma ** 2))) / (sigma * np.sqrt(np.pi * 2))

            # norm to [0,1]
            target_x[joint_id] = (target_x[joint_id] - target_x[joint_id].min()) / (
                    target_x[joint_id].max() - target_x[joint_id].min())
            target_y[joint_id] = (target_y[joint_id] - target_y[joint_id].min()) / (
                    target_y[joint_id].max() - target_y[joint_id].min())

    return target_x, target_y, target_weight


def generate_label(u, v, mask, sigma=8, sx=1280, sy=720, num_joints=13):
    joints_3d = np.zeros((num_joints, 3), dtype=np.float32)
    joints_3d_vis = np.zeros((num_joints, 3), dtype=np.float32)
    joints_3d[:, 0] = u
    joints_3d[:, 1] = v
    joints_3d_vis[:, 0] = mask
    joints_3d_vis[:, 1] = mask

    gt_x, gt_y, gt_joints_weight = generate_sa_simdr(joints_3d, joints_3d_vis, sigma, sx=sx, sy=sy)

    return gt_x, gt_y, gt_joints_weight


class DHP19EPC(Dataset):
    def __init__(self, args, root_data_dir=None, root_label_dir=None,
                 root_3Dlabel_dir=None, root_dict_dir=None, min_EventNum=1024, Test3D=False):
        self.root_data_dir = root_data_dir
        self.root_label_dir = root_label_dir
        self.root_3Dlabel_dir = root_3Dlabel_dir
        self.Test3D = Test3D
        self.sample_point_num = args.num_points
        self.label = args.label
        self.sx = args.sensor_sizeW
        self.sy = args.sensor_sizeH

        self.Point_Num_Dict = np.load(root_dict_dir, allow_pickle=True).item()
        self.Frame_Dict = []
        for name in self.Point_Num_Dict:
            self.Frame_Dict.append(name)

    def __getitem__(self, item):
        pointcloud, xlabel, ylabel, wlabel = self.load_sample(item)
    
        return pointcloud, xlabel, ylabel, wlabel

    def __len__(self):
        return len(self.Frame_Dict)

    def load_sample(self, item=None):
        pcdata = np.load(os.path.join(self.root_data_dir, self.Frame_Dict[item]), allow_pickle=True)  # [N, 4]: [x, y, t, p]
        pcdata = pcdata[:, [0, 1, 3, 2]]

        pclabel = np.load(os.path.join(self.root_label_dir, self.Frame_Dict[item][:-8] +
                                       "{:0>4d}_label.npy".format(int(self.Frame_Dict[item][-8:-4]) + 600)), allow_pickle=True)

        data = self.RasEventCloud_preprocess(pcdata)
        u, v = pclabel.T[:].astype(np.float32)
        mask = np.ones(13)
        x, y, weight = generate_label(u, v, mask)

        return data, x, y, weight

    def RasEventCloud_preprocess(self, data):

        if data.size == 0:

            data = np.zeros((1, 5))
            num_sample = self.sample_point_num

            if num_sample != 0:
                data_sample, select_index = random_sample_point(data, num_sample)
                data = data_sample

            return data

        data = data[:, 0:4]
        EventCloudDHP = RasEventCloud(input_size=(4, self.sy, self.sx))

        data = EventCloudDHP.convert(data).numpy()  # [x, y, t_avg, p_acc, event_cnt]

        num_sample = self.sample_point_num
        if num_sample != 0:
            index = data[-1, :].astype(np.int32)
            # print(index)
            res = []
            for i in range(4):
                data_sample, select_index = random_sample_point(data[index[i]:index[i + 1], :], num_sample)
                res.append(data_sample)
                # data = data_sample  # [num_sample, C]
        # print(data.shape)
        res_data = np.stack(res)
        return res_data
