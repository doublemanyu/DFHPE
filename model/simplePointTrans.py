import torch
import torch.nn as nn
from model.PointTrans_tools import PointNetSetAbstraction, TransformerBlock
import numpy as np
from model.Transformer import PoseTransBlock


class TransitionDown(nn.Module):
    def __init__(self, k, nneighbor, channels):
        super().__init__()
        # k is the number of center points in furthest point sample
        self.sa = PointNetSetAbstraction(k, 0, nneighbor, channels[0], channels[1:], group_all=False, knn=True)

    def forward(self, xyz, points):
        return self.sa(xyz, points)


class Backbone(nn.Module):
    def __init__(self, num_points):
        super().__init__()
        self.n_points = num_points
        self.n_blocks = 4
        self.n_neighbor = 16   # 16 for default
        self.d_points = 5  # dim for input points
        self.transformer_dim = 256
        self.fc1 = nn.Sequential(
            nn.Linear(self.d_points, 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        )
        self.transformer1 = TransformerBlock(32, self.transformer_dim, self.n_neighbor)
        self.transition_downs = nn.ModuleList()
        self.transformers = nn.ModuleList()
        for i in range(4):
            channel = 32 * 2 ** (i + 1)
            self.transition_downs.append(
                TransitionDown(self.n_points // 4 ** (i + 1), self.n_neighbor, [channel // 2 + 3, channel, channel]))
            self.transformers.append(TransformerBlock(channel, self.transformer_dim, self.n_neighbor))

    def forward(self, x):
        # N*5
        xyz = x[..., :3]
        points = self.transformer1(xyz, self.fc1(x))[0]

        for i in range(self.n_blocks):
            xyz, points = self.transition_downs[i](xyz, points)
            points = self.transformers[i](xyz, points)[0]

        return points


class PosePointTransformer(nn.Module):
    def __init__(self, size_h=720, size_w=1280, num_joints=13, num_points=400):
        super().__init__()
        self.backbone = Backbone(num_points)
        self.sizeH = size_h
        self.sizeW = size_w
        self.num_joints = num_joints
        nblocks = 4

        self.fc2 = nn.Sequential(
            nn.Linear(32 * 2 ** nblocks, 256),
            nn.ReLU(),
            nn.Linear(256, self.num_joints * 128),
            nn.ReLU(),
        )

        self.mlp_head_x = nn.Linear(128, self.sizeW)
        self.mlp_head_y = nn.Linear(128, self.sizeH)

    def forward(self, x):
        batch_size = x.size(0)

        points = self.backbone(x)
        # print(points.shape)
        res = self.fc2(points.mean(1))
        res = res.view(batch_size, 13, -1)
        pred_x = self.mlp_head_x(res)
        pred_y = self.mlp_head_y(res)

        return pred_x, pred_y


class PoseTrans(nn.Module):
    def __init__(self, size_h=720, size_w=1280, num_joints=13, num_points=400):
        super().__init__()
        self.backbone = Backbone(num_points)
        self.sizeH = size_h
        self.sizeW = size_w
        self.num_joints = num_joints
        nblocks = 4

        self.transformer = PoseTransBlock(img_size=(720, 1280), num_joints=13, dim=512, depth=2, heads=2, mlp_dim=512)

    def forward(self, x):
        # print(x.shape)
        # index = x[:, -1, :]
        
        points1 = self.backbone(x[:, 0, :, :])
        points2 = self.backbone(x[:, 1, :, :])
        points3 = self.backbone(x[:, 2, :, :])
        points4 = self.backbone(x[:, 3, :, :])
        # print(points4.shape)

        data = torch.cat((points1, points2, points3, points4), dim=1)
        pred_x, pred_y = self.transformer(data)

        return pred_x, pred_y


if __name__ == "__main__":
    # test()
    # load_data = RasEventCloud((4, 720, 1280))
    # events =load_data.convert(data)
    data = torch.ones((4, 4, 1000, 5))
    model = PoseTrans(num_points=400)
    model(data)

