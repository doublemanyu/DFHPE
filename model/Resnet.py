import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse


# 残差块
class CommonBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(CommonBlock, self).__init__()
        # self.channel = channel
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel,
                      out_channels=out_channel,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(out_channel),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=out_channel,
                      out_channels=out_channel,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(out_channel)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.relu(out)
        return out


class BottleNeck(nn.Module):
    def __init__(self, in_channel, out_channel, stride):
        super(BottleNeck, self).__init__()
        self.down_sample = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride[0], padding=0, bias=False),
            nn.BatchNorm2d(out_channel)
        )
        # self.down_sample = nn.Sequential(
        #     nn.MaxPool2d(kernel_size=2, stride=stride[0], padding=padding[-1]),
        #     nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0, bias=False),
        #     nn.BatchNorm2d(out_channel)
        # )
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride[0], padding=1, bias=False),
            nn.BatchNorm2d(out_channel))
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=stride[1], padding=1, bias=False),
            nn.BatchNorm2d(out_channel))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.down_sample(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out += identity
        out = self.relu(out)
        return out


class DeconvModule(nn.Module):
    def __init__(self):
        super(DeconvModule, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(
                    in_channels=512,
                    out_channels=256,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=(0, 1))
        self.deconv2 = nn.ConvTranspose2d(
                    in_channels=256,
                    out_channels=64,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(
                    in_channels=64,
                    out_channels=17,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1)

    def forward(self, x):
        out = self.deconv1(x)
        out = self.deconv2(out)
        out = self.deconv3(out)
        return out


# 残差网络
class ResNet(nn.Module):
    def __init__(self, args):
        super(ResNet, self).__init__()
        self.num_joints = args.num_joints
        self.sizeW = args.sensor_sizeW
        self.sizeH = args.sensor_sizeH
        self.in_channel = args.in_channel

        # 1280*720->640*360->320*180->320*180->160*90->80*45->40*23
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channel, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.conv2 = nn.Sequential(
            CommonBlock(64, 64),
            CommonBlock(64, 64)
        )
        self.conv3 = nn.Sequential(
            BottleNeck(64, 128, [2, 1]),
            CommonBlock(128, 128)
        )
        self.conv4 = nn.Sequential(
            BottleNeck(128, 256, [2, 1]),
            CommonBlock(256, 256)
        )
        self.conv5 = nn.Sequential(
            BottleNeck(256, 512, [2, 1]),
            CommonBlock(512, 512)
        )
        self.deconv = DeconvModule()
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        #
        # # # test
        # # self.fc2 = nn.Sequential(
        # #     nn.Linear(512, 256),
        # #     nn.ReLU(),
        # #     nn.Linear(256, self.num_joints * 128),
        # #     nn.ReLU(),
        # # )
        # #
        # # self.mlp_head_x = nn.Linear(128, self.sizeW)
        # # self.mlp_head_y = nn.Linear(128, self.sizeH)
        # self.out_conv1 = nn.Sequential(nn.Linear(256 * 2, 256, bias=False),
        #                                nn.BatchNorm1d(256),
        #                                nn.LeakyReLU(),
        #                                nn.Dropout(p=0.1))
        # self.out_conv2 = nn.Sequential(nn.Linear(256, self.num_joints * 128, bias=False),
        #                                nn.BatchNorm1d(self.num_joints * 128),
        #                                nn.LeakyReLU(),
        #                                nn.Dropout(p=0.1))
        #
        # self.mlp_head_x = nn.Linear(128, self.sizeW)
        # self.mlp_head_y = nn.Linear(128, self.sizeH)

    # def forward(self, x):
    #     out = self.conv1(x)
    #     out = self.conv5(self.conv4(self.conv3(self.conv2(out))))
    #     out = self.avgpool(out)
    #
    #     return out
    def forward(self, x):
        out = self.conv3(self.conv2(self.conv1(x)))
        out = self.conv5(self.conv4(out))
        out = self.deconv(out)
        return out
        # out = self.avgpool(out).view(x.shape[0], -1)
        # # out = self.fc2(out)
        # # print(out.shape)
        # out = self.out_conv1(out)
        # out = self.out_conv2(out)

        # out = out.view(x.shape[0], self.num_joints, -1)
        #
        # pred_x = self.mlp_head_x(out)
        # pred_y = self.mlp_head_y(out)
        # return pred_x, pred_y


if __name__ == "__main__":
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

    data = torch.ones((4, 2, 1280, 720))
    model = ResNet(args)
    out = model(data)
    print(out[0].shape)


