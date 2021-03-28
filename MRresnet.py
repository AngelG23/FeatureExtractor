import torch
import torch.nn as nn
import torch.nn.functional as F
from utils2 import PreActBlock_MR

class PreActResNet_MR(nn.Module):
    def __init__(self, block, num_classes=10):
        super(PreActResNet_MR, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)

        self.layer1_left = self.make_layer(block, 16, 16)
        self.layer1_right = self.make_layer(block, 16, 16)

        self.shortcut_conv1 = nn.Conv2d(32, 32, kernel_size=1, stride=2, padding=0, bias=False)
        self.layer20_left = self.make_layer(block, 16, 32)
        self.layer20_right = self.make_layer(block, 16, 32)
        self.layer21_left = self.make_layer(block, 32, 32)
        self.layer21_right = self.make_layer(block, 32, 32)

        self.shortcut_conv2 = nn.Conv2d(64, 64, kernel_size=1, stride=2, padding=0, bias=False)
        self.layer30_left = self.make_layer(block, 32, 64)
        self.layer30_right = self.make_layer(block, 32, 64)
        self.layer31_left = self.make_layer(block, 64, 64)
        self.layer31_right = self.make_layer(block, 64, 64)

#         self.shortcut_conv3 = nn.Conv2d(512, 512, kernel_size=1, stride=2, padding=0, bias=False)
#         self.layer40_left = self.make_layer(block, 256, 512)
#         self.layer40_right = self.make_layer(block, 256, 512)
#         self.layer41_left = self.make_layer(block, 512, 512)
#         self.layer41_right = self.make_layer(block, 512, 512)
        
        self.bn4 = nn.BatchNorm2d(64)
        self.fc = nn.Linear(64, num_classes, bias=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.bn5 = nn.BatchNorm1d(num_classes)
        self.ls = nn.LogSoftmax(dim=-1)

    def make_layer(self, block, in_planes, out_planes):
        layer = block(in_planes, out_planes)
        return nn.Sequential(layer)

    def forward(self, x):
        x = self.conv1(x)

        shortcut = x
        left_out = self.layer1_left(x)
        right_out = self.layer1_right(x)
        left_out = left_out + shortcut
        right_out = right_out + shortcut

        shortcut = torch.cat([left_out, right_out], dim=1)
        shortcut = self.shortcut_conv1(shortcut)
        left_out = self.layer20_left(left_out)
        right_out = self.layer20_right(right_out)
        left_out = left_out + shortcut
        right_out = right_out + shortcut
        shortcut = (left_out + right_out) * 1/2
        left_out = self.layer21_left(left_out)
        right_out = self.layer21_right(right_out)
        left_out = left_out + shortcut
        right_out = right_out + shortcut

        shortcut = torch.cat([left_out, right_out], dim=1)
        shortcut = self.shortcut_conv2(shortcut)
        left_out = self.layer30_left(left_out)
        right_out = self.layer30_right(right_out)
        left_out = left_out + shortcut
        right_out = right_out + shortcut
        shortcut = (left_out + right_out) * 1/2
        left_out = self.layer31_left(left_out)
        right_out = self.layer31_right(right_out)
        left_out = left_out + shortcut
        right_out = right_out + shortcut

#         shortcut = torch.cat([left_out, right_out], dim=1)
#         shortcut = self.shortcut_conv3(shortcut)
#         left_out = self.layer40_left(left_out)
#         right_out = self.layer40_right(right_out)
#         left_out = left_out + shortcut
#         right_out = right_out + shortcut
#         shortcut = (left_out + right_out) * 1/2
#         left_out = self.layer41_left(left_out)
#         right_out = self.layer41_right(right_out)
#         left_out = left_out + shortcut
#         right_out = right_out + shortcut

        out = left_out + right_out + shortcut

       
        out = self.bn4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        out = self.bn5(out)
        out = self.ls(out)
        return out

def PreActResNet21_MR():
    return PreActResNet_MR(PreActBlock_MR)