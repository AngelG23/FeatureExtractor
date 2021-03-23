import torch
import torch.nn as nn

affine_par = True


def conv(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class MonoDepth(nn.Module):
    def __init__(self, pyramid=[3, 6, 12]):
        self.pyramid = pyramid
        super(MonoDepth, self).__init__()
        self.conv1 = conv(3, 64)
        self.relu1 = nn.ReLU(inplace=False)

        self.block1_up = self._make_block(64)
        self.block2_up = self._make_block(128)
        self.block3_up = self._make_block(256)
        self.block4_up = self._make_block(512)
        self.block5_up = self._make_block(1024)
        self.block6_up = self._make_block(2048)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.fc = nn.Linear(in_features=64, out_features=10, bias=True)
        self.SoftMax = nn.LogSoftmax(dim=-1)

    def _make_block(self, filters, dilation=2):

        layers = [conv(filters, filters), nn.ReLU(inplace=False)]

        for i in range(7):
            layers.append(conv(filters, filters, dilation))
            layers.append(nn.ReLU(inplace=False))

        layers.append(conv(filters, filters*2))
        layers.append(nn.ReLU(inplace=False))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu1(self.conv1(x))

        av0 = (x + x)/2

        b1 = self.block1_up(av0)
        b2 = self.block1_up(av0)
        
        av0 = av0.view(av0.size(0), -1)

        add1 = b1 + av0
        add2 = b2 + av0

        b1 = self.block2_up(add1)
        b2 = self.block2_up(add2)

        av1 = (add1 + add2)/2

        add1 = b1 + av1
        add2 = b2 + av1

        b1 = self.block3_up(add1)
        b2 = self.block3_up(add2)

        av1 = (add1 + add2) / 2

        add1 = b1 + av1
        add2 = b2 + av1

        b1 = self.block4_up(add1)
        b2 = self.block4_up(add2)

        av1 = (add1 + add2) / 2

        add1 = b1 + av1
        add2 = b2 + av1

        b1 = self.block5_up(add1)
        b2 = self.block5_up(add2)

        av1 = (add1 + add2) / 2

        add1 = b1 + av1
        add2 = b2 + av1

        b1 = self.block6_up(add1)
        b2 = self.block6_up(add2)

        av1 = (add1 + add2) / 2

        x = (b1 + b2 + av1)/3
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.SoftMax(x)

        return x


class MonoDepth80(nn.Module):

    def __init__(self, pretrained=True):
        super().__init__()
        self.model = MonoDepth()

        if pretrained:
            saved_state_dict = torch.load(
                '/mnt/volume/dorn_trilobyte/datasets/KITTI/depth_prediction/pretrained/mono_depth.tar',
                map_location="cpu")
            new_params = self.model.state_dict().copy()
            for i in saved_state_dict:
                i_parts = i.split('.')
                if not i_parts[0] == 'fc':
                    new_params['.'.join(i_parts[0:])] = saved_state_dict[i]

            self.model.load_state_dict(new_params)

    def forward(self, input):
        return self.model(input)
