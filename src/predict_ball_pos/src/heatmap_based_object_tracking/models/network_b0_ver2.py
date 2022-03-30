import math
from re import X
import sys
from pathlib import Path

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix()) 

import torch
import torch.nn as nn

from layers import conv_bn_act
from layers import SamePadConv2d
from layers import Flatten
from layers import SEModule
from layers import DropConnect

class Swish(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(x)

class MBConv(nn.Module):
    def __init__(self, in_, out_, expand,
                 kernel_size, stride, skip,
                 se_ratio, dc_ratio=0.2):
        super().__init__()
        mid_ = in_ * expand
        self.expand_conv = conv_bn_act(in_, mid_, kernel_size=1, bias=False) if expand != 1 else nn.Identity()

        self.depth_wise_conv = conv_bn_act(mid_, mid_,
                                           kernel_size=kernel_size, stride=stride,
                                           groups=mid_, bias=False)

        self.se = SEModule(mid_, int(in_ * se_ratio)) if se_ratio > 0 else nn.Identity()

        self.project_conv = nn.Sequential(
            SamePadConv2d(mid_, out_, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_, 1e-3, 0.01)
        )

        # if _block_args.id_skip:
        # and all(s == 1 for s in self._block_args.strides)
        # and self._block_args.input_filters == self._block_args.output_filters:
        self.skip = skip and (stride == 1) and (in_ == out_)

        # DropConnect
        # self.dropconnect = DropConnect(dc_ratio) if dc_ratio > 0 else nn.Identity()
        # Original TF Repo not using drop_rate
        # https://github.com/tensorflow/tpu/blob/05f7b15cdf0ae36bac84beb4aef0a09983ce8f66/models/official/efficientnet/efficientnet_model.py#L408
        self.dropconnect = nn.Identity()

    def forward(self, inputs):
        expand = self.expand_conv(inputs)
        x = self.depth_wise_conv(expand)
        x = self.se(x)
        x = self.project_conv(x)
        if self.skip:
            x = self.dropconnect(x)
            x = x + inputs
        return x


class MBBlock(nn.Module):
    def __init__(self, in_, out_, expand, kernel, stride, num_repeat, skip, se_ratio, drop_connect_ratio=0.2):
        super().__init__()
        layers = [MBConv(in_, out_, expand, kernel, stride, skip, se_ratio, drop_connect_ratio)]
        for i in range(1, num_repeat):
            layers.append(MBConv(out_, out_, expand, kernel, 1, skip, se_ratio, drop_connect_ratio))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class EfficientNet_b0(nn.Module):
    def __init__(self, width_coeff, depth_coeff,
                 depth_div=8, min_depth=None,
                 dropout_rate=0.3, drop_connect_rate=0.3):
                 
        super().__init__()
        min_depth = min_depth or depth_div
        
        def renew_ch(x):
            if not width_coeff:
                return x

            x *= width_coeff
            new_x = max(min_depth, int(x + depth_div / 2) // depth_div * depth_div)
            if new_x < 0.9 * x:
                new_x += depth_div
            return int(new_x)

        def renew_repeat(x):
            return int(math.ceil(x * depth_coeff))

        self.init = conv_bn_act(9, 3, kernel_size=3, stride=1, bias=False)

        self.stem = conv_bn_act(3, renew_ch(32), kernel_size=3, stride=2, bias=False)
        
        self.MBBlock_1 = MBBlock(renew_ch(32), renew_ch(16), 1, 3, 1, renew_repeat(1), True, 0.25, drop_connect_rate)
        self.MBBlock_2 = MBBlock(renew_ch(16), renew_ch(24), 6, 3, 2, renew_repeat(2), True, 0.25, drop_connect_rate)
        self.MBBlock_3 = MBBlock(renew_ch(24), renew_ch(40), 6, 5, 2, renew_repeat(2), True, 0.25, drop_connect_rate)
        self.MBBlock_4 = MBBlock(renew_ch(40), renew_ch(80), 6, 3, 2, renew_repeat(3), True, 0.25, drop_connect_rate)
        self.MBBlock_5 = MBBlock(renew_ch(80), renew_ch(112), 6, 5, 1, renew_repeat(3), True, 0.25, drop_connect_rate)
        self.MBBlock_6 = MBBlock(renew_ch(112), renew_ch(192), 6, 5, 2, renew_repeat(4), True, 0.25, drop_connect_rate)
        self.MBBlock_7 = MBBlock(renew_ch(192), renew_ch(320), 6, 3, 1, renew_repeat(1), True, 0.25, drop_connect_rate)

        self.Upsample = nn.Upsample(scale_factor=float(2), mode='bilinear', align_corners=False)

        self.Up_Block_1 = nn.Sequential(
                            nn.Conv2d(432, 216, 1, stride=1, bias=False),
                            nn.BatchNorm2d(216, momentum=0.99, eps=1e-3),
                            Swish(),

                            nn.Conv2d(216, 216, 3, stride=1, padding="same", bias=False),
                            nn.BatchNorm2d(216, momentum=0.99, eps=1e-3),
                            Swish(),

                            nn.Conv2d(216, 108, 3, stride=1, padding="same", bias=False),
                            nn.BatchNorm2d(108, momentum=0.99, eps=1e-3),
                            Swish()
                            )

        self.Up_Block_2 = nn.Sequential(
                            nn.Conv2d(148, 74, 1, stride=1, bias=False),
                            nn.BatchNorm2d(74, momentum=0.99, eps=1e-3),
                            Swish(),

                            nn.Conv2d(74, 74, 3, stride=1, padding="same", bias=False),
                            nn.BatchNorm2d(74, momentum=0.99, eps=1e-3),
                            Swish(),

                            nn.Conv2d(74, 37, 3, stride=1, padding="same", bias=False),
                            nn.BatchNorm2d(37, momentum=0.99, eps=1e-3),
                            Swish()
                            )

        self.Up_Block_3 = nn.Sequential(
                            nn.Conv2d(61, 30, 1, stride=1, bias=False),
                            nn.BatchNorm2d(30, momentum=0.99, eps=1e-3),
                            Swish(),

                            # nn.Conv2d(40, 40, 3, stride=1, padding="same", bias=False),
                            # nn.BatchNorm2d(40, momentum=0.99, eps=1e-3),
                            # Swish(),

                            nn.Conv2d(30, 15, 3, stride=1, padding="same", bias=False),
                            nn.BatchNorm2d(15, momentum=0.99, eps=1e-3),
                            Swish()
                            )

        self.Up_Block_4 = nn.Sequential(
                            nn.Conv2d(31, 16, 1, stride=1, bias=False),
                            nn.BatchNorm2d(16, momentum=0.99, eps=1e-3),
                            Swish(),


                            # nn.Conv2d(128, 64, 3, stride=1, padding="same", bias=False),
                            # nn.BatchNorm2d(64, momentum=0.99, eps=1e-3),
                            # Swish(),

                            nn.Conv2d(16, 16, 3, stride=1, padding="same", bias=False),
                            nn.BatchNorm2d(16, momentum=0.99, eps=1e-3),
                            Swish()
                            )

        self.Up_Block_5 = nn.Sequential(
                            nn.Conv2d(19, 10, 1, stride=1, bias=False),
                            nn.BatchNorm2d(10, momentum=0.99, eps=1e-3),
                            Swish(),

                            # nn.Conv2d(16, 16, 3, stride=1, padding="same", bias=False),
                            # nn.BatchNorm2d(16, momentum=0.99, eps=1e-3),
                            # Swish(),

                            # nn.Conv2d(16, 16, 3, stride=1, padding="same", bias=False),
                            # nn.BatchNorm2d(16, momentum=0.99, eps=1e-3),
                            # Swish(),

                            nn.Conv2d(10, 3, 3, stride=1, padding="same", bias=False),
                            nn.BatchNorm2d(3, momentum=0.99, eps=1e-3),
                            Swish(),

                            # nn.Conv2d(3, 3, 1, stride=1, bias=False),
                            nn.Conv2d(3, 1, 1, stride=1, bias=False)

                            )

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, SamePadConv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
            elif isinstance(m, nn.Linear):
                init_range = 1.0 / math.sqrt(m.weight.shape[1])
                nn.init.uniform_(m.weight, -init_range, init_range)

    def forward(self, inputs):

        x_in = self.init(inputs)

        stem = self.stem(x_in) #torch.Size([1, 40, 144, 256])

        x_1 = self.MBBlock_1(stem) #torch.Size([1, 16, 144, 256])
        x_2 = self.MBBlock_2(x_1)  #torch.Size([1, 24, 72, 128])
        x_3 = self.MBBlock_3(x_2)  #torch.Size([1, 40, 36, 64])
        x_4 = self.MBBlock_4(x_3)  #torch.Size([1, 80, 18, 32])
        x_5 = self.MBBlock_5(x_4)  #torch.Size([1, 112, 18, 32])
        x_6 = self.MBBlock_6(x_5)  #torch.Size([1, 192, 9, 16])
        x_7 = self.MBBlock_7(x_6)  #torch.Size([1, 320, 9, 16])

        x = self.Upsample(x_7)
        x = torch.cat((x, x_5), 1)
        x = self.Up_Block_1(x)

        x = self.Upsample(x)
        x = torch.cat((x, x_3), 1)
        x = self.Up_Block_2(x)

        x = self.Upsample(x)
        x = torch.cat((x, x_2), 1)
        x = self.Up_Block_3(x)

        x = self.Upsample(x)
        x = torch.cat((x, x_1), 1)
        x = self.Up_Block_4(x)

        x = self.Upsample(x)
        x = torch.cat((x, x_in), 1)
        x = self.Up_Block_5(x)
        
        x = torch.sigmoid(x)

        return x


if __name__ == "__main__":
    print("Efficient B0 Summary")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('GPU Use : ',torch.cuda.is_available())
    

    model = EfficientNet_b0(1, 1).to(device)
    epoch = 150
    time_list =[]
    model.eval()
    
    data = torch.randn(epoch, 1, 9, 288, 512).to(device)

    from torchsummary import summary
    summary(model.cuda(), (9, 288, 512))

    import time
    import numpy as np

    with torch.no_grad():
        for i in range(epoch):
            t0 = time.time()
            torch.cuda.synchronize()
            output = model(data[i])

            h_pred = (output * 255).cpu().numpy()

            torch.cuda.synchronize()
            h_pred = (h_pred[0]).astype('uint8')
            h_pred = np.asarray(h_pred).transpose(1, 2, 0)
            t1 = time.time()        

            #print(i)

            time_list.append(t1-t0)

    print("output_shape : ",output.size())
    print("avg time : ", np.mean(time_list))
    print("avg FPS : ", 1 / np.mean(time_list))




    """data = torch.randn(50, 1, 9, 288, 512)

    model = EfficientNet(1.2, 1.4)
    epoch = 50
    time_list =[]

    from torchsummary import summary
    summary(model, (9, 288, 512), device = 'cpu')

    def dfs_freeze(model):
        for name, child in model.named_children():
            if name not in ['Up_Block_1','Up_Block_2','Up_Block_3','Up_Block_4','Up_Block_5']:
                for param in child.parameters():
                    param.requires_grad_(False)

                dfs_freeze(child)

        return model

    if True:
        print("==================back bone freeze==================")
        model = dfs_freeze(model)


        summary(model, (9, 288, 512), device = 'cpu')"""
