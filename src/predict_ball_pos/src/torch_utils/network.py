# import package

# model
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from torch import optim

# dataset and transformation
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import models
import os

# display images
from torchvision import utils
import matplotlib.pyplot as plt

# utils
import numpy as np
from torchsummary import summary
import time
import copy




# Swish activation function
class Swish(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(x)

    
# SE Block
class SEBlock(nn.Module):
    def __init__(self, in_channels, r=4):
        super().__init__()

        self.squeeze = nn.AdaptiveAvgPool2d((1,1))
        self.excitation = nn.Sequential(
            nn.Linear(in_channels, in_channels * r),
            Swish(),
            nn.Linear(in_channels * r, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.squeeze(x)
        x = x.view(x.size(0), -1)
        x = self.excitation(x)
        x = x.view(x.size(0), x.size(1), 1, 1)
        return x

class MBConv(nn.Module):
    expand = 6
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, se_scale=4, p=0.5):
        super().__init__()
        # first MBConv is not using stochastic depth
        self.p = torch.tensor(p).float() if (in_channels == out_channels) else torch.tensor(1).float()

        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * MBConv.expand, 1, stride=stride, padding=0, bias=False),
            nn.BatchNorm2d(in_channels * MBConv.expand, momentum=0.99, eps=1e-3),
            Swish(),
            nn.Conv2d(in_channels * MBConv.expand, in_channels * MBConv.expand, kernel_size=kernel_size,
                      stride=1, padding=kernel_size//2, bias=False, groups=in_channels*MBConv.expand),
            nn.BatchNorm2d(in_channels * MBConv.expand, momentum=0.99, eps=1e-3),
            Swish()
        )

        self.se = SEBlock(in_channels * MBConv.expand, se_scale)

        self.project = nn.Sequential(
            nn.Conv2d(in_channels*MBConv.expand, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels, momentum=0.99, eps=1e-3)
        )

        self.shortcut = (stride == 1) and (in_channels == out_channels)

    def forward(self, x):
        # stochastic depth
        if self.training:
            if not torch.bernoulli(self.p):
                return x

        x_shortcut = x
        x_residual = self.residual(x)
        x_se = self.se(x_residual)

        x = x_se * x_residual
        x = self.project(x)

        if self.shortcut:
            x= x_shortcut + x

        return x
    
class SepConv(nn.Module):
    expand = 1
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, se_scale=4, p=0.5):
        super().__init__()
        # first SepConv is not using stochastic depth
        self.p = torch.tensor(p).float() if (in_channels == out_channels) else torch.tensor(1).float()

        self.residual = nn.Sequential(
            nn.Conv2d(in_channels * SepConv.expand, in_channels * SepConv.expand, kernel_size=kernel_size,
                      stride=1, padding=kernel_size//2, bias=False, groups=in_channels*SepConv.expand),
            nn.BatchNorm2d(in_channels * SepConv.expand, momentum=0.99, eps=1e-3),
            Swish()
        )

        self.se = SEBlock(in_channels * SepConv.expand, se_scale)

        self.project = nn.Sequential(
            nn.Conv2d(in_channels*SepConv.expand, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels, momentum=0.99, eps=1e-3)
        )

        self.shortcut = (stride == 1) and (in_channels == out_channels)

    def forward(self, x):
        # stochastic depth
        if self.training:
            if not torch.bernoulli(self.p):
                return x

        x_shortcut = x
        x_residual = self.residual(x)
        x_se = self.se(x_residual)

        x = x_se * x_residual
        x = self.project(x)

        if self.shortcut:
            x= x_shortcut + x

        return x



class EfficientNet(nn.Module):
    def __init__(self, num_classes=10, width_coef=1., depth_coef=1., scale=1., dropout=0.2, se_scale=4, stochastic_depth=False, p=0.5):
        super().__init__()
        channels = [32, 16, 24, 40, 80, 112, 192, 320, 1280]
        repeats = [1, 2, 2, 3, 3, 4, 1]
        strides = [1, 2, 2, 2, 1, 2, 1]
        kernel_size = [3, 3, 5, 3, 5, 5, 3]
        depth = depth_coef
        width = width_coef

        channels = [int(x*width) for x in channels]
        repeats = [int(x*depth) for x in repeats]

        # stochastic depth
        if stochastic_depth:
            self.p = p
            self.step = (1 - 0.5) / (sum(repeats) - 1)
        else:
            self.p = 1
            self.step = 0


        # efficient net
        

        self.stage1 = nn.Sequential(
            nn.Conv2d(9, channels[0],3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channels[0], momentum=0.99, eps=1e-3)
        )

        self.stage2 = self._make_Block(SepConv, repeats[0], channels[0], channels[1], kernel_size[0], strides[0], se_scale)

        self.stage3 = self._make_Block(MBConv, repeats[1], channels[1], channels[2], kernel_size[1], strides[1], se_scale)

        self.stage4 = self._make_Block(MBConv, repeats[2], channels[2], channels[3], kernel_size[2], strides[2], se_scale)

        self.stage5 = self._make_Block(MBConv, repeats[3], channels[3], channels[4], kernel_size[3], strides[3], se_scale)

        self.stage6 = self._make_Block(MBConv, repeats[4], channels[4], channels[5], kernel_size[4], strides[4], se_scale)

        self.stage7 = self._make_Block(MBConv, repeats[5], channels[5], channels[6], kernel_size[5], strides[5], se_scale)

        self.stage8 = self._make_Block(MBConv, repeats[6], channels[6], channels[7], kernel_size[6], strides[6], se_scale)

        self.stage9 = nn.Sequential(
            nn.Conv2d(channels[7], channels[8], 1, stride=1, bias=False),
            nn.BatchNorm2d(channels[8], momentum=0.99, eps=1e-3),
            Swish()
        ) 


        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(channels[8], num_classes)
        
        self.upsample = nn.Upsample(scale_factor=float(scale), mode='bilinear', align_corners=False)

        self.upsample2 = nn.Sequential(
                            nn.Upsample(scale_factor=float(2), mode='bilinear', align_corners=False),
                            nn.Conv2d(1536, 134, 1, stride=1, bias=False),
                            nn.BatchNorm2d(134, momentum=0.99, eps=1e-3),
                            Swish(),

                            nn.Conv2d(134, 134, 3, stride=1, padding="same", bias=False),
                            nn.BatchNorm2d(134, momentum=0.99, eps=1e-3),
                            Swish()

                            ) 


        self.upsample3 = nn.Sequential(
                            nn.Upsample(scale_factor=float(2), mode='bilinear', align_corners=False),
                            nn.Conv2d(134, 48, 1, stride=1, bias=False),
                            nn.BatchNorm2d(48, momentum=0.99, eps=1e-3),
                            Swish(),

                            nn.Conv2d(48, 48, 3, stride=1, padding="same", bias=False),
                            nn.BatchNorm2d(48, momentum=0.99, eps=1e-3),
                            Swish(),

                            nn.Conv2d(48, 48, 3, stride=1, padding="same", bias=False),
                            nn.BatchNorm2d(48, momentum=0.99, eps=1e-3),
                            Swish(),
                            ) 

        self.upsample4 = nn.Sequential(
                            nn.Upsample(scale_factor=float(2), mode='bilinear', align_corners=False),
                            nn.Conv2d(48, 28, 1, stride=1, bias=False),
                            nn.BatchNorm2d(28, momentum=0.99, eps=1e-3),
                            Swish(),

                            nn.Conv2d(28, 28, 3, stride=1, padding="same", bias=False),
                            nn.BatchNorm2d(28, momentum=0.99, eps=1e-3),
                            Swish(),

                            nn.Conv2d(28, 28, 3, stride=1, padding="same", bias=False),
                            nn.BatchNorm2d(28, momentum=0.99, eps=1e-3),
                            Swish()
                            ) 

        self.upsample5 = nn.Sequential(
                            nn.Upsample(scale_factor=float(2), mode='bilinear', align_corners=False),
                            nn.Conv2d(28, 19, 1, stride=1, bias=False),
                            nn.BatchNorm2d(19, momentum=0.99, eps=1e-3),
                            Swish(),
                            nn.Conv2d(19, 19, 3, stride=1, padding="same", bias=False),
                            nn.BatchNorm2d(19, momentum=0.99, eps=1e-3),
                            Swish(),

                            nn.Conv2d(19, 19, 3, stride=1, padding="same", bias=False),
                            nn.BatchNorm2d(19, momentum=0.99, eps=1e-3),
                            Swish()
                            ) 

        self.upsample6 = nn.Sequential(
                            nn.Upsample(scale_factor=float(2), mode='bilinear', align_corners=False),
                            nn.Conv2d(19, 64, 1, stride=1, bias=False),
                            nn.BatchNorm2d(64, momentum=0.99, eps=1e-3),
                            Swish(),

                            nn.Conv2d(64, 64, 3, stride=1, padding="same", bias=False),
                            nn.BatchNorm2d(64, momentum=0.99, eps=1e-3),
                            Swish(),

                            nn.Conv2d(64, 64, 3, stride=1, padding="same", bias=False),
                            nn.BatchNorm2d(64, momentum=0.99, eps=1e-3),
                            Swish(),

                            nn.Conv2d(64, 64, 3, stride=1, padding="same", bias=False),
                            nn.BatchNorm2d(64, momentum=0.99, eps=1e-3),
                            Swish(),

                            nn.Conv2d(64, 3, 1, stride=1, bias=False),
                            nn.Conv2d(3, 1, 1, stride=1, bias=False)
                            ) 
    def forward(self, x):
        #x = self.upsample(x)
        x_1 = self.stage1(x)
        x_2 = self.stage2(x_1)
        x_3 = self.stage3(x_2)
        x_4 = self.stage4(x_3)
        x = self.stage5(x_4)
        x_6 = self.stage6(x)
        x = self.stage7(x_6)
        x = self.stage8(x)
        x = self.stage9(x)

        x= self.upsample2(x)
        x= x_6 + x

        x= self.upsample3(x)
        x= x_4 + x

        x= self.upsample4(x)
        x= x_3 + x

        x= self.upsample5(x)
        x= x_2 + x
        
        x= self.upsample6(x)

        
        #x= self.upsample2(x)
        #x= self.upsample2(x)
        #x= self.upsample2(x)

        #x = self.avgpool(x)
        #x = x.view(x.size(0), -1)
        #x = self.dropout(x)
        #x = self.linear(x)
        x = torch.sigmoid(x)
        return x


    def _make_Block(self, block, repeats, in_channels, out_channels, kernel_size, stride, se_scale):
        strides = [stride] + [1] * (repeats - 1)
        layers = []
        for stride in strides:
            layers.append(block(in_channels, out_channels, kernel_size, stride, se_scale, self.p))
            in_channels = out_channels
            self.p -= self.step

        return nn.Sequential(*layers)


def efficientnet_b0(num_classes=10):
    return EfficientNet(num_classes=num_classes, width_coef=1.0, depth_coef=1.0, scale=1.0,dropout=0.2, se_scale=4)

def efficientnet_b1(num_classes=10):
    return EfficientNet(num_classes=num_classes, width_coef=1.0, depth_coef=1.1, scale=240/224, dropout=0.2, se_scale=4)

def efficientnet_b2(num_classes=10):
    return EfficientNet(num_classes=num_classes, width_coef=1.1, depth_coef=1.2, scale=260/224., dropout=0.3, se_scale=4)

def efficientnet_b3(num_classes=10):
    return EfficientNet(num_classes=num_classes, width_coef=1.2, depth_coef=1.4, scale=300/224, dropout=0.3, se_scale=4)
    #return EfficientNet(num_classes=num_classes, width_coef=1.2, depth_coef=1.4, scale=2, dropout=0.3, se_scale=4)

def efficientnet_b4(num_classes=10):
    return EfficientNet(num_classes=num_classes, width_coef=1.4, depth_coef=1.8, scale=380/224, dropout=0.4, se_scale=4)


def efficientnet_b5(num_classes=10):
    return EfficientNet(num_classes=num_classes, width_coef=1.6, depth_coef=2.2, scale=456/224, dropout=0.4, se_scale=4)

def efficientnet_b6(num_classes=10):
    return EfficientNet(num_classes=num_classes, width_coef=1.8, depth_coef=2.6, scale=528/224, dropout=0.5, se_scale=4)

def efficientnet_b7(num_classes=10):
    return EfficientNet(num_classes=num_classes, width_coef=2.0, depth_coef=3.1, scale=600/224, dropout=0.5, se_scale=4)


# check
if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('GPU Use : ',torch.cuda.is_available())

    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = torch.randn(1, 9, 288, 512).to(device)
    model = efficientnet_b3().to(device)

    time_list =[]

    epoch = 50

    summary(model, (9, 288, 512),device='cuda')
    
    with torch.no_grad():
        for i in range(epoch):
            torch.cuda.synchronize()
            t0 = time.time()
            output = model(data)
            torch.cuda.synchronize()
            t1 = time.time()

            #print(i)

            time_list.append(t1-t0)
            #print('output size:', output.size())
    print("output_shape : ",output.size())
    print("avg time : ", np.mean(time_list))
    print("avg FPS : ", 1 / np.mean(time_list))
