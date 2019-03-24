from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

def dconv2d(in_channels, out_channels,kernel_size=1,padding=0,stride=1,att=0):
    return nn.Sequential(*[
            nn.Conv2d(in_channels, in_channels,kernel_size,stride=stride,padding=padding,groups=in_channels,bias=False),
            nn.Conv2d(in_channels, out_channels,1,bias=False),
            nn.BatchNorm2d(out_channels)
        ])
def conv2d(in_channels, out_channels,kernel_size=3,padding=1,stride=1,att=0,relu=1,groups=1):
    ls = [  nn.Conv2d(in_channels, out_channels,kernel_size,padding=padding,stride=stride,groups=groups),
            nn.BatchNorm2d(out_channels) ]
    if relu>0:
        ls.append(nn.ReLU(inplace=True))
    if att>0:
        ls.append(Attention(out_channels))
    return nn.Sequential(*ls)

def conv1x1(in_channels, out_channels,kernel_size=1,padding=0,stride=0,att=0,groups=1):
    return conv2d(in_channels, out_channels,kernel_size,padding,stride,att,0,groups)
def conv3x3(in_channels, out_channels,kernel_size=3,padding=1,stride=1,att=0,groups=1):
    return conv2d(in_channels, out_channels,kernel_size,padding,stride,att,0,groups)

# Attention
class Attention(nn.Module):
    def __init__(self, in_channels):
        super(Attention, self).__init__()
        self.fc1 = nn.Linear(in_features=in_channels, out_features=round(in_channels / 4))
        self.fc2 = nn.Linear(in_features=round(in_channels / 4), out_features=in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = F.avg_pool2d(x,x.size()[2:])
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        out = out.view(out.size(0), out.size(1), 1, 1)
        out = out * x
        return out

# Mobile block
class MobileBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, att=False, groups=1):
        super(MobileBlock, self).__init__()
        self.layer1 = dconv2d(in_channels, out_channels, 3, 1, stride=stride, att=att)
        self.layer2 = dconv2d(out_channels, out_channels, 3, 1, att=att)
        self.layer3 = dconv2d(out_channels, out_channels, 3, 1, att=att) 
        self.layer4 = dconv2d(out_channels, out_channels, 3, 1, att=att)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.layer1(x)
        out = self.relu(out)
        out = self.layer2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        residual = out
        out = self.layer3(out)
        out = self.relu(out)
        out = self.layer4(out)
        out += residual
        out = self.relu(out)
        return out 

# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, att=False, groups=1):
        super(ResidualBlock, self).__init__()
        self.layer1 = conv2d(in_channels, out_channels, stride=stride, att=att, groups=groups)
        self.layer2 = conv3x3(out_channels, out_channels, att=att, groups=groups)
        self.layer3 = conv2d(out_channels, out_channels, att=att, groups=groups) 
        self.layer4 = conv3x3(out_channels, out_channels, att=att, groups=groups)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.layer1(x)
        out = self.layer2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        residual = out
        out = self.layer3(out)
        out = self.layer4(out)
        out += residual
        out = self.relu(out)
        return out 

# Dense block
class DenseBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, att=False,groups=1):
        super(DenseBlock, self).__init__()
        self.layer1 = conv2d(in_channels, out_channels, stride=stride, att=att)
        self.layer2 = conv2d(out_channels, out_channels, att=att)
        self.layer3 = conv2d(out_channels*2, out_channels, att=att) 
        self.layer4 = conv2d(out_channels*3, out_channels, att=att)

    def forward(self, x):
        out = self.layer1(x)
        out2 = self.layer2(out)
        out3 = torch.cat([out,out2],1)
        out4 = self.layer3(out3)
        out4 = torch.cat([out3,out4],1)
        out = self.layer4(out4)
        return out 
