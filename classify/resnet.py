from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from blocks import *


# ResNet
class ResNet(nn.Module):
    def __init__(self, block, num_classes=10, att=0, groups=1):
        super(ResNet, self).__init__()
        self.in_channels = 16*groups
        self.conv = conv3x3(3, 16*groups)
        self.layer1 = self.make_layer(block, 16*groups, att=att, groups=groups)
        self.layer2 = self.make_layer(block, 32*groups, 2, att=att, groups=groups)
        self.layer3 = self.make_layer(block, 64*groups, 2, att=att, groups=groups)
        self.layer4 = self.make_layer(block, 64*groups, 2, att=att, groups=groups)
        self.avg_pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64*groups, num_classes)
        
    def make_layer(self, block, out_channels, stride=1, att=0, groups=1):
        downsample = None
        n = self.in_channels
        self.in_channels = out_channels
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = conv3x3(n, out_channels, stride=stride)
        return block(n, out_channels, stride, downsample, att=att, groups=groups)
    
    def forward(self, x):
        out = self.conv(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

def init_model(name):
    if name=="resnet":
        return ResNet(ResidualBlock)
    elif name=="resnext":
        return ResNet(ResidualBlock, groups = 2)
    elif name=="densenet":
        return ResNet(DenseBlock)
    elif name=="senet":
        return ResNet(ResidualBlock, att = 1)
    elif name=="mobilenet":
        return ResNet(MobileBlock)
        
if __name__ == '__main__':
    pass
