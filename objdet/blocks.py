import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from torchvision import models

def Conv2d(in_channels, out_channels,kernel_size=1,padding=0,dilation=1,stride=1):
    return nn.Sequential(*[
            nn.Conv2d(in_channels, out_channels,kernel_size,padding=padding,bias=False,dilation=dilation,stride=stride), #
            nn.BatchNorm2d(out_channels),
            nn.PReLU() #inplace=True)
        ])
def Conv1x1(in_channels, out_channels,kernel_size=1,padding=0):
    return nn.Sequential(*[
            nn.Conv2d(in_channels, out_channels,kernel_size,padding=padding),
            nn.BatchNorm2d(out_channels),
            #nn.ReLU(inplace=True)
        ])

def init_weights(modules):
    for m in modules:
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)
def fix_module(m, grad=False):
    for param in m.parameters():
        param.requires_grad = grad
def get_pretrain_vgg(name):
    if name=='16':
        vgg = models.vgg16(pretrained=True)
    elif name=='19':
        vgg = models.vgg19(pretrained=True)
    else:
        return None
    for i,m in enumerate(vgg.modules()):
        if isinstance(m, nn.Conv2d):
            fix_module(m)
    index = nn.ModuleList()
    feats = []
    for f in vgg.features:
        feats.append(f)
        if isinstance(f,nn.MaxPool2d):
            index.append(nn.Sequential(*feats))
            feats = []
    for i in range(0):
        j = 0
        for m in index[4-i].modules():
            if isinstance(m, nn.Conv2d):
                if j>=0:
                    fix_module(m, True)
                j += 1
    channels = [64, 128, 256, 512, 512]
    return index, channels

def get_pretrain_resnet(name):
    if name=='18':
        resnet = models.resnet18(pretrained=True)
    elif name=='34':
        resnet = models.resnet34(pretrained=True)
    elif name=='50':
        resnet = models.resnet50(pretrained=True)
    else:
        return None
    index = nn.ModuleList()
    a = [resnet.conv1,resnet.bn1,resnet.relu,resnet.maxpool]
    for m in resnet.modules():
        if isinstance(m, nn.Conv2d):
            fix_module(m)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    index.append(nn.Sequential(*a))
    index.append(resnet.layer1)
    index.append(resnet.layer2)
    index.append(resnet.layer3)
    index.append(resnet.layer4)
    channels = [64, 64*resnet.layer1[0].expansion, 128*resnet.layer2[0].expansion, 256*resnet.layer3[0].expansion, 512*resnet.layer4[0].expansion]
    return index, channels

def get_pretrain(name):
    if name[0:3]=='vgg':
        return get_pretrain_vgg(name[3:5])
    if name[0:6]=='resnet':
        return get_pretrain_resnet(name[6:8])
    return None

class UpConv(nn.Module):

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()
        if stride>1:
            self.conv1=nn.ConvTranspose2d(inplanes,planes,3,stride=2,output_padding=1,padding=1)
            self.upsample = Conv1x1(inplanes, planes)
        else:
            self.conv1 = Conv2d(inplanes, planes)
            self.upsample = None
        self.conv2 = Conv1x1(planes, planes, 3, 1)
        init_weights(self.modules())

    def forward(self, x):
        idx = x
        if self.upsample is not None:
            idx = F.interpolate(idx, scale_factor=2)
            idx = self.upsample(idx)
        x = self.conv1(x)
        x = self.conv2(x)
        return F.relu(x+idx)

# Dense block
class DenseBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(DenseBlock, self).__init__()
        self.layer1 = Conv2d(in_channels, out_channels, stride=stride)
        self.layer2 = Conv2d(out_channels, out_channels)
        self.layer3 = Conv2d(out_channels*2, out_channels) 
        self.layer4 = Conv2d(out_channels*3, out_channels)

    def forward(self, x):
        out = self.layer1(x)
        out2 = self.layer2(out)
        out3 = torch.cat([out,out2],1)
        out4 = self.layer3(out3)
        out4 = torch.cat([out3,out4],1)
        out = self.layer4(out4)
        return out

class Dilation(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Dilation, self).__init__()
        self.layer1 = Conv2d(in_channels, in_channels)
        self.layer2 = Conv2d(in_channels, in_channels, dilation=2)
        self.layer3 = Conv2d(in_channels, in_channels, dilation=3) 
        self.final = Conv1x1(in_channels*3, out_channels)

    def forward(self, x):
        out = self.layer1(x)
        out2 = self.layer2(x)
        out3 = self.layer3(x)
        #print(out.size(),out2.size(),out3.size())
        outx = torch.cat([out, out2, out3],1)
        return self.final(outx)