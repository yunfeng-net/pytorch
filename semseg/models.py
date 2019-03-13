import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from torchvision import models

def Conv2d(in_channels, out_channels,kernel_size=1,padding=0):
    return nn.Sequential(*[
            nn.Conv2d(in_channels, out_channels,kernel_size,padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ])
def Conv1x1(in_channels, out_channels,kernel_size=1,padding=0):
    return nn.Sequential(*[
            nn.Conv2d(in_channels, out_channels,kernel_size,padding=padding),
            #nn.BatchNorm2d(out_channels),
            #nn.ReLU(inplace=True)
        ])
def fix_pretrain(modules):
    for m in modules:
        if isinstance(m, nn.Conv2d):
            m.requires_grad = False

class FCNX(nn.Module):

    def __init__(self, classes):
        super().__init__()
        vgg16 = models.vgg16(pretrained=True) #.to(torch.device('cuda'))
        self.features = vgg16.features
        fix_pretrain(self.modules())
        idx = [[0,5],[5,10],[10,17],[17,24],[24,31]] # w/o max_pool
        self.feats = nn.ModuleList()
        for i in idx:
            self.feats.append(self.features[i[0]: i[1]])
        score_idx = [64, 128, 256, 512, 512]
        self.score_feats = nn.ModuleList()
        for i in score_idx:
            self.score_feats.append(Conv1x1(i, classes*2,3,1))
        self.gaus =nn.ModuleList()
        for i in range(4):
            self.gaus.append(Conv1x1(classes*2,classes*2))
        self.final = Conv2d(classes*2, classes, 3, 1)

    def forward(self, x):
        fs = []
        xs = []
        for i in range(len(self.feats)):
            x = self.feats[i](x)
            xs.append(x)
        for i in range(len(xs)):
            e = xs[4-i]
            f = self.score_feats[4-i](e)
            if i<=0:
                fx = f
            else:
                fx = F.interpolate(fx, scale_factor=2)
                y = F.max_pool2d(fx,kernel_size=fx.size()[2:])
                y = self.gaus[4-i-1](y)
                fx = fx + f*y
                #fx = fx +f
            #print("f{}: ".format(i),fs[-1].size())

        return self.final(F.interpolate(fx, scale_factor=2))

class SegNetX(nn.Module):

    def __init__(self, classes):
        super().__init__()
        vgg16 = models.vgg16(pretrained=True) #.to(torch.device('cuda'))
        self.features = vgg16.features
        fix_pretrain(self.modules())
        idx = [[0,4],[5,9],[10,16],[17,23],[24,30]] # w/o max_pool
        self.feats = nn.ModuleList()
        for i in idx:
            self.feats.append(self.features[i[0]: i[1]])
        score_idx = [64, 128, 256, 512, 512]
        self.score_feats = nn.ModuleList()
        for i in score_idx:
            self.score_feats.append(Conv1x1(i, classes*2))

        self.final = Conv2d(classes*10, classes, 3, 1)

    def forward(self, x):
        fs = []
        for i in range(len(self.feats)):
            x = self.feats[i](x)
            d,m = F.max_pool2d(x, kernel_size=2, stride=2, return_indices=True)
            e = F.max_unpool2d(d, m, kernel_size=2, stride=2, output_size=x.size())
            x = d
            f = self.score_feats[i](e)
            k = 1<<i
            if k>1:
                f = F.interpolate(f, scale_factor=k)
            fs.append(f)
            #print("f{}: ".format(i),fs[-1].size())

        return self.final(torch.cat(fs,1))

class PSPNetX(nn.Module):

    def __init__(self, classes):
        super().__init__()
        vgg16 = models.vgg16(pretrained=True)
        self.features = vgg16.features
        fix_pretrain(self.modules())
        idx = [[0,4],[5,9],[10,16],[17,23],[24,30]] # w/o max_pool
        self.feats = nn.ModuleList()
        for i in idx:
            self.feats.append(self.features[i[0]: i[1]])
        score_idx = [64, 128, 256, 512, 512]
        self.score_feats = nn.ModuleList()
        for i in score_idx:
            self.score_feats.append(Conv1x1(i, classes*2, 2, 1))
        self.final = Conv2d(classes*10, classes, 3, 1)

    def forward(self, x):
        fs = []
        for i in range(len(self.feats)):
            x = self.feats[i](x)
            d = F.max_pool2d(x, kernel_size=2, stride=2)
            e = F.max_pool2d(x, kernel_size=2, stride=1)
            x = d
            f = self.score_feats[i](e)
            k = 1<<i
            if k>1:
                f = F.interpolate(f, scale_factor=k)
            fs.append(f)
            #print("f{}: ".format(i),fs[-1].size())

        return self.final(torch.cat(fs,1))

if __name__ == "__main__":
    input = torch.randn(4, 3, 160, 160)
    output = SegNetX(2)(input)
    output2=PSPNetX(2)(input)
    output3 = FCNX(2)(input)
