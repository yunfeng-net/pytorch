import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from torchvision import models
from blocks import *

class FCNX(nn.Module):

    def __init__(self, classes):
        super().__init__()
        self.feats, score_idx = get_pretrain('vgg16')
        self.score_feats = nn.ModuleList()
        N = classes*4*2
        M = classes*4
        for i in score_idx:
            self.score_feats.append(Conv2d(i, M, 3,1))
        self.gaus =nn.ModuleList()
        for i in range(4):
            self.gaus.append(Conv1x1(M,M))
        #self.upsample=nn.ConvTranspose2d(classes*4,classes,3,stride=2,output_padding=1,padding=1)
        #self.upsample = UpConv(N,classes, 2)
        self.block = DenseBlock(N,N)
        self.block2 = DenseBlock(N,N)
        #self.block = Dilation(N,N)
        #self.block2 = Dilation(N,N)
        self.final4 = Conv2d(N, M, 3,1)
        self.final3 = Conv2d(N, M, 3,1)
        self.final = Conv2d(N, classes,3,1)
        init_weights(self.score_feats)
        init_weights(self.final)
        init_weights(self.block.modules())
        init_weights(self.gaus)
        init_weights(self.final4)
        init_weights(self.final3)

    def forward(self, x):
        s = x.size()[2:]
        fs = []
        xs = []
        for i in range(len(self.feats)):
            x = self.feats[i](x)
            xs.append(x)
            if i>0:
                fs.append(self.score_feats[i](F.dropout2d(x)))

        #f = self.block(f)
        #f = self.block2(f)
        f5 = F.interpolate(fs[3], size=xs[3].size()[2:]) # layer 5 -> upsampling
        y = F.max_pool2d(f5,kernel_size=f5.size()[2:])
        f4 = fs[2] * self.gaus[3](y)

        fx = torch.cat([f4,f5],1)
        fx = self.final4(fx)
        fx = F.interpolate(fx, size=xs[2].size()[2:])
        y = F.max_pool2d(fx,kernel_size=fx.size()[2:])
        f3 = fs[1]*self.gaus[2](y)

        fx = torch.cat([f3,fx],1)
        fx = self.final3(fx)
        fx = F.interpolate(fx, size=xs[1].size()[2:])
        y = F.max_pool2d(fx,kernel_size=fx.size()[2:])
        f2 = fs[0]*self.gaus[1](y)
        fx = torch.cat([f2,fx],1)
        #fx = self.block(fx)
        fx = self.final(fx)
        return F.interpolate(fx, size=s)
        #return F.interpolate(self.final(self.upsample(f)), size=s)
        #for i in range(1): #len(xs)):
        #    e = xs[4-i]
        #    f = self.score_feats[4-i](F.dropout2d(e))
            #
        #    if i<=0:
        #        fx = f
        #    else:
        #        fx = F.interpolate(fx, scale_factor=2)
                #y = F.max_pool2d(fx,kernel_size=fx.size()[2:])
                #y = self.gaus[3](y)
                #fx = fx + f*y
        #        fx = fx +f
            #print("f{}: ".format(i),fs[-1].size())
        #fx = self.fc(F.dropout2d(fx))
        return F.interpolate(self.final(F.dropout2d(fx)), size=s)
        #return F.interpolate(self.final(self.upsample(fx)), size=s)
        #fx = self.upsample(fx)
        #return self.final(F.interpolate(fx,scale_factor=16))
class SegNetX(nn.Module):

    def __init__(self, classes):
        super().__init__()
        vgg16 = models.vgg16(pretrained=True) #.to(torch.device('cuda'))
        self.features = vgg16.features
        for m in self.modules():
            fix_module(m)
        idx = [[0,4],[5,9],[10,16],[17,23],[24,30]] # w/o max_pool
        self.feats = nn.ModuleList()
        for i in idx:
            self.feats.append(self.features[i[0]: i[1]])
        score_idx = [64, 128, 256, 512, 512]
        self.score_feats = nn.ModuleList()
        for i in range(len(score_idx)-1):
            self.score_feats.append(Conv2d(score_idx[i+1], score_idx[i],3,1))

        self.final = Conv2d(64, classes, 3, 1)

    def forward(self, x):
        ms = []
        ss = []
        for i in range(len(self.feats)):
            ss.append(x.size())
            x = self.feats[i](x)
            x,m = F.max_pool2d(x, kernel_size=2, stride=2, return_indices=True)
            ms.append(m)
        for i in range(len(self.feats)):
            #print(x.size(),ms[4-i].size(),ss[4-i])
            e = F.max_unpool2d(x, ms[4-i], kernel_size=2, stride=2, output_size=ss[4-i])
            if i<4:
                x = self.score_feats[3-i](x)
            else:
                x = self.final(x)
            x = F.interpolate(x, scale_factor=2)
            #k = 1<<i
            #if k>1:
            #    f = F.interpolate(f, scale_factor=k)
            #fs.append(f)
            #print("f{}: ".format(i),fs[-1].size())
        return x
        return self.final(torch.cat(fs,1))

class PSPNetX(nn.Module):

    def __init__(self, classes):
        super().__init__()
        vgg16 = models.vgg16(pretrained=True) #.to(torch.device('cuda'))
        self.features = vgg16.features
        for m in self.modules():
            fix_module(m)
        idx = [[0,4],[5,9],[10,16],[17,23],[24,30]] # w/o max_pool
        self.feats = nn.ModuleList()
        for i in idx:
            self.feats.append(self.features[i[0]: i[1]])
        score_idx = [64, 128, 256, 512, 512]
        self.score_feats = nn.ModuleList()
        for i in score_idx:
            self.score_feats.append(Conv2d(i, classes*2,2,1))
        self.final = Conv2d(classes*6, classes, 3, 1)

    def forward(self, x):
        fs = []
        for i in range(len(self.feats)):
            x = self.feats[i](x)
            #print(x.size())
            d = F.max_pool2d(x, kernel_size=2, stride=2)
            e = F.max_pool2d(x, kernel_size=2, stride=1)
            x = d
            f = self.score_feats[i](F.dropout2d(e))
            k = 1<<i
            if k>1:
                f = F.interpolate(f, scale_factor=k)
            #print(f.size())
            fs.append(f)
            #print("f{}: ".format(i),fs[-1].size())

        return self.final(torch.cat(fs[2:],1))

if __name__ == "__main__":
    input = torch.randn(4, 3, 160, 160)
    output = SegNetX(2)(input)
    output2=PSPNetX(2)(input)
    output3 = FCNX(2)(input)
