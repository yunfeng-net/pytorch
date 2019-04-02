import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from torchvision import models

def Conv2d(in_channels, out_channels,kernel_size=1,padding=0):
    return nn.Sequential(*[
            nn.Conv2d(in_channels, out_channels,kernel_size,padding=padding,bias=False), #
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
        for m in index[4-i].modules():
            if isinstance(m, nn.Conv2d):
                fix_module(m, True)
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

class FCNX(nn.Module):

    def __init__(self, classes):
        super().__init__()
        self.feats, score_idx = get_pretrain('vgg16')
        self.score_feats = nn.ModuleList()
        for i in score_idx:
            self.score_feats.append(Conv2d(i, classes*4, 3,1))
        #self.gaus =nn.ModuleList()
        #for i in range(4):
        #    self.gaus.append(Conv1x1(classes*4,classes*4))
        #self.upsample=nn.ConvTranspose2d(classes*4,classes,3,stride=2,output_padding=1,padding=1)
        self.upsample = UpConv(classes*4,classes, 2)
        self.final = Conv2d(classes, classes,3,1)
        init_weights(self.score_feats)
        init_weights(self.final)

    def forward(self, x):
        s = x.size()[2:]
        fs = []
        xs = []
        for i in range(len(self.feats)):
            x = self.feats[i](x)
            xs.append(x)
        for i in range(1): #len(xs)):
            e = xs[4-i]
            f = self.score_feats[4-i](F.dropout2d(e))
            #f = self.block(e)
            if i<=0:
                fx = f
            else:
                fx = F.interpolate(fx, scale_factor=2)
                y = F.max_pool2d(fx,kernel_size=fx.size()[2:])
                y = self.gaus[4-i-1](y)
                fx = fx + f*y
                #fx = fx +f
            #print("f{}: ".format(i),fs[-1].size())
        #fx = self.fc(F.dropout2d(fx))
        #return F.interpolate(self.final(F.dropout2d(fx)), size=s)
        return F.interpolate(self.final(self.upsample(fx)), size=s)
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
