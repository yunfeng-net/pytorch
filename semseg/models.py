import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from torchvision import models

class SegNet(nn.Module):

    def __init__(self, classes):
        super().__init__()
        vgg16 = models.vgg16(pretrained=True)
        features = vgg16.features
        self.dec1 = features[0: 4]
        self.dec2 = features[5: 9]
        self.dec3 = features[10: 16]
        self.dec4 = features[17: 23]
        self.dec5 = features[24: 30]

        #for m in self.modules():
        #    if isinstance(m, nn.Conv2d):
        #        m.requires_grad = False


        self.final = nn.Sequential(*[
            nn.Conv2d(classes*10, classes, 3, padding=1),
            #nn.BatchNorm2d(classes),
            nn.ReLU(inplace=True)
        ])
        self.g4 = nn.Sequential(*[
            nn.Conv2d(512, 256, 1)
        ])
        self.g3 = nn.Sequential(*[
            nn.Conv2d(256, 128, 1)
        ])
        self.g2 = nn.Sequential(*[
            nn.Conv2d(128, 64, 1)
        ])
        self.score_feat1 = nn.Conv2d(64, classes*2, 1)
        self.score_feat2 = nn.Conv2d(128, classes*2, 1)
        self.score_feat3 = nn.Conv2d(256, classes*2, 1)
        self.score_feat4 = nn.Conv2d(512, classes*2, 1)
        self.score_feat5 = nn.Conv2d(512, classes*2, 1)       

    def forward(self, x):
        x1 = self.dec1(x)
        d1, m1 = F.max_pool2d(x1, kernel_size=2, stride=2, return_indices=True)
        x2 = self.dec2(d1)
        d2, m2 = F.max_pool2d(x2, kernel_size=2, stride=2, return_indices=True)
        x3 = self.dec3(d2)
        d3, m3 = F.max_pool2d(x3, kernel_size=2, stride=2, return_indices=True)
        x4 = self.dec4(d3)
        d4, m4 = F.max_pool2d(x4, kernel_size=2, stride=2, return_indices=True)
        x5 = self.dec5(d4)
        d5, m5 = F.max_pool2d(x5, kernel_size=2, stride=2, return_indices=True)
        #print("d5 ",d5.size())

        f5 =  F.max_unpool2d(d5, m5, kernel_size=2, stride=2, output_size=x5.size())
        f5 = F.interpolate(self.score_feat5(f5), scale_factor=16)
        #print("c5 ",f5.size())
        f4 = F.max_unpool2d(d4, m4, kernel_size=2, stride=2, output_size=x4.size())
        f4 = F.interpolate(self.score_feat4(f4), scale_factor=8)
        #print("c4 ",f4.size())
        f3 = F.max_unpool2d(d3, m3, kernel_size=2, stride=2, output_size=x3.size())
        f3 = F.interpolate(self.score_feat3(f3), scale_factor=4)
        #print("c3 ",f3.size())
        f2 = F.max_unpool2d(d2, m2, kernel_size=2, stride=2, output_size=x2.size())
        f2 = F.interpolate(self.score_feat2(f2), scale_factor=2)
        #print("c2 ",f2.size())
        f1 = F.max_unpool2d(d1, m1, kernel_size=2, stride=2, output_size=x1.size())
        f1 = self.score_feat1(f1)
        #print("c1 ",f1.size())

        return self.final(torch.cat([f1,f2,f3,f4,f5],1))


class PSPNet(nn.Module):

    def __init__(self, classes):
        super().__init__()
        vgg16 = models.vgg16(pretrained=True)
        features = vgg16.features
        self.dec1 = features[0: 4]
        self.dec2 = features[5: 9]
        self.dec3 = features[10: 16]
        self.dec4 = features[17: 23]
        self.dec5 = features[24: 30]

        #for m in self.modules():
        #    if isinstance(m, nn.Conv2d):
        #        m.requires_grad = False


        self.final = nn.Sequential(*[
            nn.Conv2d(classes*10, classes, 3, padding=1),
            #nn.BatchNorm2d(classes),
            nn.ReLU(inplace=True)
        ])
        self.score_feat1 = nn.Conv2d(64, classes*2, 2, padding=1)
        self.score_feat2 = nn.Conv2d(128, classes*2, 2, padding=1)
        self.score_feat3 = nn.Conv2d(256, classes*2, 2, padding=1)
        self.score_feat4 = nn.Conv2d(512, classes*2, 2, padding=1)
        self.score_feat5 = nn.Conv2d(512, classes*2, 2, padding=1)       

    def forward(self, x):
        x1 = self.dec1(x)
        d1, m1 = F.max_pool2d(x1, kernel_size=2, stride=2, return_indices=True)
        e1 = F.max_pool2d(x1, kernel_size=2, stride=1)
        f1 = self.score_feat1(e1)
        #print("f1: ",f1.size())

        x2 = self.dec2(d1)
        d2, m2 = F.max_pool2d(x2, kernel_size=2, stride=2, return_indices=True)
        e2 = F.max_pool2d(x2, kernel_size=2, stride=1)
        f2 = F.interpolate(self.score_feat2(e2), scale_factor=2)
        #print("f2: ",f2.size())

        x3 = self.dec3(d2)
        d3, m3 = F.max_pool2d(x3, kernel_size=2, stride=2, return_indices=True)
        e3 = F.max_pool2d(x3, kernel_size=2, stride=1)
        f3 = F.interpolate(self.score_feat3(e3), scale_factor=4)
        #print("f3: ",f3.size())

        x4 = self.dec4(d3)
        d4, m4 = F.max_pool2d(x4, kernel_size=2, stride=2, return_indices=True)
        e4 = F.max_pool2d(x4, kernel_size=2, stride=1)
        f4 = F.interpolate(self.score_feat4(e4), scale_factor=8)
        #print("f4: ",f4.size())

        x5 = self.dec5(d4)
        e5 = F.max_pool2d(x5, kernel_size=2, stride=1)
        f5 = F.interpolate(self.score_feat5(e5), scale_factor=16)
        #print("f5 ",f5.size())

        return self.final(torch.cat([f1,f2,f3,f4,f5],1))

