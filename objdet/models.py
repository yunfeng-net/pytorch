import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from torchvision import models
from blocks import *

class YOLO(nn.Module):

    def __init__(self, classes):
        super().__init__()
        vgg16 = models.vgg16_bn(pretrained=True)
        self.features = vgg16.features[:-1]
        for m in self.modules():
            fix_module(m)

        self.detect = nn.Sequential(
            #nn.Flatten(),
            nn.Linear(in_features=7*7*1024, out_features=4096),
            nn.Dropout(),
            nn.LeakyReLU(0.1),
            nn.Linear(in_features=4096, out_features=7 * 7 * (2 * 5 + classes))
        )

    def forward(self, x):
        y = self.features(x)
        return self.detect(F.flatten(y))

if __name__ == "__main__":
    input = torch.randn(4, 3, 160, 160)
    output = YOLO(2)(input)