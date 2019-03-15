from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

class Inception_base(nn.Module):
    def __init__(self, depth_dim, input_size, config):
        super(Inception_base, self).__init__()

        self.depth_dim = depth_dim

        #mixed 'name'_1x1
        self.conv1 = nn.Conv2d(input_size, out_channels=config[0][0], kernel_size=1, stride=1, padding=0)

        #mixed 'name'_3x3_bottleneck
        self.conv3_1 = nn.Conv2d(input_size, out_channels=config[1][0], kernel_size=1, stride=1, padding=0)
        #mixed 'name'_3x3
        self.conv3_3 = nn.Conv2d(config[1][0], config[1][1], kernel_size=3, stride=1, padding=1)

        # mixed 'name'_5x5_bottleneck
        self.conv5_1 = nn.Conv2d(input_size, out_channels=config[2][0], kernel_size=1, stride=1, padding=0)
        # mixed 'name'_5x5
        self.conv5_5 = nn.Conv2d(config[2][0], config[2][1], kernel_size=5, stride=1, padding=2)

        self.max_pool_1 = nn.MaxPool2d(kernel_size=config[3][0], stride=1, padding=1)
        #mixed 'name'_pool_reduce
        self.conv_max_1 = nn.Conv2d(input_size, out_channels=config[3][1], kernel_size=1, stride=1, padding=0)

        self.apply(layer_init)

    def forward(self, input):
        #print(input.size())
        output1 = F.relu(self.conv1(input))

        output2 = F.relu(self.conv3_1(input))
        output2 = F.relu(self.conv3_3(output2))

        output3 = F.relu(self.conv5_1(input))
        output3 = F.relu(self.conv5_5(output3))

        output4 = F.relu(self.conv_max_1(self.max_pool_1(input)))

        return torch.cat([output1, output2, output3, output4], dim=self.depth_dim)
# 3x3 convolution
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     stride=stride, padding=1, bias=False)
def layer_init(m):
    classname = m.__class__.__name__
    classname = classname.lower()
    if classname.find('conv') != -1 or classname.find('linear') != -1:
        gain = nn.init.calculate_gain(classname)
        nn.init.xavier_uniform(m.weight, gain=gain)
        if m.bias is not None:
            nn.init.constant(m.bias, 0)
    elif classname.find('batchnorm') != -1:
        nn.init.constant(m.weight, 1)
        if m.bias is not None:
            nn.init.constant(m.bias, 0)
    elif classname.find('embedding') != -1:
        # The default initializer in the TensorFlow embedding layer is a truncated normal with mean 0 and
        # standard deviation 1/sqrt(sparse_id_column.length). Here we use a normal truncated with 3 std dev
        num_columns = m.weight.size(1)
        sigma = 1/(num_columns**0.5)
        m.weight.data.normal_(0, sigma).clamp_(-3*sigma, 3*sigma)

# weights available at t https://github.com/antspy/inception_v1.pytorch
class Inception_v1(nn.Module):
    def __init__(self, num_classes=10):
        super(Inception_v1, self).__init__()

        #conv2d0
        self.conv1 = conv3x3(3, 6)
        self.max_pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.lrn1 = nn.BatchNorm2d(6)

        self.inception_3a = Inception_base(1, 6, [[16], [16,32], [8, 16], [3, 16]]) #3a
        self.inception_3b = Inception_base(1, 80, [[40], [32,48], [12, 16], [3, 16]]) #3b
        self.max_pool_inc3= nn.MaxPool2d(kernel_size=3, stride=2, padding=0)

        self.inception_5a = Inception_base(1, 120, [[40], [32,48], [12, 16], [3, 16]]) #5a
        self.inception_5b = Inception_base(1, 120, [[40], [32,48], [12, 16], [3, 16]]) #5b
        self.avg_pool5 = nn.AvgPool2d(kernel_size=3, stride=2, padding=0)

        self.dropout_layer = nn.Dropout(0.4)
        self.fc = nn.Linear(120*9, num_classes)

        self.apply(layer_init)

    def forward(self, input):

        output = self.max_pool1(F.relu(self.conv1(input)))
        output = self.lrn1(output)
        #print(output.size())
        output = self.inception_3a(output)
        output = self.inception_3b(output)
        output = self.max_pool_inc3(output)
        #print(output.size())

        output = self.inception_5a(output)
        output = self.inception_5b(output)
        output = self.avg_pool5(output)
        #print(output.size())

        output = output.view(-1, 120*9)

        if self.fc is not None:
            output = self.dropout_layer(output)
            output = self.fc(output)

        return output

model = Inception_v1()

if __name__ == '__main__':
    pass
