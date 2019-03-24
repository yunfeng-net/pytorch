from __future__ import print_function
import argparse
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

class Reshape(nn.Module): 
    def __init__(self, *args): 
        super(Reshape, self).__init__() 
        self.shape = args 
    def forward(self, x): 
        x =  x.view((x.size(0),)+self.shape)
        return x

# For updating learning rate
def update_lr(optimizer, lr):    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_step = len(train_loader)
    curr_lr = args.lr
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print ("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}"
                .format(epoch,args. epochs, i+1, total_step, loss.item()))

        # Decay learning rate
        #if (epoch+1) % 30 == 0:
        #    curr_lr /= 3
        #    update_lr(optimizer, curr_lr)

def test(args, model, device, test_loader):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        t = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            prev_time = datetime.now()
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            cur_time = datetime.now()
            t += (cur_time - prev_time).microseconds/1e6
        print('Accuracy of the model on the test images: {} % time: {}'.format(100 * correct / total,t
        ))


def parse_args():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Example')
    parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=80, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--model', default="vgg",
                        help='model name')
    parser.add_argument('--load', 
                        help='model parameters file')
    return parser.parse_args()

