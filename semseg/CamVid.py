# -*- coding: utf-8 -*-

from __future__ import print_function

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import scipy.misc
import random
import os

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import utils

batch_size = 3
num_class = 32
means     = np.array([103.939, 116.779, 123.68]) / 255. # mean of three channels in the order of BGR
h, w      = 720, 960
train_h   = int(h * 2 / 3)  # 480
train_w   = int(w * 2 / 3)  # 640
val_h     = int(h/32) * 32  # 704
val_w     = w               # 960

def crop_image(image, lbl = 0): 
    h = ((image.shape[0] >>5)<<5)
    w = ((image.shape[1] >>5)<<5)
    # h,w = 320, 224
    h_top = int((image.shape[0] - h) / 2) 
    h_bottom = h_top + h
    w_left = int((image.shape[1] - w) / 2) 
    w_right = w_left + w
    if lbl==0:
        return image[h_top:h_bottom, w_left:w_right, :] 
    else:
        return image[h_top:h_bottom, w_left:w_right] 

class CamVidDataset(Dataset):

    def __init__(self, csv_file, phase, n_class=num_class, crop=True, flip_rate=0.5,root=""):
        self.data      = pd.read_csv(csv_file)
        self.means     = means
        self.n_class   = n_class
        self.root = root

        self.flip_rate = flip_rate
        self.crop      = crop
        if phase == 'train':
            self.new_h = train_h
            self.new_w = train_w
        elif phase == 'val':
            self.flip_rate = 0.
            self.crop = False
            self.new_h = val_h
            self.new_w = val_w


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name   = self.data.ix[idx, 0]
        img_file = os.path.join(self.root, img_name)
        img        = scipy.misc.imread(img_file, mode='RGB')
        label_name = self.data.ix[idx, 1]
        label_file = os.path.join(self.root, label_name)
        label      = np.load(label_file)

        if self.crop:
            h, w, _ = img.shape
            top   = random.randint(0, h - self.new_h)
            left  = random.randint(0, w - self.new_w)
            img   = img[top:top + self.new_h, left:left + self.new_w]
            label = label[top:top + self.new_h, left:left + self.new_w]

        if random.random() < self.flip_rate:
            img   = np.fliplr(img)
            label = np.fliplr(label)
        img = crop_image(img,0)
        label = crop_image(label,1)
        # reduce mean
        img = img[:, :, ::-1]  # switch to BGR
        img = np.transpose(img, (2, 0, 1)) / 255.
        img[0] -= self.means[0]
        img[1] -= self.means[1]
        img[2] -= self.means[2]

        # convert to tensor
        img = torch.from_numpy(img.copy()).float()
        label = torch.from_numpy(label.copy()).long()

        # create one-hot encoding
        #h, w = label.size()
        #target = torch.zeros(self.n_class, h, w).long()
        #for c in range(self.n_class):
        #    target[c][label == c] = 1
        #sample = {'X': img, 'l': label}

        return img,label


def show_batch(batch):
    img_batch = batch['X']
    img_batch[:,0,...].add_(means[0])
    img_batch[:,1,...].add_(means[1])
    img_batch[:,2,...].add_(means[2])
    batch_size = len(img_batch)

    grid = utils.make_grid(img_batch)
    plt.imshow(grid.numpy()[::-1].transpose((1, 2, 0)))

    plt.title('Batch from dataloader')

root_dir   = "/home/yunfeng/code/FCN-pytorch/"
train_file = os.path.join(root_dir, "CamVid/train.csv")
val_file   = os.path.join(root_dir, "CamVid/val.csv")

train_data = CamVidDataset(csv_file=train_file, phase='train',root=root_dir)
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=8)

test_data = CamVidDataset(csv_file=val_file, phase='val', flip_rate=0,root=root_dir)
test_dataloader = DataLoader(test_data, batch_size=1, num_workers=8)

if __name__ == "__main__":

    train_data = CamVidDataset(csv_file=train_file, phase='train',root=root_dir)

    # show a batch
    batch_size = 4
    for i in range(batch_size):
        sample = train_data[i]
        #print(i, sample['X'].size(), sample['Y'].size())
        plt.figure()
        show_batch(sample)
        plt.axis('off')
        plt.ioff()
        plt.show()
        plt.imshow(sample['l'])
        plt.show()



