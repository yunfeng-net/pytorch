#!/usr/bin/env python

import os.path as osp

import numpy as np
import PIL.Image as Image
import scipy.io
import torch
import torchvision
from torch.utils import data
from matplotlib import pyplot as plt
import random
import cv2
from torch.utils.data import DataLoader

uni_size = (160,160)
def set_uni_size():
     h = np.random.randint(5, size=1)
     uni_size = (96+h[0]*32, 96+h[0]*32)

def uint82bin(n, count=8):
    """returns the binary of integer n, count refers to amount of bits"""
    return ''.join([str((n >> y) & 1) for y in range(count - 1, -1, -1)])


def labelcolormap(N):
    cmap = np.zeros((N, 3), dtype=np.uint8)
    for i in range(N):
        r = 0
        g = 0
        b = 0
        id = i
        for j in range(7):
            str_id = uint82bin(id)
            r = r ^ (np.uint8(str_id[-1]) << (7 - j))
            g = g ^ (np.uint8(str_id[-2]) << (7 - j))
            b = b ^ (np.uint8(str_id[-3]) << (7 - j))
            id = id >> 3
        cmap[i, 0] = r
        cmap[i, 1] = g
        cmap[i, 2] = b
    return cmap


index2color = labelcolormap(21)
index2color = [list(hh) for hh in index2color]
index2name = ['background', 'airplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
              'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'potted plant',
              'sheep', 'sofa', 'train', 'tv/monitor']

def crop_image(image, flag, lbl = 0): 

    h_top = flag[0]*32
    h_bottom = h_top + 160
    w_left = flag[1]*32
    w_right = w_left + 160
    if lbl==0:
        return image[h_top:h_bottom, w_left:w_right, :] 
    else:
        return image[h_top:h_bottom, w_left:w_right] 

def crop_image2(image, lbl = 0): 
    h = ((image.shape[0] >>5)<<5)
    w = ((image.shape[1] >>5)<<5)
    #h,w = 96,96
    h_top = int((image.shape[0] - h) / 2) 
    h_bottom = h_top + h
    w_left = int((image.shape[1] - w) / 2) 
    w_right = w_left + w
    if lbl==0:
        return image[h_top:h_bottom, w_left:w_right, :] 
    else:
        return image[h_top:h_bottom, w_left:w_right] 


class VOCClassSegBase(data.Dataset):

    class_names = np.array([
        'background',
        'aeroplane',
        'bicycle',
        'bird',
        'boat',
        'bottle',
        'bus',
        'car',
        'cat',
        'chair',
        'cow',
        'diningtable',
        'dog',
        'horse',
        'motorbike',
        'person',
        'potted plant',
        'sheep',
        'sofa',
        'train',
        'tv/monitor',
    ])
    mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
    def __init__(self, root='/home/yunfeng/data/VOCdevkit/VOC2012', phase='train'):
        self.root = root
        self.flip_rate = 0
        if phase == 'train':
            self.flip_rate = 0.5

        # VOC2011 and others are subset of VOC2012
        dataset_dir = root
        self.files = []
        imgsets_file = osp.join(
            dataset_dir, 'ImageSets/Segmentation/%s.txt' % phase)
        for did in open(imgsets_file):
            did = did.strip()
            img_file = osp.join(dataset_dir, 'JPEGImages/%s.jpg' % did)
            label_file = osp.join(
                dataset_dir, 'SegmentationClass/%s.png' % did)
            self.files.append({
                'img': img_file,
                'label': label_file,
            })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        data_file = self.files[index]
        # load image
        img_file = data_file['img']
        img =  Image.open(img_file)
        us = (160,160)
        if self.flip_rate>0:
            us = uni_size
        img = img.resize(us)
        #img = np.array(img, dtype=np.uint8)
        # load label
        label_file = data_file['label']
        #label = cv2.imread(label_file)
        label = Image.open(label_file)
        label = label.resize(us)
        if self.flip_rate>0: # training
            #label = label.resize((5,5))
            #label = label.resize((160,160))
            pass
        rd = random.random() 
        if rd < self.flip_rate:
            img   = img.transpose(Image.FLIP_LEFT_RIGHT)
            label = label.transpose(Image.FLIP_LEFT_RIGHT)
        if self.flip_rate>0:
            p = torchvision.transforms.Pad(32)
            img = p(img)
            label = p(label)
        h = np.random.randint(3, size=2)
        img = np.array(img, dtype=np.uint8)
        label = np.array(label, dtype=np.int32)
        if self.flip_rate>0:
            img = crop_image(img,h)
            label = crop_image(label,h,1)
        label[label >20] = 0
        lbl = label
        img,lbl = self.transform(img, lbl)
        assert lbl.max()<21
        return img, lbl

    def transform(self, img, label):
        #img = img[:, :, ::-1]  # RGB -> BGR
        img = img.astype(np.float64)
        img -= self.mean_bgr
        img = img.transpose(2, 0, 1)/ 255
        img = torch.from_numpy(img).float()
        label = torch.from_numpy(label).long()
        return img, label

    def untransform(self, img, label):
        img = img.numpy()
        img = img.transpose(1, 2, 0)
        img += self.mean_bgr
        img = img.astype(np.uint8)
        img = img[:, :, ::-1]
        label = label.numpy()
        return img, label
num_class = 21
batch_size = 16
train_data = VOCClassSegBase(phase='train')
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=8)
test_data = VOCClassSegBase(phase='val')
test_dataloader = DataLoader(test_data, batch_size=batch_size, num_workers=8)
if __name__ == "__main__":
    sb = VOCClassSegBase(phase='train')
    for i in range(len(sb)):
        sample,label = sb[i]
        #print(sample.size(), label.size())
        #assert sample['X'].size()[1]==96
        #print(i, sample['X'].size(), sample['Y'].size())
        #plt.figure()
        #show_batch(sample)
        #plt.axis('off')
        #plt.ioff()
        #plt.imshow(sample['l'])
        #plt.show()