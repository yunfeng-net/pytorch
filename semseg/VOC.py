#!/usr/bin/env python

import os.path as osp

import numpy as np
import PIL.Image as Image
import scipy.io
import torch
from torch.utils import data
from matplotlib import pyplot as plt
import random
import cv2
from torch.utils.data import DataLoader

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

def crop_image(image, lbl = 0): 
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
    def __init__(self, root='/home/yunfeng/data/VOCdevkit/VOC2012', phase='train', transform=False):
        self.root = root
        self._transform = transform
        self.flip_rate = 0
        #if phase == 'train':
        #        self.flip_rate = 0.5

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
        #print(img_file)
        img =  cv2.imread(img_file)
        img = cv2.resize(img, (160, 160))
        #img = np.array(img, dtype=np.uint8)
        #rd = random.random() 
        #if rd < self.flip_rate:
        #    img   = np.fliplr(img)
        #assert img.shape[0]>96
        #img = crop_image(img)
        #assert img.shape[0]==96
        # load label
        label_file = data_file['label']
        #label = cv2.imread(label_file)
        label = Image.open(label_file)
        label =label.resize((160,160))
        label = np.array(label, dtype=np.int32)
        #if rd< self.flip_rate:
        #   label = np.fliplr(label)
        #label = crop_image(label,1)
        #lbl = -1*np.ones((label.shape[0], label.shape[1]), dtype=np.int64)
        #for i, color in enumerate(index2color):
        #    lbl[np.where(np.all(label == color, axis=2))] = i
        label[label >20] = 0
        lbl = label
        #if self._transform:
        img,lbl = self.transform(img, lbl)
        assert lbl.max()<21
        #print(index,label.size(),img.size())
        #assert lbl.size()[0]==img.size()[1] and lbl.size()[1]==img.size()[2] 
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
batch_size = 5
train_data = VOCClassSegBase(phase='train')
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=8)
test_data = VOCClassSegBase(phase='val')
test_dataloader = DataLoader(test_data, batch_size=batch_size, num_workers=8)
if __name__ == "__main__":
    sb = VOCClassSegBase(phase='train')
    for i in range(len(sb)):
        sample,label = sb[i]
        #assert sample['X'].size()[1]==96
        #print(i, sample['X'].size(), sample['Y'].size())
        #plt.figure()
        #show_batch(sample)
        #plt.axis('off')
        #plt.ioff()
        #plt.imshow(sample['l'])
        #plt.show()