import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from torchvision import models
from blocks import *
from box_utils import jaccard,point_form
import numpy as np

class YOLO(nn.Module):

    def __init__(self, classes, B = 1):
        super().__init__()
        vgg16 = models.vgg16_bn(pretrained=True)
        self.features = vgg16.features
        for m in self.modules():
            fix_module(m)
        self.grid = grid = 5
        self.b = B
        self.classes = classes

        self.detect = nn.Sequential(
            #nn.Flatten(),
            nn.Linear(in_features=grid*grid*512, out_features=6400),
            nn.Dropout(),
            nn.LeakyReLU(0.1),
            nn.Linear(in_features=6400, out_features=grid* grid * (B * 5 + classes))
        )

    def forward(self, x):
        y = self.features(x)
        z = y.view(y.size(0), -1)
        r = self.detect(z)
        return r

    def attr(self):
        return self.grid,self.b,self.classes

    def loss(self, n_batch, l_coord, l_noobj):
        return YoloLoss(self, n_batch, l_coord, l_noobj)
    

class YoloLoss(nn.Module):
    def __init__(self, yolo, n_batch, l_coord, l_noobj, use_gpu=True):
        """
        :param n_batch: number of batches
        :param B: number of bounding boxes
        :param C: number of bounding classes
        :param l_coord: factor for loss which contain objects
        :param l_noobj: factor for loss which do not contain objects
        """
        super(YoloLoss, self).__init__()
        self.n_batch = n_batch
        self.S, self.B, self.C = yolo.attr()
        self.l_coord = l_coord
        self.l_noobj = l_noobj
        self.use_gpu = use_gpu
        self.kind_loss = nn.CrossEntropyLoss()

    def encode(self,labels):
        '''
        labels list(tensor) [[x1,y1,x2,y2,class],[]]
        return SxSx(Bx5+C)
        '''

        grid_num = self.S
        target = torch.zeros((len(labels),grid_num,grid_num,self.B*5+self.C),device=torch.device('cuda' if self.use_gpu else 'cpu'))
        cell_size = 1./grid_num
        for j,data in enumerate(labels):
            boxes = data[:,:4]
            wh = boxes[:,2:]-boxes[:,:2]
            cxcy = (boxes[:,2:]+boxes[:,:2])/2
            for i in range(cxcy.size()[0]):
                cxcy_sample = cxcy[i]
                ij = (cxcy_sample/cell_size).ceil()-1 #
                for k in range(self.B):
                    target[j,int(ij[1]),int(ij[0]),4+self.S*k] = 1
                target[j,int(ij[1]),int(ij[0]),self.B*5] = int(data[i,4])
                xy = ij*cell_size 
                delta_xy = (cxcy_sample -xy)/cell_size
                for k in range(self.B):
                    s = self.S*k
                    target[j,int(ij[1]),int(ij[0]),s+2:s+4] = wh[i]
                    target[j,int(ij[1]),int(ij[0]),s:s+2] = delta_xy
        return target
    
    def forward(self, prediction, target):
        """
        :param prediction: Tensor [batch,SxSx(Bx5+C))]
        :param target: [batch,[bbox,...]]
        :return: total loss
        """
        n_elements = self.B * 5 + self.C
        target = self.encode(target) # Tensor [batch,SxSx(Bx5+C)]

        batch = target.size(0)
        target = target.view(batch,-1,n_elements)
        prediction = prediction.view(batch,-1,n_elements)

        # compute class loss
        mask = prediction.view(-1,n_elements)
        class_pred = mask[:,self.B*5:]
        mask2 = target.view(-1,n_elements)
        class_target = mask2[:,self.B*5]
        class_loss = self.kind_loss(class_pred, class_target.long())

        # compute location loss
        coord_mask = target[:,:,self.B*5] > 0
        coord_pred = prediction[coord_mask].view(-1,n_elements)
        box_pred = coord_pred[:,:self.B*5].contiguous().view(-1,5)
        coord_target = target[coord_mask].view(-1,n_elements)
        box_target = coord_target[:,:self.B*5].contiguous().view(-1,5)
        loc_loss = F.mse_loss(box_pred[:, :2], box_target[:, :2]) +\
                   F.mse_loss(box_pred[:, 2:4], box_target[:, 2:4])

        return class_loss+self.l_coord * loc_loss

    def post_process(self, prediction, target):
        """ prediction: [batch, -1, self.B * 5 + self.C ]
            target: [ [x,y,x2,y2,class] ]
            return [tp, fp, confidence, class]

            confidence = prob(class)*iou(bbox)
            tp = class==given and iou(bbox)>0.5 and max confidence
            fp = ~tp
        """

        n_elements = self.B * 5 + self.C
        batch = prediction.size(0)
        prediction = prediction.view(batch,-1,n_elements)
        #print(prediction.shape)
        pred = prediction[:,:,self.B*5:]
        output = torch.softmax(pred,2) 
        output_np = output.cpu().detach().numpy().copy()
        kind = np.argmax(output_np, axis=2) # class
        confidence = np.max(output_np, axis=2) # prob(class)
        gt = np.zeros(kind.shape)
        #order = np.argsort(confidence)
        bbox = prediction[:,:,:self.B*5]
        num = 0
        for i,bbs in enumerate(target): # batch
            num += len(bbs)
            for bb in bbs: # each box
                a = None
                #print(kind[i,:],bb)
                x = []
                for j,k in enumerate(kind[i,:]):
                    if k==int(bb[4]):
                        if a is None:
                            a = bbox[i,j,:4].cpu().detach().numpy().copy()
                            a = np.expand_dims(a,0)
                        else:
                            a = np.vstack((a,np.expand_dims(bbox[i,j,:4].cpu().detach().numpy().copy(),0)))
                        x.append(j)
                if a is not None:
                    a = np.hstack((a[:,:2] - a[:,2:]/2, a[:, :2] + a[:, 2:]/2))
                    #print(a) #, bbox[i,:,:][mask])
                    b = torch.Tensor(bb[:4])
                    gt_bbox = torch.unsqueeze(b,0)
                    #print(gt_bbox)
                    a = jaccard(torch.from_numpy(a), gt_bbox)
                    #print(a)
                    f = 0.5
                    j = -1
                    for k,d in enumerate(x):
                        if a[k,0]>f:
                            j = d
                            f = a[k,0]
                    if j>=0:
                        gt[i,j] = 1
        shape = (confidence.shape[0]*confidence.shape[1])
        return [np.reshape(confidence,shape),np.reshape(gt,shape),np.reshape(kind,shape),num]


if __name__ == "__main__":
    input = torch.randn(4, 3, 160, 160)
    net = YOLO(2)
    net.train()
    output = net(input)
