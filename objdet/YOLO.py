import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from torchvision import models
from blocks import *
from box_utils import jaccard,point_form

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
        self.blief_loss = nn.BCEWithLogitsLoss()

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
                target[j,int(ij[1]),int(ij[0]),10] = int(data[i,4])
                xy = ij*cell_size 
                delta_xy = (cxcy_sample -xy)/cell_size
                for k in range(self.B):
                    s = self.S*k
                    target[j,int(ij[1]),int(ij[0]),s+2:s+4] = wh[i]*grid_num
                    target[j,int(ij[1]),int(ij[0]),s:s+2] = delta_xy
        return target
    
    def forward(self, prediction, target):
        """
        :param prediction: Tensor [batch,SxSx(Bx5+C))]
        :param target: [batch,[bbox,...]]
        :return: total loss
        """
        n_elements = self.B * 5 + self.C
        target = self.encode(target) # Tensor [batch,SxSx(Bx5+20)]

        batch = target.size(0)
        target = target.view(batch,-1,n_elements)
        prediction = prediction.view(batch,-1,n_elements)
        coord_mask = target[:,:,4] > 0
        noobj_mask = target[:,:,4] == 0
        coord_mask = coord_mask.unsqueeze(-1).expand_as(target)
        noobj_mask = noobj_mask.unsqueeze(-1).expand_as(target)


        coord_pred = prediction[coord_mask].view(-1,n_elements)
        class_pred = coord_pred[:,self.B*5:]
        box_pred = coord_pred[:,:self.B*5].contiguous().view(-1,5)
        noobj_pred = prediction[noobj_mask].view(-1,n_elements)

        coord_target = target[coord_mask].view(-1,n_elements)
        class_target = coord_target[:,self.B*5]
        box_target = coord_target[:,:self.B*5].contiguous().view(-1,5)
        noobj_target = target[noobj_mask].view(-1,n_elements)

        # compute loss which do not contain objects
        noobj_target_mask = torch.zeros(noobj_target.size(),dtype=torch.bool)

        if self.use_gpu:
            noobj_target_mask.to(torch.device('cuda')) 
        #else:
        #    noobj_target_mask = torch.ByteTensor(noobj_target.size())
        for i in range(self.B):
            noobj_target_mask[:,i*5+4] = True
        noobj_target_c = noobj_target[noobj_target_mask] # only compute loss of c size [2*B*noobj_target.size(0)]
        noobj_pred_c = noobj_pred[noobj_target_mask]
        noobj_loss = F.mse_loss(noobj_pred_c, noobj_target_c, size_average=False)

        # compute loss which contain objects
        if self.use_gpu:
            coord_response_mask = torch.cuda.ByteTensor(box_target.size())
            coord_not_response_mask = torch.cuda.ByteTensor(box_target.size())
        else:
            coord_response_mask = torch.ByteTensor(box_target.size())
            coord_not_response_mask = torch.ByteTensor(box_target.size())
        coord_response_mask.zero_()
        coord_not_response_mask = ~coord_not_response_mask.zero_()
        for i in range(0,box_target.size()[0],self.B):
            box1 = box_pred[i:i+self.B]
            box2 = box_target[i:i+self.B]
            iou = jaccard(point_form(box1[:, :4]), point_form(box2[:, :4]))
            max_iou, max_index = iou.max(0)
            if self.use_gpu:
                max_index = max_index.data.cuda()
            else:
                max_index = max_index.data
            coord_response_mask[i+max_index]=1
            coord_not_response_mask[i+max_index]=0

        # 1. response loss
        box_pred_response = box_pred[coord_response_mask].view(-1, 5)
        box_target_response = box_target[coord_response_mask].view(-1, 5)
        #contain_loss = F.mse_loss(box_pred_response[:, 4], box_target_response[:, 4], size_average=False)
        contain_loss = self.blief_loss(box_pred_response[:, 4], box_target_response[:, 4])
        loc_loss = F.mse_loss(box_pred_response[:, :2], box_target_response[:, :2]) +\
                   F.mse_loss(box_pred_response[:, 2:4], box_target_response[:, 2:4])
        # 2. not response loss
        box_pred_not_response = box_pred[coord_not_response_mask].view(-1, 5)
        box_target_not_response = box_target[coord_not_response_mask].view(-1, 5)
        box_target_not_response[:,4]= 0
        #not_contain_loss = F.mse_loss(box_pred_not_response[:,4], box_target_not_response[:,4],size_average=False)
        #not_contain_loss = self.blief_loss(box_pred_not_response[:, 4], box_target_not_response[:, 4])
        # compute class prediction loss
        #class_loss = F.mse_loss(class_pred, class_target, size_average=False)
        class_loss = self.kind_loss(class_pred, class_target.long())

        # compute total loss
        total_loss = self.l_coord * loc_loss + contain_loss + self.l_noobj * noobj_loss + class_loss
        #return total_loss
        return class_loss+contain_loss+self.l_coord * loc_loss #+not_contain_loss

if __name__ == "__main__":
    input = torch.randn(4, 3, 160, 160)
    net = YOLO(2)
    net.train()
    output = net(input)
