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

class YoloLoss(nn.Module):
    def __init__(self, n_batch, B, C, lambda_coord, lambda_noobj, use_gpu=False):
        """
        :param n_batch: number of batches
        :param B: number of bounding boxes
        :param C: number of bounding classes
        :param lambda_coord: factor for loss which contain objects
        :param lambda_noobj: factor for loss which do not contain objects
        """
        super(YoloLoss, self).__init__()
        self.n_batch = n_batch
        self.B = B # assume there are two bounding boxes
        self.C = C
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.use_gpu = use_gpu
    def forward(self, pred_tensor, target_tensor):
        """
        :param pred_tensor: [batch,SxSx(Bx5+20))]
        :param target_tensor: [batch,S,S,Bx5+20]
        :return: total loss
        """
        n_elements = self.B * 5 + self.C
        batch = target_tensor.size(0)
        target_tensor = target_tensor.view(batch,-1,n_elements)
        #print(target_tensor.size())
        #print(pred_tensor.size())
        pred_tensor = pred_tensor.view(batch,-1,n_elements)
        coord_mask = target_tensor[:,:,5] > 0
        noobj_mask = target_tensor[:,:,5] == 0
        coord_mask = coord_mask.unsqueeze(-1).expand_as(target_tensor)
        noobj_mask = noobj_mask.unsqueeze(-1).expand_as(target_tensor)

        coord_target = target_tensor[coord_mask].view(-1,n_elements)
        coord_pred = pred_tensor[coord_mask].view(-1,n_elements)
        class_pred = coord_pred[:,self.B*5:]
        class_target = coord_target[:,self.B*5:]
        box_pred = coord_pred[:,:self.B*5].contiguous().view(-1,5)
        box_target = coord_target[:,:self.B*5].contiguous().view(-1,5)

        noobj_target = target_tensor[noobj_mask].view(-1,n_elements)
        noobj_pred = pred_tensor[noobj_mask].view(-1,n_elements)

        # compute loss which do not contain objects
        if self.use_gpu:
            noobj_target_mask = torch.cuda.ByteTensor(noobj_target.size())
        else:
            noobj_target_mask = torch.ByteTensor(noobj_target.size())
        noobj_target_mask.zero_()
        for i in range(self.B):
            noobj_target_mask[:,i*5+4] = 1
        noobj_target_c = noobj_target[noobj_target_mask] # only compute loss of c size [2*B*noobj_target.size(0)]
        noobj_pred_c = noobj_pred[noobj_target_mask]
        noobj_loss = functional.mse_loss(noobj_pred_c, noobj_target_c, size_average=False)

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
            iou = self.compute_iou(box1[:, :4], box2[:, :4])
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
        contain_loss = functional.mse_loss(box_pred_response[:, 4], box_target_response[:, 4], size_average=False)
        loc_loss = functional.mse_loss(box_pred_response[:, :2], box_target_response[:, :2], size_average=False) +\
                   functional.mse_loss(box_pred_response[:, 2:4], box_target_response[:, 2:4], size_average=False)
        # 2. not response loss
        box_pred_not_response = box_pred[coord_not_response_mask].view(-1, 5)
        box_target_not_response = box_target[coord_not_response_mask].view(-1, 5)

        # compute class prediction loss
        class_loss = functional.mse_loss(class_pred, class_target, size_average=False)

        # compute total loss
        total_loss = self.lambda_coord * loc_loss + contain_loss + self.lambda_noobj * noobj_loss + class_loss
        return total_loss

if __name__ == "__main__":
    input = torch.randn(4, 3, 160, 160)
    output = YOLO(2)(input)
