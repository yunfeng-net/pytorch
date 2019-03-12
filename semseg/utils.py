import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import visdom
from datetime import datetime

def iou(pred, target,n_class):
    ious = []
    for cls in range(n_class):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = pred_inds[target_inds].sum()
        union = pred_inds.sum() + target_inds.sum() - intersection
        if union == 0:
            ious.append(float('nan'))  # if there is no ground truth, do not include in evaluation
        else:
            ious.append(float(intersection) / max(union, 1))
        # print("cls", cls, pred_inds.sum(), target_inds.sum(), intersection, float(intersection) / max(union, 1))
    return ious


def pixel_acc(pred, target):
    correct = (pred == target).sum()
    total   = (target == target).sum()
    return correct / total

def restore_label(img): # B,H,W -> B,C,H,W
    color_index = [(64, 128, 64) ,
        (192, 0, 128) ,
        (0, 128, 192) ,
        (0, 128, 64) ,
        (128, 0, 0) ,
        (64, 0, 128) ,
        (64, 0, 192) ,
        (192, 128, 64) ,
        (192, 192, 128) ,
        (64, 64, 128) ,
        (128, 0, 192) ,
        (192, 0, 64) ,
        (128, 128, 64) ,
        (192, 0, 192) ,
        (128, 64, 64) ,
        (64, 192, 128) ,
        (64, 64, 0) ,
        (128, 64, 128) ,
        (128, 128, 192) ,
        (0, 0, 192) ,
        (192, 128, 128) ,
        (128, 128, 128) ,
        (64, 128, 192) ,
        (0, 0, 64) ,
        (0, 64, 64) ,
        (192, 64, 128) ,
        (128, 128, 0) ,
        (192, 128, 192) ,
        (64, 0, 64) ,
        (192, 192, 0) ,
        (0, 0, 0) ,
        (64, 192, 0) ]
    b,h,w = img.shape
    output_img = np.zeros((b,h, w, 3), dtype=np.uint8)
    for i, color in enumerate(color_index):
        output_img[img == i, :] = color
    #print(output_img.shape)
    return output_img.transpose(0,3,1,2)

def compute(input, label, model, criterion, optimizer=None):

    if optimizer:
        optimizer.zero_grad()
    output = model(input)
    loss = criterion(output, label)

    if optimizer:
        loss.backward()
        optimizer.step()
    return loss.item(),output

def measure(output,label,num_class):
    output = torch.softmax(output,1) 

    output_np = output.cpu().detach().numpy().copy()
    output_np = np.argmax(output_np, axis=1)
    label = label.cpu().detach().numpy().copy()
    pa = pixel_acc(output_np,label)
    iu = iou(output_np,label,num_class)
    miu=np.nanmean(iu)
    return output_np,label,pa,miu

def train(fcn_model,test_dataloader, train_dataloader,num_class,opt):

    vis = visdom.Visdom()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    fcn_model = fcn_model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(fcn_model.parameters(), lr=opt.lr, momentum=opt.momentum)

    all_train_iter_loss = []
    all_test_iter_loss = []
    all_test_iter_pa = []
    all_test_iter_miou = []

    # start timing
    prev_time = datetime.now()
    for epo in range(opt.e):
        
        train_loss = 0
        fcn_model.train()
        for index, (bag, bag_msk) in enumerate(train_dataloader):
            # bag.shape is torch.Size([4, 3, 160, 160])
            # bag_msk.shape is torch.Size([4, 160, 160])
            bag = bag.to(device)
            bag_msk = bag_msk.to(device)

            iter_loss, _ = compute(bag, bag_msk,fcn_model,criterion, optimizer)
            all_train_iter_loss.append(iter_loss)
            train_loss += iter_loss

            if np.mod(index, 15) == 0:
                print('epoch {}, {}/{},train loss is {}'.format(epo, index, len(train_dataloader), iter_loss))
                vis.line(all_train_iter_loss, win='train_iter_loss',opts=dict(title='train iter loss'))

   
        test_loss = 0
        test_pa = 0
        test_miou = 0
        fcn_model.eval()
        with torch.no_grad():
            for index, (bag, bag_msk) in enumerate(test_dataloader):

                bag = bag.to(device)
                bag_msk = bag_msk.to(device)

                iter_loss, output = compute(bag, bag_msk,fcn_model,criterion)
                all_test_iter_loss.append(iter_loss)
                test_loss += iter_loss
                
                output_np,bag_msk_np,pa,miu, = measure(output,bag_msk,num_class)

                test_pa += pa
                all_test_iter_pa.append(pa)
                test_miou += miu
                all_test_iter_miou.append(miu)

                if np.mod(index, 15) == 0:
                    print(r'Testing... Open http://localhost:8097/ to see test result. pixel_acc:{},mIOU:{}'.format(pa,miu))
                    output_np = restore_label(output_np)
                    bag_msk_np = restore_label(bag_msk_np)
                    vis.images(output_np[:, :, :, :], win='test_pred', opts=dict(title='test prediction')) 
                    vis.images(bag_msk_np[:, :, :, :], win='test_label', opts=dict(title='label'))
                    vis.line(all_test_iter_loss, win='test_iter_loss', opts=dict(title='test iter loss'))
                    vis.line(all_test_iter_pa, win='test_iter_pa', opts=dict(title='test iter PA'))
                    vis.line(all_test_iter_miou, win='test_iter_miou', opts=dict(title='test iter mIOU'))


        cur_time = datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = "Time %02d:%02d:%02d" % (h, m, s)
        prev_time = cur_time
        N = len(test_dataloader)
        print('epoch train loss = %f, epoch test loss = %f, pixel-acc = %f, mIOU = %f , %s'
                %(train_loss/len(train_dataloader), test_loss/N, 
                test_pa/N, test_miou/N, time_str))

        if np.mod(epo, 5) == 0:
            s = 'checkpoints/{}_{}.pt'.format(opt.model,epo)
            torch.save(fcn_model, s)
            print('saveing {}'.format(s))
