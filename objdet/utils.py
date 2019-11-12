import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import visdom
from datetime import datetime

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
    miu = iou(output_np,label,num_class)
    return output_np,label,pa,miu

def train(vis, network,test_dataloader, train_dataloader,criterion,num_class,opt):
    #from VOC import set_uni_size


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    network = network.to(device)
    optimizer = optim.Adam(network.parameters(), lr=opt.lr)
    #optimizer = optim.SGD(network.parameters(), lr=1e-3, momentum=0.9,weight_decay=2e-4)

    all_train_loss = []
    all_test_loss = []
    all_test_pa = []
    all_test_miou = []

    # start timing
    prev_time = datetime.now()
    for epo in range(opt.e):
        
        train_loss = 0
        network.train()
        #set_uni_size()
        for index, (sample, label) in enumerate(train_dataloader):
            # sample.shape is torch.Size([4, 3, 160, 160])
            # label.shape is torch.Size([4, 160, 160])
            sample = sample.to(device)
            #label = label.to(device)

            iter_loss, _ = compute(sample, label,network,criterion, optimizer)
            train_loss += iter_loss

            if np.mod(index, 100) == 0:
                print('epoch {}, {}/{},train loss is {}'.format(epo, index, len(train_dataloader), iter_loss))
        train_loss /= len(train_dataloader)
        all_train_loss.append(train_loss)
        if vis:
            vis.line(all_train_loss, win='train_loss',opts=dict(title='train loss'))

        test_loss = 0
        test_miou = 0
        '''
        network.eval()
        with torch.no_grad():
            for index, (sample, label) in enumerate(test_dataloader):

                sample = sample.to(device)
                #label = label.to(device)

                iter_loss, output = compute(sample, label,network,criterion)
                test_loss += iter_loss
                
                output_np,label_np, = measure(output,label,num_class)

                test_miou += miu

                if np.mod(index+epo, 90) == 0:
                    print(r'Testing... Open http://localhost:8097/ to see test result. pixel_acc:{},mIOU:{}'.format(pa,miu))
                    output_np = restore_label(output_np)
                    label_np = restore_label(label_np)
                    if vis:
                        vis.images(output_np[:, :, :, :], win='test_pred', opts=dict(title='test prediction')) 
                        vis.images(label_np[:, :, :, :], win='test_label', opts=dict(title='label'))

        cur_time = datetime.now()
        t = (cur_time - prev_time).seconds
        h, remainder = divmod(t, 3600)
        m, s = divmod(remainder, 60)
        time_str = "Time %02d:%02d:%02d" % (h, m, s)
        prev_time = cur_time
        N = len(test_dataloader)
        test_miou /= N
        test_loss /= N
        print('|%.3f|%.3f|%.3f|%.0f|'
                %(train_loss, test_loss, test_miou, t))
        all_test_miou.append(test_miou)
        all_test_loss.append(test_loss)
        if vis:
            vis.line(all_test_loss, win='test_loss',opts=dict(title='test loss'))
            vis.line(all_test_miou, win='test_mIOU',opts=dict(title='test mIOU'))
        '''
        if np.mod(epo+1, 5) == 0:
            s = 'checkpoints/{}_voc_{}.pt'.format(opt.model,epo)
            torch.save(network, s)
            print('saveing {}'.format(s))
