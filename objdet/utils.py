import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import visdom
from datetime import datetime

def measure(results, compute_ap, classes):
    # input: [confidence,gt,kind,num]
    # return: mAP
    
    ap = 0
    for i in range(1,classes):
        data = None
        tp = None
        num = 0
        for result in results:
            mask = result[2]==i

            #print(result[0].shape,result[0][mask].shape)
            d = result[0][mask]
            if len(d)==0:
                continue
            d = np.expand_dims(d,-1)
            e = result[1][mask]
            #print("kind,d,e", i,d,e)
            e = np.expand_dims(e,-1)
            num += result[3][i]
            #print(d.shape)
            if data is None:
                data = d
                tp = e
            else:
                data = np.vstack((data,d))
                #print(tp.shape,e.shape)
                tp = np.vstack((tp,e))
                #print(tp.shape)
        if data is None:
            continue
        #print("data",data)
        index = np.argsort(-data,0)
        #print(index.shape,data.shape,gt.shape,index[0:10],gt[:100])
        tp = np.squeeze(tp[index],-1)
        fp = np.ones(tp.shape)
        fp = fp-tp
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        recall = tp / float(num)
        # avoid divide by zero in case the first detection matches a difficult
        # ground truth
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        #print("tp ", num, prec, recall)
        ap += compute_ap(prec,recall)

    return ap/(classes-1)

def compute(input, label, model, criterion, optimizer=None):
    if optimizer:
        optimizer.zero_grad()
    output = model(input)
    loss = criterion(output, label)

    if optimizer:
        loss.backward()
        optimizer.step()
    return loss.item(),output

def get_time(prev_time):
    cur_time = datetime.now() # FPS
    t = (cur_time - prev_time).seconds
    h, remainder = divmod(t, 3600)
    m, s = divmod(remainder, 60)
    time_str = "Time %02d:%02d:%02d" % (h, m, s)
    return time_str, cur_time

def train(vis, network,test_dataloader, train_dataloader,criterion,ap,num_class,opt):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    network = network.to(device)
    optimizer = optim.Adam(network.parameters(), lr=opt.lr)
    #optimizer = optim.SGD(network.parameters(), lr=1e-3, momentum=0.9,weight_decay=2e-4)

    all_train_loss = []
    all_mAP = []

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
            #if np.mod(index, 100) == 0:
            #    print('epoch {}, {}/{},train loss is {}'.format(epo, index, len(train_dataloader), iter_loss))
        train_loss /= len(train_dataloader)
        all_train_loss.append(train_loss)
        if vis:
            vis.line(all_train_loss, win='train_loss',opts=dict(title='train loss'))
            #print(sample.permute(0, 2, 3, 1).shape)
            

        time_str, prev_time = get_time(prev_time)
        print("epoch {}, averge train loss: {}, {}".format(epo,train_loss,time_str))

        if np.mod(epo+1, 5) == 0:
            s = 'checkpoints/{}_voc_{}.pt'.format(opt.model,epo)
            torch.save(network, s)
            #print('saveing {}'.format(s))
            result_list = []
            network.eval()
            with torch.no_grad():
                for index, (sample, label) in enumerate(test_dataloader):
                    sample = sample.to(device)
                    output = network(sample)
                    result = criterion.post_process(output, label)
                    if result is not None:
                        result_list.append(result)
                score = measure(result_list, ap, num_class)*100

            time_str, prev_time = get_time(prev_time)
            print("saveing {}, mAP: {:2.2f}%, {}".format(s,score,time_str))
            all_mAP.append(score)
            if vis:
                vis.line(all_mAP, win='mAP',opts=dict(title='mean AP %%'))
                #vis.images(sample[:4, :, :, :], win='target', opts=dict(title='target'))


