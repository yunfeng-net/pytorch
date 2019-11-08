
import argparse
import torch
from utils import *
from blocks import fix_module

parser = argparse.ArgumentParser()
parser.add_argument('--load')  
parser.add_argument('--eval')  
parser.add_argument('--lr', type=float,default=1e-3)  
parser.add_argument('--momentum', type=float, default=0.7)   
parser.add_argument('--model', default='YOLO') 
parser.add_argument('--e', type=int, default=50)
parser.add_argument('--visdom', default=False, type=str2bool,
                    help='Use visdom for loss visualization')
opt = parser.parse_args()

if __name__ == "__main__":
    

    from VOC import test_dataloader, train_dataloader,num_class
    if opt.load:
        model = torch.load(opt.load)
    elif opt.model=='YOLO':
        from models import YOLO
        model = YOLO(num_class)
    else:
        print("the model name is wrong {}".format(opt.model))
        exit(-1)
    if opt.eval:
        print("not ready")
    else:
        train(model,test_dataloader, train_dataloader,num_class,opt)
