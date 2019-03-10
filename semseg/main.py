
import argparse
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--load')  
parser.add_argument('--eval')  
parser.add_argument('--data', default='bag')  
parser.add_argument('--model', default='FCN') 
parser.add_argument('--b', type=int, default=24) 
parser.add_argument('--e', type=int, default=400)
opt = parser.parse_args()

if __name__ == "__main__":

    if opt.data=='bag':
        from BagData import test_dataloader, train_dataloader,num_class
    elif opt.data=='CamVid':
        from CamVid import test_dataloader, train_dataloader,num_class
    elif opt.data=='VOC2007':
        from VOC import test_dataloader, train_dataloader,num_class
    elif opt.data=='VOC2012':
        from VOC import test_dataloader, train_dataloader,num_class
    if opt.model=='FCN':
        from FCN import FCNs
        model = FCNs(num_class)
    elif opt.model=='SegNetX':
        from models import SegNet
        model = SegNet(num_class)
    elif opt.model=='PSPNetX':
        from models import PSPNet
        model = PSPNet(num_class)
    elif opt.model=='SegNet':
        from network import SegNet
        model = SegNet(num_class)
    else:
        print("the model name is wrong {}".format(opt.model))
        exit(-1)
    if opt.eval:
        print("not ready")
    else:
        train(model,test_dataloader, train_dataloader,num_class,load=opt.load,epo_num=50)