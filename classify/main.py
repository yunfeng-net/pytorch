from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from utils import *

def main():
    args = parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 2, 'pin_memory': True} if use_cuda else {}
    # Image preprocessing modules
    transform = transforms.Compose([
        transforms.Pad(4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32),
        transforms.ToTensor()])
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('~/Download', train=True, download=True,
                       transform=transform),  batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('~/Download', train=False, transform=transforms.ToTensor()),
        batch_size=args.test_batch_size, **kwargs)
    if args.model=="vgg":
        from vgg import model
    elif args.model=="resnet":
        from resnet import model
    elif args.model=="inception":
        from inception import model
    if args.load:
        model = torch.load(args.load)
    model = model.to(device)
    print('# generator parameters:', sum(param.numel() for param in model.parameters()))
    #optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer,epoch)
        test(args, model, device, test_loader)

    s = 'checkpoints/{}_{}.pt'.format(args.model,"cifar_100")
    torch.save(model, s)
        
if __name__ == '__main__':
    main()
