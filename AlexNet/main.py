import argparse

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dset
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchsummary import summary

from dataset import CustomDataset
from model import get_alexnet
from train import TrainModel

def get_args_parser():
    parser = argparse.ArgumentParser(description='Training AlexNet', add_help=False)
    parser.add_argument('--use_benchmark', type=bool, required=True,
                        help='if you use benchmark (ex: cifar10), then write True, or False')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='directory where your dataset is located, write it if you set use_benchmakr to False')
    parser.add_argument('--pretrained', type=bool, default=True,
                        help='use pretrained model')
    parser.add_argument('--resize_size', type=int, default=256,
                        help='image resize size before applying cropping')
    parser.add_argument('--crop_size', type=int, default=224,
                        help='image crop size after resizing image')
    parser.add_argument('--lr', default=1e-2, type=float,
                        help='learning rate')
    parser.add_argument('--epochs', default=90, type=int,
                        help='Epochs for training model')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Batch Size for training model')
    parser.add_argument('--weight_decay', default=5e-4, type=float,
                        help='weight decay of optimizer SGD')
    parser.add_argument('--num_classes', type=int, default=10,
                        help='class number of dataset')
    parser.add_argument('--lr_scheduling', default=True, type=bool,
                        help='apply learning rate scheduler')
    parser.add_argument('--check_point', default=True, type=bool,
                        help='save weight file when achieve the best score in validation phase')
    parser.add_argument('--train_log_step', type=int, default=40,
                        help='print log of iteration in training loop')
    parser.add_argument('--valid_log_step', type=int, default=10,
                        help='print log of iteration in validating loop')
    return parser

def main(args):
    assert (args.use_benchmark==True and args.data_dir is None) or \
        (args.use_benchmark==False and args.data_dir is not None)

    # Load dataset and Apply augmentation options
    train_transforms_ = transforms.Compose([
        transforms.Resize((args.resize_size, args.resize_size)),
        transforms.RandomCrop((args.crop_size, args.crop_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    valid_transforms_ = transforms.Compose([
        transforms.Resize((args.crop_size, args.crop_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    if args.use_benchmark:
        train_data = dset.CIFAR10(
            root='./data', 
            train=True, 
            download=True, 
            transform=train_transforms_
        )

        train_loader = DataLoader(
            train_data,
            batch_size=16,
            shuffle=True,
            drop_last=True,
        )

        test_data = dset.CIFAR10(
            root='./data',
            train=False,
            download=True,
            transform=valid_transforms_,
        )

        test_loader = DataLoader(
            test_data,
            batch_size=16,
            shuffle=False,
            drop_last=True,
        )

    else:
        train_folder = ImageFolder(
            root=args.data_dir+'/train',
            transform=train_transforms_,
        )

        valid_folder = ImageFolder(
            root=args.data_dir+'/valid',
            transform=valid_transforms_,
        )

        train_loader = DataLoader(
            train_folder,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
        )

        valid_loader = DataLoader(
            valid_folder,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
        )

    # Load AlexNet and check summary
    if args.pretrained:
        if args.use_benckmark:
            alexnet = get_alexnet(pretrained=True, num_classes=10)
        else:
            alexnet = get_alexnet(pretrained=True, num_classes=args.num_classes)
    
    else:
        alexnet = get_alexnet(pretrained=False, num_classes=args.num_classes)

    summary(alexnet, (3, args.crop_size, args.crop_size), device='cpu')

    model = TrainModel(
        model=alexnet,
        lr=args.lr,
        epochs=args.epochs,
        weight_decay=args.weight_decay,
        lr_scheduling=args.lr_scheduling,
        check_point=args.check_point,
        train_log_step=args.train_log_step,
        valid_log_step=args.valid_log_step,
    )
    
    # Train model
    history = model.fit(train_loader, valid_loader)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('AlexNet training', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)