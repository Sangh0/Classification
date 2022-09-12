import argparse

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchsummary import summary

from dataset import CustomDataset
from model import AlexNet
from train import TrainModel

def get_args_parser():
    parser = argparse.ArgumentParser(description='Training AlexNet', add_help=False)
    parser.add_argument('--data_dir', type=str, required=True,
                        help='directory where your dataset is located')
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
    parser.add_argument('--num_classes', type=int, default=2,
                        help='class number of dataset')
    parser.add_argument('--lr_scheduling', default=True, type=bool,
                        help='apply learning rate scheduler')
    parser.add_argument('--train_log_step', type=int, default=40,
                        help='print log of iteration in training loop')
    parser.add_argument('--valid_log_step', type=int, default=10,
                        help='print log of iteration in validating loop')
    return parser

def main(args):
    # Load dataset and Apply augmentation options
    transforms_ = transforms.Compose([
        transforms.Resize((args.resize_size, args.resize_size)),
        transforms.RandomCrop((args.crop_size, args.crop_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    train_loader = DataLoader(
        CustomDataset(path=args.data_dir, subset='train', transforms_=transforms_),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )

    valid_loader = DataLoader(
        CustomDataset(path=args.data_dir, subset='valid', transforms_=transforms_),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )

    # Load AlexNet and check summary
    alexnet = AlexNet(num_classes=args.num_classes)
    summary(alexnet, (3, args.crop_size, args.crop_size), device='cpu')

    model = TrainModel(
        model=alexnet,
        lr=args.lr,
        epochs=args.epochs,
        weight_decay=args.weight_decay,
        lr_scheduling=args.lr_scheduling,
        train_log_step=args.train_log_step,
        valid_log_step=args.valid_log_step,
    )
    
    # Train model
    history = model.fit(train_loader, valid_loader)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('AlexNet training', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)