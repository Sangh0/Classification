import argparse
import time
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from model import AlexNet

@torch.no_grad()
def eval(model, dataset, loss_func=nn.CrossEntropyLoss()):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.eval()
    batch_loss, batch_acc = 0, 0
    pbar = tqdm(enumerate(dataset), total=len(dataset))
    
    start = time.time()
    for batch, (images, labels) in pbar:
        images, labels = images.to(device), labels.to(device)
    
        outputs = model(images)
        loss = loss_func(outputs, labels)
        output_index = torch.argmax(outputs, dim=1)
        acc = (output_index==labels).sum()/len(outputs)

        batch_loss += loss.item()
        batch_acc += acc.item()

        del images; del labels; del outputs
        torch.cuda.empty_cache()

    end = time.time()

    print(f'\nTotal time for testing is {end-start:.2f}s')
    print(f'\nAverage loss: {batch_loss/(batch+1):.3f}  accuracy: {batch_acc/(batch+1):.3f}')
    return {
        'loss': batch_loss/(batch+1),
        'accuracy': batch_acc/(batch+1),
    }

def get_args_parser():
    parser = argparse.ArgumentParser(description='Evaluating AlexNet', add_help=False)
    parser.add_argument('--data_dir', type=str, required=True,
                        help='directory where your dataset is located')
    parser.add_argument('--weight', type=str, required=True,
                        help='load weight file of trained model')
    parser.add_argument('--num_classes', type=int, required=True,
                        help='class number of dataset')
    parser.add_argument('--img_size', type=int, default=224,
                        help='image size used when training')
    return 

def main(args):
    transforms_ = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    test_folder = ImageFolder(
        root=args.data_dir+'/test',
        transform=transforms_,
    )

    test_loader = DataLoader(
        test_folder,
        batch_size=1,
        shuffle=False,
        drop_last=False,
    )

    model = AlexNet(num_classes=args.num_classes)
    model.load_state_dict(torch.load(args.weight))

    result = eval(model, test_loader)