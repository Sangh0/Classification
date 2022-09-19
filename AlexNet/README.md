# AlexNet Implementation  

### Paper Link : [AlexNet](https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)

### [Paper Review](https://github.com/Sangh0/Classification/blob/main/AlexNet/alexnet_paper_review.ipynb)

### AlexNet Architecture  
<img src = "https://github.com/Sangh0/Classification/blob/main/AlexNet/figure/figure2.png?raw=true" width=600>

### Install and Prepare environment on anaconda
```
conda create -n alexnet python=3.8
conda activate alexnet
cd AlexNet
pip install -r requirements.txt
```

### dataset directory guide
```
path : dataset/

├── train
│    ├─ class 0
│       ├─ image1.jpg
│       ├─ ...
│    ├─ class 1
│       ├─ image3.jpg
│       ├─ ...
│    ├─ ...
│       ├─ image9.jpg
│       ├─ ...
├── valid
│    ├─ class 0
│       ├─ image11.jpg
│       ├─ ...
│    ├─ class 1
│       ├─ image13.jpg
│       ├─ ...
│    ├─ ...
│       ├─ image19.jpg
│       ├─ ...
├── test
│    ├─ class 0
│       ├─ image111.jpg
│       ├─ ...
│    ├─ class 1
│       ├─ image113.jpg
│       ├─ ...
│    ├─ ...
│       ├─ image119.jpg
│       ├─ ...
```

### Train
```
usage: main.py [-h] [--use_benchmark USE_BENCHMARK] [--data_dir DATA_DIR] [--pretrained PRETRAINED] \
               [--resize_size RESIZE_SIZE] [--crop_size CROP_SIZE] [--lr LR] [--epochs EPOCHS] \
               [--batch_size BATCH_SIZE] [--weight_decay WEIGHT_DECAY] [--num_classes NUM_CLASSES] \
               [--lr_scheduling LR_SCHEDULING] [--train_log_step TRAIN_LOG_STEP] [--valid_log_step VALID_LOG_STEP]

example: python main.py --use_benchmark True --pretrained True --batch_size 32 --train_log_step 50 --valid_log_step 20
```

### Evaluate
```
usage: eval.py [-h] [--use_benchmark USE_BENCHMARK] [--data_dir DATA_DIR] [--weight WEIGHT] \
               [--num_classes NUM_CLASSES] [--img_size IMG_SIZE]

example: python eval.py --use_benchmark True --weight ./weights/best_weight.pt
```

### Run on Jupyter Notebook for training model
```python
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dset
from torch.utils.data import DataLoader
from torchsummary import summary

from dataset import CustomDataset
from model import get_alexnet
from train import TrainModel

Config = {
    'resize_size': 256,
    'crop_size': 224,
    'batch_size': 32,
    'num_classes': 10,
    'lr': 1e-2,
    'epochs': 90,
    'weight_decay': 5e-5,
    'lr_scheduling': True,
    'check_point': True,
    'train_log_step': 50,
    'valid_log_step': 20,
}

train_transforms_ = transforms.Compose([
    transforms.Resize((Config['resize_size'], Config['resize_size'])),
    transforms.RandomCrop((Config['crop_size'], Config['crop_size'])),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

valid_transforms_ = transforms.Compose([
    transforms.Resize((Config['crop_size'], Config['crop_size'])),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

train_data = dset.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=train_transforms_,
)

test_data = dset.CIFAR10(
    root='./data',
    train=False,
    download=True,
    transform=valid_transforms_,
)

train_loader = DataLoader(
    train_data,
    batch_size=Config['batch_size'],
    shuffle=True,
    drop_last=True,
)

valid_loader = DataLoader(
    test_data,
    batch_size=Config['batch_size'],
    shuffle=True,
    drop_last=True,
)

# Load AlexNet and check summary
alexnet = get_alexnet(pretrained=True, num_classes=Config['num_classes'])
summary(alexnet, (3, Config['crop_size'], Config['crop_size']), device='cpu')

model = TrainModel(
    model=alexnet,
    lr=Config['lr'],
    epochs=Config['epochs'],
    weight_decay=Config['weight_decay'],
    lr_scheduling=Config['lr_scheduling'],
    check_point=Config['check_point'],
    train_log_step=Config['train_log_step'],
    valid_log_step=Config['valid_log_step'],
)

# Train model
history = model.fit(train_loader, valid_loader)
```

### Run on Jupyter Notebook to test model
```python
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from model import AlexNet
from eval import eval

Config = {
    'weight': './weights/best_weight.pt',
    'num_classes': 10,
    'img_size': 224,
}

transforms_ = transforms.Compose([
    transforms.Resize((Config['img_size'], Config['img_size'])),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

test_folder = ImageFolder(
    root=Config['data_dir']+'/test',
    transform=transforms_,
)

test_data = dset.CIFAR10(
    root='./data',
    train=False,
    download=True,
    transform=transforms_,
)

test_loader = DataLoader(
    test_data,
    batch_size=1,
    shuffle=False,
    drop_last=False,
)

model = AlexNet(num_classes=Config['num_classes'])
model.load_state_dict(torch.load(Config['weight']))

result = eval(model, test_loader)
```