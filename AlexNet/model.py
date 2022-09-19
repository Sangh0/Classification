import torch
import torch.nn as nn
import torchvision.models as models

class AlexNet_self(nn.Module):
    
    def __init__(
        self,
        in_dim=3,
        filters=[96, 256, 384, 384, 256],
        num_classes=1000,
        dropout=0.5,
    ):
        super(AlexNet_self, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(in_dim, filters[0], kernel_size=11, stride=4, padding=2),
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            nn.Conv2d(filters[0], filters[1], kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2), 
            
            nn.Conv2d(filters[1], filters[2], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            
            nn.Conv2d(filters[2], filters[3], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            
            nn.Conv2d(filters[3], filters[4], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(filters[4]*6*6, 4096),
            nn.ReLU(),
            nn.Dropout(dropout),
        
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(dropout),
        
            nn.Linear(4096, num_classes),
        )

        self._init_weights_()
        
    def forward(self, x):
        B = x.size(0)
        x = self.features(x)
        x = x.view(B, -1)
        x = self.classifier(x)
        return x

    def _init_weights_(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                nn.init.constant_(m.bias, 1)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 1)
        
        nn.init.constant_(self.features[0].bias, 0)
        nn.init.constant_(self.features[8].bias, 0)

class AlexNet_pretrained(nn.Module):

    def __init__(
        self,
        num_classes=1000,
    ):
        super().__init__()
        self.model = models.alexnet(pretrained=True)
        self.model.classifier[-1] = nn.Linear(4096, num_classes)

    def forward(self, x):
        return self.model(x)


def get_alexnet(
    num_classes: int, 
    pretrained: bool=True, 
    in_dim: int=None,
    dropout: float=None, 
    filters: list=None,
):
    if pretrained==True:
        if (in_dim, dropout, filters) is not None:
            raise ValueError('if pretrained parameter is True, then other parameters must be None')
        else:
            model = AlexNet_pretrained(num_classes=num_classes)
    
    else:
        model = AlexNet_self(num_classes=num_classes)

    return model