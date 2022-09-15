import torch
import torch.nn as nn

class AlexNet(nn.Module):
    
    def __init__(
        self,
        in_dim=3,
        filters=[96, 256, 384, 384, 256],
        num_classes=1000,
        dropout=0.5,
    ):
        super(AlexNet, self).__init__()
        
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