import torch
import torch.nn as nn

class AlexNet(nn.Module):
    
    def __init__(
        self,
        in_dim=3,
        num_classes=1000,
    ):
        super(AlexNet, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(in_dim, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2), 
            
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(256*6*6, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
        
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
        
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