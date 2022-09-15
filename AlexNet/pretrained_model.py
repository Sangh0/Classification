import torch.nn as nn
import torchvision.models as models

class AlexNet(nn.Module):

    def __init__(
        self,
        num_classes,
        pretrained=True,
    ):
        super().__init__()
        model = models.alexnet(pretrained=pretrained)
        self.features = model.features
        self.avgpool = model.avgpool
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256*6*6, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes),
        )

    def forward(self, inputs):
        x = self.features(inputs)
        x = self.avgpool(x)
        x = x.view(inputs.size(0), -1)
        x = self.classifier(x)
        return x