import torch
import torch.nn as nn
import torchvision.models as models


class AlexNet(nn.Module):

    def __init__(
        self,
        num_classes=1000,
    ):
        super().__init__()
        self.model = models.alexnet(pretrained=True)
        self.model.classifier[-1] = nn.Linear(4096, num_classes)

    def forward(self, x):
        return self.model(x)