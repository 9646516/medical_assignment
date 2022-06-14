from collections import OrderedDict

import torch.nn.functional
import torchvision
from torchvision.ops import FeaturePyramidNetwork


class model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torchvision.models.resnet50(pretrained=True)
        self.backbone.avgpool = None
        self.backbone.fc = None

        self.feat = [256, 512, 1024, 2048]
        self.fpn = FeaturePyramidNetwork(self.feat, 256)

        self.head = torch.nn.Sequential(
            torch.nn.Flatten(start_dim=1),
            torch.nn.Linear(256 * 16 * 16, 2048),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(2048, 2048),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(2048, 4),
        )

    def forward(self, x):
        x = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(x)

        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        f1 = self.backbone.layer1(x)
        f2 = self.backbone.layer2(f1)
        f3 = self.backbone.layer3(f2)
        f4 = self.backbone.layer4(f3)

        x = OrderedDict()
        x['0'] = f1
        x['1'] = f2
        x['2'] = f3
        x['3'] = f4

        x = self.fpn(x)
        F = x['3']
        F = self.head(F)
        return F
