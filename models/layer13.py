import numpy as np

import torch
import torch.nn as nn

__all__ = [ 'layer13' ]


class layer13(nn.Module):
    r"""
    13-layer CNN.
    We can choose local pooling type either of 'avg'- or 'max'-pooling.
    """
    def __init__(self, num_classes=10, ptype='max'):
        super(layer13, self).__init__()
        cstride = 2 if ptype == 'skip' else 1

        self.features = nn.Sequential(
            #conv1
            nn.Conv2d(  3, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=cstride, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            self._pooling(ptype, 128, kernel_size=2, stride=2),
            #conv2
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=cstride, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            self._pooling(ptype, 256, kernel_size=2, stride=2),
            #conv3
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(128, num_classes)

        # initialize parameters
        self._initialize_weights()

    def _pooling(self, ptype, num_features, kernel_size, stride):
        if ptype == 'max':
            pool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride)
        elif ptype == 'avg':
            pool = nn.AvgPool2d(kernel_size=kernel_size, stride=stride)
        else:
            raise ValueError("pooling type of {} is not supported.".format(ptype))

        return pool

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)