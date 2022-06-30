import os
import math
import torch
import torch.nn as nn
import torchvision.models
from .resnet import resnet18, resnet34, resnet50, resnet101, resnet152
__all__ = ['SwitchNet', 'switchnet18', 'switchnet34', 'switchnet50', 'switchnet101', 'switchnet152']


class SwitchNet(nn.Module):

    def __init__(self, backbone='resnet18', num_classes=1000):
        super(SwitchNet, self).__init__()
        if backbone == 'resnet18':
            self.resnet = resnet18(pretrained=True, num_classes=num_classes)
        elif backbone == 'resnet34':
            self.resnet = resnet34(pretrained=True, num_classes=num_classes)
        elif backbone == 'resnet50':
            self.resnet = resnet50(pretrained=True, num_classes=num_classes)
        elif backbone == 'resnet101':
            self.resnet = resnet101(pretrained=True, num_classes=num_classes)
        elif backbone == 'resnet152':
            self.resnet = resnet152(pretrained=True, num_classes=num_classes)

        # Fixed some layers
        self.resnet.layer1.requires_grad = True
        self.resnet.layer2.requires_grad = False
        self.resnet.layer3.requires_grad = False
        self.resnet.layer4.requires_grad = False

        self.pre_proc = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu,
            self.resnet.maxpool
        )
        self.backbone = nn.Sequential(
            self.resnet.layer2,
            self.resnet.layer3,
            self.resnet.layer4
        )
        self.head = nn.Sequential(
            self.resnet.avgpool,
            nn.Flatten(),
            self.resnet.fc
        )
        # There are two modes: one is Sensitive and one is Robustness
        self.switch_layer = nn.Sequential(
            nn.Embedding(2, 784),
            nn.Unflatten(1, (1, 28, 28)),
            nn.Conv2d(1, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            # 56
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        for m in self.switch_layer:
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x, mode):
        mode = self.switch_layer(mode)
        x = self.pre_proc(x)
        x = self.resnet.layer1(x+mode)
        with torch.no_grad():
            x = self.backbone(x)
        x = self.head(x)
        return x


def switchnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SwitchNet(backbone='resnet18', num_classes=1000)

    return model


def switchnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SwitchNet(backbone='resnet34', num_classes=1000)

    return model


def switchnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SwitchNet(backbone='resnet50', num_classes=1000)

    return model


def switchnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SwitchNet(backbone='resnet101', num_classes=1000)

    return model


def switchnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SwitchNet(backbone='resnet152', num_classes=1000)

    return model
