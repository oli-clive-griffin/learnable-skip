import torch
import torch.nn as nn
from torch import Tensor
import torchvision.models as models
from typing import Union, Type, Callable, Optional, List


class LearnableSkipLayer(nn.Module):
    def __init__(self, channels, activation=nn.ReLU()):
        super(LearnableSkipLayer, self).__init__()
        self.transform = nn.Sequential(
            nn.Linear(channels, channels),
            activation
        )
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        a = self.transform(x)

        asdf = a * self.alpha

        asdf2 = x * (1 - self.alpha)

        return asdf + asdf2

class SimpleLearnableSkipNet(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleLearnableSkipNet, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 32 * 3, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.output = nn.Linear(256, num_classes)
        self.activation = nn.ReLU()
        self.ls1 = LearnableSkipLayer(256)
        self.ls2 = LearnableSkipLayer(256)

    def forward(self, x):
        x = self.flatten(x)
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.ls1(x)  # Learnable skip connection 1
        x = self.activation(self.fc3(x))
        x = self.ls2(x)  # Learnable skip connection 2
        x = self.output(x)
        return x

class SimpleSkipNet(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleSkipNet, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 32 * 3, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.output = nn.Linear(256, num_classes)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.flatten(x)
        x1 = self.activation(self.fc1(x))
        x2 = self.activation(self.fc2(x1))
        x = x1 + x2  # Skip connection 1
        x3 = self.activation(self.fc3(x))
        x = x + x3  # Skip connection 2
        x = self.output(x)
        return x

class LearnableSkipBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(LearnableSkipBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        out = nn.ReLU(inplace=True)(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        skip = self.shortcut(x)
        out = self.alpha * out + (1 - self.alpha) * skip
        out = nn.ReLU(inplace=True)(out)
        return out

# class LearnableSkipResNet18(nn.Module):
#     def __init__(self, block, num_classes=1000):
#         super(LearnableSkipResNet18, self).__init__()
#         self.in_planes = 64

#         self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.layer1 = self._make_layer(block, 64, 2, stride=1)
#         self.layer2 = self._make_layer(block, 128, 2, stride=2)
#         self.layer3 = self._make_layer(block, 256, 2, stride=2)
#         self.layer4 = self._make_layer(block, 512, 2, stride=2)
#         self.linear = nn.Linear(512 * block.expansion, num_classes)

#     def _make_layer(self, block, planes, num_blocks, stride):
#         strides = [stride] + [1] * (num_blocks - 1)
#         layers = []
#         for stride in strides:
#             layers.append(block(self.in_planes, planes, stride))
#             self.in_planes = planes * block.expansion
#         return nn.Sequential(*layers)

#     def forward(self, x):
#         out = nn.ReLU(inplace=True)(self.bn1(self.conv1(x)))
#         out = self.layer1(out)
#         out = self.layer2(out)
#         out = self.layer3(out)
#         out = self.layer4(out)
#         out = nn.AdaptiveAvgPool2d((1, 1))(out)
#         out = out.view(out.size(0), -1)
#         out = self.linear(out)
#         return out

# def learnable_skip_resnet18(num_classes=1000):
#     return LearnableSkipResNet18(LearnableSkipBasicBlock, num_classes=num_classes)

