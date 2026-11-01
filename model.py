import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.nn.init as init
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim

class VGG(nn.Module):
    def __init__(self, features, num_classes=10):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(128, num_classes),
        )
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))  # fixed output size
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
def get_activation(act_fn):
  if act_fn == "relu":
    return nn.ReLU(inplace=True)
  elif act_fn == "tanh":
    return nn.Tanh()
  elif act_fn == "sigmoid":
    return nn.Sigmoid()
  elif act_fn == "silu":
    return nn.SiLU()
  elif act_fn == "gelu":
    return nn.GELU()

def make_layers(cfg, batch_norm=True):
    layers = []
    in_channels = 3
    ACTIVATION="silu"
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), get_activation(ACTIVATION)]
            else:
                layers += [conv2d, get_activation(ACTIVATION)]
            in_channels = v
    return nn.Sequential(*layers)

def vgg(cfg, num_classes=10, batch_norm=True):
    return VGG(make_layers(cfg, batch_norm=batch_norm), num_classes=num_classes)

# VGG-6 configuration
cfg_vgg6 = [64, 64, 'M', 128, 128, 'M']

model = vgg(cfg_vgg6, num_classes=10, batch_norm=True)
print(model)