# resnet56_imagenet.py
import torch.nn as nn
import torch.hub

class ResNet56_ImageNet(nn.Module):
    def __init__(self, num_classes:int=10):
        super().__init__()

        # 1) load CIFAR-ResNet56 to grab its conv1.out_channels
        cifar_r56 = torch.hub.load(
            "chenyaofo/pytorch-cifar-models",
            "cifar10_resnet56",
            pretrained=False
        )
        stem_ch = cifar_r56.conv1.out_channels  # == 16

        # 2) build an ImageNet-style stem that lands in 16 channels
        self.stem = nn.Sequential(
            nn.Conv2d(3, stem_ch, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(stem_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # 3) remove the CIFAR “tiny‐stem”
        cifar_r56.conv1   = nn.Identity()
        cifar_r56.bn1     = nn.Identity()
        cifar_r56.relu    = nn.Identity()
        cifar_r56.maxpool = nn.Identity()

        # 4) swap in new classifier
        in_feats = cifar_r56.fc.in_features
        cifar_r56.fc = nn.Linear(in_feats, num_classes)

        self.backbone = cifar_r56

    def forward(self, x):
        x = self.stem(x)
        return self.backbone(x)
