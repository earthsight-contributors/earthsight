import torch.nn as nn
from torchvision.models import efficientnet_b0, efficientnet_b1, efficientnet_b2, efficientnet_b3, efficientnet_b4, efficientnet_b5, efficientnet_b6, efficientnet_b7

class EfficientNet(nn.Module):
    def __init__(self, num_classes, backbone_model):
        super(EfficientNet, self).__init__()
        if backbone_model == 'b0':
            model = efficientnet_b0(weights="DEFAULT")
        elif backbone_model == 'b1':
            model = efficientnet_b1(weights="DEFAULT")
        elif backbone_model == 'b2':
            model = efficientnet_b2(weights="DEFAULT")
        elif backbone_model == 'b3':
            model = efficientnet_b3(weights="DEFAULT")
        elif backbone_model == 'b4':
            model = efficientnet_b4(weights="DEFAULT")
        elif backbone_model == 'b5':
            model = efficientnet_b5(weights="DEFAULT")
        elif backbone_model == 'b6':
            model = efficientnet_b6(weights="DEFAULT")
        elif backbone_model == 'b7':
            model = efficientnet_b7(weights="DEFAULT")
        else:
            raise ValueError(f"Unsupported backbone model: {backbone_model}")
        self.backbone = model.features
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), 
            nn.Flatten(),  
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(self.backbone[-1].out_channels, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x